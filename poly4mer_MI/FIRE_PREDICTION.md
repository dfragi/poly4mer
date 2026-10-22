# Fire-property prediction from SAE features

`predict_fire_properties.py` predicts four cone-calorimeter properties from a
polymer pSMILES, using the **sparse features of a Top-K SAE** rather than the raw
poly4mer embedding.

| property | meaning | units |
|---|---|---|
| `tig` | time to ignition | s |
| `pkhrr` | peak heat release rate | kW/m² |
| `sea` | specific extinction area | — |
| `co` | CO yield | — |

## Pipeline

```
pSMILES
  --[frozen smi-ted + star_encoder + autoencoder.encoder]-->  z ∈ R^768
  --[frozen Top-K SAE encoder]------------------------------>  c ∈ R^(alpha*768), k nonzeros
  --[linear head, + thickness/flux for tig and pkhrr]------->  property
```

Only the head is trained. The CLLM and the SAE are both frozen and loaded from
existing checkpoints.

> **Relation to the built-in heads.** `checkpoint/poly4mer_v3.0.0.ckpt` already
> carries fire-property regressors (`property_regressors`, `property_stats`) that
> run on `z` directly — 770 inputs for tig/pkhrr (`z` + normalized thickness and
> flux), 768 for sea/co. This script does **not** use them. It is the SAE-feature
> counterpart: same targets, same conditioning, `c` in place of `z`.

## Quick start

```bash
python predict_fire_properties.py --alpha 8 --k 160
python predict_fire_properties.py --alpha 8 --k 160 --solver ridge --ridge_lambda 1
python predict_fire_properties.py --properties sea co --alpha 32 --k 1000
```

First run embeds every pSMILES through the CLLM and caches the result under
`checkpoint/sae_cache/fire_psmiles_embeddings_<hash>.pt`, keyed by the content of
the SMILES list. Later runs with the same molecules reuse it, so only the first
invocation pays the cost of loading the 3.9 GB checkpoint.

## Data

| file | rows | unique pSMILES | columns used |
|---|---:|---:|---|
| `data/experimental_flux_thickness.smi` | 203 | 49 | `tig`, `pkhrr` + `thickness`, `flux` |
| `data/experimental_sea_avg.smi` | 44 | 44 | `sea` |
| `data/experimental_co_avg.smi` | 32 | 32 | `co` |

tig and pkhrr are measured per (polymer, thickness, flux), so the same pSMILES
appears on several rows with very different targets — e.g. tig 251 s at flux 35
vs 86 s at flux 50. Thickness and flux are therefore fed to the head alongside
`c`, log-scaled and standardized. Without them one input has several correct
answers and that share of the error is irreducible; `--no_conditions` shows the
difference.

sea and co are already averaged per polymer in their own files and have no
conditions. The `sea` and `co` columns inside `experimental_flux_thickness.smi`
are mostly empty (46 and 107 nulls) and are not used.

## How the head is fit

- **Features.** `c` is passed to the head **raw** — it is never centred or
  scaled, so its sparsity is preserved. Only the appended condition columns
  (thickness, flux) are standardized. They are log-transformed first by default
  (`--condition_transform log`): with a log target, a linear head can only
  express the power laws ignition physics follows (`tig ∝ flux^b · thickness^c`)
  if the conditions enter as logs too. `--condition_transform none` uses the raw
  values, still standardized.
- **Dead features dropped.** SAE features that are zero for every molecule in a
  dataset are removed first — mathematically a no-op, but it cuts 6144 columns to
  ~800–1030 and makes the solve cheap.
- **Targets.** All four properties are positive and span orders of magnitude, so
  targets are log-transformed then standardized. Predictions are mapped back to
  the original units before any error is computed.
- **No train/validation split.** Every pSMILES is used for training and the errors
  are measured in-distribution on those same rows.

### Solvers

`--solver sgd` (default) trains an `nn.Linear` with Adam. Note this is **not**
unregularized: `weight_decay` adds an L2 penalty to the gradient, so it is ridge
reached iteratively and stopped after a fixed epoch budget.

`--solver ridge` solves it exactly, with an unpenalized intercept:

```
w = (Xc' Xc + lambda I)^-1 Xc' yc        b = ybar - xbar' w
```

Because N < D here, the code uses the algebraically identical dual form, which
inverts an N×N matrix instead of D×D (203×203 rather than 24576×24576 at
alpha=32):

```
w = Xc' (Xc Xc' + lambda I_n)^-1 yc
```

## Options

| flag | default | meaning |
|---|---|---|
| `--alpha` | 8 | SAE expansion factor, D = alpha·768. Trained: 4, 8, 16, 32 |
| `--k` | 160 | SAE sparsity. Trained: 80, 100, 160, 200, 250, 500, 1000 |
| `--sae_ckpt` | — | override the `topk_sae_alpha{A}_k{K}.pt` path |
| `--properties` | all four | subset to run |
| `--solver` | `sgd` | `sgd` (Adam) or `ridge` (closed form) |
| `--epochs` / `--lr` / `--weight_decay` | 3000 / 1e-3 / 1e-2 | sgd solver |
| `--ridge_lambda` | 10.0 | ridge solver |
| `--target_transform` | `log` | `log` or `none` |
| `--condition_transform` | `log` | `log` or `none` for thickness/flux; standardized either way |
| `--no_conditions` | — | drop thickness/flux from tig and pkhrr |
| `--aggregate` | `none` | `mean` collapses repeats to one row per pSMILES |
| `--out` | — | write per-row predictions to csv |

`--alpha` and `--k` resolve to `checkpoint/topk_sae/topk_sae_alpha{A}_k{K}.pt`
and are validated against the config stored inside the checkpoint.

## Output

Console summary of MSE, RMSE and relative error per property. With `--out`, a csv
of every input row plus:

| column | meaning |
|---|---|
| `predicted` | prediction in the property's original units |
| `relative_error_pct` | per-row `|pred − y| / |y| × 100` |
| `property` | which property the row belongs to |

The headline relative error in the console is the **relative L2 norm**:

```
rel err % = ||y_pred - y_true||_2 / ||y_true||_2 x 100
```

This is a single aggregate per property, not a per-row average, so it is weighted
by magnitude and is not decomposable to individual molecules — use the
`relative_error_pct` column for that.

## Results

alpha=8, k=160, log targets, conditions on, in-distribution:

| property | n | MSE | RMSE | rel err % |
|---|---:|---:|---:|---:|
| tig | 203 | 3950.14 | 62.85 | 36.04 |
| pkhrr | 203 | 25337.6 | 159.18 | 20.41 |
| sea | 44 | 3.19e-08 | 1.78e-04 | 0.26 |
| co | 32 | 1.99e-05 | 4.46e-03 | 3.92 |

Closed-form ridge, same settings, swept over lambda:

| lambda | tig | pkhrr | sea | co |
|---|---:|---:|---:|---:|
| 0.001 | 36.11 | 20.38 | 0.00 | 0.00 |
| 0.1 | 36.11 | 20.38 | 0.10 | 0.08 |
| 1 | 36.06 | 20.40 | 0.96 | 0.81 |
| 10 | 36.08 | 20.71 | 7.03 | 7.02 |
| 100 | 46.22 | 26.00 | 24.59 | 33.44 |

## Reading the numbers

**The two property groups are not comparable.** What matters is the rank of the
design matrix, not its width. Every row for the same polymer carries an identical
code vector, so rank is capped by the number of *distinct molecules*:

| property | rows | columns | rank(X) | unique pSMILES |
|---|---:|---:|---:|---:|
| tig | 203 | 6146 | 50 | 49 |
| pkhrr | 203 | 6146 | 50 | 49 |
| sea | 44 | 6144 | 43 | 44 |
| co | 32 | 6144 | 31 | 32 |

(49 molecules + 2 condition columns − 1 for centering = 50; 44 − 1 = 43; 32 − 1 = 31.)

- **tig and pkhrr**: 50 free parameters fitting 203 observations, 4:1
  overdetermined. Nothing can interpolate them, so 36% and 20% measure how well
  `c` plus the test conditions actually explain the targets. These are the two
  numbers worth trusting.
- **sea and co**: ~43 and ~31 free parameters fitting 44 and 32 observations —
  roughly one parameter per data point. A small `lambda` lets the solution pass
  through essentially every point, so the near-zero error is the fit having enough
  freedom to interpolate. It would look much the same if `c` were random noise.

This is why the lambda sweep splits the way it does: sea and co are controlled
entirely by the penalty, while tig and pkhrr are flat until lambda=100 starts to
underfit.

A related numerical trap: for sea the smallest singular value of the centred
design sits at ~1.8e-05 against a largest of ~136 — the direction that centering
should annihilate exactly, left at floating-point residue. Ridge inverts
`sigma^2 + lambda`, so once lambda drops below `sigma^2 ~ 3e-10` that direction is
amplified by roughly `1/lambda`. The perfect interpolation at lambda=0.001 is the
fit loading weight onto rounding error. `lambda=1` is where all four properties
are simultaneously sensible.

## Caveats

- In-distribution error only, by design. These numbers are not a generalization
  estimate, and for sea and co they are close to meaningless as one — see above.
- `--aggregate mean` collapses the 203 tig/pkhrr rows to 49, putting all four
  properties on the same footing, at the cost of discarding the flux dependence.
- At `lambda = 0` exactly the dual matrix is singular and the solve fails;
  `lambda = 0.001` is already effectively interpolating.
- The sgd solver's result depends on `--epochs` as much as on `--weight_decay`,
  since early stopping acts as extra regularization. Use `--solver ridge` when you
  want the penalty to be the only thing controlling the fit.

---

# 10-fold evaluation: `predict_fire_properties_10folds.py`

Same pipeline, run over the pre-made splits in `data/10folds/`. For each property
and each fold `data_{0..9}`, a head is trained on `train.csv`, evaluated on
`test.csv` after every epoch, and the epoch with the lowest **test** relative
error is kept. The 5 folds with the lowest test error are then summarized as
mean ± std (ddof=1). Every selected head is saved.

```bash
python predict_fire_properties_10folds.py --alpha 8 --k 160
python predict_fire_properties_10folds.py --alpha 8 --k 160 --out data/fire_10folds_alpha8_k160.csv
```

## Data

| property | folder | columns | train / test per fold |
|---|---|---|---|
| tig | `tig_exp_data` | `smiles_names, smiles_canonicalized, thickness, flux, tig` | 205–206 / 22–23 |
| pkhrr | `pkhrr_exp_data` | same, target `pkhrr` | 205–206 / 22–23 |
| sea | `SEA_exp_data_avg` | `smiles_canonicalized, SEA` | 39–40 / 4–5 |
| co | `CO_exp_data_avg` | `smiles_canonicalized, CO` | 28–29 / 3–4 |

All folds across all properties draw on the same **49 unique pSMILES**, so the
script embeds once and reuses the codes everywhere.

Note the two split designs differ. The tig/pkhrr folds split by *measurement
row*: 15–21 molecules appear in both train and test of the same fold, at
different flux. The SEA/CO folds split by *molecule*: zero overlap. A tig test
error therefore measures "predict this polymer at a new flux", while a SEA test
error measures "predict a polymer never seen".

## What is fit on train only

The dead-feature mask, the condition mean/std and the target log-statistics are
all computed from `train.csv`. Nothing from `test.csv` enters the fit. The
sparse code is never scaled.

The *epoch* is chosen on test error, as specified. That is selection on the
evaluation set, and it matters most for sea and co — see below.

## Saved heads

`checkpoint/fire_heads_10folds/alpha{A}_k{K}/<property>_fold<i>.pt`, 40 per run,
~1.3 MB total. Each carries:

| key | meaning |
|---|---|
| `state_dict` | the selected head (lowest test rel err) |
| `state_dict_final` | the head at the last epoch |
| `code_mask` | which of the α·768 SAE features survived dead-feature dropping |
| `condition_cols`, `condition_transform` | `["thickness","flux"]` + `log`/`none`, or empty |
| `cond_mu`, `cond_sd` | standardization for the condition columns (identity on code) |
| `mu_y`, `sd_y`, `target_transform` | target log-statistics |
| `alpha`, `k`, `in_features`, `n_code`, `n_cond` | geometry |
| `best_epoch`, `train_rel`, `test_rel`, `test_final` | the fold's numbers |
| `train_csv`, `test_csv` | provenance |

`load_head(path)` in the script reloads one; its docstring gives the exact
inference sequence. Reloaded heads reproduce their recorded test error to
float32 precision. `--no_save` skips saving; `--save_dir` relocates it.

## Results, α=8, k=160, 3000 epochs

| property | top-5 folds | train rel % | test rel % | all-10 test rel % |
|---|---|---:|---:|---:|
| tig | [8, 1, 5, 0, 9] | 41.72 ± 5.20 | **27.69 ± 3.30** | 32.78 ± 6.73 |
| pkhrr | [4, 6, 8, 0, 7] | 22.53 ± 0.45 | **21.31 ± 4.30** | 26.40 ± 6.96 |
| sea | [0, 1, 3, 2, 8] | 23.00 ± 27.69 | **25.49 ± 9.89** | 39.78 ± 18.01 |
| co | [9, 8, 0, 7, 6] | 41.56 ± 30.32 | **20.59 ± 16.20** | 42.75 ± 25.96 |

Per-fold detail in `data/fire_10folds_alpha8_k160.csv`.

Two patterns in the per-fold table are worth knowing before tuning:

- **`test@final` vs `test rel`.** For tig/pkhrr the two are nearly equal (pkhrr
  fold 4: 16.70 vs 16.70) — epoch selection changes little. For sea/co they
  diverge badly (sea fold 9: 50.12 selected vs 283.76 at the final epoch; co
  fold 5: 70.93 vs 167.45), and `best_epoch` is often 1–16. Those runs are
  diverging and the selected number is the test set catching it early.
- **Train error above test error** on tig and co. Same cause: the best-test
  epoch arrives before the head has fit the training rows. It is not a sign of
  unusually good generalization.

## Hyperparameters and how to tune them

### SAE settings — `--alpha`, `--k`

These select which frozen SAE the codes come from; nothing is retrained. They
resolve to `checkpoint/topk_sae/topk_sae_alpha{A}_k{K}.pt`, and the available
grid is α ∈ {4, 8, 16, 32} × k ∈ {80, 100, 160, 200, 250, 500, 1000}.

- **α** sets the code width D = α·768 and so the number of candidate features.
  After dead-feature dropping on ~200 training rows the head sees ~1000 live
  features at α=8; expect roughly proportional growth with α.
- **k** sets how many features are active per molecule. Small k gives a very
  sparse, low-rank code; large k is closer to a dense embedding.

Note that the design matrix rank is capped by the number of distinct molecules
in the training split (≤ 49, see the single-split section), regardless of α.
Raising α adds columns but not independent directions, so gains from α are
bounded and the main effect is on which features are available to the head.

Sweep recipe:

```bash
for A in 4 8 16 32; do for K in 80 160 500 1000; do
  python predict_fire_properties_10folds.py --alpha $A --k $K --no_save \
    --out data/sweep/fire_10folds_alpha${A}_k${K}.csv \
    | grep -A5 "^property   top folds"
done; done
```

Each run is a couple of minutes once embeddings are cached; the α=32 runs are
slowest because of per-epoch evaluation over ~4000 features.

### Head optimization — `--epochs`, `--lr`, `--weight_decay`

The head is `nn.Linear` trained with Adam, full-batch. `weight_decay` is Adam's
coupled L2 penalty, so the solution is ridge-regularized; `epochs` acts as an
additional early-stopping regularizer because training is not run to
convergence.

| | default | guidance |
|---|---|---|
| `--epochs` | 3000 | tig/pkhrr typically select epochs 2300–3000, so they are still improving at the end; try 5000–10000. sea/co select 1–100 — more epochs will not help them. |
| `--lr` | 1e-3 | lower (1e-4) with more epochs is the first thing to try for sea/co divergence. |
| `--weight_decay` | 1e-2 | the main capacity control. sea/co have ~30–40 training rows against ~750–970 features and want more; tig/pkhrr tolerate less. Sweep 1e-3, 1e-2, 1e-1. |
| `--eval_every` | 1 | evaluate every N epochs. Raise it to speed up large-α runs; the selected epoch gets coarser. |

The sea/co regime is the hard one. With rows ≈ rank the fit can interpolate the
training data, and a fixed-step optimizer on unscaled sparse inputs can
overshoot. Treat `(lr, weight_decay)` jointly for those two properties rather
than carrying over the tig/pkhrr settings.

### Input design — `--condition_transform`, `--no_conditions`

`--condition_transform` chooses how thickness and flux enter the tig/pkhrr head.
Both options are standardized afterwards; only the transform differs:

| | tig test rel % | pkhrr test rel % |
|---|---:|---:|
| `log` (default) | **27.69 ± 3.30** | **21.31 ± 4.30** |
| `none` | 31.62 ± 4.96 | 22.97 ± 2.83 |

`log` wins on both, by ~4 points on tig where flux dependence is strongest. The
reason is functional form: ignition time is a power law in flux
(`t_ig ∝ 1/flux²` thermally thick, `∝ thickness/flux` thin). With a log target,
`log(tig) = a + b·log(flux)` recovers that; `log(tig) = a + b·flux` is an
exponential and cannot.

`--no_conditions` drops thickness and flux entirely. Because the same polymer
appears at several fluxes with very different targets, this introduces an
irreducible error floor and is only useful as a baseline.

### Target — `--target_transform`

`log` (default) fits in log space and exponentiates predictions back. All four
properties are positive and span one to two orders of magnitude, so `none` will
let the largest values dominate the loss. Keep `log` unless there is a specific
reason not to.

### Selection — `--top`, `--folds`

`--top 5` of `--folds 10` is the specification. Reporting all 10 as well
(printed above the top-5 line) shows how much the selection is doing: for co it
halves the mean (42.75 → 20.59) and the two numbers should not be quoted
interchangeably.

### What not to tune

- **Dead-feature dropping** (`--drop_dead_features`, on) is a no-op on the fit;
  it only removes all-zero columns so the head is smaller. Leave it on.
- **`--seed`** changes the head initialization per fold (`seed + fold`). Worth
  checking that a conclusion survives a different seed, but not a knob to
  optimize.
- **The SAE itself.** α and k pick a checkpoint; the SAE is never trained here.
  To try an (α, k) outside the grid, train it first with `train_topk_sae.py
  --expansion_factor A --k K`.
