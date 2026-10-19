---
license: mit
library_name: pytorch
tags:
  - chemistry
  - materials
  - polymers
  - molecular-property-prediction
  - inverse-design
  - smiles
  - fire-safety
---

# poly4mer v3.0.0

A chemical language model for polymer **fire property prediction** and **inverse design**,
conditioned on the test conditions of the experiment (sample thickness and incident heat flux).

poly4mer maps a polymer pSMILES string to a single 768-dimensional latent vector `z`. From
that latent it both reconstructs the pSMILES and predicts four cone-calorimeter properties, so
the same representation supports forward prediction and gradient-based design in one space.

| | |
|---|---|
| Code | https://github.com/fishmoon1234/poly4mer (branch `dev`) |
| Backbone | [SMI-TED](https://github.com/IBM/materials) Light, frozen |
| Latent | 768 dimensions |
| Max sequence | 202 tokens, vocabulary 2393 |
| Properties | time to ignition, peak heat release rate, smoke extinction area, CO yield |
| Conditions | sample thickness (mm), incident heat flux (kW/m²) |
| License | MIT |

## Predicted properties

| Key | Property | Unit |
|---|---|---|
| `tig` | time to ignition | s |
| `pkhrr` | peak heat release rate | kW/m² |
| `sea` | smoke extinction area | m²/kg |
| `co` | carbon monoxide yield | kg/kg |

## Files

| File | Size | Destination in the repo |
|---|---:|---|
| `poly4mer_v3.ckpt` | 3.7 GB | `checkpoint/` |
| `smi-ted-Light_40.pt` | 1.1 GB | `smi_ted_light/` |
| `fire_property_heads/<property>/fold_{0..9}.ckpt` | 126 MB | `checkpoint/` — optional, for reproducing the cross-validation tables |
| `poly4mer_monotone/<property>/last_two_linear/fold_{0..9}.ckpt` | 63 MB | `poly4mer_monotone/checkpoint/` — optional, see below |

Only the first two are needed for prediction, reconstruction or design.

## What is in `poly4mer_v3.ckpt`

| Key | Contents |
|---|---|
| `star_encoder` | embeds the `*` attachment-point token |
| `autoencoder_encoder` | token embeddings → latent `z` |
| `decoder` | latent `z` → token embeddings |
| `predictor` | token embeddings → token logits (the **token** predictor, not the property one) |
| `property_regressors` | four MLP heads for tig, pHRR, SEA, CO |
| `property_stats` | standardization statistics for thickness, flux and the four targets |

> **Two kinds of "predictor".** `predictor` is the token predictor, the last stage of the
> decoder. The property predictors live under `property_regressors`.

The property heads consume the **raw** latent. Only thickness and flux are standardized.

| Head | Input | Dimension |
|---|---|---:|
| tig | `[z, thickness_norm, flux_norm]` | 770 |
| pHRR | `[z, thickness_norm, flux_norm]` | 770 |
| SEA | `z` | 768 |
| CO | `z` | 768 |

Each head emits a standardized value; the physical value is `output * y_std[k] + y_mean[k]`.

## Input format

Inputs must be **RDKit-canonical pSMILES with bare `*` attachment points**, for example
`*CC(*)c1ccccc1`. Bracketed stars such as `[*]`, and non-canonical orderings of the same
molecule, do not encode faithfully and will not reconstruct.

This is the single most common cause of a failed installation reporting near-zero
reconstruction. The repository ships `canonical_psmiles()` and a `--canonicalize` flag for
exactly this.

The model is trained on single repeat units. Multi-component strings joined by `.`, used for
blends and some thermoset formulations, do not reconstruct: the fragments come back but the
separator does not.

## Quick start

```bash
git clone -b dev https://github.com/fishmoon1234/poly4mer.git && cd poly4mer
pip install -r requirements.txt
# place poly4mer_v3.ckpt in checkpoint/ and smi-ted-Light_40.pt in smi_ted_light/

python fire_property_prediction_V3.py --smiles '*CC(*)C' --thickness 3 --flux 50
```

```
>> Predicted Tig      30.0235  s
>> Predicted pHRR   2106.3787  kW/m^2
>> Predicted SEA     0.053634  m^2/kg
>> Predicted CO      0.028990  kg/kg
```

Inverse design from a seed polymer, and a reconstruction check:

```bash
./run_design.sh
python evaluate_reconstruction_v3.py --files homo=example_pSMILES_homo.txt
```

## Performance

### Reconstruction

Canonical pSMILES in, canonical pSMILES out, exact string match.

| Set | n | Exact match | Mean Tanimoto |
|---|---:|---:|---:|
| Training corpus sample | 100,000 | 99.99% | 1.000 |
| Simulated set | 109,993 | 100.00% | 1.000 |
| Commodity homopolymers | 16 | 100.00% | 1.000 |

The 16 homopolymers include seven absent from the training corpus; all seven reconstruct
exactly, including an 87-token polyetherimide.

### Property prediction

Mean absolute relative error on experimental cone-calorimeter data, 10-fold cross-validation.

| Property | Best-5 folds | All 10 folds |
|---|---:|---:|
| tig | 15.76% | 19.11% |
| pHRR | 12.61% | 16.39% |
| SEA | 1.58% | 3.63% |
| CO | 0.93% | 3.94% |

**Read these with the protocol in mind.** The training epoch and learning rate are selected on
each fold's own test rows, so per-fold test errors are optimistically biased. Prefer the
all-10 column when a single number is needed. The repository ships the fold data, the 40
per-fold heads and `evaluate_fire_prediction_heads.py` so both columns can be checked, plus a
`--cv` mode that re-runs the training; re-training reproduces the best-5 figure to about 0.1
points for tig and 0.03 for CO.

Sanity against measurement, at 50 kW/m²:

| Polymer | Measured pHRR | Predicted |
|---|---:|---:|
| nylon 6,6 | 1226 | 1228 |
| polypropylene | 2141 | 1947 |
| PMMA | 1115 | 1144 |

## Intended use

Screening and ranking candidate polymer repeat units for fire behaviour, and exploring the
latent neighbourhood of a known polymer. Suitable for research.

**Not suitable** as a substitute for cone-calorimeter testing, for safety certification, or
for any decision where a wrong fire-performance estimate carries risk. The experimental data
behind the property heads is small, a few hundred rows, and errors on unseen polymers are tens
of percent.

## Limitations

**Inverse design: report the molecule, not the latent point.** Properties read off the
optimization path are predictions at a point in latent space, and they do not match the
predictions for the molecule that point decodes to. Across a benchmark of 17 designs the
median surviving fraction of the predicted gain was about one third, and four of 17 reversed
sign once re-encoded. Always re-encode a candidate and predict again before quoting a number.

**Design yield is modest.** On 16 commodity homopolymers the loop returned about 1.1 valid
designs per seed, and most were chemically implausible on inspection. Large, already
fire-resistant seeds such as polyetherimides yield little, because the objective terms they
already satisfy supply almost no gradient.

**Weight the objective by headroom, not importance.** On a seed already near optimal for a
property, weighting that property heavily starves the search.

**The decodable region is thin.** Latents far from a real polymer decode to unparsable strings,
and the property heads can return physically impossible values there, such as a negative heat
release rate. A trust region on the latent displacement is advisable.

**Out-of-distribution polymer databases.** Reconstruction on an external polymer database is
far below the figures above, around 56%, because the chemistry differs from the training
distribution.

## Optional: monotone heads

The repository also offers alternative tig and pHRR heads that are **hard-monotone in heat
flux** by construction: tig can only fall as flux rises, pHRR can only rise. The constraint is
structural, not a penalty, so it holds for every input. The encoder is unchanged.

Use them when a prediction feeds an optimizer and a non-physical flux trend would be costly.
On held-out polymers they are **level with**, not better than, the standard heads under
leakage-free selection, so the case for them is the guarantee rather than the accuracy.

## Citation

```bibtex
@article{liu2025harnessing,
  title={Harnessing large language models for data-scarce learning of polymer properties},
  author={Liu, Ning and Jafarzadeh, Siavash and Lattimer, Brian Y and Ni, Shuna and Lua, Jim and Yu, Yue},
  journal={Nature Computational Science},
  volume={5},
  number={3},
  pages={245--254},
  year={2025},
  publisher={Nature Publishing Group US New York}
}

@inproceedings{yin2025fake,
  title={Fake It Till You Make It: Multi-Physics Synthesis Breaks the Data Barrier in Chemical Language Models},
  author={Yin, Naiyu and Liu, Ning and Chen, Jiuzhou and Lattimer, Brian Y and Lua, Jim and Yu, Yue},
  booktitle={NeurIPS 2025 Machine Learning and the Physical Sciences Workshop}
}
```
