# Poly4mer-SAE: A Controllable Polymer LLM for Understanding Molar Structures and Beyond

This directory contains the workflow for studying chemical structure and fire-property information in Poly4mer embeddings using Top-K sparse autoencoders (SAEs).

The workflow learns sparse features, identifies features associated with molar groups, tests their effects through zero-ablation, and trains fire-property predictors on the sparse representations.

## 1. Workflow overview

```text
Polymer pSMILES
      |
      v
Frozen Poly4mer encoder
      |
      v
Dense embedding z, dimension 768
      |
      v
Top-K sparse autoencoder
      |
      v
Sparse code c, dimension D = alpha × 768
      |
      +--> Evaluate embedding and pSMILES reconstruction
      |
      +--> Train molar-group binary classifiers
      |          |
      |          v
      |    Rank and select essential features
      |          |
      |          v
      |    Zero selected features and decode pSMILES
      |
      +--> Train linear fire-property prediction heads
```

The pretrained Poly4mer encoder stays frozen. SAE training learns a sparse representation of its embeddings. Subsequent classifier and property-head training keep the SAE frozen.

The reconstruction path is:

```text
Sparse code c → SAE decoder → reconstructed z → Poly4mer decoder → pSMILES
```

## 2. Files in this directory

| File | Purpose |
|---|---|
| `train_topk_sae.py` | Extract/cache Poly4mer embeddings and train a Top-K SAE. Optionally evaluate pSMILES reconstruction. |
| `sweep_topk_sae.sh`  | Sweep the Top-K SAE over (alpha, K) with tunable LR, epochs, and LR decay.
| `eval_sae_psmiles.py` | Evaluate reconstruction through an existing SAE checkpoint. |
| `eval_sae_dce_loss_recovery.py` | Measure decoder cross-entropy changes and loss recovery relative to clean and zero-embedding baselines. |
| `train_group_binary_classifier.py` | Train binary classifiers for fragment/molar-group presence using frozen SAE features. |
| `select_features_binary_pipeline.py` | Rank features using activation–weight scores and marginal AUROC; combine rankings and optionally fit L1-regularized heads. |
| `zero_ablate_sweep_pipeline.py` | Zero individual active, selected features and record the decoded polymer structures. |
| `predict_fire_properties.py` | Fit fire-property heads on the full experimental dataset using Adam or closed-form ridge regression. |
| `predict_fire_properties_10folds.py` | Train and evaluate property heads using the existing 10-fold splits. |
| `analyze_fire_properties.py` | Produce top-20 polymer tables from verified full-data ridge runs. |
| `README_Masterdoc.md` | Broader project documentation, checkpoint descriptions, and historical examples. |
| `FIRE_PREDICTION.md` | Detailed fire-prediction methodology and experimental discussion. |


Some helper modules and data-preparation scripts remain in the parent repository.

## 3. Environment and directory setup

Use the existing `poly4mer` environment:

```bash
conda activate poly4mer
cd poly4mer_MI
```


### Running scripts from this subdirectory

The current scripts import parent-level modules such as `models.py`, `utils1.py`, and `train_group_classifier.py`. Several also resolve data and model paths relative to their own location.


Run all examples below from the repository root.

## 4. Required inputs

| Input | Role |
|---|---|
| `checkpoint/poly4mer_v3.0.0.ckpt` | Pretrained Poly4mer encoder/decoder checkpoint. |
| `smi_ted_light/` | Frozen smi-ted weights, vocabulary, and implementation. |
| `data/PSMILES.smi` | Polymer corpus used for SAE training. |
| `data/v1_fragment_ids_correct.smi` | Fragment-labelled polymer source used by the group-analysis workflow. |
| `data/v1_fragments_ids_correct_binary.smi` | pSMILES followed by 196 group-presence columns. |
| `data/fragments_reference.csv` | Fragment/group ID reference. |
| `data/experimental_flux_thickness.smi` | Ignition and peak heat-release targets with thickness and flux. |
| `data/experimental_sea_avg.smi` | SEA targets. |
| `data/experimental_co_avg.smi` | CO targets. |
| `data/10folds/` | Existing train/test splits for property evaluation. |

### Included pretrained checkpoints

The following checkpoints are included inside `poly4mer_MI/`:

```text
poly4mer_MI/
└── checkpoint/
    ├── poly4mer_v3.0.0.ckpt
    └── top_k_sae/
        └── topk_sae_alpha4_k1000.pt
```

| Checkpoint | Purpose |
|---|---|
| `checkpoint/poly4mer_v3.0.0.ckpt` | Pretrained Poly4mer model used to encode pSMILES and decode reconstructed embeddings. |
| `checkpoint/top_k_sae/topk_sae_alpha4_k1000.pt` | Pretrained Top-K SAE with expansion factor 4 and K = 1000 |

These checkpoints allow the workflow to begin with SAE evaluation or downstream analysis without retraining the SAE. Data files, smi-ted resources, and parent-level helper modules are still required. Group classifiers and fire-property heads are trained separately.

## Directory structure

```text

poly4mer_MI/
    ├── README.md                       
    ├── FIRE_PREDICTION.md               # Detailed fire-prediction documentation
    │
    ├── train_topk_sae.py                
    ├── eval_sae_psmiles.py              
    ├── eval_sae_dce_loss_recovery.py     
    ├── train_group_binary_classifier.py 
    ├── select_features_binary_pipeline.py
    ├── sweep_topk_sae.sh 
    ├── zero_ablate_sweep_pipeline.py    
    ├── predict_fire_properties.py       
    ├── predict_fire_properties_10folds.py
    │                                   
    ├── analyze_fire_properties.py       
    │
    ├── checkpoint/
    │   ├── poly4mer_v3.0.0.ckpt         # Included Poly4mer checkpoint
    │   ├── topk_sae/
    │   │   └── topk_sae_alpha4_k1000.pt # Included pretrained SAE
    │   ├── sae_cache/                  # Generated embedding/token caches
    │   ├── group_binary_classifier/    # Generated classifiers and feature sets
    │   ├── fire_heads/                 # Generated full-data property heads
    │   └── fire_heads_10folds/          # Default output for fold-specific heads
    │
    ├── data/           -> ../data/
    ├── smi_ted_light/  -> ../smi_ted_light/
    ├── results/        -> ../results/
    └── logs/           -> ../logs/
```

## 5. Hyperparameters explored

The table separates recorded search values from fixed settings and example configurations. Listed search values do not imply that every possible combination was run.

| Stage | Parameter | Recorded values or settings | Role |
|---|---|---|---|
| SAE architecture | `--expansion_factor` | 4, 8, 16, 32 | Sets dictionary width: `D = alpha × 768`. |
| SAE sparsity | `--k` | 80, 100, 160, 200, 250, 500, 1000 | Recorded 28-model grid with the four expansion factors. |
| SAE training | `--epochs` | 50 | Documented grid-training setting. 1000 used for final training   |
| SAE training | `--lr` | `3e-4` | Documented grid-training setting. |
| Group classifier | `--epochs` | 20 by default; 1000 for  Training duration. |
| Group classifier | `--lr`, `--weight_decay` | `1e-3`, `1e-4` | Current defaults. |
| Feature selection | `--top_n` | 30 and 100 in examples | Size of the selected global feature set. |
| Feature selection | `--combine_mode` | `borda`; also supports `intersection` | Combines feature rankings. |
| L1 head fitting | `--l1_lambda` | `1e-4`, `1e-3`, `0.01`, `0.1`, `0.2`, `0.3`, `0.4` | Documented regularization values used to see number of selected features. |
| Zero-ablation | `--top_k_features` | 5 | Default number of individual feature interventions per polymer. |
| Fire-head solver | `--solver` | `sgd`, `ridge` | `sgd` uses Adam; `ridge` uses a closed-form . |
| Fire-head tuning | `--lr` | `1e-4`, `3e-4`, `1e-3` |Explored values. |
| Fire-head tuning | `--weight_decay` | `1e-3`, `1e-2`, `1e-1` | Explored values |
| Fire-head tuning | `--epochs` | 1000, 3000, 5000, 10000 | Explored values |
| Final 10-fold configuration | Learning rate / weight decay / epochs | `1e-3` / `1e-2` / 10000 | Documented final configuration applied across the SAE grid. |
| Ridge tuning | `--ridge_lambda` | `0.001`, `0.01`, `0.1`, `1`, `10`, `100` | Values explored for ridge sweep files. |
| Final full-data ridge runs | `--ridge_lambda` | `0.001` | Setting represented in `results/ridge_final/`. |


## 6. Example workflow

### Stage 1: Train a Top-K SAE

Example: expansion factor 4 and K = 1000.

```bash
python poly4mer_MI/train_topk_sae.py \
    --expansion_factor 4 \
    --k 1000 \
    --epochs 1000 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --eval_psmiles
```

for full sweep : 
```bash
EPOCHS=1000 LR=3e-5 ALPHAS="4 8 16 32" KS="80 100 160 200 250 500 1000" bash sweep_topk_sae.sh
```

Expected artifacts:

```text
checkpoint/topk_sae/topk_sae_alpha4_k1000.pt
checkpoint/topk_sae/topk_sae_alpha4_k1000.metrics.tsv
```

Embedding extraction is cached in `checkpoint/sae_cache/` and reused across SAE runs.

Compare models using validation embedding MSE, active-feature count (L0), fraction of features alive, and pSMILES reconstruction accuracy.

### Stage 2: Evaluate reconstruction and loss recovery

```bash
SAE=checkpoint/topk_sae/topk_sae_alpha4_k1000.pt

python3 eval_sae_psmiles.py   --sae_ckpt /home/poly4mer3.0/checkpoint/topk_sae/topk_sae_alpha8_k40.pt   --data_path /home/poly4mer3.0/data/PSMILES.smi   --cache_dir /home/poly4mer3.0/checkpoint/sae_cache

python poly4mer_MI/eval_sae_dce_loss_recovery.py \
    --sae_ckpt "$SAE" \
    --out_tsv checkpoint/sae_ce_metrics.tsv
```

### Stage 3: Train a molar-group binary classifier

Example: Hydroxy (OH) group indicated as group 7 in data file.

```bash
python poly4mer_MI/train_group_binary_classifier.py \
    --group_ids 7 \
    --epochs 1000 \
    --lr 1e-3 \
    --warmup_steps 200 \
```

Expected checkpoint:

```text
checkpoint/group_binary_classifier/
    group_bin_clf_topk_sae_alpha4_k1000_ids_7.pt
```

### Stage 4: Select essential features

```bash
CLF=checkpoint/group_binary_classifier/group_bin_clf_topk_sae_alpha4_k1000_ids_7.pt

python poly4mer_MI/select_features_binary_pipeline.py \
    --classifier_ckpt "$CLF" \
    --top_n 100 --combine_mode borda \
    --act_sample_size 200000 \
    --l1_lambda 0.3 --l1_lambda 1e-3 --l1_lambda 1e-2 --l1_lambda 1e-1  --l1_lambda 0.2\
    --l1_epochs 20 --save_l1_ckpts \
    > logs/select_id7_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```


### Stage 5: Test features through zero-ablation

Using borda feature directly from validation.json under checkpoint/group_binary_classifier/

```bash

python zero_ablate_sweep_pipeline.py   --group_id 7   --sae_ckpt checkpoint/topk_sae/topk_sae_alpha4_k1000.pt   --borda_indices 76,100,374,400,415,651,669,695,737,806,821,910,924,956,1012,1026,1507,1551,1777,2018,2040,2270,2272,2379,2519,2670,2817,2950,3012,3038   --limit 200 
```
or 

```bash 
VAL=checkpoint/group_binary_classifier/validation_group_bin_clf_topk_sae_alpha4_k1000_ids_7.json

python poly4mer_MI/zero_ablate_sweep_pipeline.py \
    --group_id 7 \
    --sae_ckpt "$SAE" \
    --borda_json "$VAL" \
    --borda_top_n 30 \
    --limit 100 \
    --out_csv results/zero_ablation/ablation_group7.csv
```


### Stage 6: Fit fire-property heads

| Property | Meaning | Inputs |
|---|---|---|
| `tig` | Time to ignition | SAE code + thickness + flux |
| `pkhrr` | Peak heat release rate | SAE code + thickness + flux |
| `sea` | Specific extinction area | SAE code |
| `co` | CO yield | SAE code |

#### Full-data ridge fit

```bash
mkdir -p results/ridge_final logs/ridge_final

python poly4mer_MI/predict_fire_properties.py \
    --alpha 4 \
    --k 1000 \
    --solver ridge \
    --ridge_lambda 0.001 \
    --out results/ridge_final/a4_k1000_lam0.001.csv \
    > logs/ridge_final/a4_k1000_lam0.001.log 2>&1
```

alternatively, to sweep all value 

```bash
mkdir -p results/ridge_final logs/ridge_final
for A in 4 8 16 32; do   for K in 80 100 160 200 250 500 1000; do     python predict_fire_properties.py --alpha $A --k $K       --solver ridge --ridge_lambda 0.001       --out results/ridge_final/a${A}_k${K}_lam0.001.csv       > logs/ridge_final/a${A}_k${K}_lam0.001.log;   done; done

```

The current predictor saves heads under:

```text
checkpoint/fire_heads/<property>_alpha4_k1000_ridge.pt
```

#### Existing 10-fold evaluation

```bash
mkdir -p results/10folds_final/lr1e-3_wd1e-2_ep10000

python poly4mer_MI/predict_fire_properties_10folds.py \
    --alpha 8 \
    --k 160 \
    --epochs 10000 \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --top 5 \
    --save_dir checkpoint/fire_heads_10folds_final/lr1e-3_wd1e-2_ep10000/alpha8_k160 \
    --out results/10folds_final/lr1e-3_wd1e-2_ep10000/a8_k160.csv
```



### Stage 7: Generate top-20 ridge tables

After producing the full-data ridge CSVs, logs, and saved heads:

```bash
python poly4mer_MI/analyze_fire_properties.py \
    'results/ridge_final/a*_k*_lam0.001.csv'
```

The current analyzer requires:

- Input CSVs inside a directory named `ridge_final`.
- Matching logs under `logs/ridge_final/`.
- Logs that identify completed full-data ridge runs.
- Existing SAE checkpoints and saved property heads.


Outputs are written to the current working directory:

```text
table_a2_top20_ridge_tig.csv
table_a2_top20_ridge_pkhrr.csv
table_a2_top20_ridge_sea.csv
table_a2_top20_ridge_co.csv
```


## 7. Output locations

Paths below are relative to the repository root after the shared-directory setup.

| Directory | Contents |
|---|---|
| `checkpoint/topk_sae/` | SAE checkpoints and per-epoch metrics. |
| `checkpoint/sae_cache/` | Cached dense embeddings and token IDs. |
| `checkpoint/group_binary_classifier/` | Group classifiers, feature rankings, validation JSONs, and L1 heads. |
| `checkpoint/fire_heads/` | Full-data property heads. |
| `checkpoint/fire_heads_10folds*/` | Fold-specific property heads. |
| `results/zero_ablation/` | Decoded intervention results. |
| `results/ridge_final/` | Final full-data ridge predictions. |
| `results/10folds_final/` | Final fold tables across SAE configurations. |
| `logs/` | Run logs, including provenance required by the ridge analyzer. |

