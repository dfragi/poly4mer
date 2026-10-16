#!/usr/bin/env bash
# End-to-end demo of the fire property heads.
#
#   Stage 1  score the 40 bundled per-fold heads and print the reported tables
#   Stage 2  re-train one property's 10 folds from scratch with the same recipe
#   Stage 3  score the freshly trained heads and compare them with stage 1
#
# Stage 1 takes under a minute. Stage 2 takes about 90 s per fold for CO, so roughly
# 15 minutes for the default. Set DEMO_PROPERTY=tig for the slowest (hours) or
# DEMO_FOLDS=0,3 to train just a couple.
#
# Usage (from anywhere):
#   bash run_demo_property_heads.sh
#   PYTHON=/path/to/python bash run_demo_property_heads.sh
#   DEMO_PROPERTY=sea bash run_demo_property_heads.sh
#   DEMO_FOLDS=0,1,2 bash run_demo_property_heads.sh
#   SKIP_TRAIN=1 bash run_demo_property_heads.sh          # stage 1 only
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
DEMO_PROPERTY=${DEMO_PROPERTY:-co}
DEMO_FOLDS=${DEMO_FOLDS:-0-9}
OUT_DIR=${OUT_DIR:-runs/demo_cv}

hr() { printf '=%.0s' {1..74}; echo; }

hr
echo " poly4mer v3: fire property head demo"
echo " retrain property : $DEMO_PROPERTY   folds: $DEMO_FOLDS"
hr

# ---------------------------------------------------------------- preflight --
MISSING=0
for f in checkpoint/poly4mer_v3.ckpt smi_ted_light/smi-ted-Light_40.pt; do
    [ -e "$f" ] || { echo "!! missing $f"; MISSING=1; }
done
N_HEADS=$(ls checkpoint/fire_property_heads/*/fold_*.ckpt 2>/dev/null | wc -l)
if [ "$N_HEADS" -ne 40 ]; then
    echo "!! found $N_HEADS of 40 per-fold heads in checkpoint/fire_property_heads/"
    MISSING=1
fi
if [ "$MISSING" -ne 0 ]; then
    echo
    echo "Download the weights first; see the Pretrained models section of README.md."
    exit 1
fi
echo ">> weights present: poly4mer_v3.ckpt, smi-ted, and 40 per-fold heads"

# ------------------------------------------------- stage 1: shipped heads --
echo
hr
echo " STAGE 1  score the 40 bundled heads"
hr
echo "Each head is run on its own held-out fold. The 'recorded' column is the value"
echo "stored with the head when it was trained, so the two should agree."
echo
"$PYTHON" evaluate_fire_prediction_heads.py --csv "${OUT_DIR}_shipped.csv"

if [ -n "${SKIP_TRAIN:-}" ]; then
    echo; echo ">> SKIP_TRAIN set, stopping after stage 1."; exit 0
fi

# ------------------------------------------------------ stage 2: retrain --
echo
hr
echo " STAGE 2  re-train $DEMO_PROPERTY, folds $DEMO_FOLDS"
hr
echo "Same recipe as the original run: the per-property learning-rate grid is swept on"
echo "each fold, the epoch with the lowest error on that fold's test rows is kept, and"
echo "every head starts from the one already in poly4mer_v3.ckpt."
echo
"$PYTHON" training/train_property_predictor_finetune.py \
    --cv --properties "$DEMO_PROPERTY" --cv_folds "$DEMO_FOLDS" --out_dir "$OUT_DIR"

# ------------------------------------------- stage 3: score the new heads --
echo
hr
echo " STAGE 3  score the heads just trained"
hr
"$PYTHON" evaluate_fire_prediction_heads.py \
    --properties "$DEMO_PROPERTY" --heads_dir "$OUT_DIR" --csv "${OUT_DIR}_retrained.csv"

# --------------------------------------------------------- side by side --
echo
hr
echo " COMPARISON  $DEMO_PROPERTY, bundled vs re-trained"
hr
"$PYTHON" - "$DEMO_PROPERTY" "${OUT_DIR}_shipped.csv" "${OUT_DIR}_retrained.csv" <<'PY'
import sys, pandas as pd, numpy as np
prop, a_path, b_path = sys.argv[1], sys.argv[2], sys.argv[3]
a = pd.read_csv(a_path).query("property == @prop").set_index("fold")
b = pd.read_csv(b_path).query("property == @prop").set_index("fold")
common = sorted(set(a.index) & set(b.index))
print(f"{'fold':>5}{'bundled %':>12}{'retrained %':>14}{'diff':>9}   lr bundled / retrained")
for k in common:
    d = b.loc[k, 'test_rel'] - a.loc[k, 'test_rel']
    print(f"{k:>5}{a.loc[k,'test_rel']:>12.2f}{b.loc[k,'test_rel']:>14.2f}{d:>+9.2f}"
          f"   {a.loc[k,'best_lr']:.0e} / {b.loc[k,'best_lr']:.0e}")
for label, f in (("bundled", a), ("retrained", b)):
    t = f.loc[common, 'test_rel'].values
    top = np.sort(t)[:min(5, len(t))]
    print(f"  {label:<10} best-{len(top)} {top.mean():6.2f}%   all-{len(t)} {t.mean():6.2f}%")
if len(common) >= 2:
    d = (b.loc[common, 'test_rel'] - a.loc[common, 'test_rel']).values
    print(f"  mean |difference| over {len(common)} folds: {np.abs(d).mean():.2f} points")
print()
print("The retrained heads should land close to the bundled ones but not identically.")
print("The epoch and learning rate are selected on each fold's own test rows, and several")
print("learning rates often finish within a few tenths of a point, so which one wins is")
print("close to a tie-break. Differences of a fraction of a point are expected.")
PY

echo
hr
echo " done. CSVs: ${OUT_DIR}_shipped.csv and ${OUT_DIR}_retrained.csv"
echo " new heads: $OUT_DIR/$DEMO_PROPERTY/"
hr
