#!/usr/bin/env bash
# Inverse design with poly4mer v3 for four homopolymer seeds: PU, NYLON, PMMA and PP.
#
# Each seed is read by name from example_pSMILES_homo.txt and optimized for MAX_EPOCHS
# latent steps on the weighted fire objective, at its own learning rate.
#
# Usage (from anywhere):
#   bash run_design.sh
#   HEADS_CKPT=/path/to/heads.ckpt bash run_design.sh   # use a separate property-head file
#   PYTHON=/path/to/python RESULTS_DIR=my_results bash run_design.sh
#
# By default the property heads stored in poly4mer_v3.ckpt are used. The designs found
# depend on which heads drive the objective, so a different head file will give a
# different set of candidates for the same seed and learning rate.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
INIT_FILE=example_pSMILES_homo.txt
RESULTS_DIR=${RESULTS_DIR:-results}

# objective weights: tig, pHRR, SEA, CO
C_TIG=0.2
C_QPUA=0.6
C_SEA=0.1
C_CO=0.1

# fire test conditions
THICKNESS=3.0      # mm
FLUX=50.0          # kW/m^2
MAX_EPOCHS=5000

# polymer name (as in INIT_FILE) and its learning rate
RUNS="PU 0.1
NYLON 0.3
PMMA 0.3
PP 1.0"

HEADS_ARG=()
if [ -n "${HEADS_CKPT:-}" ]; then HEADS_ARG=(--heads_ckpt "$HEADS_CKPT"); fi

while read -r NAME LR; do
    # 1-based data row of this polymer in INIT_FILE (header excluded)
    IDX=$(awk -F, -v n="$NAME" 'NR > 1 && $2 == n { print NR - 1; exit }' "$INIT_FILE")
    if [ -z "$IDX" ]; then echo "polymer $NAME not found in $INIT_FILE" >&2; exit 1; fi
    echo "=== $NAME (row $IDX), lr $LR ==="
    "$PYTHON" fire_inverse_design_v3.py \
        --init_file "$INIT_FILE" --smiles_idx "$IDX" --name "$NAME" \
        --lr_start "$LR" --max_epochs "$MAX_EPOCHS" \
        --c_tig "$C_TIG" --c_qpua "$C_QPUA" --c_sea "$C_SEA" --c_co "$C_CO" \
        --thickness "$THICKNESS" --flux "$FLUX" \
        --results_dir "$RESULTS_DIR" "${HEADS_ARG[@]}"
done <<< "$RUNS"
