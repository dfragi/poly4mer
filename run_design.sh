#!/usr/bin/env bash
# Inverse design demo with poly4mer v3: five worked examples.
#
# Each run prints the seed's predicted properties, then every design found along the
# latent path with its discovery step, attachment-point count and predicted properties.
# Expected output for each seed is listed in the table below, so you can see at a glance
# whether your installation reproduces it.
#
#   seed        lr     weights tig/pHRR/SEA/CO   designs expected
#   PMMA        0.3    0.2/0.6/0.1/0.1           2   linear
#   PP          1.0    0.2/0.6/0.1/0.1           2   linear
#   NYLON       0.3    0.2/0.6/0.1/0.1           2   linear
#   BPA-ether   0.1    0.2/0.6/0.1/0.1           1   linear
#   PEI-N       0.01   1/1/1/1                   2   1 linear, 1 branch point
#
# PEI-N runs at equal weights on purpose. It already sits close to the objective's
# optimum on pHRR and SEA, so those terms supply almost no gradient; weighting them
# heavily starves the search and it yields only the branch point. Equal weights let the
# CO term, which has the most headroom, drive the path. See poly4mer_monotone/../README.
#
# Usage (from anywhere):
#   bash run_design.sh
#   PYTHON=/path/to/python bash run_design.sh
#   HEADS_CKPT=/path/to/heads.ckpt bash run_design.sh     # drive with different heads
#   ONLY="PMMA PP" bash run_design.sh                     # run a subset
#   MAX_EPOCHS=500 bash run_design.sh                     # quick smoke test, finds less
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
RESULTS_DIR=${RESULTS_DIR:-results}
MAX_EPOCHS=${MAX_EPOCHS:-5000}

# fire test conditions, shared by every run
THICKNESS=3.0      # mm
FLUX=50.0          # kW/m^2

# name  seed-file  lr  c_tig  c_pHRR  c_SEA  c_CO
RUNS="PMMA      example_pSMILES_homo.txt   0.3   0.2 0.6 0.1 0.1
PP        example_pSMILES_homo.txt   1.0   0.2 0.6 0.1 0.1
NYLON     example_pSMILES_homo.txt   0.3   0.2 0.6 0.1 0.1
BPA-ether example_pSMILES_gavit.txt  0.1   0.2 0.6 0.1 0.1
PEI-N     example_pSMILES_gavit.txt  0.01  1.0 1.0 1.0 1.0"

HEADS_ARG=()
if [ -n "${HEADS_CKPT:-}" ]; then HEADS_ARG=(--heads_ckpt "$HEADS_CKPT"); fi

echo "================================================================"
echo " poly4mer v3 inverse design: 5 examples"
echo " conditions: ${THICKNESS} mm, ${FLUX} kW/m^2, ${MAX_EPOCHS} steps"
echo "================================================================"

while read -r NAME FILE LR CT CP CS CC; do
    [ -z "${NAME:-}" ] && continue
    if [ -n "${ONLY:-}" ] && ! grep -qw -- "$NAME" <<< "$ONLY"; then continue; fi
    IDX=$(awk -F, -v n="$NAME" 'NR > 1 && $2 == n { print NR - 1; exit }' "$FILE")
    if [ -z "$IDX" ]; then echo "polymer $NAME not found in $FILE" >&2; exit 1; fi
    echo
    echo "---------------- $NAME  (lr $LR, weights $CT/$CP/$CS/$CC) ----------------"
    "$PYTHON" fire_inverse_design_v3.py \
        --init_file "$FILE" --smiles_idx "$IDX" --name "$NAME" \
        --lr_start "$LR" --max_epochs "$MAX_EPOCHS" \
        --c_tig "$CT" --c_qpua "$CP" --c_sea "$CS" --c_co "$CC" \
        --thickness "$THICKNESS" --flux "$FLUX" \
        --results_dir "$RESULTS_DIR" "${HEADS_ARG[@]}"
done <<< "$RUNS"

echo
echo "================================================================"
echo " done. designs, trajectories and property logs are under $RESULTS_DIR/"
echo "================================================================"
