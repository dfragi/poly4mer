#!/usr/bin/env bash
# Fire-property prediction with poly4mer v3.
#
# Predicts tig, pHRR, SEA and CO for a set of pSMILES at one thickness and heat flux.
# Weights come from checkpoint/poly4mer_v3.ckpt.
#
# Usage (from anywhere):
#   bash run_property_prediction.sh
#   PYTHON=/path/to/python bash run_property_prediction.sh
#   HEADS_CKPT=/path/to/heads.ckpt bash run_property_prediction.sh
#
# Edit THICKNESS, FLUX and the polymer list below to change what is predicted.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
INIT_FILE=example_pSMILES_homo.txt
OUT_CSV=${OUT_CSV:-predictions.csv}

# fire test conditions
THICKNESS=3.0      # mm
FLUX=50.0          # kW/m^2

# polymers to predict, by name as it appears in INIT_FILE.
# Set to "ALL" to predict for every row of the file instead.
POLYMERS="PMMA PP PVC PC"

HEADS_ARG=()
if [ -n "${HEADS_CKPT:-}" ]; then HEADS_ARG=(--heads_ckpt "$HEADS_CKPT"); fi

echo -e "PERFORMING PROPERTY PREDICTION.\n"

if [ "$POLYMERS" = "ALL" ]; then
    "$PYTHON" fire_property_prediction_V3.py \
        --init_file "$INIT_FILE" --all \
        --thickness "$THICKNESS" --flux "$FLUX" \
        --csv "$OUT_CSV" "${HEADS_ARG[@]}"
else
    # collect the row indices of the named polymers, then predict them in one pass
    IDX=""
    for NAME in $POLYMERS; do
        I=$(awk -F, -v n="$NAME" 'NR > 1 && $2 == n { print NR - 1; exit }' "$INIT_FILE")
        if [ -z "$I" ]; then echo "polymer $NAME not found in $INIT_FILE" >&2; exit 1; fi
        IDX="$IDX $I"
    done
    SMILES=$(for I in $IDX; do awk -F, -v r="$I" 'NR == r + 1 { print $3 }' "$INIT_FILE"; done | paste -sd,)
    echo "Running: $POLYMERS"
    "$PYTHON" fire_property_prediction_V3.py \
        --smiles "$SMILES" \
        --thickness "$THICKNESS" --flux "$FLUX" \
        --csv "$OUT_CSV" "${HEADS_ARG[@]}"
fi

echo 'ALL RUNS FINISHED.'
