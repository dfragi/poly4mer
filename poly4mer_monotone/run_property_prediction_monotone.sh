#!/usr/bin/env bash
# Monotone fire-property prediction: Tig and pHRR, hard-monotone in heat flux.
#
# The encoder comes from ../checkpoint/poly4mer_v3.ckpt; only the property heads differ
# from the main model. SEA and CO have no monotone heads, so for those use
# ../run_property_prediction.sh instead.
#
# Usage (from anywhere):
#   bash run_property_prediction_monotone.sh
#   PYTHON=/path/to/python bash run_property_prediction_monotone.sh
#   FOLD=3 ENSEMBLE=0 bash run_property_prediction_monotone.sh
#   FLUX_SWEEP=20,100,10 bash run_property_prediction_monotone.sh   # check monotonicity
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
INIT_FILE=../example_pSMILES_homo.txt
OUT_CSV=${OUT_CSV:-monotone_predictions.csv}

# fire test conditions
THICKNESS=3.0      # mm
FLUX=50.0          # kW/m^2

# head variant sub-folder; only last_two_linear ships
VARIANT=${VARIANT:-last_two_linear}

# ENSEMBLE=1 averages every fold present, which is the safer choice for a polymer
# outside the experimental set. Otherwise FOLD selects one checkpoint.
ENSEMBLE=${ENSEMBLE:-1}
FOLD=${FOLD:-0}

# polymers to predict, by name as it appears in INIT_FILE. "ALL" uses every row.
POLYMERS="PMMA PP PVC PC"

# optional: sweep flux instead of a single value, as START,STOP,STEP
FLUX_SWEEP=${FLUX_SWEEP:-}

EXTRA=()
if [ "$ENSEMBLE" = "1" ]; then EXTRA+=(--ensemble); else EXTRA+=(--fold "$FOLD"); fi
if [ -n "$FLUX_SWEEP" ]; then EXTRA+=(--flux_sweep "$FLUX_SWEEP"); fi

echo -e "PERFORMING MONOTONE PROPERTY PREDICTION.\n"

if [ "$POLYMERS" = "ALL" ]; then
    "$PYTHON" fire_property_prediction_monotone.py \
        --init_file "$INIT_FILE" --all \
        --variant "$VARIANT" \
        --thickness "$THICKNESS" --flux "$FLUX" \
        --csv "$OUT_CSV" "${EXTRA[@]}"
else
    IDX=""
    for NAME in $POLYMERS; do
        I=$(awk -F, -v n="$NAME" 'NR > 1 && $2 == n { print NR - 1; exit }' "$INIT_FILE")
        if [ -z "$I" ]; then echo "polymer $NAME not found in $INIT_FILE" >&2; exit 1; fi
        IDX="$IDX $I"
    done
    SMILES=$(for I in $IDX; do awk -F, -v r="$I" 'NR == r + 1 { print $3 }' "$INIT_FILE"; done | paste -sd,)
    echo "Running: $POLYMERS"
    "$PYTHON" fire_property_prediction_monotone.py \
        --smiles "$SMILES" \
        --variant "$VARIANT" \
        --thickness "$THICKNESS" --flux "$FLUX" \
        --csv "$OUT_CSV" "${EXTRA[@]}"
fi

echo 'ALL RUNS FINISHED.'
