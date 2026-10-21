#!/usr/bin/env bash
# Upload the poly4mer v3 release artifacts to the HuggingFace model repository.
#
# You run this; it never runs itself. Log in first with:
#     hf auth login
#
# What it uploads, and where:
#
#   poly4mer_v3_release.ckpt   ->  poly4mer_pretrained_v3/poly4mer_v3.ckpt
#   monotone tig heads  (10)   ->  poly4mer_pretrained_v3/tig_10folds/
#   monotone pHRR heads (10)   ->  poly4mer_pretrained_v3/phrr_10folds/
#   HUGGINGFACE_MODEL_CARD.md  ->  README.md
#
# The local file is named *_release.ckpt to distinguish it from the original on this
# machine. It is uploaded under the plain name, which is what the code expects.
#
# Usage:
#   bash upload_to_huggingface.sh                 # show what would be sent, change nothing
#   CONFIRM=1 bash upload_to_huggingface.sh       # actually upload
#   CONFIRM=1 ONLY=card bash upload_to_huggingface.sh    # card only
#   CONFIRM=1 ONLY=ckpt bash upload_to_huggingface.sh    # checkpoint only
#   CONFIRM=1 ONLY=heads bash upload_to_huggingface.sh   # monotone heads only
set -euo pipefail
cd "$(dirname "$0")"

REPO=${REPO:-fishmoon1234/poly4mer}
HF=${HF:-hf}
ONLY=${ONLY:-all}

CKPT=${CKPT:-/home/office5090/Desktop/paper/poly4merv3/poly4mer_v3_release.ckpt}
MONO_SRC=${MONO_SRC:-/home/office5090/Desktop/paper/poly4mer_monotone/checkpoints}
CARD=HUGGINGFACE_MODEL_CARD.md

# sha256 of the release checkpoint this script was written against; a mismatch means the
# file was rebuilt, which is fine, but check it is the one you meant to publish
EXPECT_SHA=c31c80811049fb99b372b0400e939fd577799960d6a13ddf38b3c6f1d18fbcfc

say() { printf '%s\n' "$*"; }
hr()  { printf -- '-%.0s' {1..70}; echo; }

hr
say " target repository : $REPO"
say " scope             : $ONLY"
say " mode              : ${CONFIRM:+UPLOAD}${CONFIRM:-dry run, nothing will be sent}"
hr

# ------------------------------------------------------------- preflight --
FAIL=0
if [ "$ONLY" = all ] || [ "$ONLY" = ckpt ]; then
    if [ ! -f "$CKPT" ]; then say "!! missing $CKPT"; FAIL=1; else
        say ">> checkpoint $(du -h "$CKPT" | cut -f1)  $CKPT"
        SHA=$(sha256sum "$CKPT" | cut -d' ' -f1)
        if [ "$SHA" = "$EXPECT_SHA" ]; then
            say "   sha256 matches the verified release build"
        else
            say "   !! sha256 $SHA"
            say "      differs from the verified build $EXPECT_SHA"
            say "      the file was rebuilt; confirm it is the one you want before uploading"
        fi
    fi
fi
if [ "$ONLY" = all ] || [ "$ONLY" = heads ]; then
    for P in tig phrr; do
        N=$(ls "$MONO_SRC/$P/last_two_linear"/fold_*_best_fp32.ckpt 2>/dev/null | wc -l)
        say ">> monotone $P heads: $N of 10"
        [ "$N" -eq 10 ] || { say "   !! expected 10"; FAIL=1; }
    done
fi
if [ "$ONLY" = all ] || [ "$ONLY" = card ]; then
    [ -f "$CARD" ] && say ">> model card $CARD" || { say "!! missing $CARD"; FAIL=1; }
fi
[ "$FAIL" -eq 0 ] || { say; say "preflight failed, nothing sent."; exit 1; }

if [ -z "${CONFIRM:-}" ]; then
    say
    say "Dry run. Re-run with CONFIRM=1 to upload."
    say
    say "After replacing the checkpoint, squash the repository history so the previous"
    say "blob stops being downloadable: repository Settings -> Super-squash history."
    exit 0
fi

command -v "$HF" >/dev/null || { say "!! '$HF' not on PATH; pip install -U huggingface_hub"; exit 1; }

# --------------------------------------------------------------- upload --
if [ "$ONLY" = all ] || [ "$ONLY" = ckpt ]; then
    say; say ">> uploading the checkpoint (3.6 GiB, this takes a while)"
    "$HF" upload "$REPO" "$CKPT" poly4mer_pretrained_v3/poly4mer_v3.ckpt \
        --commit-message "poly4mer v3.0.0 checkpoint"
fi

if [ "$ONLY" = all ] || [ "$ONLY" = heads ]; then
    for P in tig phrr; do
        say; say ">> uploading the monotone $P heads"
        "$HF" upload "$REPO" "$MONO_SRC/$P/last_two_linear" \
            "poly4mer_pretrained_v3/${P}_10folds" \
            --include "fold_*_best_fp32.ckpt" \
            --commit-message "monotone $P heads, 10 folds"
    done
fi

if [ "$ONLY" = all ] || [ "$ONLY" = card ]; then
    say; say ">> uploading the model card as README.md"
    "$HF" upload "$REPO" "$CARD" README.md --commit-message "model card"
fi

say
hr
say " done."
say
say " One step remains and it is not optional if the previous checkpoint carried"
say " metadata you did not intend to publish: HuggingFace keeps git history, so the"
say " old blob is still reachable at its commit. Open the repository Settings and run"
say " Super-squash history, which collapses history to a single commit and drops it."
hr
