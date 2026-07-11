#!/usr/bin/env bash
# Manifest-native release wrapper; verifies immutable provenance before legacy hook.
set -euo pipefail

REPO=${REPO:-${VERL_REPO_ROOT:-/data-1/code/verl}}
: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${FINAL_STEP:?FINAL_STEP required}"
: "${TRAIN_FILE:?TRAIN_FILE required}"
: "${EXPERIMENT_NORMALIZED_MANIFEST:?EXPERIMENT_NORMALIZED_MANIFEST required}"
: "${EXPERIMENT_RUN_PROVENANCE:?EXPERIMENT_RUN_PROVENANCE required}"
: "${EXPERIMENT_PREFLIGHT_RECEIPT:?EXPERIMENT_PREFLIGHT_RECEIPT required}"

python3 "$REPO/scripts/verify_manifest_release_provenance.py" \
    --normalized-manifest "$EXPERIMENT_NORMALIZED_MANIFEST" \
    --provenance "$EXPERIMENT_RUN_PROVENANCE" \
    --run-prefix "$RUN_PREFIX" \
    --final-step "$FINAL_STEP" \
    --train-file "$TRAIN_FILE" \
    --preflight-receipt "$EXPERIMENT_PREFLIGHT_RECEIPT"

exec bash "$REPO/scripts/code_task_training_release_hook.sh"
