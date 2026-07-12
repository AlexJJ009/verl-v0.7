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
: "${EXPERIMENT_DEPLOYABILITY_RECEIPT:?EXPERIMENT_DEPLOYABILITY_RECEIPT required}"
: "${EXPERIMENT_CALIBRATION_REPORT:?EXPERIMENT_CALIBRATION_REPORT required}"
: "${EXPERIMENT_CALIBRATION_POLICY:?EXPERIMENT_CALIBRATION_POLICY required}"
: "${EXPERIMENT_CALIBRATION_HISTORY_INDEX:?EXPERIMENT_CALIBRATION_HISTORY_INDEX required}"
: "${EXPERIMENT_CALIBRATION_PREDICTION_CONTRACT:?EXPERIMENT_CALIBRATION_PREDICTION_CONTRACT required}"
: "${EXPERIMENT_FORMAL_QUEUE_ID:?EXPERIMENT_FORMAL_QUEUE_ID required}"
: "${EXPERIMENT_EXPECTED_PROFILE_HASH:?EXPERIMENT_EXPECTED_PROFILE_HASH required}"

if python3 - "$EXPERIMENT_DEPLOYABILITY_RECEIPT" <<'PY'
import json,sys
try:
    data=json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("receipt_type") == "code_task_operational_calibration_stage12_producer" else 1)
PY
then
    echo "limited_receipt_scope_mismatch" >&2
    exit 1
fi

deployability_args=(
    --receipt "$EXPERIMENT_DEPLOYABILITY_RECEIPT"
    --normalized-manifest "$EXPERIMENT_NORMALIZED_MANIFEST"
    --preflight-receipt "$EXPERIMENT_PREFLIGHT_RECEIPT"
    --report "$EXPERIMENT_CALIBRATION_REPORT"
    --policy "$EXPERIMENT_CALIBRATION_POLICY"
    --history-index "$EXPERIMENT_CALIBRATION_HISTORY_INDEX"
    --prediction-contract "$EXPERIMENT_CALIBRATION_PREDICTION_CONTRACT"
    --queue-identity "$EXPERIMENT_FORMAL_QUEUE_ID"
    --profile-hash "$EXPERIMENT_EXPECTED_PROFILE_HASH"
)
if [ -n "${EXPERIMENT_CALIBRATION_SEMANTIC_CONTRACT:-}" ]; then
    deployability_args+=(--semantic-contract "$EXPERIMENT_CALIBRATION_SEMANTIC_CONTRACT")
fi
python3 "$REPO/recipe/on_policy_wdl_sft/code_task/stage123_deployability_receipt.py" "${deployability_args[@]}"

python3 "$REPO/scripts/verify_manifest_release_provenance.py" \
    --normalized-manifest "$EXPERIMENT_NORMALIZED_MANIFEST" \
    --provenance "$EXPERIMENT_RUN_PROVENANCE" \
    --run-prefix "$RUN_PREFIX" \
    --final-step "$FINAL_STEP" \
    --train-file "$TRAIN_FILE" \
    --preflight-receipt "$EXPERIMENT_PREFLIGHT_RECEIPT"

exec bash "$REPO/scripts/code_task_training_release_hook.sh"
