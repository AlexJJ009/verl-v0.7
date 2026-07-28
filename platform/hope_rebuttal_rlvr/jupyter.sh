#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ROOT:?ROOT must be set by run.hope}"
: "${REPO_SUBPATH:?REPO_SUBPATH must be set by run.hope}"
: "${REPO_COMMIT:?REPO_COMMIT must be set by run.hope}"
: "${REPO_SUBMODULE_RECEIPT:?REPO_SUBMODULE_RECEIPT must be set by run.hope}"
: "${REPO_SUBMODULE_RECEIPT_HASH:?REPO_SUBMODULE_RECEIPT_HASH must be set by run.hope}"
: "${SUBMITTER_SOURCE_HASH:?SUBMITTER_SOURCE_HASH must be set by run.hope}"
: "${IMAGE_DIGEST:?IMAGE_DIGEST must be set by run.hope}"
: "${ARM:?ARM must be set by run.hope}"
: "${EXPERIMENT:?EXPERIMENT must be set by run.hope}"
: "${INIT_MODEL_PATH:?INIT_MODEL_PATH must be set by run.hope}"
: "${PAIRED_INIT_MANIFEST:?PAIRED_INIT_MANIFEST must be set by run.hope}"
: "${PAIRED_INIT_MANIFEST_HASH:?PAIRED_INIT_MANIFEST_HASH must be set by run.hope}"
: "${CHECKPOINT_RECEIPT:?CHECKPOINT_RECEIPT must be set by run.hope}"
: "${CHECKPOINT_RECEIPT_HASH:?CHECKPOINT_RECEIPT_HASH must be set by run.hope}"
: "${TRAIN_RECEIPT:?TRAIN_RECEIPT must be set by run.hope}"
: "${TRAIN_RECEIPT_HASH:?TRAIN_RECEIPT_HASH must be set by run.hope}"
: "${MATH7_RECEIPT:?MATH7_RECEIPT must be set by run.hope}"
: "${MATH7_RECEIPT_HASH:?MATH7_RECEIPT_HASH must be set by run.hope}"
: "${GRADER_RECEIPT:?GRADER_RECEIPT must be set by run.hope}"
: "${GRADER_RECEIPT_HASH:?GRADER_RECEIPT_HASH must be set by run.hope}"
: "${JOB_TAG:?JOB_TAG must be set by run.hope}"
: "${CELL_HASH:?CELL_HASH must be set by run.hope}"
: "${ATTEMPT_ID:?ATTEMPT_ID must be set by run.hope}"
: "${ALGORITHM_CONFIG_HASH:?ALGORITHM_CONFIG_HASH must be set by run.hope}"
: "${H20_PROFILE_PATH:?H20_PROFILE_PATH must be set by run.hope}"
: "${H20_PROFILE_HASH:?H20_PROFILE_HASH must be set by run.hope}"
: "${H20_CALIBRATION_RECEIPT:?H20_CALIBRATION_RECEIPT must be set by run.hope}"
: "${H20_CALIBRATION_RECEIPT_HASH:?H20_CALIBRATION_RECEIPT_HASH must be set by run.hope}"
: "${PATH_OVERRIDE_RECEIPT:?PATH_OVERRIDE_RECEIPT must be set by run.hope}"
: "${PATH_OVERRIDE_RECEIPT_HASH:?PATH_OVERRIDE_RECEIPT_HASH must be set by run.hope}"
: "${RUN_MODE:?RUN_MODE must be set by run.hope}"

if [[ "$REPO_SUBPATH" == /* || "/$REPO_SUBPATH/" == *"/../"* || "/$REPO_SUBPATH/" == *"/./"* || "$REPO_SUBPATH" == *"//"* ]]; then
    echo "ERROR: unsafe REPO_SUBPATH: $REPO_SUBPATH" >&2
    exit 2
fi

# Platform jobs never inherit a caller-selected checkout or parent root.
REPO_ROOT="$ROOT/$REPO_SUBPATH"
LGX="$ROOT"
export ROOT REPO_SUBPATH REPO_COMMIT REPO_ROOT LGX
export REQUIRE_PLATFORM_RECEIPTS=1

if [ ! -d "$REPO_ROOT/.git" ] && [ ! -f "$REPO_ROOT/.git" ]; then
    echo "ERROR: immutable repo worktree not found: $REPO_ROOT" >&2
    exit 2
fi
if [ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$REPO_COMMIT" ]; then
    echo "ERROR: repo commit does not match approved manifest" >&2
    exit 2
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
    echo "ERROR: formal worker checkout is dirty" >&2
    exit 2
fi

verify_file_hash() {
    local label=$1
    local path=$2
    local expected=$3
    if [ ! -f "$path" ]; then
        echo "ERROR: $label is missing: $path" >&2
        exit 2
    fi
    if [ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]; then
        echo "ERROR: $label hash mismatch" >&2
        exit 2
    fi
}

verify_file_hash "submodule receipt" "$REPO_SUBMODULE_RECEIPT" "$REPO_SUBMODULE_RECEIPT_HASH"
verify_file_hash "paired-init manifest" "$PAIRED_INIT_MANIFEST" "$PAIRED_INIT_MANIFEST_HASH"
verify_file_hash "checkpoint receipt" "$CHECKPOINT_RECEIPT" "$CHECKPOINT_RECEIPT_HASH"
verify_file_hash "train receipt" "$TRAIN_RECEIPT" "$TRAIN_RECEIPT_HASH"
verify_file_hash "Math-7 receipt" "$MATH7_RECEIPT" "$MATH7_RECEIPT_HASH"
verify_file_hash "grader receipt" "$GRADER_RECEIPT" "$GRADER_RECEIPT_HASH"
verify_file_hash "H20 profile" "$H20_PROFILE_PATH" "$H20_PROFILE_HASH"
verify_file_hash "path-override receipt" "$PATH_OVERRIDE_RECEIPT" "$PATH_OVERRIDE_RECEIPT_HASH"

H20_CALIBRATION_ARGS=()
if [ "$RUN_MODE" = "formal" ]; then
    if [ "$H20_CALIBRATION_RECEIPT" = "NONE" ] || [ "$H20_CALIBRATION_RECEIPT_HASH" = "NONE" ]; then
        echo "ERROR: formal worker requires a signed H20 calibration receipt" >&2
        exit 2
    fi
    verify_file_hash "H20 calibration receipt" "$H20_CALIBRATION_RECEIPT" "$H20_CALIBRATION_RECEIPT_HASH"
    H20_CALIBRATION_ARGS=(--h20-calibration-receipt "$H20_CALIBRATION_RECEIPT")
elif [ "$H20_CALIBRATION_RECEIPT" != "NONE" ] || [ "$H20_CALIBRATION_RECEIPT_HASH" != "NONE" ]; then
    echo "ERROR: smoke worker must not claim formal H20 calibration admission" >&2
    exit 2
fi

python3 - "$REPO_ROOT" "$REPO_SUBMODULE_RECEIPT" <<'PY'
import hashlib
import json
import subprocess
import sys

repo, receipt_path = sys.argv[1:]
with open(receipt_path, encoding="utf-8") as handle:
    receipt = json.load(handle)
status = subprocess.run(
    ["git", "-C", repo, "submodule", "status", "--recursive"],
    check=True,
    capture_output=True,
    text=True,
).stdout.encode()
if receipt.get("status_sha256") != hashlib.sha256(status).hexdigest():
    raise SystemExit("ERROR: live recursive submodule status differs from receipt")
PY

FROZEN_CONFIG="$REPO_ROOT/recipe/on_policy_wdl_sft/rebuttal_rlvr/frozen_grpo_v2.env"
SUBMITTER_SOURCE="$REPO_ROOT/platform/hope_rebuttal_rlvr/submit_manifest.py"
verify_file_hash "frozen project-GRPO config" "$FROZEN_CONFIG" "$ALGORITHM_CONFIG_HASH"
verify_file_hash "submitter source" "$SUBMITTER_SOURCE" "$SUBMITTER_SOURCE_HASH"

# Establish the exact runtime paths once, then validate the live values before
# Layer 3 re-sources the same idempotent adapter.
# shellcheck disable=SC1091
source "$REPO_ROOT/recipe/on_policy_wdl_sft/rebuttal_rlvr/meituan/env.sh"

GRADER_PATH="$REPO_ROOT/recipe/joint_training/custom_reward_function_latex_verify.py"
VALIDATOR="$REPO_ROOT/recipe/on_policy_wdl_sft/rebuttal_rlvr/validate_inputs.py"
REVIEWER_ALLOWLIST="$REPO_ROOT/platform/hope_rebuttal_rlvr/g3_reviewer_keys.json"

H20_ENV_FILE=$(mktemp /tmp/rebuttal-rlvr-h20-env.XXXXXX)
trap 'rm -f "$H20_ENV_FILE"' EXIT
python3 "$VALIDATOR" platform-artifacts \
    --train-receipt "$TRAIN_RECEIPT" \
    --train-file "$TRAIN_FILE" \
    --math7-receipt "$MATH7_RECEIPT" \
    --math7-aime-2025-file "$MATH7_AIME_FILE" \
    --math7-math-500-file "$MATH7_MATH500_FILE" \
    --math7-amc23-file "$MATH7_AMC23_FILE" \
    --math7-aqua-file "$MATH7_AQUA_FILE" \
    --math7-gsm8k-file "$MATH7_GSM8K_FILE" \
    --math7-mawps-file "$MATH7_MAWPS_FILE" \
    --math7-svamp-file "$MATH7_SVAMP_FILE" \
    --grader-receipt "$GRADER_RECEIPT" \
    --grader-path "$GRADER_PATH" \
    --h20-profile "$H20_PROFILE_PATH" \
    "${H20_CALIBRATION_ARGS[@]}" \
    --reviewer-allowlist "$REVIEWER_ALLOWLIST" \
    --rendered-hope "$SCRIPT_DIR/run.hope" \
    --path-override-receipt "$PATH_OVERRIDE_RECEIPT" \
    --image-digest "$IMAGE_DIGEST" \
    --repo-root "$REPO_ROOT" \
    --root "$ROOT" \
    --repo-subpath "$REPO_SUBPATH" \
    --init-model-path "$INIT_MODEL_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint-root "$BASE_CKPT_DIR" \
    --eval-root "$EVAL_ROOT" \
    --log-root "$LOG_DIR" \
    --wandb-root "$WANDB_DIR" \
    --receipt-root "$RECEIPT_ROOT" \
    --hf-home "$HF_HOME" \
    --huggingface-hub-cache "$HUGGINGFACE_HUB_CACHE" \
    --hf-datasets-cache "$HF_DATASETS_CACHE" \
    --xdg-cache-home "$XDG_CACHE_HOME" \
    --ray-tmpdir "$RAY_TMPDIR" \
    --tmpdir "$TMPDIR" \
    --vllm-config-root "$VLLM_CONFIG_ROOT" \
    --zmq-ipc-dir "$VERL_ZMQ_IPC_DIR" \
    --run-mode "$RUN_MODE" \
    --h20-env-output "$H20_ENV_FILE"

# validate_inputs.py emits only single-quoted values selected from strict JSON
# enums/consts. This is the sole source of platform system-knob overrides.
# shellcheck disable=SC1090
source "$H20_ENV_FILE"
rm -f "$H20_ENV_FILE"
trap - EXIT
export \
    ROLLOUT_GPU_MEMORY_UTILIZATION GENERATION_MICRO_BATCH_SIZE \
    LOG_PROB_MICRO_BATCH_SIZE ACTOR_PPO_MAX_TOKEN_LEN ROLLOUT_TP_SIZE \
    ROLLOUT_AGENT_NUM_WORKERS ROLLOUT_MAX_NUM_SEQS ROLLOUT_ENFORCE_EAGER \
    ROLLOUT_ENABLE_CHUNKED_PREFILL ROLLOUT_MAX_MODEL_LEN \
    LOG_PROB_MAX_TOKEN_LEN_PER_GPU ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD

exec bash "$REPO_ROOT/recipe/on_policy_wdl_sft/rebuttal_rlvr/meituan/jupyter.sh"
