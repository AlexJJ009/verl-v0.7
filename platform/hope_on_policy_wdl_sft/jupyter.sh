#!/usr/bin/env bash
set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set in run.hope}"

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}
REPO_SUBPATH=${REPO_SUBPATH:-verl08/verl-v0.7-feature-on-policy-wdl-sft}
REPO="$LGX/$REPO_SUBPATH"
EXPERIMENT_LC="${EXPERIMENT,,}"

if [ ! -d "$REPO" ]; then
    echo "ERROR: repo not found at $REPO" >&2
    exit 1
fi

if [ "${SMOKE:-0}" = "1" ]; then
    export TOTAL_TRAINING_STEPS=10
    export SAVE_FREQ=5
    export TEST_FREQ=5
    echo "[hope_on_policy_wdl_sft] SMOKE mode: TOTAL_TRAINING_STEPS=10 SAVE_FREQ=5 TEST_FREQ=5"
else
    export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
    echo "[hope_on_policy_wdl_sft] FULL run: keeping latest checkpoint plus best model-only checkpoint"
fi

export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@1}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}

echo "[hope_on_policy_wdl_sft] REPO       = $REPO"
echo "[hope_on_policy_wdl_sft] EXPERIMENT = $EXPERIMENT_LC"
export EXPERIMENT="$EXPERIMENT_LC"

case "$EXPERIMENT_LC" in
    1a|1b|1c|1a-joint|1b-joint|1c-joint)
        exec bash "$REPO/recipe/on_policy_wdl_sft/meituan/jupyter.sh"
        ;;
    2a-base|2a-sft|2b-base|2b-sft|2c-base|2c-sft|2z-base|2z-sft|2g-base|2g-sft)
        exec bash "$REPO/recipe/on_policy_wdl_sft/ablation_single_model/meituan/jupyter.sh"
        ;;
    *)
        echo "[hope_on_policy_wdl_sft] ERROR: unsupported EXPERIMENT='$EXPERIMENT'" >&2
        echo "[hope_on_policy_wdl_sft] Supported: 1a/1b/1c, 2a/2b/2c/2z/2g with -base or -sft." >&2
        exit 1
        ;;
esac
