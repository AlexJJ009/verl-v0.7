#!/usr/bin/env bash
set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set in run.hope}"

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}
REPO_SUBPATH=${REPO_SUBPATH:-verl08/verl-v0.7-feature-on-policy-wdl-sft}
REPO="$LGX/$REPO_SUBPATH"
EXPERIMENT_LC="${EXPERIMENT,,}"

[ -d "$REPO" ] || { echo "ERROR: repo not found at $REPO" >&2; exit 1; }

export SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
    export SAVE_FREQ=${SAVE_FREQ:-5}
    export TEST_FREQ=${TEST_FREQ:-5}
    export VAL_N=${VAL_N:-3}
    export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
    echo "[hope_staged_v1] SMOKE mode propagated"
else
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
    export TEST_FREQ=${TEST_FREQ:-5}
    export SAVE_FREQ=${SAVE_FREQ:-5}
    export VAL_N=${VAL_N:-3}
    export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
    export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
fi

export KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}
export BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@3}
export BEST_CKPT_METRIC_MODE=${BEST_CKPT_METRIC_MODE:-max}
export BEST_CKPT_STRIP_OPTIMIZER=${BEST_CKPT_STRIP_OPTIMIZER:-True}

echo "[hope_staged_v1] REPO       = $REPO"
echo "[hope_staged_v1] EXPERIMENT = $EXPERIMENT_LC"
echo "[hope_staged_v1] SMOKE      = $SMOKE"
export EXPERIMENT="$EXPERIMENT_LC"

exec bash "$REPO/recipe/on_policy_wdl_sft/staged_v1/meituan/jupyter.sh"
