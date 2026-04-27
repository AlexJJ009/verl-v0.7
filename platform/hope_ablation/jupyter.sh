#!/usr/bin/env bash
set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set in run.hope}"

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}
REPO_SUBPATH=${REPO_SUBPATH:-verl08/verl-v0.7-feature-on-policy-wdl-sft}
REPO="$LGX/$REPO_SUBPATH"

if [ ! -d "$REPO" ]; then
    echo "ERROR: repo not found at $REPO" >&2
    exit 1
fi

if [ "${SMOKE:-0}" = "1" ]; then
    export TOTAL_TRAINING_STEPS=10
    export SAVE_FREQ=5
    echo "[hope_ablation] SMOKE mode: TOTAL_TRAINING_STEPS=10, SAVE_FREQ=5"
else
    export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-13}
    echo "[hope_ablation] FULL run: keeping up to $MAX_ACTOR_CKPTS_TO_KEEP actor checkpoints"
fi

echo "[hope_ablation] REPO       = $REPO"
echo "[hope_ablation] EXPERIMENT = $EXPERIMENT"

exec bash "$REPO/recipe/on_policy_wdl_sft/ablation_single_model/meituan/jupyter.sh"
