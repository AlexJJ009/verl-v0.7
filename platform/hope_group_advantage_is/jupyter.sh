#!/usr/bin/env bash
set -xeuo pipefail

: "${EXPERIMENT:?EXPERIMENT must be set in run.hope}"

LGX=${LGX:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx}
REPO_SUBPATH=${REPO_SUBPATH:-verl08/verl-v0.7-feature-on-policy-wdl-sft}
REPO="$LGX/$REPO_SUBPATH"

[ -d "$REPO" ] || { echo "ERROR: repo not found at $REPO" >&2; exit 1; }

export SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
    export SAVE_FREQ=${SAVE_FREQ:-5}
    echo "[hope_group_advantage_is] SMOKE mode propagated"
else
    export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}
fi

echo "[hope_group_advantage_is] REPO       = $REPO"
echo "[hope_group_advantage_is] EXPERIMENT = $EXPERIMENT"
echo "[hope_group_advantage_is] SMOKE      = $SMOKE"

exec bash "$REPO/recipe/on_policy_wdl_sft/group_advantage_is/meituan/jupyter.sh"
