#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT:?Set EXPERIMENT in run.hope}"
: "${LGX:?Set the private dolphinfs root in the submission environment}"
: "${REPO_SUBPATH:?Set REPO_SUBPATH to the staged verl checkout under LGX}"

REPO=${LGX}/${REPO_SUBPATH}
if [ ! -d "${REPO}" ]; then
  echo "ERROR: staged repository not found: ${REPO}" >&2
  exit 2
fi

if [ "${SMOKE:-0}" = 1 ]; then
  export TOTAL_TRAINING_STEPS=${SMOKE_STEPS:-10}
  export SAVE_FREQ=${SMOKE_SAVE_FREQ:-5}
fi

exec bash "${REPO}/recipe/on_policy_wdl_sft/standard_grpo/meituan/jupyter.sh"
