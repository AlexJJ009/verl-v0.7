#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SCRATCH=${EXPERIMENT_WORKFLOW_SCRATCH:-/data-1/tmp/verl_agent_scratch/experiment_workflow/full_gate}
mkdir -p "$SCRATCH"
REPO_HOST="$REPO" /data-1/verl07/run_train.sh python -m pytest -q \
  "$REPO/tests/experiment_workflow" \
  "$REPO/tests/joint_training/regression/test_validation_generation_logging.py"
