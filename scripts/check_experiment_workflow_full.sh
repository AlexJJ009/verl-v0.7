#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SCRATCH=${EXPERIMENT_WORKFLOW_SCRATCH:-/data-1/tmp/verl_agent_scratch/experiment_workflow/full_gate}
mkdir -p "$SCRATCH"
run() { bash "$REPO/scripts/run_checked_pipeline.sh" "$SCRATCH/$1.log" "${@:2}"; }
run fast bash "$REPO/scripts/check_experiment_workflow_fast.sh"
run imports /data-1/verl07/run_train.sh /opt/venv/bin/python -c 'import verl; import torch; import ray'
run reward /data-1/verl07/run_train.sh /opt/venv/bin/python -m pytest -q tests/on_policy_wdl_sft/test_code_task_reward_and_metrics.py
run queue env DRY_RUN=1 bash "$REPO/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh"
run monitor python3 "$REPO/scripts/stage123_manifest_monitor.py" --manifest "$SCRATCH/stage123.normalized.json" --checkpoint-root "$SCRATCH/checkpoints" --queue-tmux no-such-queue --poll-seconds 0 --ledger "$SCRATCH/notifications.jsonl" --policy "$REPO/scripts/experiment_notification_policy.py" --once
run release /data-1/verl07/run_train.sh /opt/venv/bin/python -m pytest -q tests/experiment_workflow/test_manifest_release_gate.py
