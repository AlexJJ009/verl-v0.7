#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python3 "$REPO/scripts/experiment_manifest.py" validate "$REPO/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
python3 "$REPO/scripts/check_engineering_rule_catalog.py" "$REPO/docs/joint_training/constraints/principles/engineering_rule_catalog.md"
python3 -m py_compile "$REPO/scripts/experiment_execution_core.py" "$REPO/scripts/execution_results.py" "$REPO/scripts/check_code_task_operational_calibration.py"
bash -n "$REPO/scripts/run_code_task_operational_calibration_queue.sh" "$REPO/scripts/run_code_task_operational_calibration.sh" "$REPO/recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh"
REPO_HOST="$REPO" /data-1/verl07/run_train.sh python -m pytest -q \
  "$REPO/tests/experiment_workflow/test_operational_calibration_runner.py" \
  "$REPO/tests/experiment_workflow/test_operational_calibration_checker.py" \
  "$REPO/tests/experiment_workflow/test_evidence_compatibility.py"
