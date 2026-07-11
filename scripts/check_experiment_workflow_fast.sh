#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASELINE_ROOT=${BASELINE_ROOT:-/data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline}
python3 "$REPO/scripts/experiment_manifest.py" validate "$REPO/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
python3 "$REPO/scripts/check_new_experiment_gate.py" --repo "$REPO" --inventory "$REPO/docs/joint_training/manifests/superproject_legacy_runnables.json" --dirty-baseline "$BASELINE_ROOT/superproject.json"
python3 "$REPO/scripts/check_new_experiment_gate.py" --repo "$REPO/recipe" --inventory "$REPO/docs/joint_training/manifests/recipe_legacy_runnables.json" --dirty-baseline "$BASELINE_ROOT/recipe.json"
python3 "$REPO/scripts/check_goal_git_isolation.py" --superproject "$REPO" --submodule "$REPO/recipe" --baseline-root "$BASELINE_ROOT" --adoption-manifest "$REPO/docs/joint_training/manifests/stage123_dirty_adoption.json" --superproject-adoption-manifest "$REPO/docs/joint_training/manifests/goal_contract_dirty_adoption.json"
/data-1/verl07/run_train.sh /opt/venv/bin/python -m pytest -q "$REPO/tests/experiment_workflow"
