from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_agent_timer_or_tuning_authority_in_batch_paths() -> None:
    core = (ROOT / "scripts/experiment_execution_core.py").read_text().lower()
    queue = (
        (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh")
        .read_text()
        .lower()
    )
    monitor = (ROOT / "scripts/stage123_manifest_monitor.py").read_text().lower()
    for forbidden in ("codex", "crontab", "systemd", "wandb", "auto_tune", "parameter_mutation"):
        assert forbidden not in core + queue + monitor
    assert "max_attempts=1" in core
    assert "resumable_failure_codes=()" in core
    assert "forbids retry/resume" in queue


def test_deletion_budget_and_inventory_are_consistent() -> None:
    budget = json.loads(
        (ROOT / "docs/joint_training/goals/experiment-batch-orchestration/deletion-budget.json").read_text()
    )
    inventory = json.loads(
        (ROOT / "docs/joint_training/goals/experiment-batch-orchestration/authority-inventory.json").read_text()
    )
    assert budget["sole_authority"] == "scripts/experiment_execution_core.py"
    assert inventory["sole_transition_authority"] == budget["sole_authority"]
    assert any(item["path"].endswith("stage123_queue_impl.sh") for item in inventory["constructs"])
    assert budget["deletion_outside_budget_authorized"] is False


def test_batch_manifest_fixture_is_validated_by_core_command() -> None:
    manifest = ROOT / "tests/experiment_workflow/fixtures/experiment_batch_v1.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text())["schema_version"] == 1


def test_mutated_protected_binding_is_rejected_without_touching_asset(tmp_path: Path) -> None:
    core_path = ROOT / "scripts/experiment_execution_core.py"
    spec = importlib.util.spec_from_file_location("batch_policy_core", core_path)
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader
    import sys

    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    bundle = json.loads((ROOT / "tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json").read_text())
    bundle["bindings"]["protected_asset_hashes"]["test_data"] = "0" * 64
    bundle["bundle_sha256"] = tool.sha256_json({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    try:
        tool.validate_admission_bundle(bundle, path, ROOT)
    except tool.BatchValidationError as exc:
        assert "protected asset hash mismatch" in str(exc)
    else:
        raise AssertionError("mutated protected binding was accepted")
