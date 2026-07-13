from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh"
QUEUE_IMPL = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh"
GATE = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh"
MONITOR = ROOT / "scripts/stage123_manifest_monitor.py"
POLICY = ROOT / "config/experiment_execution/stage123_recovery_policy_v1.json"


def test_public_queue_is_thin_python_core_delegate() -> None:
    text = QUEUE.read_text()
    assert "experiment_execution_core.py" in text
    assert "run_code_task_qwen3_1p7b_stage123_queue_impl.sh" in text
    assert "--resume" in text
    assert "verl.trainer" not in text


def test_active_queue_and_gate_have_no_legacy_receipt_authority() -> None:
    combined = QUEUE_IMPL.read_text() + GATE.read_text()
    forbidden = ["DEPLOYABILITY_RECEIPT", "STAGE12_PRODUCER_RECEIPT", "PREFLIGHT_RECEIPT", "RECEIPT_MAX_AGE_SECONDS", "CALIBRATION_REPORT"]
    assert not [name for name in forbidden if name in combined]
    assert "execution_results.py\" admission validate" in GATE.read_text()
    assert "STAGE123_ADMISSION_BUNDLE" in combined


def test_monitor_consumes_persisted_execution_state() -> None:
    text = MONITOR.read_text()
    assert "persisted_states" in text
    assert "--state-root" in text
    assert "execution_status" in text


def test_recovery_policy_is_frozen_one_resume_schema() -> None:
    value = json.loads(POLICY.read_text())
    assert value["policy_id"] == "stage123-recovery-v1"
    assert value["max_attempts"] == 2
    assert value["resumable_failure_codes"] == [
        "checkpoint_available_child_exit",
        "container_runtime_interruption",
        "host_interruption",
    ]
    assert set(value["required_attempt_fields"]) == {
        "attempt", "max_attempts", "resume_from_checkpoint", "failure_code",
        "manifest_sha256", "implementation_tree_sha256", "bundle_sha256",
        "started_at", "completed_at",
    }
