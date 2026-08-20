# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh"
QUEUE_IMPL = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh"
GATE = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh"
MONITOR = ROOT / "scripts/stage123_manifest_monitor.py"
POLICY = ROOT / "config/experiment_execution/stage123_recovery_policy_v1.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_public_queue_is_thin_python_core_delegate() -> None:
    public = QUEUE.read_text()
    adapter = QUEUE_IMPL.read_text()
    assert "run_code_task_qwen3_1p7b_stage123_queue_impl.sh" in public
    assert "experiment_execution_core.py" in adapter
    assert "batch-run" in adapter
    assert "--resume|--recovery-policy" in adapter
    assert "verl.trainer" not in public + adapter


def test_stage123_batch_adapter_rejects_resume_and_missing_manifest() -> None:
    resume = subprocess.run(["bash", str(QUEUE), "--resume"], text=True, capture_output=True, check=False)
    assert resume.returncode == 2
    assert "forbids retry/resume" in resume.stderr
    missing = subprocess.run(["bash", str(QUEUE)], text=True, capture_output=True, check=False)
    assert missing.returncode == 2
    assert "EXPERIMENT_BATCH_MANIFEST" in missing.stderr


def test_active_queue_and_gate_have_no_legacy_receipt_authority() -> None:
    combined = QUEUE_IMPL.read_text() + GATE.read_text()
    forbidden = [
        "DEPLOYABILITY_RECEIPT",
        "STAGE12_PRODUCER_RECEIPT",
        "PREFLIGHT_RECEIPT",
        "RECEIPT_MAX_AGE_SECONDS",
        "CALIBRATION_REPORT",
    ]
    assert not [name for name in forbidden if name in combined]
    assert 'execution_results.py" admission validate' in GATE.read_text()
    assert "STAGE123_ADMISSION_BUNDLE" not in QUEUE_IMPL.read_text()
    for forbidden in ("status.tsv", "launch_and_wait", "validation_deadline_controller", "tmux", "latest_checkpoint"):
        assert forbidden not in QUEUE_IMPL.read_text()


def test_monitor_consumes_persisted_execution_state() -> None:
    text = MONITOR.read_text()
    assert "persisted_states" in text
    assert "persisted_events" in text
    assert "--state-root" in text
    assert "execution_status" in text
    for forbidden in ("tmux", "checkpoint-root", "queue-tmux", "latest_checkpoint", "validation_deadlines"):
        assert forbidden not in text


def test_monitor_maps_only_persisted_core_events() -> None:
    monitor = load(MONITOR, "stage123_monitor_core_mapping")
    policy = load(ROOT / "scripts/experiment_notification_policy.py", "stage123_notification_mapping")
    base = {
        "schema_version": 1,
        "run_id": "stage123-primary-queue",
        "attempt": 1,
        "transition": {"from": "pending", "to": "running", "at": 1.0},
        "failure": None,
        "cleanup": None,
    }
    expected = {"pending": None, "running": "run_started", "succeeded": None}
    for status, event_name in expected.items():
        state = monitor.notification_state_from_event(
            {**base, "status": status}, ["frac25-stage2", "frac25-stage3"], Path("/state")
        )
        assert state["manifest_run_ids"] == ["frac25-stage2", "frac25-stage3"]
        assert policy.event_for(state) == event_name
    for status in ("failed", "deadline_exceeded", "cleanup_failed"):
        event = {
            **base,
            "status": status,
            "failure": {"code": status, "message": "failed", "context": {}},
            "cleanup": {"resources_released": status != "cleanup_failed"},
        }
        state = monitor.notification_state_from_event(event, ["frac25-stage2", "frac25-stage3"], Path("/state"))
        assert state["failure"] == event["failure"] and state["cleanup"] == event["cleanup"]
        assert policy.event_for(state) == "run_failed"


def test_monitor_replays_core_events_without_external_runtime_probes(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": [{"id": "frac25-stage2"}, {"id": "frac25-stage3"}]}))
    events = [
        {
            "schema_version": 1,
            "run_id": "stage123-primary-queue",
            "status": "running",
            "attempt": 1,
            "transition": {"from": "pending", "to": "running", "at": 1},
            "failure": None,
            "cleanup": None,
        },
        {
            "schema_version": 1,
            "run_id": "stage123-primary-queue",
            "status": "failed",
            "attempt": 1,
            "transition": {"from": "running", "to": "failed", "at": 2},
            "failure": {"code": "child_exit", "message": "failed", "context": {}},
            "cleanup": {"resources_released": True},
        },
        {
            "schema_version": 1,
            "run_id": "stage123-primary-queue",
            "status": "succeeded",
            "attempt": 2,
            "transition": {"from": "running", "to": "succeeded", "at": 3},
            "failure": None,
            "cleanup": {"resources_released": True},
        },
    ]
    (state_root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (state_root / "stage123-primary-queue.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "stage123-primary-queue", "status": "succeeded", "attempt": 2})
    )
    ledger = tmp_path / "notifications.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(MONITOR),
            "--manifest",
            str(manifest),
            "--state-root",
            str(state_root),
            "--ledger",
            str(ledger),
            "--policy",
            str(ROOT / "scripts/experiment_notification_policy.py"),
            "--once",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(item["run_id"], item["event"]) for item in records] == [
        ("stage123-primary-queue", "run_started"),
        ("stage123-primary-queue", "run_failed"),
    ]


def test_legacy_atomic_recovery_policy_remains_but_stage123_batch_rejects_it() -> None:
    value = json.loads(POLICY.read_text())
    assert value["policy_id"] == "stage123-recovery-v1"
    assert value["max_attempts"] == 2
    assert value["resumable_failure_codes"] == [
        "checkpoint_available_child_exit",
        "container_runtime_interruption",
        "host_interruption",
    ]
    assert set(value["required_attempt_fields"]) == {
        "attempt",
        "max_attempts",
        "resume_from_checkpoint",
        "failure_code",
        "manifest_sha256",
        "implementation_tree_sha256",
        "bundle_sha256",
        "started_at",
        "completed_at",
    }
    adapter = QUEUE_IMPL.read_text()
    assert "forbids retry/resume" in adapter
