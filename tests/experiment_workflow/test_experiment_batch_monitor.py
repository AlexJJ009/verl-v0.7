from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "scripts/stage123_manifest_monitor.py"
POLICY = ROOT / "scripts/experiment_notification_policy.py"


def load_monitor():
    spec = importlib.util.spec_from_file_location("experiment_batch_monitor", MONITOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_events_are_read_from_core_schema_and_cursor_is_idempotent(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{"expected_run_ids": ["phase-a", "phase-b"]}]}))
    batch_state = {
        "schema_version": 1,
        "batch_id": "batch-a",
        "status": "shared_failure",
        "batch_manifest_sha256": "1" * 64,
        "items": [],
    }
    (state_root / "batch-a.json").write_text(json.dumps(batch_state))
    events = [
        {"schema_version": 1, "batch_id": "batch-a", "item_id": "item-a", "event": "item_started", "state": "running", "batch_revision": 1},
        {"schema_version": 1, "batch_id": "batch-a", "event": "batch_shared_failure", "state": "shared_failure", "batch_revision": 2, "failure": {"code": "corrupt", "message": "failed", "context": {}}},
    ]
    (state_root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    ledger = tmp_path / "ledger.jsonl"
    cursor = tmp_path / "cursor.json"
    command = [
        sys.executable,
        str(MONITOR),
        "--manifest",
        str(manifest),
        "--state-root",
        str(state_root),
        "--ledger",
        str(ledger),
        "--policy",
        str(POLICY),
        "--cursor",
        str(cursor),
        "--once",
    ]
    assert subprocess.run(command, check=False).returncode == 0
    assert subprocess.run(command, check=False).returncode == 0
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(record["run_id"], record["event"]) for record in records] == [
        ("item-a", "run_started"),
        ("batch-a", "run_failed"),
    ]
    assert len(json.loads(cursor.read_text())["event_digests"]) == 2


def test_malformed_batch_event_fails_closed(tmp_path: Path) -> None:
    monitor = load_monitor()
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "events.jsonl").write_text('{"schema_version":1,"batch_id":"batch-a","state":"running"}\n')
    try:
        monitor.persisted_events(state_root)
    except ValueError as exc:
        assert "batch event schema" in str(exc)
    else:
        raise AssertionError("malformed batch event was accepted")


def test_monitor_source_has_no_transition_or_runtime_inference_authority() -> None:
    source = MONITOR.read_text()
    for forbidden in ("tmux", "latest_checkpoint", "latest_checkpointed_iteration", "wandb", "registry", "transition(state", "terminate("):
        assert forbidden not in source.lower()
    assert "persisted_events" in source
    assert "event_digests" in source
