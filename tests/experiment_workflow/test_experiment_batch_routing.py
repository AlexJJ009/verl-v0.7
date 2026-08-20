# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/experiment_workflow/test_experiment_batch_core.py"


def support():
    spec = importlib.util.spec_from_file_location("batch_core_support_routing", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module.load_core()


def test_success_and_local_failure_fallback_are_ordered(tmp_path: Path) -> None:
    support_module, tool = support()
    items = [
        support_module.make_item(tool, "failed", "run-failed", phases=2),
        support_module.make_item(tool, "next", "run-next"),
    ]
    state = tool.BatchExecutor(
        support_module.make_manifest(tool, tmp_path, items),
        tmp_path / "state",
        support_module.FakeAdapter([7, 0]),
        support_module.FakeClock(),
    ).run()
    assert state["status"] == "completed_with_failures"
    assert [item["item_id"] for item in state["items"]] == ["failed", "next"]
    assert state["items"][0]["skipped_phases"] == ["run-failed-phase-2"]


def test_repeated_failure_policy_reaches_shared_failure_after_restart(tmp_path: Path) -> None:
    support_module, tool = support()
    manifest = support_module.make_manifest(
        tool,
        tmp_path,
        [
            support_module.make_item(tool, "one", "run-one"),
            support_module.make_item(tool, "two", "run-two"),
            support_module.make_item(tool, "three", "run-three"),
        ],
    )

    def pause_after_first_failure() -> None:
        with manifest.operator_control_path.open("a") as handle:
            import json

            handle.write(json.dumps(support_module.control(tool, manifest, 1, 2, "pause_after_current")) + "\n")

    first = tool.BatchExecutor(
        manifest,
        tmp_path / "state",
        support_module.FakeAdapter([7], on_start=pause_after_first_failure),
        support_module.FakeClock(),
    ).run()
    assert first["status"] == "paused_after_current"
    import json

    with manifest.operator_control_path.open("a") as handle:
        handle.write(
            json.dumps(support_module.control(tool, manifest, 2, first["batch_revision"], "continue_remaining")) + "\n"
        )
    second = tool.BatchExecutor(
        manifest, tmp_path / "state", support_module.FakeAdapter([7, 0]), support_module.FakeClock()
    ).run()
    assert second["status"] == "shared_failure"
    assert len(second["items"]) == 2


def test_cleanup_failure_is_shared_failure_stop(tmp_path: Path) -> None:
    support_module, tool = support()

    class CleanupFailure(support_module.FakeAdapter):
        def terminate(self, child_id: str, grace_seconds: float):
            return {"resources_released": False, "term_sent": True, "kill_sent": True}

    item = support_module.make_item(tool, "one", "run-one")
    state = tool.BatchExecutor(
        support_module.make_manifest(tool, tmp_path, [item]),
        tmp_path / "state",
        CleanupFailure([7]),
        support_module.FakeClock(),
    ).run()
    assert state["status"] == "shared_failure"


def test_corrupt_state_and_event_ledgers_fail_closed(tmp_path: Path) -> None:
    support_module, tool = support()
    manifest = support_module.make_manifest(tool, tmp_path, [support_module.make_item(tool, "one", "run-one")])
    state_root = tmp_path / "state-corrupt"
    state_root.mkdir()
    (state_root / f"{manifest.batch_id}.json").write_text("{broken")
    state = tool.BatchExecutor(manifest, state_root, support_module.FakeAdapter([0]), support_module.FakeClock()).run()
    assert state["status"] == "shared_failure"
    assert (state_root / f"{manifest.batch_id}.corrupt.json").exists()

    event_root = tmp_path / "event-corrupt"
    event_root.mkdir()
    (event_root / "events.jsonl").write_text("{broken\n")
    event_state = tool.BatchExecutor(
        manifest, event_root, support_module.FakeAdapter([0]), support_module.FakeClock()
    ).run()
    assert event_state["status"] == "shared_failure"
    assert event_state["failure"]["code"] == "event_corruption"

    atomic_root = tmp_path / "atomic-event-corrupt"
    atomic_root.mkdir()
    (atomic_root / "events.jsonl").write_text(
        json.dumps({"schema_version": 1, "run_id": "phase-a", "status": "unknown", "attempt": 1}) + "\n"
    )
    atomic_state = tool.BatchExecutor(
        manifest, atomic_root, support_module.FakeAdapter([0]), support_module.FakeClock()
    ).run()
    assert atomic_state["status"] == "shared_failure"
    assert atomic_state["failure"]["code"] == "event_corruption"
