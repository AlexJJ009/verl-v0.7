from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/experiment_workflow/test_experiment_batch_core.py"


def support():
    spec = importlib.util.spec_from_file_location("batch_core_support_control", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module.load_core()


def test_stale_and_replayed_controls_do_not_mutate_state(tmp_path: Path) -> None:
    support_module, tool = support()
    manifest = support_module.make_manifest(tool, tmp_path, [support_module.make_item(tool, "one", "run-one")])
    executor = tool.BatchExecutor(manifest, tmp_path / "state", support_module.FakeAdapter([0]), support_module.FakeClock())
    manifest.operator_control_path.write_text(json.dumps(support_module.control(tool, manifest, 1, 0, "pause_after_current")) + "\n")
    result = executor.run()
    assert result["status"] == "paused_after_current"
    with manifest.operator_control_path.open("a") as handle:
        handle.write(json.dumps(support_module.control(tool, manifest, 1, result["batch_revision"], "stop_now")) + "\n")
    executor._read_controls()
    assert executor.control_rejection and "replay" in executor.control_rejection["message"]


def test_pause_then_continue_uses_persisted_control_cursor(tmp_path: Path) -> None:
    support_module, tool = support()
    manifest = support_module.make_manifest(tool, tmp_path, [support_module.make_item(tool, "one", "run-one")])
    first = tool.BatchExecutor(manifest, tmp_path / "state", support_module.FakeAdapter([0]), support_module.FakeClock())
    manifest.operator_control_path.write_text(__import__("json").dumps(support_module.control(tool, manifest, 1, 0, "pause_after_current")) + "\n")
    assert first.run()["status"] == "paused_after_current"
    with manifest.operator_control_path.open("a") as handle:
        handle.write(__import__("json").dumps(support_module.control(tool, manifest, 2, 1, "continue_remaining")) + "\n")
    second = tool.BatchExecutor(manifest, tmp_path / "state", support_module.FakeAdapter([0]), support_module.FakeClock())
    assert second.run()["status"] == "completed"
