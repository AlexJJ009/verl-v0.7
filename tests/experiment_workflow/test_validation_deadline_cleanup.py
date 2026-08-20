from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tool():
    path = ROOT / "scripts/experiment_execution_core.py"
    spec = importlib.util.spec_from_file_location("execution_core_deadline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_failure_shape_is_stable_under_message_formatting_changes():
    module = tool()
    first = module.failure("deadline_exceeded", "deadline exceeded", observed_at=10)
    second = module.failure("deadline_exceeded", "Deadline exceeded.", observed_at=10)
    assert first["code"] == second["code"] == "deadline_exceeded"
    assert first["context"] == second["context"] == {"observed_at": 10}


def test_terminal_states_include_cleanup_failure():
    module = tool()
    assert {"succeeded", "failed", "deadline_exceeded", "cleanup_failed"} <= module.TERMINAL_STATES
