from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location("stage123_matrix_admission", ROOT / "scripts/stage123_matrix_admission.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_facts_are_structured(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "0, NVIDIA L40S, 46068\n1, NVIDIA L40S, 46068\n")
    assert module.gpu_facts() == [
        {"index": 0, "name": "NVIDIA L40S", "memory_total_mib": 46068},
        {"index": 1, "name": "NVIDIA L40S", "memory_total_mib": 46068},
    ]
