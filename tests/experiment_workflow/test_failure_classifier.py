from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("failure_classifier", ROOT / "scripts/experiment_failure_classifier.py")
classifier = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(classifier)


FIXTURES = ROOT / "tests/experiment_workflow/fixtures/failure_classes"


def test_primary_failure_classes():
    paths = sorted(FIXTURES.glob("*.txt"))
    assert len(paths) == 8
    for path in paths:
        expected = path.stem
        text = path.read_text()
        assert classifier.classify(text)["primary_reason"] == expected


def test_unknown_is_not_fabricated():
    assert classifier.classify("ordinary informational log")["primary_reason"] == "unknown"


def test_priority_is_deterministic_and_secondary_is_retained():
    result = classifier.classify("CUDA out of memory; Reward computation timed out")
    assert result["primary_reason"] == "gpu_oom"
    assert result["secondary_signals"] == ["scorer_timeout"]
