from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_stage123_baseline_contract():
    module = load_module("experiment_workflow_baseline", ROOT / "scripts/experiment_workflow_baseline.py")
    report = module.collect(ROOT / "tests/experiment_workflow/fixtures/stage123_scorer_timeout")
    assert report["scorer"]["submitted_count"] == 4
    assert report["scorer"]["timeout_count"] == 2
    assert report["scorer"]["timeout_rate"] == 0.5
    assert report["scorer"]["score_distribution"] == {"positive": 1, "zero": 1}
    assert report["resources"]["reward_worker_peak_rss_gib"] == 191.25
    assert report["resources"]["gpu_idle_seconds"] == 60.0
    assert report["resources"]["gpu_idle_fraction"] == 0.75
    assert report["failure"]["primary_reason"] == "scorer_timeout"
    assert report["secret_scan"] == {"ok": True, "matches": []}


def test_missing_observations_are_explicit_unknown(tmp_path: Path):
    module = load_module("experiment_workflow_baseline_unknown", ROOT / "scripts/experiment_workflow_baseline.py")
    (tmp_path / "metadata.json").write_text(json.dumps({"run": "minimal"}))
    (tmp_path / "events.jsonl").write_text("")
    report = module.collect(tmp_path)
    assert report["scorer"]["timeout_rate"] == "unknown"
    assert report["resources"]["reward_worker_peak_rss_gib"] == "unknown"
    assert report["resources"]["gpu_idle_fraction"] == "unknown"
