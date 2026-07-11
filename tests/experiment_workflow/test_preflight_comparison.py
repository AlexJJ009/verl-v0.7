from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def base_report():
    runner = load("comparison_runner", ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_benchmark.py")
    return runner.render(ROOT / "tests/experiment_workflow/fixtures/preflight_three_phase")


def optimized(report):
    result = copy.deepcopy(report)
    for phase in result["phases"]:
        phase["metrics"]["valid_scores_per_minute"] *= 1.2
        phase["metrics"]["validation_elapsed_seconds"] = 900
    return result


def test_accepted_optimization():
    checker = load("comparison_checker", ROOT / "scripts/compare_code_task_preflight.py")
    before = base_report()
    result = checker.compare(before, optimized(before))
    assert result["decision"] == "optimized"
    assert all(item["optimized"] for item in result["phases"])


def test_semantic_downscope_is_rejected_even_if_faster():
    checker = load("comparison_checker_semantic", ROOT / "scripts/compare_code_task_preflight.py")
    before = base_report(); after = optimized(before)
    after["semantic_hash"] = "0" * 64
    assert checker.compare(before, after)["decision"] == "rejected_semantic_change"


def test_neutral_or_over_30_minutes_is_not_optimized():
    checker = load("comparison_checker_neutral", ROOT / "scripts/compare_code_task_preflight.py")
    before = base_report(); after = copy.deepcopy(before)
    assert checker.compare(before, after)["decision"] == "neutral_or_regressed"
    after = optimized(before)
    after["phases"][1]["metrics"]["validation_elapsed_seconds"] = 1801
    result = checker.compare(before, after)
    assert result["decision"] == "neutral_or_regressed"
    assert result["phases"][1]["hard_wall_pass"] is False
