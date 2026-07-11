from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def report_and_policy():
    runner = load("budget_runner", ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_benchmark.py")
    report = runner.render(ROOT / "tests/experiment_workflow/fixtures/preflight_three_phase")
    policy = json.loads((ROOT / "tests/experiment_workflow/fixtures/preflight_policy.json").read_text())
    return report, policy


def test_passing_report():
    checker = load("budget_checker", ROOT / "scripts/check_code_task_preflight_budget.py")
    report, policy = report_and_policy()
    assert checker.check(report, policy) == {"schema_version": 1, "ok": True, "decision": "pass", "failures": []}


def test_soft_failure_requires_user_decision():
    checker = load("budget_checker_soft", ROOT / "scripts/check_code_task_preflight_budget.py")
    report, policy = report_and_policy()
    report["phases"][0]["metrics"]["gpu_wait_fraction"] = 0.9
    result = checker.check(report, policy)
    assert result["decision"] == "user_decision_required"
    assert result["failures"][0]["tier"] == "soft"


def test_hard_and_missing_metrics_fail_closed():
    checker = load("budget_checker_hard", ROOT / "scripts/check_code_task_preflight_budget.py")
    report, policy = report_and_policy()
    report["contract"]["max_response_length"] = 4096
    del report["phases"][1]["metrics"]["validation_elapsed_seconds"]
    result = checker.check(report, policy)
    assert result["decision"] == "blocked"
    assert {item["tier"] for item in result["failures"]} == {"hard"}


def test_force_like_arguments_are_rejected(tmp_path: Path):
    report, policy = report_and_policy()
    report_path, policy_path = tmp_path / "report.json", tmp_path / "policy.json"
    report_path.write_text(json.dumps(report)); policy_path.write_text(json.dumps(policy))
    result = subprocess.run(["python3", str(ROOT / "scripts/check_code_task_preflight_budget.py"), "--report", str(report_path), "--policy", str(policy_path), "--force"], text=True, capture_output=True)
    assert result.returncode == 2
    assert "unknown arguments" in result.stderr
