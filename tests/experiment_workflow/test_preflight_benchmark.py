from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "recipe/on_policy_wdl_sft/code_task"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_three_phase_fixture_contract():
    runner = load("preflight_runner", RECIPE / "run_code_task_preflight_benchmark.py")
    report = runner.render(ROOT / "tests/experiment_workflow/fixtures/preflight_three_phase")
    schema = json.loads((RECIPE / "preflight_benchmark_schema.json").read_text())
    jsonschema.validate(report, schema)
    assert [phase["phase"] for phase in report["phases"]] == ["stage1", "stage2", "stage3"]
    assert report["contract"]["max_response_length"] == 8192
    assert report["contract"]["validation_datasets"] == ["HumanEval+", "MBPP+", "LiveCodeBench"]
    assert all(len(phase["repetitions"]) == 4 for phase in report["phases"])
    assert all(sum(not rep["warmup"] for rep in phase["repetitions"]) == 3 for phase in report["phases"])
    assert report["phases"][1]["model_topology"] == "fixed_model2_rollout_joint_fused_loss"
    assert report["phases"][1]["fixed_model2_source"]
    assert all(len(dataset["row_ids"]) in (16, 32) for phase in report["phases"] for dataset in phase["datasets"])


def test_warmup_is_excluded_from_aggregation():
    runner = load("preflight_runner_warmup", RECIPE / "run_code_task_preflight_benchmark.py")
    report = runner.render(ROOT / "tests/experiment_workflow/fixtures/preflight_three_phase")
    assert report["phases"][0]["metrics"]["timeout_rate"] == 2 / 64
    assert report["phases"][0]["metrics"]["validation_elapsed_seconds"] == 430.0
