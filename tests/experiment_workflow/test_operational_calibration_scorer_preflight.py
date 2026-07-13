from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "recipe/on_policy_wdl_sft/code_task"


def load_probe():
    path = RECIPE / "check_official_scorer_dependencies.py"
    spec = importlib.util.spec_from_file_location("scorer_dependencies", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_probe_accepts_complete_official_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_probe()
    monkeypatch.setenv(
        "LCB_INPUT_OUTPUT_INDEX",
        "/data-2/evaluator_assets/livecodebench_cache/index/release_v5_input_output.sqlite",
    )
    result = module.validate("/workspace/verl:/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench")
    assert result["ok"] is True
    assert "evalplus.evaluate" in result["imports"]
    assert "lcb_runner.evaluation.compute_code_generation_metrics" in result["imports"]


def test_probe_rejects_missing_lcb_path_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_probe()
    monkeypatch.setenv("LCB_INPUT_OUTPUT_INDEX", "/no/such/index.sqlite")
    with pytest.raises(RuntimeError, match="PYTHONPATH is incomplete"):
        module.validate("/workspace/verl:/data-1/code_eval_envs/official_site")
    with pytest.raises(RuntimeError, match="input/output index missing"):
        module.validate("/workspace/verl:/data-1/code_eval_envs/official_site:/data-1/code_eval_envs/LiveCodeBench")


def test_phase_probes_dependencies_before_starting_ray() -> None:
    text = (RECIPE / "run_code_task_operational_calibration_phase.sh").read_text()
    assert text.index("check_official_scorer_dependencies.py") < text.index("ray start --head")
    assert "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH" not in text


def test_stage_wrappers_propagate_scorer_pythonpath() -> None:
    for name in ("run_s1_code_base.sh", "run_s2_code_model2_rollout_common.sh"):
        text = (RECIPE / name).read_text()
        override = '+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="\'${PYTHONPATH}\'"'
        assert text.count(override) == 1


def test_phase_dependency_failure_prevents_ray_start(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "ray-started"
    ray = fake_bin / "ray"
    ray.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    ray.chmod(0o700)
    result = subprocess.run(
        ["bash", str(RECIPE / "run_code_task_operational_calibration_phase.sh"), "stage1"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CODE_EVAL_OFFICIAL_SITE": str(tmp_path / "missing-evalplus"),
            "LCB_REPO_DIR": str(tmp_path / "missing-lcb"),
            "LCB_INPUT_OUTPUT_INDEX": str(tmp_path / "missing.sqlite"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "dependency_failure" in result.stderr
    assert not marker.exists()
