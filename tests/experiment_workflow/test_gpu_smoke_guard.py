# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_gpu_smoke.sh"


def run(env):
    return subprocess.run(["bash", str(SCRIPT)], text=True, capture_output=True, env={**os.environ, **env})


def test_unapproved_smoke_fails_before_runtime(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    result = run({"PREFLIGHT_MANIFEST": str(manifest), "PREFLIGHT_OUTPUT_ROOT": str(tmp_path / "scratch")})
    assert result.returncode != 0
    assert "ALLOW_CODE_PREFLIGHT_GPU_SMOKE=1" in result.stderr


def test_dry_run_is_scratch_only_and_labeled(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    scratch = Path("/data-1/tmp/verl_agent_scratch/experiment_workflow/test_gpu_smoke_guard")
    result = run(
        {
            "ALLOW_CODE_PREFLIGHT_GPU_SMOKE": "1",
            "PREFLIGHT_GPU_SMOKE_DRY_RUN": "1",
            "PREFLIGHT_MANIFEST": str(manifest),
            "PREFLIGHT_OUTPUT_ROOT": str(scratch),
        }
    )
    assert result.returncode == 0
    assert "evidence_class=infrastructure_preflight" in result.stdout


def test_non_scratch_output_is_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    result = run(
        {
            "ALLOW_CODE_PREFLIGHT_GPU_SMOKE": "1",
            "PREFLIGHT_GPU_SMOKE_DRY_RUN": "1",
            "PREFLIGHT_MANIFEST": str(manifest),
            "PREFLIGHT_OUTPUT_ROOT": "/data-1/checkpoints/not-allowed",
        }
    )
    assert result.returncode != 0
    assert "declared scratch root" in result.stderr
