from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path):
    return None if not path.exists() else hashlib.sha256(path.read_bytes()).hexdigest()


def test_dry_run_is_scratch_only_and_manifest_consistent(tmp_path: Path):
    registry = Path("/data-1/experiment_registry/experiment_registry.sqlite")
    release = Path("/data-1/experiment_registry/training_release_gate.jsonl")
    before = (digest(registry), digest(release))
    scratch = tmp_path / "scratch"
    status = tmp_path / "status.tsv"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "STAGE123_MANIFEST_PYTHON": "/opt/venv/bin/python",
        "STAGE123_SCRATCH_ROOT": str(scratch),
        "QUEUE_STATUS_FILE": str(status),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    normalized = json.loads((scratch / "stage123.normalized.json").read_text())
    provenance = list(scratch.glob("**/*.provenance.json"))
    stage2_runs = [run for run in normalized["runs"] if run["phase"] == "stage2"]
    stage3_runs = [run for run in normalized["runs"] if run["phase"] == "stage3"]
    assert len(provenance) == len(stage2_runs)
    assert {path.stem.removesuffix(".provenance") for path in provenance} == {run["id"] for run in stage2_runs}
    assert all(json.loads(path.read_text())["release_eligible"] is False for path in provenance)
    rows = (scratch / "status.tsv").read_text().splitlines()
    assert sum("\tstage3\tpending_producer\t" in row for row in rows) == len(stage3_runs)
    assert not list(scratch.glob("**/stage2_final_model2/*"))
    assert "Stage3 blocked: pending current" in result.stdout
    assert before == (digest(registry), digest(release))
    assert "tmux new-session" not in result.stdout and "docker run" not in result.stdout
