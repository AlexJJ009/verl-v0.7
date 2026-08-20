from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_checked_pipeline_preserves_child_failure_through_tee(tmp_path: Path):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_checked_pipeline.sh"),
            str(tmp_path / "log"),
            "bash",
            "-c",
            "echo before; exit 23",
        ]
    )
    assert result.returncode == 23 and "before" in (tmp_path / "log").read_text()


def test_socket_deny_layer_blocks_attempted_network():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "tests/experiment_workflow")}
    result = subprocess.run(
        ["python3", "-c", "import socket_deny,socket; socket.socket()"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "network disabled" in result.stderr


def test_full_gate_runs_one_nonduplicative_cpu_suite():
    text = (ROOT / "scripts/check_experiment_workflow_full.sh").read_text()
    assert text.count("python -m pytest") == 1
    assert "tests/experiment_workflow" in text
    assert "test_validation_generation_logging.py" in text
    assert "run_code_task_qwen3_1p7b_stage123_queue.sh" not in text
