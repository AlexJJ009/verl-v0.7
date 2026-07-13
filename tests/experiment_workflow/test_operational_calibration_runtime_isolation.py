from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "scripts/run_code_task_operational_calibration_queue.sh"
PHASE = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh"
PROFILE = ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh"


def _load_port_checker():
    path = ROOT / "scripts/check_calibration_port_quiet.py"
    spec = importlib.util.spec_from_file_location("check_calibration_port_quiet", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_calibration_uses_disjoint_supported_port_interfaces() -> None:
    phase = PHASE.read_text()
    queue = QUEUE.read_text()
    assert "ray start --head" in phase
    assert '--min-worker-port="$CALIBRATION_RAY_WORKER_PORT_MIN"' in phase
    assert '--max-worker-port="$CALIBRATION_RAY_WORKER_PORT_MAX"' in phase
    assert '+trainer.ray_master_port_range="[$CALIBRATION_TCPSTORE_PORT_MIN,$CALIBRATION_TCPSTORE_PORT_MAX]"' in phase
    assert "ray.init.min_worker_port" not in phase
    assert "assert_port_ranges" in queue
    assert "wait_for_runtime_quiet" in queue


def test_controlled_port_parser_and_busy_listener() -> None:
    module = _load_port_checker()
    assert module.controlled_ports("21000-21002,35000-35001") == {21000, 21001, 21002, 35000, 35001}
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    result = subprocess.run(
        ["python3", str(ROOT / "scripts/check_calibration_port_quiet.py")],
        env={**os.environ, "CALIBRATION_PORT_DOMAINS": f"{port}-{port}"},
        text=True,
        capture_output=True,
        check=False,
    )
    sock.close()
    assert result.returncode != 0
    assert str(port) in result.stderr


def test_runtime_isolation_does_not_change_resource_profile() -> None:
    before = subprocess.run(
        ["bash", "-lc", f"source {PROFILE}; stage123_profile_hash"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    after = subprocess.run(
        [
            "bash",
            "-lc",
            f"export CALIBRATION_RAY_WORKER_PORT_MIN=21000 CALIBRATION_TCPSTORE_PORT_MIN=35000; source {PROFILE}; stage123_profile_hash",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert before == after


def test_cleanup_never_uses_global_ray_stop() -> None:
    cleanup = (ROOT / "scripts/validation_deadline_controller.py").read_text()
    assert 'command(["ray", "stop", "--force"])' not in cleanup


def test_ppo_propagates_master_port_range_to_worker_group() -> None:
    trainer = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    assert 'OmegaConf.select(self.config.trainer, "ray_master_port_range")' in trainer
    assert 'wg_kwargs["master_port_range"]' in trainer
    assert "**wg_kwargs" in trainer
