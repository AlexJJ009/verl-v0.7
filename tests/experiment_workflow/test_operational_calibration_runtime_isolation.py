# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_port_checker():
    path = ROOT / "scripts/check_calibration_port_quiet.py"
    spec = importlib.util.spec_from_file_location("check_calibration_port_quiet", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


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


def test_python_execution_core_does_not_use_global_ray_stop() -> None:
    source = (ROOT / "scripts/experiment_execution_core.py").read_text()
    assert "ray stop" not in source
    assert "docker system" not in source
