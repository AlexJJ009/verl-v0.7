from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check_pm2_ci_keepalive.py"
CONTRACT = ROOT / "config/pm2/experiment-reliability-ci.json"


def fake_pm2(path: Path, fail_on: str = "") -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        f"fail_on={fail_on!r}\n"
        "command=sys.argv[1]\n"
        "home=pathlib.Path(os.environ['PM2_HOME']); home.mkdir(parents=True, exist_ok=True)\n"
        "log=home/'commands.jsonl'\n"
        "with log.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if command == fail_on: raise SystemExit(7)\n"
        "if command == 'start': (home/'running').write_text('1')\n"
        "elif command == 'restart': (home/'restarted').write_text('1')\n"
        "elif command == 'save': (home/'dump.pm2').write_text('saved')\n"
        "elif command == 'resurrect' and not (home/'dump.pm2').is_file(): raise SystemExit(8)\n"
        "elif command == 'status': print('[]')\n"
    )
    path.chmod(0o755)
    return path


def run(contract: Path, pm2: Path | None = None):
    command = [
        sys.executable,
        str(CHECKER),
        "--contract",
        str(contract),
        "--repo-root",
        str(ROOT),
        "--require-no-systemd",
    ]
    if pm2 is not None:
        command.extend(["--pm2-bin", str(pm2)])
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_real_contract_and_fake_pm2_lifecycle_pass(tmp_path):
    result = run(CONTRACT, fake_pm2(tmp_path / "pm2"))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("needle", ["systemctl", "pm2 startup", "bad.service"])
def test_systemd_and_pm2_startup_surfaces_are_rejected(tmp_path, needle):
    contract = json.loads(CONTRACT.read_text())
    contract["processes"][0]["args"].append(needle)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    assert run(path).returncode != 0


def test_missing_non_systemd_bootstrap_is_explicit_fail_closed_state():
    contract = json.loads(CONTRACT.read_text())
    assert contract["reboot_restore"] == {
        "bootstrap": None,
        "reboot_restore_available": False,
        "restore_command": "pm2 resurrect",
    }
    assert run(CONTRACT).returncode == 0


def test_pm2_command_failure_is_not_hidden(tmp_path):
    result = run(CONTRACT, fake_pm2(tmp_path / "pm2", fail_on="restart"))
    assert result.returncode != 0
    assert "restart" in result.stdout
