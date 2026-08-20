#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate and sandbox-exercise the PM2-only CI keepalive contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FORBIDDEN = ("systemctl", "pm2 startup", ".service")


def load_contract(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("contract must be a JSON object")
    return data


def validate(contract: dict, repo_root: Path, require_no_systemd: bool) -> list[str]:
    failures: list[str] = []
    if contract.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    processes = contract.get("processes")
    if not isinstance(processes, list) or not processes:
        failures.append("processes must be a non-empty list")
        processes = []
    names: set[str] = set()
    for index, process in enumerate(processes):
        if not isinstance(process, dict):
            failures.append(f"process {index} must be an object")
            continue
        name = process.get("name")
        if not isinstance(name, str) or not name or name in names:
            failures.append(f"process {index} has invalid or duplicate name")
        names.add(str(name))
        if process.get("autorestart") is not True:
            failures.append(f"{name}: autorestart must be true")
        command = process.get("command")
        args = process.get("args")
        if not isinstance(command, str) or not command or not isinstance(args, list):
            failures.append(f"{name}: command/args are invalid")
        elif command == "bash" and args:
            target = repo_root / str(args[0])
            if not target.is_file():
                failures.append(f"{name}: command target does not exist")
        for key in ("output_log", "error_log"):
            value = process.get(key)
            if not isinstance(value, str) or not value.startswith("/data-1/tmp/verl_agent_scratch/"):
                failures.append(f"{name}: {key} must use the declared scratch root")
    reboot = contract.get("reboot_restore")
    if not isinstance(reboot, dict) or reboot.get("restore_command") != "pm2 resurrect":
        failures.append("reboot_restore must declare pm2 resurrect")
    elif reboot.get("reboot_restore_available") is True:
        bootstrap = reboot.get("bootstrap")
        if not isinstance(bootstrap, str) or not bootstrap or any(token in bootstrap for token in FORBIDDEN):
            failures.append("available reboot restore requires a verified non-systemd bootstrap")
    elif reboot.get("reboot_restore_available") is not False or reboot.get("bootstrap") is not None:
        failures.append("unavailable reboot restore must fail closed with bootstrap=null")
    if require_no_systemd:
        serialized = json.dumps(contract, sort_keys=True).lower()
        if any(token in serialized for token in FORBIDDEN):
            failures.append("contract contains forbidden systemd/startup surface")
        service_files = list(repo_root.rglob("*.service"))
        if service_files:
            failures.append(f"systemd unit is forbidden: {service_files[0]}")
    return failures


def exercise(pm2_bin: Path, contract_path: Path, contract: dict) -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pm2-ci-check-") as temp:
        env = {**os.environ, "PM2_HOME": str(Path(temp) / "home")}
        commands = (
            [str(pm2_bin), "start", str(contract_path)],
            [str(pm2_bin), "status", "--json"],
            [str(pm2_bin), "restart", contract["processes"][0]["name"]],
            [str(pm2_bin), "save"],
            [str(pm2_bin), "resurrect"],
        )
        for command in commands:
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
            if result.returncode != 0:
                failures.append(f"PM2 lifecycle failed: {' '.join(command[1:])}")
                break
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--require-no-systemd", action="store_true")
    parser.add_argument("--pm2-bin", type=Path)
    args = parser.parse_args()
    try:
        contract = load_contract(args.contract)
        failures = validate(contract, args.repo_root.resolve(), args.require_no_systemd)
        if args.pm2_bin is not None and not failures:
            failures.extend(exercise(args.pm2_bin.resolve(), args.contract.resolve(), contract))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures = [str(exc)]
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
