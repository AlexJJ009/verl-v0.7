#!/usr/bin/env python3
"""Enforce validation hard walls using explicit run ownership evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return len(fields) > 2 and fields[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def group_alive(pgid: int) -> bool:
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text().split()
            if len(fields) > 4 and int(fields[4]) == pgid and fields[2] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return False


def terminate_group(pgid: int, grace_seconds: float, sleep=time.sleep) -> dict:
    result = {"pgid": pgid, "term_sent": False, "kill_sent": False}
    try: os.killpg(pgid, signal.SIGTERM); result["term_sent"] = True
    except ProcessLookupError: return result
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not group_alive(pgid): return result
        sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    try: os.killpg(pgid, signal.SIGKILL); result["kill_sent"] = True
    except ProcessLookupError: pass
    return result


def command(args: list[str]) -> dict:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"command": args, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}


def cleanup(ownership: dict, grace_seconds: float = 10.0) -> dict:
    actions = []
    pgid = int(ownership.get("process_group_id") or 0)
    if pgid > 0: actions.append({"kind": "process_group", **terminate_group(pgid, grace_seconds)})
    for pid in ownership.get("descendant_pids", []):
        if alive(int(pid)):
            try: os.kill(int(pid), signal.SIGKILL); actions.append({"kind": "descendant", "pid": int(pid), "returncode": 0})
            except OSError as exc: actions.append({"kind": "descendant", "pid": int(pid), "returncode": 1, "error": str(exc)})
    for session in ownership.get("tmux_sessions", []): actions.append({"kind": "tmux", **command(["tmux", "kill-session", "-t", session])})
    for container in ownership.get("docker_containers", []): actions.append({"kind": "docker", **command(["docker", "rm", "-f", container])})
    if ownership.get("ray_address"):
        actions.append({"kind": "ray", **command(["ray", "stop", "--force"])})
    reap_deadline = time.monotonic() + min(max(grace_seconds, 0.1), 2.0)
    while time.monotonic() < reap_deadline:
        residual_pids = [int(pid) for pid in ownership.get("descendant_pids", []) if alive(int(pid))]
        if not residual_pids:
            break
        time.sleep(0.02)
    residual_pids = [int(pid) for pid in ownership.get("descendant_pids", []) if alive(int(pid))]
    gpu_pids = set(int(pid) for pid in ownership.get("gpu_pids", []))
    live_gpu_pids = []
    query = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if query["returncode"] == 0:
        current = {int(line.strip()) for line in query["stdout"].splitlines() if line.strip().isdigit()}
        live_gpu_pids = sorted(gpu_pids & current)
    elif gpu_pids:
        live_gpu_pids = sorted(gpu_pids)
    command_failures = [item for item in actions if item.get("returncode", 0) != 0 and not (item["kind"] in {"tmux", "docker"} and "not found" in (item.get("stderr", "") + item.get("stdout", "")).lower())]
    attribution_proven = bool(ownership.get("docker_containers")) and bool(ownership.get("container_init_pid"))
    released = attribution_proven and not residual_pids and not live_gpu_pids and not command_failures
    return {"actions": actions, "residual_pids": residual_pids, "live_run_gpu_pids": live_gpu_pids, "cleanup_failures": command_failures, "ownership_attribution_proven": attribution_proven, "resources_released": released}


def evaluate(ownership: dict, now_s: float) -> dict:
    complete = bool(ownership.get("complete_validation_metrics"))
    deadline = float(ownership["validation_ready_epoch_s"]) + float(ownership.get("deadline_seconds", 1800))
    return {"complete": complete, "deadline_epoch_s": deadline, "timed_out": not complete and now_s >= deadline}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--ownership", type=Path, required=True); parser.add_argument("--report", type=Path, required=True); parser.add_argument("--now-epoch-s", type=float); parser.add_argument("--grace-seconds", type=float, default=10); args = parser.parse_args()
    ownership = json.loads(args.ownership.read_text()); state = evaluate(ownership, args.now_epoch_s if args.now_epoch_s is not None else time.time())
    report = {"schema_version": 1, "run_id": ownership.get("run_id"), "generated_at": datetime.now(timezone.utc).isoformat(), **state}
    if state["timed_out"]:
        report.update({"status": "blocked", "reason": "validation_deadline_exceeded", **cleanup(ownership, args.grace_seconds)})
    elif state["complete"]:
        report.update({"status": "complete", "resources_released": False})
    else:
        report.update({"status": "active", "resources_released": False})
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__": sys.exit(main())
