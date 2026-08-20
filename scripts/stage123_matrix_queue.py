#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage123_matrix_manifest import load

ROOT = Path(__file__).resolve().parents[1]


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state_path(root: Path, run_id: str) -> Path:
    return root / f"{run_id}.json"


def wait_for_operator(root: Path) -> bool:
    while (root / "PAUSE").exists():
        if (root / "STOP").exists():
            return False
        time.sleep(5)
    return not (root / "STOP").exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load(args.manifest)
    args.state_root.mkdir(parents=True, exist_ok=True)
    rendered_manifest = args.state_root / "rendered_manifest.json"
    atomic_write(rendered_manifest, manifest)
    statuses: dict[str, str] = {}
    for run in manifest["runs"]:
        source_run = run.get("source", {}).get("run_id")
        if source_run and statuses.get(source_run) != "succeeded":
            status = {
                "schema_version": 1,
                "run_id": run["id"],
                "status": "skipped",
                "attempt": 0,
                "failure": {"code": "dependency_failed", "source_run_id": source_run},
                "transitions": [{"from": "pending", "to": "skipped", "at": time.time()}],
            }
            atomic_write(state_path(args.state_root, run["id"]), status)
            statuses[run["id"]] = "skipped"
            continue
        if not wait_for_operator(args.state_root):
            return 2

        command = [
            sys.executable,
            str(ROOT / "scripts/stage123_phase_adapter.py"),
            "--manifest",
            str(rendered_manifest),
            "--run-id",
            run["id"],
        ]
        if args.dry_run:
            command.append("--dry-run")
        started_at = time.time()
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), start_new_session=True)
        state = {
            "schema_version": 1,
            "run_id": run["id"],
            "status": "running",
            "attempt": 1,
            "child_id": str(process.pid),
            "started_at": started_at,
            "transitions": [{"from": "pending", "to": "running", "at": started_at}],
        }
        atomic_write(state_path(args.state_root, run["id"]), state)
        return_code = process.wait()
        completed_at = time.time()
        state["return_code"] = return_code
        state["completed_at"] = completed_at
        state["status"] = "succeeded" if return_code == 0 else "failed"
        state["transitions"].append({"from": "running", "to": state["status"], "at": completed_at})
        if return_code != 0:
            state["failure"] = {"code": "child_exit_nonzero", "return_code": return_code}
        atomic_write(state_path(args.state_root, run["id"]), state)
        statuses[run["id"]] = state["status"]

    return 0 if all(status == "succeeded" for status in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
