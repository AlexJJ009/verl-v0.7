#!/usr/bin/env python3
"""Read-only persisted-event monitor for atomic, batch, and Stage123 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ATOMIC_TERMINAL_STATES = {"succeeded", "failed", "deadline_exceeded", "cleanup_failed"}
ATOMIC_STATES = ATOMIC_TERMINAL_STATES | {"pending", "running"}
BATCH_TERMINAL_STATES = {"completed", "completed_with_failures", "shared_failure", "stopped"}
BATCH_STATES = BATCH_TERMINAL_STATES | {"pending", "running", "paused_after_current", "stopping"}


def persisted_states(state_root: Path) -> dict[str, dict]:
    result = {}
    for path in state_root.glob("*.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid persisted execution state: {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"invalid persisted execution state schema: {path}")
        identity = value.get("run_id") or value.get("batch_id")
        status = value.get("status")
        if not isinstance(identity, str) or status not in ATOMIC_STATES | BATCH_STATES:
            raise ValueError(f"invalid persisted execution state schema: {path}")
        result[identity] = value
    return result


def event_identity(event: dict) -> str:
    identity = event.get("run_id") or event.get("item_id") or event.get("batch_id")
    if not isinstance(identity, str):
        raise ValueError("execution event lacks identity")
    return identity


def persisted_events(state_root: Path) -> list[dict]:
    path = state_root / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid persisted execution event line {index}: {exc}") from exc
        if event.get("schema_version") != 1:
            raise ValueError(f"invalid persisted execution event schema at line {index}")
        if "batch_id" in event:
            if not isinstance(event.get("event"), str) or event.get("state") not in BATCH_STATES:
                raise ValueError(f"invalid persisted batch event schema at line {index}")
        elif "run_id" in event:
            if event.get("status") not in ATOMIC_STATES:
                raise ValueError(f"invalid persisted atomic event schema at line {index}")
        else:
            raise ValueError(f"invalid persisted execution event schema at line {index}")
        event_identity(event)
        events.append(event)
    return events


def manifest_run_ids(manifest: dict) -> list[str]:
    if isinstance(manifest.get("runs"), list):
        return [run["id"] for run in manifest["runs"]]
    if isinstance(manifest.get("items"), list):
        return [run_id for item in manifest["items"] for run_id in item.get("expected_run_ids", [])]
    raise ValueError("monitor manifest has no runs or batch items")


def notification_state_from_event(event: dict, run_ids: list[str], state_root: Path) -> dict:
    identity = event_identity(event)
    status = event.get("status") or event.get("state")
    failed = status in {"failed", "deadline_exceeded", "cleanup_failed", "shared_failure"}
    return {
        "authority_type": "experiment_execution_core_event_v1",
        "run_id": identity,
        "manifest_run_ids": run_ids,
        "execution_status": "failed" if status == "shared_failure" else status,
        "execution_attempt": event.get("attempt", 0),
        "transition": event.get("transition") or event.get("event"),
        "failure": event.get("failure"),
        "cleanup": event.get("cleanup") or ({"resources_released": True} if status == "shared_failure" else None),
        "background": (event.get("failure") or {}).get("message", "Experiment execution lifecycle event"),
        "evidence": json.dumps({"status": status, "event": event.get("event"), "failure": event.get("failure"), "cleanup": event.get("cleanup")}, sort_keys=True),
        "cost": "Execution stopped" if failed else "",
        "recommendation": "Inspect persisted execution state and events" if failed else "",
        "local_paths": f"execution_events={state_root / 'events.jsonl'}",
    }


def emit(policy: Path, ledger: Path, sender: list[str] | None, state: dict, scratch: Path) -> None:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(json.dumps(state))
    command = ["python3", str(policy), "--state", str(scratch), "--ledger", str(ledger)]
    if sender:
        command += ["--sender", *sender]
    subprocess.run(command, check=False)


def event_digest(event: dict) -> str:
    return hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def monitor_once(manifest: dict, state_root: Path, ledger: Path, policy: Path, sender: list[str] | None, cursor_path: Path) -> tuple[dict[str, dict], list[dict]]:
    states = persisted_states(state_root)
    events = persisted_events(state_root)
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else {"schema_version": 1, "event_digests": []}
    seen = set(cursor.get("event_digests", []))
    scratch = ledger.with_suffix(".state.json")
    run_ids = manifest_run_ids(manifest)
    for event in events:
        digest = event_digest(event)
        if digest in seen:
            continue
        emit(policy, ledger, sender, notification_state_from_event(event, run_ids, state_root), scratch)
        seen.add(digest)
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text(json.dumps({"schema_version": 1, "event_digests": sorted(seen)}, indent=2, sort_keys=True) + "\n")
    return states, events


def all_terminal(states: dict[str, dict]) -> bool:
    return bool(states) and all(state.get("status") in ATOMIC_TERMINAL_STATES | BATCH_TERMINAL_STATES for state in states.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--sender", nargs="+")
    parser.add_argument("--cursor", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cursor = args.cursor or args.ledger.with_suffix(".cursor.json")
    while True:
        states, events = monitor_once(manifest, args.state_root, args.ledger, args.policy, args.sender, cursor)
        if args.once or all_terminal(states):
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
