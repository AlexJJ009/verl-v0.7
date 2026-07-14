#!/usr/bin/env python3
"""Manifest-native Stage123 monitor with reviewed lifecycle notifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


TERMINAL_STATES = {"succeeded", "failed", "deadline_exceeded", "cleanup_failed"}
KNOWN_STATES = TERMINAL_STATES | {"pending", "running"}


def persisted_states(state_root: Path) -> dict[str, dict]:
    result = {}
    for path in state_root.glob("*.json"):
        if path.name == "events.jsonl":
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid persisted execution state: {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("run_id"), str):
            raise ValueError(f"invalid persisted execution state schema: {path}")
        result[value["run_id"]] = value
    return result


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
        if event.get("schema_version") != 1 or not isinstance(event.get("run_id"), str) or event.get("status") not in KNOWN_STATES:
            raise ValueError(f"invalid persisted execution event schema at line {index}")
        events.append(event)
    return events


def notification_state_from_event(event: dict, manifest_run_ids: list[str], state_root: Path) -> dict:
    return {
        "authority_type": "experiment_execution_core_event_v1",
        "run_id": event["run_id"],
        "manifest_run_ids": manifest_run_ids,
        "execution_status": event["status"],
        "execution_attempt": event.get("attempt", 0),
        "transition": event.get("transition"),
        "failure": event.get("failure"),
        "cleanup": event.get("cleanup"),
        "background": (event.get("failure") or {}).get("message", "Stage123 queue lifecycle event"),
        "evidence": json.dumps({"status": event["status"], "failure": event.get("failure"), "cleanup": event.get("cleanup")}, sort_keys=True),
        "cost": "GPU queue stopped" if event["status"] in {"failed", "deadline_exceeded", "cleanup_failed"} else "",
        "recommendation": "Inspect persisted execution state and events" if event["status"] in {"failed", "deadline_exceeded", "cleanup_failed"} else "",
        "local_paths": f"execution_state={state_root / (event['run_id'] + '.json')}; execution_events={state_root / 'events.jsonl'}",
    }


def emit(policy: Path, ledger: Path, sender: list[str] | None, state: dict, scratch: Path) -> None:
    scratch.write_text(json.dumps(state))
    cmd = ["python3", str(policy), "--state", str(scratch), "--ledger", str(ledger)]
    if sender: cmd += ["--sender", *sender]
    subprocess.run(cmd, check=False)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--state-root',type=Path,required=True); p.add_argument('--poll-seconds',type=float,default=60); p.add_argument('--ledger',type=Path,required=True); p.add_argument('--policy',type=Path,required=True); p.add_argument('--sender',nargs='+'); p.add_argument('--once',action='store_true'); args=p.parse_args()
    manifest=json.loads(args.manifest.read_text()); scratch=args.ledger.with_suffix('.state.json'); manifest_run_ids=[run['id'] for run in manifest['runs']]
    while True:
        states=persisted_states(args.state_root); events=persisted_events(args.state_root)
        for event in events:
            emit(args.policy,args.ledger,args.sender,notification_state_from_event(event,manifest_run_ids,args.state_root),scratch)
        if args.once or (states and all(state.get('status') in TERMINAL_STATES for state in states.values())): return 0
        if not states and not events: return 0
        time.sleep(args.poll_seconds)

if __name__=='__main__': raise SystemExit(main())
