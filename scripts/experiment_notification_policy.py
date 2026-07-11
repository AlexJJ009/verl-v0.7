#!/usr/bin/env python3
"""Emit only reviewed experiment lifecycle events through an injectable sender."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ALLOWED = {"run_started", "run_failed", "user_decision_required"}
SECRET = re.compile(r"(?:github_pat_[A-Za-z0-9_]+|(?:TOKEN|PASSWORD|SECRET|API_KEY)\s*[=:]\s*\S+)", re.I)


def event_for(state: dict) -> str | None:
    if state.get("decision_required"): return "user_decision_required"
    if state.get("terminal_failure") and state.get("cleanup_evidence"): return "run_failed"
    if int(state.get("training_step", 0)) >= 1 or state.get("complete_validation_metrics") is True: return "run_started"
    return None


def redact(text: str) -> str: return SECRET.sub("[REDACTED]", text)


def process(state: dict, ledger_path: Path, sender: list[str] | None) -> dict:
    event = event_for(state)
    if event not in ALLOWED: return {"event": None, "sent": False, "reason": "non_event"}
    key = hashlib.sha256(f"{state.get('run_id')}:{event}".encode()).hexdigest()
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()] if ledger_path.exists() else []
    if any(item.get("dedup_key") == key for item in ledger): return {"event": event, "sent": False, "reason": "duplicate"}
    message = redact("\n".join(str(state.get(field, "")) for field in ("background", "evidence", "cost", "recommendation", "local_paths")))
    delivery = {"returncode": 0}
    if sender:
        done = subprocess.run(sender + [event, message], text=True, capture_output=True, check=False)
        delivery = {"returncode": done.returncode, "stdout": done.stdout[-2000:], "stderr": done.stderr[-2000:]}
    record = {"run_id": state.get("run_id"), "event": event, "dedup_key": key, "message": message, "delivery": delivery}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"event": event, "sent": delivery["returncode"] == 0, "reason": "delivered" if delivery["returncode"] == 0 else "delivery_failed"}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state", type=Path, required=True); parser.add_argument("--ledger", type=Path, required=True); parser.add_argument("--sender", nargs="+"); args = parser.parse_args()
    result = process(json.loads(args.state.read_text()), args.ledger, args.sender); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": sys.exit(main())
