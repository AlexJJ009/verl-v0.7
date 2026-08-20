#!/usr/bin/env python3
"""Monitor the Code Stage123 event stream and send deduplicated milestones."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

NOTIFIER = Path("/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py")


def send(title: str, body: str) -> None:
    subprocess.run(
        ["python3", str(NOTIFIER), "--title", title, "--body", body],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def tmux_alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue-session", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    sent = (
        {json.loads(line)["key"] for line in args.ledger.read_text().splitlines() if line.strip()}
        if args.ledger.exists()
        else set()
    )
    offset = 0
    missing_polls = 0
    while True:
        lines = args.event_log.read_text().splitlines() if args.event_log.exists() else []
        events = [json.loads(line) for line in lines[offset:] if line.strip()]
        offset = len(lines)
        for event in events:
            name = event["event"]
            key = f"{name}:{event.get('run_id', '')}"
            if key in sent:
                continue
            if name == "stage_run_started":
                send(
                    f"Code run started: {event['run_id']}",
                    f"Status: running\n\nWhat happened: Code Stage123 entered {event['phase']}.\nEvidence: run={event['run_id']}.\nNext action: Continue sequential execution with full Code-3 n=3 validation.",
                )
            elif name == "queue_failed":
                send(
                    "Code Stage123 queue failed",
                    f"Status: failed\n\nWhat happened: {event.get('reason', 'Queue reported a failure.')}\nEvidence: {args.event_log}\nNext action: Inspect the queue log before resuming.",
                )
            elif name == "queue_completed":
                send(
                    "Code Stage123 queue completed",
                    f"Status: completed\n\nWhat happened: All 16 Code Stage123 runs completed.\nEvidence: {args.event_log}\nNext action: Apply the release gate before DB/W&B publication.",
                )
            else:
                continue
            with args.ledger.open("a") as handle:
                handle.write(json.dumps({"key": key, "event": event}, sort_keys=True) + "\n")
            sent.add(key)
            if name in {"queue_failed", "queue_completed"}:
                return 0
        missing_polls = 0 if tmux_alive(args.queue_session) else missing_polls + 1
        if missing_polls >= 3:
            send(
                "Code Stage123 queue stopped unexpectedly",
                f"Status: failed\n\nWhat happened: tmux session {args.queue_session} disappeared without a terminal event.\nEvidence: {args.event_log}\nNext action: Inspect the queue tmux log.",
            )
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
