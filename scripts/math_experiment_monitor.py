#!/usr/bin/env python3
"""Monitor the integrated 1.7B math queue and send deduplicated WxPusher milestones."""

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
    sent = set()
    if args.ledger.exists():
        sent = {json.loads(line)["key"] for line in args.ledger.read_text().splitlines() if line.strip()}
    offset = 0
    missing_polls = 0
    while True:
        events = []
        if args.event_log.exists():
            lines = args.event_log.read_text().splitlines()
            events = [json.loads(line) for line in lines[offset:] if line.strip()]
            offset = len(lines)
        for event in events:
            name = event["event"]
            key = f"{name}:{event.get('step', event.get('run_id', ''))}"
            if key in sent:
                continue
            title = None
            body = None
            if name == "queue_started":
                title = "1.7B Math queue launched"
                body = "Status: running\n\nWhat happened: Cold-start and the authorized Stage123 handoff queue started.\nEvidence: Full Math-7 n=1 cold validation every 5 steps.\nNext action: Monitor will notify at admission, stage transitions, failure, or completion."
            elif name == "cold_candidate_evaluated":
                title = f"Math cold-start step {event['step']} evaluated"
                body = f"Status: {'admitted' if event['passed'] else 'running'}\n\nWhat happened: Complete Math-7 n=1 validation finished.\nEvidence: complete-format rate={event['format_contract_success_rate']:.2%}; gate=95%.\nNext action: {'Select Model1 and hand off to Stage123.' if event['passed'] else 'Continue to the next 5-step checkpoint.'}"
            elif name == "model1_selected":
                title = "Math Model1 selected"
                body = f"Status: completed\n\nWhat happened: Earliest passing cold-start checkpoint was selected automatically.\nEvidence: step={event['step']}, complete-format={event['format_contract_success_rate']:.2%}.\nNext action: Start the Stage123 matrix."
            elif name == "stage_run_started":
                title = f"Math run started: {event['run_id']}"
                body = f"Status: running\n\nWhat happened: Stage123 queue entered {event['phase']}.\nEvidence: run={event['run_id']}.\nNext action: Continue sequential queue execution."
            elif name == "queue_failed":
                title = "1.7B Math queue failed"
                body = f"Status: failed\n\nWhat happened: {event.get('reason', 'Queue reported a failure.')}\nEvidence: {args.event_log}\nNext action: Inspect the queue log before relaunching."
            elif name == "queue_completed":
                title = "1.7B Math queue completed"
                body = f"Status: completed\n\nWhat happened: Cold-start selection and all Stage123 runs completed.\nEvidence: {args.event_log}\nNext action: Apply the training release gate before publishing results."
            if title and body:
                send(title, body)
                with args.ledger.open("a") as handle:
                    handle.write(json.dumps({"key": key, "event": event}, sort_keys=True) + "\n")
                sent.add(key)
            if name in {"queue_failed", "queue_completed"}:
                return 0
        if tmux_alive(args.queue_session):
            missing_polls = 0
        else:
            missing_polls += 1
            if missing_polls >= 3:
                key = "queue_session_disappeared"
                if key not in sent:
                    send(
                        "1.7B Math queue stopped unexpectedly",
                        f"Status: failed\n\nWhat happened: tmux session {args.queue_session} disappeared without a terminal event.\nEvidence: {args.event_log}\nNext action: Inspect the queue tmux log.",
                    )
                return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
