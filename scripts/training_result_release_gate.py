#!/usr/bin/env python3
"""Deterministic release gate for training-result publication.

The gate records terminal monitor events and answers whether a run is allowed
to be published to the local registry or W&B cloud. It deliberately avoids any
LLM, W&B, or network dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE = Path("/data-1/experiment_registry/training_release_gate.jsonl")
TERMINAL_STATUSES = {"success_complete", "failed", "pending"}
RELEASABLE_STATUS = "success_complete"


@dataclass(frozen=True)
class GateEvent:
    ts: str
    run_name: str
    family: str
    status: str
    source: str
    checkpoint: str | None
    metrics: str | None
    final_step: int | None
    observed_step: int | None
    notes: str | None
    host: str

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GateEvent":
        return cls(
            ts=str(payload["ts"]),
            run_name=str(payload["run_name"]),
            family=str(payload["family"]),
            status=str(payload["status"]),
            source=str(payload.get("source", "unknown")),
            checkpoint=payload.get("checkpoint"),
            metrics=payload.get("metrics"),
            final_step=_optional_int(payload.get("final_step")),
            observed_step=_optional_int(payload.get("observed_step")),
            notes=payload.get("notes"),
            host=str(payload.get("host", "unknown")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "run_name": self.run_name,
            "family": self.family,
            "status": self.status,
            "source": self.source,
            "checkpoint": self.checkpoint,
            "metrics": self.metrics,
            "final_step": self.final_step,
            "observed_step": self.observed_step,
            "notes": self.notes,
            "host": self.host,
        }


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _infer_family(run_name: str) -> str:
    if "_" in run_name:
        return run_name.rsplit("_", 1)[0]
    return run_name


def _load_events(path: Path) -> list[GateEvent]:
    if not path.exists():
        return []
    events: list[GateEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(GateEvent.from_json(json.loads(stripped)))
            except Exception as exc:  # pragma: no cover - defensive error path
                raise RuntimeError(f"Invalid gate state {path}:{lineno}: {exc}") from exc
    return events


def _append_event(path: Path, event: GateEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_json(), sort_keys=True) + "\n")


def _latest_by_run(events: list[GateEvent], run_name: str) -> GateEvent | None:
    for event in reversed(events):
        if event.run_name == run_name:
            return event
    return None


def _family_blocking_event(events: list[GateEvent], family: str) -> GateEvent | None:
    """Return the latest failed event after the latest successful family event."""
    latest_success_index = -1
    latest_failed: tuple[int, GateEvent] | None = None
    for idx, event in enumerate(events):
        if event.family != family:
            continue
        if event.status == RELEASABLE_STATUS:
            latest_success_index = idx
            latest_failed = None
        elif event.status == "failed" and idx > latest_success_index:
            latest_failed = (idx, event)
    return latest_failed[1] if latest_failed else None


def _print_event(prefix: str, event: GateEvent | None) -> None:
    if event is None:
        print(f"{prefix}: none")
        return
    print(
        f"{prefix}: run={event.run_name} family={event.family} "
        f"status={event.status} step={event.observed_step}/{event.final_step} "
        f"checkpoint={event.checkpoint or 'none'} metrics={event.metrics or 'none'} "
        f"ts={event.ts}"
    )


def cmd_record(args: argparse.Namespace) -> int:
    if args.status not in TERMINAL_STATUSES:
        print(f"invalid status: {args.status}", file=sys.stderr)
        return 2
    run_name = args.run_name
    family = args.family or _infer_family(run_name)
    event = GateEvent(
        ts=_now(),
        run_name=run_name,
        family=family,
        status=args.status,
        source=args.source,
        checkpoint=args.checkpoint,
        metrics=args.metrics,
        final_step=args.final_step,
        observed_step=args.observed_step,
        notes=args.notes,
        host=socket.gethostname(),
    )
    _append_event(args.state, event)
    _print_event("recorded", event)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        events = _load_events(args.state)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_name = args.run_name
    family = args.family or _infer_family(run_name)
    latest = _latest_by_run(events, run_name)
    _print_event("latest_run_event", latest)
    if latest is None:
        print(f"BLOCKED: no release-gate event for run {run_name}")
        return 1
    if latest.status != RELEASABLE_STATUS:
        print(f"BLOCKED: latest run status is {latest.status}, not {RELEASABLE_STATUS}")
        return 1
    blocker = _family_blocking_event(events, family)
    if blocker is not None and blocker.run_name != run_name:
        _print_event("family_blocker", blocker)
        print(
            "BLOCKED: experiment family has a later failed attempt; "
            "wait for the next successful full-flow run"
        )
        return 1
    print(f"ALLOWED: run {run_name} is releasable")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    try:
        events = _load_events(args.state)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    selected = [event for event in events if not args.family or event.family == args.family]
    for event in selected[-args.limit :]:
        _print_event("event", event)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(os.environ.get("TRAINING_RELEASE_GATE_STATE", DEFAULT_STATE)),
        help=f"Gate JSONL state path (default: {DEFAULT_STATE})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Append a terminal gate event")
    record.add_argument("--run-name", required=True)
    record.add_argument("--family")
    record.add_argument("--status", required=True, choices=sorted(TERMINAL_STATUSES))
    record.add_argument("--source", default="manual")
    record.add_argument("--checkpoint")
    record.add_argument("--metrics")
    record.add_argument("--final-step", type=int)
    record.add_argument("--observed-step", type=int)
    record.add_argument("--notes")
    record.set_defaults(func=cmd_record)

    check = sub.add_parser("check", help="Check whether a run may be released")
    check.add_argument("--run-name", required=True)
    check.add_argument("--family")
    check.set_defaults(func=cmd_check)

    summary = sub.add_parser("summary", help="Print recent gate events")
    summary.add_argument("--family")
    summary.add_argument("--limit", type=int, default=20)
    summary.set_defaults(func=cmd_summary)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
