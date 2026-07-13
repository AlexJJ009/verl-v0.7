#!/usr/bin/env python3
"""CPU-testable experiment execution state machine."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Protocol


TERMINAL_STATES = {"succeeded", "failed", "deadline_exceeded", "cleanup_failed"}


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    schema_version: int
    run_id: str
    status: str
    attempt: int
    started_at: float | None = None
    deadline_at: float | None = None
    completed_at: float | None = None
    child_id: str | None = None
    failure: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    max_attempts: int = 1
    resume_from_checkpoint: bool = False
    transitions: list[dict[str, Any]] = field(default_factory=list)


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ChildAdapter(Protocol):
    def start(self, command: list[str], env: dict[str, str]) -> str: ...
    def poll(self, child_id: str) -> int | None: ...
    def terminate(self, child_id: str, grace_seconds: float) -> dict[str, Any]: ...


class SubprocessAdapter:
    def __init__(self) -> None:
        self.children: dict[str, subprocess.Popen[Any]] = {}

    def start(self, command: list[str], env: dict[str, str]) -> str:
        process = subprocess.Popen(command, env=env, start_new_session=True)
        child_id = str(process.pid)
        self.children[child_id] = process
        return child_id

    def poll(self, child_id: str) -> int | None:
        return self.children[child_id].poll()

    def terminate(self, child_id: str, grace_seconds: float) -> dict[str, Any]:
        process = self.children.get(child_id)
        if process is None or process.poll() is not None:
            return {"resources_released": True, "term_sent": False, "kill_sent": False}
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(0.05, grace_seconds))
        kill_sent = process.poll() is None
        if kill_sent:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return {"resources_released": process.poll() is not None, "term_sent": True, "kill_sent": kill_sent}


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state(path: Path, run_id: str) -> ExecutionState:
    if not path.exists():
        return ExecutionState(schema_version=1, run_id=run_id, status="pending", attempt=0)
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("run_id") != run_id:
        raise ValueError("state identity mismatch")
    return ExecutionState(**value)


def transition(state: ExecutionState, status: str, now: float, **fields: Any) -> None:
    previous = state.status
    state.status = status
    for key, value in fields.items():
        setattr(state, key, value)
    state.transitions.append({"from": previous, "to": status, "at": now})


def failure(code: str, message: str, **context: Any) -> dict[str, Any]:
    return asdict(Failure(code=code, message=message, context=context))


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    command: list[str]
    timeout_seconds: float
    poll_seconds: float = 1.0
    cleanup_grace_seconds: float = 10.0
    env: dict[str, str] = field(default_factory=dict)
    max_attempts: int = 1
    resumable_failure_codes: tuple[str, ...] = ()


def recovery_decision(state: ExecutionState, spec: RunSpec, failure_code: str, checkpoint_available: bool) -> dict[str, Any]:
    resumable = failure_code in spec.resumable_failure_codes and state.attempt < spec.max_attempts
    if failure_code == "checkpoint_available_child_exit" and not checkpoint_available:
        resumable = False
    return {
        "resume": resumable,
        "attempt": state.attempt,
        "max_attempts": spec.max_attempts,
        "resume_from_checkpoint": resumable and checkpoint_available,
        "failure_code": failure_code,
    }


def load_recovery_policy(path: Path | None) -> tuple[int, tuple[str, ...]]:
    if path is None:
        return 1, ()
    value = json.loads(path.read_text())
    codes = value.get("resumable_failure_codes")
    if value.get("schema_version") != 1 or not isinstance(value.get("max_attempts"), int):
        raise ValueError("unsupported recovery policy")
    if value["max_attempts"] < 1 or not isinstance(codes, list) or not all(isinstance(item, str) for item in codes):
        raise ValueError("invalid recovery policy")
    return value["max_attempts"], tuple(codes)


class ExecutionCore:
    def __init__(self, state_root: Path, adapter: ChildAdapter, clock: Clock | None = None) -> None:
        self.state_root = state_root
        self.adapter = adapter
        self.clock = clock or SystemClock()

    def state_path(self, run_id: str) -> Path:
        return self.state_root / f"{run_id}.json"

    def persist(self, state: ExecutionState) -> None:
        atomic_write(self.state_path(state.run_id), asdict(state))
        event_path = self.state_root / "events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": 1,
            "run_id": state.run_id,
            "status": state.status,
            "attempt": state.attempt,
            "child_id": state.child_id,
            "failure": state.failure,
            "cleanup": state.cleanup,
            "transition": state.transitions[-1] if state.transitions else None,
        }
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    def run(self, spec: RunSpec) -> ExecutionState:
        state = load_state(self.state_path(spec.run_id), spec.run_id)
        if state.status in TERMINAL_STATES:
            return state
        if state.status == "running" and state.child_id is not None:
            return self.resume(spec, state)
        now = self.clock.now()
        state.attempt += 1
        state.max_attempts = spec.max_attempts
        child_id = self.adapter.start(spec.command, {**os.environ, **spec.env})
        transition(
            state,
            "running",
            now,
            started_at=now,
            deadline_at=now + spec.timeout_seconds,
            child_id=child_id,
            completed_at=None,
            failure=None,
            cleanup=None,
        )
        self.persist(state)
        return self.resume(spec, state)

    def resume(self, spec: RunSpec, state: ExecutionState | None = None) -> ExecutionState:
        state = state or load_state(self.state_path(spec.run_id), spec.run_id)
        if state.status in TERMINAL_STATES:
            failure_code = (state.failure or {}).get("code", "")
            checkpoint_available = failure_code == "checkpoint_available_child_exit"
            decision = recovery_decision(state, spec, failure_code, checkpoint_available)
            if not decision["resume"]:
                state.transitions.append({"from": state.status, "to": state.status, "at": self.clock.now(), "recovery": decision})
                self.persist(state)
                return state
            state.resume_from_checkpoint = decision["resume_from_checkpoint"]
            transition(state, "pending", self.clock.now(), child_id=None, completed_at=None, cleanup=None)
            self.persist(state)
            return self.run(spec)
        if state.status != "running" or state.child_id is None or state.deadline_at is None:
            transition(
                state,
                "failed",
                self.clock.now(),
                completed_at=self.clock.now(),
                failure=failure("invalid_resume_state", "cannot resume incomplete running state", status=state.status),
            )
            self.persist(state)
            return state
        while True:
            now = self.clock.now()
            returncode = self.adapter.poll(state.child_id)
            if returncode is not None:
                if returncode == 0:
                    transition(state, "succeeded", now, completed_at=now)
                else:
                    cleanup = self.adapter.terminate(state.child_id, spec.cleanup_grace_seconds)
                    status = "failed" if cleanup.get("resources_released") else "cleanup_failed"
                    transition(
                        state,
                        status,
                        now,
                        completed_at=now,
                        cleanup=cleanup,
                        failure=failure("child_exit", "child process exited unsuccessfully", returncode=returncode),
                    )
                self.persist(state)
                return state
            if now >= state.deadline_at:
                cleanup = self.adapter.terminate(state.child_id, spec.cleanup_grace_seconds)
                status = "deadline_exceeded" if cleanup.get("resources_released") else "cleanup_failed"
                transition(
                    state,
                    status,
                    now,
                    completed_at=now,
                    cleanup=cleanup,
                    failure=failure("deadline_exceeded", "execution deadline exceeded", deadline_at=state.deadline_at, observed_at=now),
                )
                self.persist(state)
                return state
            self.clock.sleep(spec.poll_seconds)


def parse_command(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
        raise ValueError("command must be a non-empty JSON string array")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("queue", "phase"))
    parser.add_argument("--run-id", default=os.environ.get("CALIBRATION_RUN_ID", "operational-calibration"))
    parser.add_argument("--state-root", type=Path, default=Path(os.environ.get("CALIBRATION_STATE_ROOT", "/data-1/tmp/verl_agent_scratch/experiment_workflow/state")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.environ.get("CALIBRATION_DEADLINE_SECONDS", "1800")))
    parser.add_argument("--command-json", default=os.environ.get("CALIBRATION_CHILD_COMMAND_JSON"))
    parser.add_argument("--recovery-policy", type=Path)
    parser.add_argument("--resume", action="store_true")
    args, legacy = parser.parse_known_args()
    command_json = args.command_json
    if command_json is None:
        result = failure("missing_child_command", "CALIBRATION_CHILD_COMMAND_JSON or --command-json is required", legacy_args=legacy)
        print(json.dumps({"ok": False, "failure": result}, sort_keys=True))
        return 2
    try:
        max_attempts, resumable_codes = load_recovery_policy(args.recovery_policy)
        spec = RunSpec(args.run_id, parse_command(command_json), args.timeout_seconds, max_attempts=max_attempts, resumable_failure_codes=resumable_codes)
        core = ExecutionCore(args.state_root, SubprocessAdapter())
        state = core.resume(spec) if args.resume else core.run(spec)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "failure": failure("invalid_execution_request", str(exc))}, sort_keys=True))
        return 2
    print(json.dumps({"ok": state.status == "succeeded", "state": asdict(state)}, sort_keys=True))
    return 0 if state.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
