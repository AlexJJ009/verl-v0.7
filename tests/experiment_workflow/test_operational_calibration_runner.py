# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "scripts/experiment_execution_core.py"
QUEUE = ROOT / "scripts/run_code_task_operational_calibration_queue.sh"
PHASE = ROOT / "scripts/run_code_task_operational_calibration.sh"


def module():
    spec = importlib.util.spec_from_file_location("experiment_execution_core", CORE)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.value = now
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class FakeAdapter:
    def __init__(self, polls: list[int | None], cleanup_released: bool = True) -> None:
        self.polls = list(polls)
        self.cleanup_released = cleanup_released
        self.calls: list[tuple] = []

    def start(self, command: list[str], env: dict[str, str]) -> str:
        self.calls.append(("start", command, env.get("FIXTURE")))
        return "child-1"

    def poll(self, child_id: str) -> int | None:
        self.calls.append(("poll", child_id))
        return self.polls.pop(0) if self.polls else None

    def terminate(self, child_id: str, grace_seconds: float) -> dict:
        self.calls.append(("terminate", child_id, grace_seconds))
        return {"resources_released": self.cleanup_released, "term_sent": True, "kill_sent": False}


def spec(tool, run_id: str = "run"):
    return tool.RunSpec(
        run_id,
        ["fake", "command"],
        timeout_seconds=2,
        poll_seconds=1,
        cleanup_grace_seconds=0.5,
        env={"FIXTURE": "yes"},
    )


def test_start_success_persists_transitions_and_call_log(tmp_path: Path) -> None:
    tool = module()
    clock = FakeClock()
    adapter = FakeAdapter([None, 0])
    state = tool.ExecutionCore(tmp_path, adapter, clock).run(spec(tool))
    assert state.status == "succeeded"
    assert [(item["from"], item["to"]) for item in state.transitions] == [
        ("pending", "running"),
        ("running", "succeeded"),
    ]
    assert adapter.calls[0] == ("start", ["fake", "command"], "yes")
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "succeeded"


def test_failed_child_is_structured_and_release_stays_failed(tmp_path: Path) -> None:
    tool = module()
    state = tool.ExecutionCore(tmp_path, FakeAdapter([7]), FakeClock()).run(spec(tool))
    assert state.status == "failed"
    assert state.failure == {
        "code": "child_exit",
        "message": "child process exited unsuccessfully",
        "context": {"returncode": 7},
    }
    assert state.cleanup["resources_released"] is True


def test_deadline_terminates_owned_child_and_records_cleanup(tmp_path: Path) -> None:
    tool = module()
    adapter = FakeAdapter([None, None, None])
    clock = FakeClock()
    state = tool.ExecutionCore(tmp_path, adapter, clock).run(spec(tool))
    assert state.status == "deadline_exceeded"
    assert state.failure["code"] == "deadline_exceeded"
    assert state.cleanup["resources_released"] is True
    assert ("terminate", "child-1", 0.5) in adapter.calls


def test_cleanup_failure_is_distinct_fail_closed_state(tmp_path: Path) -> None:
    tool = module()
    adapter = FakeAdapter([None, None, None], cleanup_released=False)
    state = tool.ExecutionCore(tmp_path, adapter, FakeClock()).run(spec(tool))
    assert state.status == "cleanup_failed"
    assert state.cleanup["resources_released"] is False


def test_resume_uses_persisted_child_without_starting_another(tmp_path: Path) -> None:
    tool = module()
    clock = FakeClock(1)
    adapter = FakeAdapter([0])
    state = tool.ExecutionState(1, "resume", "running", 1, started_at=0, deadline_at=10, child_id="existing")
    tool.atomic_write(tmp_path / "resume.json", tool.asdict(state))
    result = tool.ExecutionCore(tmp_path, adapter, clock).resume(spec(tool, "resume"))
    assert result.status == "succeeded"
    assert not any(call[0] == "start" for call in adapter.calls)


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "python.log"
    executable = tmp_path / "python3"
    executable.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable, log


def test_shell_entrypoints_delegate_once_to_python_core(tmp_path: Path) -> None:
    _, log = _fake_python(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "TMUX": "fixture"}
    subprocess.run([str(QUEUE), "--resume"], env=env, check=True)
    subprocess.run([str(PHASE), "--resume"], env=env, check=True, stderr=subprocess.PIPE)
    calls = log.read_text().splitlines()
    assert len(calls) == 2
    assert calls[0].endswith("experiment_execution_core.py queue --resume")
    assert calls[1].endswith("experiment_execution_core.py phase --resume")


def test_frozen_recovery_allows_only_one_qualified_resume() -> None:
    tool = module()
    state = tool.ExecutionState(1, "r", "failed", 1)
    spec_value = tool.RunSpec(
        "r",
        ["fake"],
        1,
        max_attempts=2,
        resumable_failure_codes=("host_interruption", "checkpoint_available_child_exit"),
    )
    assert tool.recovery_decision(state, spec_value, "host_interruption", False)["resume"] is True
    assert tool.recovery_decision(state, spec_value, "checkpoint_available_child_exit", False)["resume"] is False
    state.attempt = 2
    assert tool.recovery_decision(state, spec_value, "host_interruption", True)["resume"] is False


def test_terminal_failure_does_not_restart_without_explicit_resume(tmp_path: Path) -> None:
    tool = module()
    adapter = FakeAdapter([7, 0])
    core = tool.ExecutionCore(tmp_path, adapter, FakeClock())
    failed = core.run(spec(tool, "terminal"))
    assert failed.status == "failed"
    assert core.run(spec(tool, "terminal")).attempt == 1
    assert len([call for call in adapter.calls if call[0] == "start"]) == 1


def test_policy_loaded_resume_enforces_max_attempts(tmp_path: Path) -> None:
    tool = module()
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"schema_version": 1, "max_attempts": 2, "resumable_failure_codes": ["host_interruption"]})
    )
    assert tool.load_recovery_policy(policy) == (2, ("host_interruption",))
    adapter = FakeAdapter([0])
    core = tool.ExecutionCore(tmp_path, adapter, FakeClock())
    state = tool.ExecutionState(
        1, "recover", "failed", 1, failure=tool.failure("host_interruption", "lost host"), max_attempts=2
    )
    tool.atomic_write(tmp_path / "recover.json", tool.asdict(state))
    recovered = core.resume(
        tool.RunSpec(
            "recover",
            ["fake", "command"],
            2,
            env={"FIXTURE": "yes"},
            max_attempts=2,
            resumable_failure_codes=("host_interruption",),
        )
    )
    assert recovered.status == "succeeded" and recovered.attempt == 2
