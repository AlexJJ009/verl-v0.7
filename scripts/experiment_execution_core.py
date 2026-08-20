#!/usr/bin/env python3
"""CPU-testable experiment execution state machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


TERMINAL_STATES = {"succeeded", "failed", "deadline_exceeded", "cleanup_failed"}
BATCH_TERMINAL_STATES = {"completed", "completed_with_failures", "shared_failure", "stopped"}
CONTROL_ACTIONS = {"pause_after_current", "stop_now", "continue_remaining"}
ACCEPTED_ADAPTER_TYPES = {
    "stage123_queue_v1",
    "stage123_treatment_reuse_v1",
    "stage123_stage2_handoff_v1",
    "cpu_fixture_v1",
}


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


@dataclass(frozen=True)
class BatchItemSpec:
    item_id: str
    goal_id: str
    plan_sha256: str
    admission_bundle_path: Path
    admission_bundle_sha256: str
    adapter_type: str
    commands: tuple[tuple[str, ...], ...]
    command_sha256: str
    expected_run_ids: tuple[str, ...]
    input_hashes: dict[str, str]
    implementation_tree_sha256: str
    evidence_commit: str
    recipe_gitlink: str
    timeout_seconds: float = 1800.0
    poll_seconds: float = 1.0
    cleanup_grace_seconds: float = 10.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchManifest:
    batch_id: str
    authorization_id: str
    created_at: str
    failure_policy_id: str
    operator_control_path: Path
    items: tuple[BatchItemSpec, ...]
    batch_manifest_sha256: str
    repo_root: Path = Path(".")


class OperatorControlError(ValueError):
    pass


class BatchValidationError(ValueError):
    pass


def recovery_decision(
    state: ExecutionState, spec: RunSpec, failure_code: str, checkpoint_available: bool
) -> dict[str, Any]:
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def implementation_tree_sha256(repo_root: Path, relative_paths: list[str]) -> str:
    records: list[dict[str, str]] = []
    for relative in sorted(relative_paths):
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise BatchValidationError(f"implementation path is not a regular file: {relative}")
        records.append({"path": relative, "sha256": file_sha256(path)})
    return sha256_json(records)


def protected_asset_sha256(path: Path) -> str:
    if path.is_symlink():
        return sha256_json({"type": "symlink", "target": os.readlink(path)})
    if path.is_file():
        return sha256_json({"type": "file", "sha256": file_sha256(path)})
    if path.is_dir():
        records: list[dict[str, str]] = []
        for child in sorted(path.rglob("*")):
            relative = child.relative_to(path).as_posix()
            if child.is_symlink():
                records.append({"path": relative, "type": "symlink", "target": os.readlink(child)})
            elif child.is_file():
                records.append({"path": relative, "type": "file", "sha256": file_sha256(child)})
        return sha256_json(records)
    raise BatchValidationError(f"protected asset does not exist: {path}")


def _without_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _validate_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise BatchValidationError(f"{label} must be {length} hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise BatchValidationError(f"{label} must be hexadecimal") from exc
    return value


def validate_admission_bundle(
    bundle: dict[str, Any],
    bundle_path: Path,
    repo_root: Path,
    expected_run_ids: list[str] | None = None,
    *,
    static_after_item_start: bool = False,
) -> dict[str, Any]:
    if bundle.get("bundle_type") in {"stage123_treatment_reuse_admission", "stage123_stage2_handoff_admission"}:
        admission_hash = bundle.get("admission_sha256")
        if not isinstance(admission_hash, str) or admission_hash != sha256_json(
            _without_hash(bundle, "admission_sha256")
        ):
            raise BatchValidationError("treatment admission file hash mismatch")
        if bundle.get("status") != "authorized":
            raise BatchValidationError("treatment admission is prepared but not authorized")
        treatment_check = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/stage123_control_reuse.py"),
                "validate-treatment",
                "--admission",
                str(bundle_path),
                "--run-id",
                str(bundle.get("expected_run_ids", [""])[0]),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if treatment_check.returncode != 0:
            raise BatchValidationError(
                f"treatment admission validation failed: {treatment_check.stderr.strip() or treatment_check.stdout.strip()}"
            )
        identity = bundle.get("control_plane_identity")
        if not isinstance(identity, dict):
            raise BatchValidationError("treatment admission control-plane identity is required")
        for key, length in (
            ("plan_sha256", 64),
            ("implementation_tree_sha256", 64),
            ("evidence_commit", 40),
            ("recipe_gitlink", 40),
        ):
            value = _validate_hex(identity.get(key), length, key)
            if set(value) == {"0"}:
                raise BatchValidationError(f"treatment admission {key} cannot be all-zero")
        run_ids = tuple(bundle.get("expected_run_ids", ()))
        expected_by_type = {
            "stage123_treatment_reuse_admission": ("frac25-stage2", "frac25-stage3"),
            "stage123_stage2_handoff_admission": ("frac25-stage3",),
        }
        if run_ids != expected_by_type[bundle["bundle_type"]] or tuple(expected_run_ids or run_ids) != run_ids:
            raise BatchValidationError("treatment admission run ids do not match its admission type")
        certificate_path = Path(str(bundle.get("certificate_path", "")))
        if not certificate_path.is_file() or file_sha256(certificate_path) != bundle.get("certificate_sha256"):
            raise BatchValidationError("treatment control certificate binding mismatch")
        manifest_path = Path(str(bundle.get("treatment_manifest_path", "")))
        if not manifest_path.is_file() or file_sha256(manifest_path) != bundle.get("treatment_manifest_sha256"):
            raise BatchValidationError("treatment manifest binding mismatch")
        commands = [
            [
                "/data-1/verl07/run_train.sh",
                "python",
                "/workspace/verl/scripts/stage123_phase_adapter.py",
                "--manifest",
                str(manifest_path),
                "--run-id",
                run_id,
            ]
            for run_id in run_ids
        ]
        return {
            "adapter_type": "stage123_treatment_reuse_v1"
            if bundle["bundle_type"] == "stage123_treatment_reuse_admission"
            else "stage123_stage2_handoff_v1",
            "commands": commands,
            "command_sha256": sha256_json(commands),
            "bindings": {
                "implementation_tree_sha256": identity["implementation_tree_sha256"],
                "evidence_commit": identity["evidence_commit"],
                "recipe_gitlink": identity["recipe_gitlink"],
                "input_hashes": {
                    str(certificate_path): bundle["certificate_sha256"],
                    str(manifest_path): bundle["treatment_manifest_sha256"],
                },
            },
        }
    if bundle.get("bundle_type") == "stage123_admission_bundle":
        from execution_results import validate_admission_bundle as validate_stage123_bundle
        from execution_results import validate_current_checkout

        structural = validate_stage123_bundle(bundle, require_accepted=True)
        inputs = bundle.get("inputs", {})
        protected_baseline = Path(inputs.get("protected_baseline", ""))
        current = validate_current_checkout(
            bundle,
            repo_root,
            protected_baseline,
            require_accepted=True,
            enforce_result_freshness=not static_after_item_start,
        )
        if not structural.authorized or not current.authorized:
            decision = structural if not structural.authorized else current
            raise BatchValidationError(f"Stage123 admission rejected: {decision.code}: {decision.message}")
        all_commands = [
            [
                "/data-1/verl07/run_train.sh",
                "python",
                "/workspace/verl/scripts/stage123_phase_adapter.py",
                "--manifest",
                "/workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml",
                "--run-id",
                run_id,
            ]
            for run_id in ("frac25-stage1-control", "frac25-stage2", "frac25-stage3")
        ]
        all_run_ids = ("frac25-stage1-control", "frac25-stage2", "frac25-stage3")
        selected_run_ids = tuple(expected_run_ids or all_run_ids)
        if selected_run_ids not in (all_run_ids, ("frac25-stage2", "frac25-stage3")):
            raise BatchValidationError("unsupported Stage123 run subset")
        commands = [
            command for run_id, command in zip(all_run_ids, all_commands, strict=True) if run_id in selected_run_ids
        ]
        bindings = bundle["bindings"]
        input_hashes = {
            path: file_sha256(Path(path))
            for path in inputs.values()
            if isinstance(path, str) and path and Path(path).is_file()
        }
        if bundle.get("acceptance_report_path"):
            input_hashes[bundle["acceptance_report_path"]] = file_sha256(Path(bundle["acceptance_report_path"]))
        return {
            "adapter_type": "stage123_queue_v1",
            "commands": commands,
            "command_sha256": sha256_json(commands),
            "bindings": {
                "implementation_tree_sha256": bindings["implementation_tree_sha256"],
                "evidence_commit": bindings["readiness_evidence_commit"],
                "recipe_gitlink": bindings["recipe_gitlink"],
                "input_hashes": input_hashes,
            },
        }
    if bundle.get("schema_version") != 1 or bundle.get("bundle_type") != "experiment_batch_admission":
        raise BatchValidationError("unsupported admission bundle")
    adapter_type = bundle.get("adapter_type")
    if adapter_type not in ACCEPTED_ADAPTER_TYPES:
        raise BatchValidationError(f"unsupported adapter type: {adapter_type}")
    bindings = bundle.get("bindings")
    if not isinstance(bindings, dict):
        raise BatchValidationError("admission bundle bindings are required")
    required = (
        "implementation_tree_sha256",
        "evidence_commit",
        "recipe_gitlink",
        "input_hashes",
        "protected_asset_hashes",
    )
    missing = [key for key in required if key not in bindings]
    if missing:
        raise BatchValidationError(f"admission bundle missing bindings: {missing}")
    _validate_hex(bindings["implementation_tree_sha256"], 64, "implementation_tree_sha256")
    _validate_hex(bindings["evidence_commit"], 40, "evidence_commit")
    _validate_hex(bindings["recipe_gitlink"], 40, "recipe_gitlink")
    if not isinstance(bindings["input_hashes"], dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in bindings["input_hashes"].items()
    ):
        raise BatchValidationError("input_hashes must be a string map")
    protected = bindings["protected_asset_hashes"]
    if not isinstance(protected, dict) or not protected:
        raise BatchValidationError("protected_asset_hashes must be a non-empty string map")
    for relative, expected_hash in protected.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or not isinstance(expected_hash, str):
            raise BatchValidationError("protected asset bindings must use relative string paths")
        if protected_asset_sha256(repo_root / relative) != expected_hash:
            raise BatchValidationError(f"protected asset hash mismatch: {relative}")
    implementation_paths = bundle.get("implementation_paths")
    if (
        not isinstance(implementation_paths, list)
        or not implementation_paths
        or not all(isinstance(item, str) and item and not Path(item).is_absolute() for item in implementation_paths)
    ):
        raise BatchValidationError("implementation_paths must be relative non-empty paths")
    tree_hash = implementation_tree_sha256(repo_root, implementation_paths)
    if tree_hash != bindings["implementation_tree_sha256"]:
        raise BatchValidationError("implementation tree hash mismatch")
    commands = bundle.get("canonical_commands")
    if commands is None and bundle.get("canonical_command") is not None:
        commands = [bundle["canonical_command"]]
    if (
        not isinstance(commands, list)
        or not commands
        or not all(
            isinstance(command, list) and command and all(isinstance(value, str) for value in command)
            for command in commands
        )
    ):
        raise BatchValidationError("canonical_commands must be a non-empty array of string arrays")
    command_hash = sha256_json(commands)
    if bundle.get("command_sha256") != command_hash:
        raise BatchValidationError("canonical command hash mismatch")
    if adapter_type == "stage123_queue_v1":
        required_paths = {
            "scripts/experiment_execution_core.py",
            "scripts/stage123_manifest_monitor.py",
            "scripts/stage123_phase_adapter.py",
        }
        if not required_paths.issubset(set(implementation_paths)):
            raise BatchValidationError("stage123 admission does not bind the core, monitor, and phase adapter")
        expected_commands = [
            [
                "/data-1/verl07/run_train.sh",
                "python",
                "/workspace/verl/scripts/stage123_phase_adapter.py",
                "--manifest",
                "/workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml",
                "--run-id",
                run_id,
            ]
            for run_id in ("frac25-stage1-control", "frac25-stage2", "frac25-stage3")
        ]
        if commands != expected_commands:
            raise BatchValidationError("stage123 phase commands do not match the frozen manifest-driven adapter matrix")
    current_recipe = subprocess.run(
        ["git", "-C", str(repo_root / "recipe"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if current_recipe.returncode != 0 or current_recipe.stdout.strip() != bindings["recipe_gitlink"]:
        raise BatchValidationError("recipe gitlink mismatch")
    evidence = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", bindings["evidence_commit"], "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if evidence.returncode != 0:
        raise BatchValidationError("evidence commit is not an ancestor of current checkout")
    for input_path, expected_hash in bindings["input_hashes"].items():
        candidate = Path(input_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise BatchValidationError(f"input hash mismatch: {input_path}")
    for relative in implementation_paths:
        try:
            tracked = subprocess.run(
                ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", relative],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise BatchValidationError(f"cannot inspect tracked implementation path: {relative}") from exc
        if tracked.returncode != 0:
            raise BatchValidationError(f"implementation path is not tracked: {relative}")
    bundle_hash = bundle.get("bundle_sha256")
    if bundle_hash != sha256_json(_without_hash(bundle, "bundle_sha256")):
        raise BatchValidationError("admission bundle hash mismatch")
    return {"adapter_type": adapter_type, "commands": commands, "command_sha256": command_hash, "bindings": bindings}


def load_batch_manifest(
    path: Path,
    repo_root: Path | None = None,
    *,
    static_after_item_start: set[str] | None = None,
) -> BatchManifest:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchValidationError(f"cannot read batch manifest: {path}") from exc
    if raw.get("schema_version") != 1 or not isinstance(raw.get("items"), list) or not raw["items"]:
        raise BatchValidationError("unsupported or empty batch manifest")
    manifest_hash = raw.get("batch_manifest_sha256")
    if manifest_hash != sha256_json(_without_hash(raw, "batch_manifest_sha256")):
        raise BatchValidationError("batch manifest hash mismatch")
    item_ids: set[str] = set()
    run_ids: set[str] = set()
    items: list[BatchItemSpec] = []
    for item in raw["items"]:
        if not isinstance(item, dict):
            raise BatchValidationError("batch item must be an object")
        item_id = item.get("item_id")
        expected = item.get("expected_run_ids")
        if not isinstance(item_id, str) or item_id in item_ids:
            raise BatchValidationError("duplicate or invalid item_id")
        if not isinstance(expected, list) or not expected or not all(isinstance(value, str) for value in expected):
            raise BatchValidationError(f"invalid expected_run_ids for {item_id}")
        if run_ids.intersection(expected):
            raise BatchValidationError("duplicate expected run id")
        item_ids.add(item_id)
        run_ids.update(expected)
        bundle_path = Path(item.get("admission_bundle_path", ""))
        if bundle_path.is_absolute():
            resolved_bundle = bundle_path
        else:
            resolved_bundle = repo_root / bundle_path
        try:
            bundle = json.loads(resolved_bundle.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchValidationError(f"cannot read admission bundle for {item_id}") from exc
        bundle_hash = file_sha256(resolved_bundle)
        if item.get("admission_bundle_sha256") != bundle_hash:
            raise BatchValidationError(f"admission bundle file hash mismatch for {item_id}")
        validated = validate_admission_bundle(
            bundle,
            resolved_bundle,
            repo_root,
            expected,
            static_after_item_start=item_id in (static_after_item_start or set()),
        )
        commands = validated["commands"]
        if len(commands) != len(expected):
            raise BatchValidationError(f"phase command/run-id count mismatch for {item_id}")
        if item.get("command_sha256") != sha256_json(commands):
            raise BatchValidationError(f"item command hash mismatch for {item_id}")
        if item.get("adapter_type") != validated["adapter_type"]:
            raise BatchValidationError(f"item adapter mismatch for {item_id}")
        bindings = validated["bindings"]
        if item.get("implementation_tree_sha256") != bindings["implementation_tree_sha256"]:
            raise BatchValidationError(f"item implementation identity mismatch for {item_id}")
        items.append(
            BatchItemSpec(
                item_id=item_id,
                goal_id=str(item.get("goal_id", "")),
                plan_sha256=_validate_hex(item.get("plan_sha256"), 64, "plan_sha256"),
                admission_bundle_path=resolved_bundle,
                admission_bundle_sha256=bundle_hash,
                adapter_type=validated["adapter_type"],
                commands=tuple(tuple(value for value in command) for command in commands),
                command_sha256=validated["command_sha256"],
                expected_run_ids=tuple(expected),
                input_hashes=dict(bindings["input_hashes"]),
                implementation_tree_sha256=bindings["implementation_tree_sha256"],
                evidence_commit=bindings["evidence_commit"],
                recipe_gitlink=bindings["recipe_gitlink"],
                timeout_seconds=float(item.get("timeout_seconds", 1800.0)),
                poll_seconds=float(item.get("poll_seconds", 1.0)),
                cleanup_grace_seconds=float(item.get("cleanup_grace_seconds", 10.0)),
                env=dict(item.get("env", {})),
            )
        )
    control_path = Path(raw.get("operator_control_path", ""))
    if not control_path:
        raise BatchValidationError("operator_control_path is required")
    if not isinstance(raw.get("authorization_id"), str) or not raw["authorization_id"]:
        raise BatchValidationError("authorization_id is required")
    return BatchManifest(
        batch_id=str(raw.get("batch_id", "")),
        authorization_id=raw["authorization_id"],
        created_at=str(raw.get("created_at", "")),
        failure_policy_id=str(raw.get("failure_policy_id", "")),
        operator_control_path=control_path,
        items=tuple(items),
        batch_manifest_sha256=manifest_hash,
        repo_root=repo_root,
    )


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
                state.transitions.append(
                    {"from": state.status, "to": state.status, "at": self.clock.now(), "recovery": decision}
                )
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
                    transition(
                        state,
                        "succeeded",
                        now,
                        completed_at=now,
                        cleanup={"resources_released": True, "term_sent": False, "kill_sent": False},
                    )
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
                    failure=failure(
                        "deadline_exceeded",
                        "execution deadline exceeded",
                        deadline_at=state.deadline_at,
                        observed_at=now,
                    ),
                )
                self.persist(state)
                return state
            self.clock.sleep(spec.poll_seconds)


def normalize_failure_code(state: ExecutionState) -> str:
    code = (state.failure or {}).get("code")
    if not isinstance(code, str) or not code:
        return "unknown_failure"
    return code.split(":", 1)[0]


class BatchExecutor:
    """Deterministic ordered batch orchestration on top of the atomic core."""

    def __init__(
        self,
        manifest: BatchManifest,
        state_root: Path,
        adapter: ChildAdapter,
        clock: Clock | None = None,
    ) -> None:
        self.manifest = manifest
        self.state_root = state_root
        self.adapter = adapter
        self.clock = clock or SystemClock()
        self.batch_state_path = state_root / f"{manifest.batch_id}.json"
        self.event_path = state_root / "events.jsonl"
        self.control_offset = 0
        self.control_seq = 0
        self.batch_revision = 0
        self.pause_after_current = False
        self.stop_requested = False
        self.continue_requested = False
        self.control_rejection: dict[str, Any] | None = None

    def _admission_record_path(self, item: BatchItemSpec) -> Path:
        return self.state_root / "admitted-items" / f"{item.item_id}.json"

    def _write_item_admission(self, item: BatchItemSpec) -> dict[str, str]:
        path = self._admission_record_path(item)
        record = {
            "schema_version": 1,
            "status": "active",
            "batch_id": self.manifest.batch_id,
            "batch_manifest_sha256": self.manifest.batch_manifest_sha256,
            "batch_state_path": str(self.batch_state_path),
            "item_id": item.item_id,
            "goal_id": item.goal_id,
            "plan_sha256": item.plan_sha256,
            "expected_run_ids": list(item.expected_run_ids),
            "command_sha256": item.command_sha256,
            "admission_bundle_path": str(item.admission_bundle_path),
            "admission_bundle_sha256": item.admission_bundle_sha256,
            "implementation_tree_sha256": item.implementation_tree_sha256,
            "admitted_at": self.clock.now(),
        }
        record["record_sha256"] = sha256_json(record)
        atomic_write(path, record)
        return {"path": str(path), "sha256": record["record_sha256"]}

    def _validate_live_item_admission(self, item: BatchItemSpec, active: dict[str, Any]) -> None:
        path = Path(active.get("path", ""))
        expected_sha = active.get("sha256")
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchValidationError(f"cannot read active admission record for {item.item_id}") from exc
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != sha256_json(unsigned) or record.get("record_sha256") != expected_sha:
            raise BatchValidationError(f"active admission record hash mismatch for {item.item_id}")
        expected = {
            "status": "active",
            "batch_id": self.manifest.batch_id,
            "batch_manifest_sha256": self.manifest.batch_manifest_sha256,
            "batch_state_path": str(self.batch_state_path),
            "item_id": item.item_id,
            "plan_sha256": item.plan_sha256,
            "expected_run_ids": list(item.expected_run_ids),
            "command_sha256": item.command_sha256,
            "admission_bundle_sha256": item.admission_bundle_sha256,
            "implementation_tree_sha256": item.implementation_tree_sha256,
        }
        mismatched = [key for key, value in expected.items() if record.get(key) != value]
        if mismatched:
            raise BatchValidationError(f"active admission record binding mismatch for {item.item_id}: {mismatched}")
        if item.adapter_type == "cpu_fixture_v1":
            return
        bundle = json.loads(item.admission_bundle_path.read_text())
        validated = validate_admission_bundle(
            bundle,
            item.admission_bundle_path,
            self.manifest.repo_root,
            list(item.expected_run_ids),
            static_after_item_start=True,
        )
        if validated["command_sha256"] != item.command_sha256:
            raise BatchValidationError(f"command binding changed for active item {item.item_id}")

    def _persist(self, state: dict[str, Any], event: str, **fields: Any) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        state["batch_revision"] = self.batch_revision
        state["control_seq"] = self.control_seq
        state["control_offset"] = self.control_offset
        atomic_write(self.batch_state_path, state)
        record = {
            "schema_version": 1,
            "batch_id": self.manifest.batch_id,
            "batch_manifest_sha256": self.manifest.batch_manifest_sha256,
            "event": event,
            "batch_revision": self.batch_revision,
            "state": state.get("status"),
            **fields,
        }
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")

    def _read_controls(self) -> None:
        path = self.manifest.operator_control_path
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(self.control_offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                self.control_offset = handle.tell()
                if not line.strip():
                    continue
                try:
                    control = json.loads(line)
                    self._apply_control(control)
                except (json.JSONDecodeError, OperatorControlError) as exc:
                    self.control_rejection = {"code": "control_rejected", "message": str(exc)}
                    self._append_control_event("control_rejected", {"error": str(exc)})

    def _append_control_event(self, event: str, fields: dict[str, Any]) -> None:
        state = self._load_state(sync=False)
        self._persist(state, event, **fields)

    def _apply_control(self, control: dict[str, Any]) -> None:
        if not isinstance(control, dict) or control.get("schema_version") != 1:
            raise OperatorControlError("unsupported control schema")
        required = (
            "batch_id",
            "batch_manifest_sha256",
            "control_seq",
            "expected_batch_revision",
            "action",
            "authorization_id",
            "issued_at",
            "control_sha256",
        )
        if any(key not in control for key in required):
            raise OperatorControlError("control envelope is incomplete")
        if (
            control["batch_id"] != self.manifest.batch_id
            or control["batch_manifest_sha256"] != self.manifest.batch_manifest_sha256
        ):
            raise OperatorControlError("control batch binding mismatch")
        if control["authorization_id"] != self.manifest.authorization_id:
            raise OperatorControlError("control authorization mismatch")
        action = control["action"]
        if action not in CONTROL_ACTIONS:
            raise OperatorControlError("unsupported control action")
        if not isinstance(control["control_seq"], int) or control["control_seq"] <= self.control_seq:
            raise OperatorControlError("control sequence replay or regression")
        if control["expected_batch_revision"] != self.batch_revision:
            raise OperatorControlError("stale batch revision")
        if control["control_sha256"] != sha256_json(_without_hash(control, "control_sha256")):
            raise OperatorControlError("control hash mismatch")
        self.control_seq = control["control_seq"]
        if action == "pause_after_current":
            self.pause_after_current = True
        elif action == "stop_now":
            self.stop_requested = True
        elif action == "continue_remaining":
            state = self._load_state(sync=False)
            if state.get("status") not in {"paused_after_current", "stopped"}:
                raise OperatorControlError("continue_remaining requires paused or stopped batch")
            self.stop_requested = False
            self.pause_after_current = False
            self.continue_requested = True
        self._append_control_event("control_consumed", {"action": action, "control_seq": self.control_seq})

    def _load_state(self, sync: bool = True) -> dict[str, Any]:
        if not self.batch_state_path.exists():
            return {
                "schema_version": 1,
                "batch_id": self.manifest.batch_id,
                "batch_manifest_sha256": self.manifest.batch_manifest_sha256,
                "status": "pending",
                "batch_revision": self.batch_revision,
                "items": [],
            }
        try:
            state = json.loads(self.batch_state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            evidence_path = self.batch_state_path.with_suffix(".corrupt.json")
            if self.batch_state_path.exists() and not evidence_path.exists():
                evidence_path.write_bytes(self.batch_state_path.read_bytes())
            state = {
                "schema_version": 1,
                "batch_id": self.manifest.batch_id,
                "batch_manifest_sha256": self.manifest.batch_manifest_sha256,
                "status": "shared_failure",
                "batch_revision": self.batch_revision,
                "items": [],
                "failure": failure("state_corruption", str(exc), evidence_path=str(evidence_path)),
            }
            self._persist(state, "batch_shared_failure", failure=state["failure"])
            return state
        if state.get("batch_manifest_sha256") != self.manifest.batch_manifest_sha256:
            raise BatchValidationError("batch state manifest mismatch")
        if sync:
            self.batch_revision = int(state.get("batch_revision", 0))
            self.control_seq = int(state.get("control_seq", 0))
            self.control_offset = int(state.get("control_offset", 0))
        return state

    def _validate_event_ledger(self) -> None:
        if not self.event_path.exists():
            return
        for index, line in enumerate(self.event_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchValidationError(f"event ledger corruption at line {index}") from exc
            if event.get("schema_version") != 1:
                raise BatchValidationError(f"event ledger schema mismatch at line {index}")
            if "batch_id" in event:
                if (
                    event.get("batch_id") != self.manifest.batch_id
                    or event.get("batch_manifest_sha256") != self.manifest.batch_manifest_sha256
                ):
                    raise BatchValidationError(f"event ledger batch binding mismatch at line {index}")
                if event.get("state") not in BATCH_TERMINAL_STATES | {
                    "pending",
                    "running",
                    "paused_after_current",
                    "stopping",
                }:
                    raise BatchValidationError(f"batch event schema mismatch at line {index}")
            elif "run_id" in event:
                if not isinstance(event.get("run_id"), str) or event.get("status") not in TERMINAL_STATES | {
                    "pending",
                    "running",
                }:
                    raise BatchValidationError(f"atomic event schema mismatch at line {index}")
                if not isinstance(event.get("attempt", 0), int):
                    raise BatchValidationError(f"atomic event attempt mismatch at line {index}")
            else:
                raise BatchValidationError(f"event ledger identity missing at line {index}")

    def _validate_item_bindings(self, item: BatchItemSpec) -> None:
        if item.adapter_type == "cpu_fixture_v1":
            return
        try:
            bundle = json.loads(item.admission_bundle_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchValidationError(f"cannot revalidate admission bundle for {item.item_id}") from exc
        if file_sha256(item.admission_bundle_path) != item.admission_bundle_sha256:
            raise BatchValidationError(f"admission bundle changed for {item.item_id}")
        validated = validate_admission_bundle(bundle, item.admission_bundle_path, self.manifest.repo_root)
        if validated["command_sha256"] != item.command_sha256:
            raise BatchValidationError(f"command binding changed for {item.item_id}")

    def _item_record(self, item: BatchItemSpec, status: str, **fields: Any) -> dict[str, Any]:
        return {
            "item_id": item.item_id,
            "goal_id": item.goal_id,
            "plan_sha256": item.plan_sha256,
            "admission_bundle_sha256": item.admission_bundle_sha256,
            "implementation_tree_sha256": item.implementation_tree_sha256,
            "command_sha256": item.command_sha256,
            "status": status,
            **fields,
        }

    class _ControlledAdapter:
        def __init__(self, owner: BatchExecutor, inner: ChildAdapter, grace_seconds: float) -> None:
            self.owner = owner
            self.inner = inner
            self.grace_seconds = grace_seconds
            self.terminated_for_stop: set[str] = set()

        def start(self, command: list[str], env: dict[str, str]) -> str:
            return self.inner.start(command, env)

        def poll(self, child_id: str) -> int | None:
            self.owner._read_controls()
            if self.owner.stop_requested and child_id not in self.terminated_for_stop:
                self.inner.terminate(child_id, self.grace_seconds)
                self.terminated_for_stop.add(child_id)
                return 143
            return self.inner.poll(child_id)

        def terminate(self, child_id: str, grace_seconds: float) -> dict[str, Any]:
            return self.inner.terminate(child_id, grace_seconds)

    def _stop_for_shared_failure(self, state: dict[str, Any], code: str, message: str) -> dict[str, Any]:
        state["status"] = "shared_failure"
        state["failure"] = failure(code, message)
        self.batch_revision += 1
        self._persist(state, "batch_shared_failure", failure=state["failure"])
        return state

    def _stop_for_operator(self, state: dict[str, Any]) -> dict[str, Any]:
        state["status"] = "stopped"
        state["failure"] = failure("operator_stop_now", "operator requested batch stop")
        self.batch_revision += 1
        self._persist(state, "batch_stopped", reason="operator_stop_now", failure=state["failure"])
        return state

    def run(self) -> dict[str, Any]:
        state = self._load_state()
        try:
            self._validate_event_ledger()
        except BatchValidationError as exc:
            return self._stop_for_shared_failure(state, "event_corruption", str(exc))
        self._read_controls()
        state = self._load_state()
        if state.get("status") in BATCH_TERMINAL_STATES and state.get("status") != "stopped":
            return state
        if state.get("status") in {"paused_after_current", "stopped"} and not self.continue_requested:
            return state
        self.continue_requested = False
        state["status"] = "running"
        state.setdefault("items", [])
        state.setdefault("phases", [])
        failure_codes = [
            str(record.get("failure_code") or ((record.get("failure") or {}).get("code", "unknown_failure"))).split(
                ":", 1
            )[0]
            for record in state["items"]
            if record.get("status") == "inconclusive_operational_failure"
        ]
        for item in self.manifest.items:
            self._read_controls()
            if self.stop_requested:
                return self._stop_for_operator(state)
            if self.pause_after_current:
                state["status"] = "paused_after_current"
                self.batch_revision += 1
                self._persist(state, "batch_paused", reason="operator_pause_after_current")
                return state
            if any(record.get("item_id") == item.item_id for record in state["items"]):
                continue
            active_admission = state.get("current_item_admission")
            continuing_started_item = state.get("current_item_id") == item.item_id and isinstance(
                active_admission, dict
            )
            try:
                if continuing_started_item:
                    self._validate_live_item_admission(item, active_admission)
                else:
                    self._validate_item_bindings(item)
                    active_admission = self._write_item_admission(item)
                    state["current_item_id"] = item.item_id
                    state["current_item_admission"] = active_admission
                    self.batch_revision += 1
                    self._persist(
                        state, "item_started", item_id=item.item_id, admission_record_sha256=active_admission["sha256"]
                    )
            except BatchValidationError as exc:
                return self._stop_for_shared_failure(state, "admission_binding_changed", str(exc))
            item_failed = False
            phase_records: list[dict[str, Any]] = []
            for phase_index, (run_id, command) in enumerate(zip(item.expected_run_ids, item.commands, strict=True)):
                previous = next((record for record in state["phases"] if record.get("run_id") == run_id), None)
                if previous and previous.get("status") == "succeeded":
                    phase_records.append(previous)
                    continue
                state["current_run_id"] = run_id
                self.batch_revision += 1
                self._persist(state, "phase_started", item_id=item.item_id, run_id=run_id, phase_index=phase_index)
                phase_env = {
                    **item.env,
                    "STAGE123_BATCH_EXECUTION": "1",
                    "STAGE123_BATCH_ID": self.manifest.batch_id,
                    "STAGE123_BATCH_MANIFEST_SHA256": self.manifest.batch_manifest_sha256,
                    "STAGE123_BATCH_ITEM_ID": item.item_id,
                    "STAGE123_BATCH_ADMISSION_BUNDLE_SHA256": item.admission_bundle_sha256,
                    "STAGE123_ADMISSION_BUNDLE": str(item.admission_bundle_path),
                }
                if item.adapter_type in {"stage123_queue_v1", "cpu_fixture_v1"}:
                    phase_env.update(
                        {
                            "STAGE123_BATCH_COMMAND_SHA256": item.command_sha256,
                            "STAGE123_BATCH_ADMISSION_RECORD": active_admission["path"],
                            "STAGE123_BATCH_ADMISSION_RECORD_SHA256": active_admission["sha256"],
                        }
                    )
                if item.adapter_type in {"stage123_treatment_reuse_v1", "stage123_stage2_handoff_v1"}:
                    phase_env["STAGE123_TREATMENT_REUSE_ADMISSION"] = str(item.admission_bundle_path)
                    phase_env["STAGE123_ADMISSION_BUNDLE"] = str(item.admission_bundle_path)
                spec = RunSpec(
                    run_id=run_id,
                    command=list(command),
                    timeout_seconds=item.timeout_seconds,
                    poll_seconds=item.poll_seconds,
                    cleanup_grace_seconds=item.cleanup_grace_seconds,
                    env=phase_env,
                    max_attempts=1,
                    resumable_failure_codes=(),
                )
                try:
                    controlled_adapter = self._ControlledAdapter(self, self.adapter, item.cleanup_grace_seconds)
                    atomic_state = ExecutionCore(self.state_root, controlled_adapter, self.clock).run(spec)
                except (OSError, ValueError, KeyError) as exc:
                    return self._stop_for_shared_failure(state, "state_or_execution_error", str(exc))
                self._read_controls()
                phase_record = {
                    "item_id": item.item_id,
                    "run_id": run_id,
                    "phase_index": phase_index,
                    "status": atomic_state.status,
                    "failure": atomic_state.failure,
                    "cleanup": atomic_state.cleanup,
                    "attempt": atomic_state.attempt,
                }
                state["phases"].append(phase_record)
                phase_records.append(phase_record)
                self.batch_revision += 1
                self._persist(
                    state, "phase_terminal", item_id=item.item_id, run_id=run_id, phase_status=atomic_state.status
                )
                if atomic_state.status == "succeeded" and not self.stop_requested:
                    continue
                if atomic_state.status == "succeeded":
                    skipped = list(item.expected_run_ids[phase_index + 1 :])
                    status = "succeeded" if not skipped else "inconclusive_operational_failure"
                    state["items"].append(self._item_record(item, status, phases=phase_records, skipped_phases=skipped))
                    return self._stop_for_operator(state)
                code = normalize_failure_code(atomic_state)
                failure_codes.append(code)
                state["items"].append(
                    self._item_record(
                        item,
                        "inconclusive_operational_failure",
                        phases=phase_records,
                        failure=atomic_state.failure,
                        failure_code=code,
                        cleanup=atomic_state.cleanup,
                        skipped_phases=list(item.expected_run_ids[phase_index + 1 :]),
                        attempt=atomic_state.attempt,
                    )
                )
                item_failed = True
                if atomic_state.status == "cleanup_failed":
                    return self._stop_for_shared_failure(state, "shared_failure", f"batch stopped after {code}")
                if self.stop_requested:
                    return self._stop_for_operator(state)
                if len(failure_codes) >= 2 and failure_codes[-1] == failure_codes[-2]:
                    return self._stop_for_shared_failure(state, "shared_failure", f"batch stopped after {code}")
                self.batch_revision += 1
                self._persist(state, "item_failed_fallback", item_id=item.item_id, failure_code=code, next_item=True)
                break
            if not item_failed:
                state["items"].append(self._item_record(item, "succeeded", phases=phase_records))
                self.batch_revision += 1
                self._persist(state, "item_succeeded", item_id=item.item_id)
            if self.pause_after_current:
                state["status"] = "paused_after_current"
                self.batch_revision += 1
                self._persist(state, "batch_paused", reason="operator_pause_after_current")
                return state
        state["status"] = (
            "completed_with_failures" if any(item["status"] != "succeeded" for item in state["items"]) else "completed"
        )
        self.batch_revision += 1
        self._persist(state, "batch_completed")
        return state


def parse_command(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
        raise ValueError("command must be a non-empty JSON string array")
    return parsed


def active_item_ids_for_restart(manifest_path: Path, state_root: Path) -> set[str]:
    try:
        raw = json.loads(manifest_path.read_text())
        batch_id = raw["batch_id"]
        state = json.loads((state_root / f"{batch_id}.json").read_text())
    except (OSError, KeyError, json.JSONDecodeError):
        return set()
    item_id = state.get("current_item_id")
    active = state.get("current_item_admission")
    if state.get("status") == "running" and isinstance(item_id, str) and isinstance(active, dict):
        return {item_id}
    return set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("queue", "phase", "batch-validate", "batch-run"))
    parser.add_argument("--run-id", default=os.environ.get("CALIBRATION_RUN_ID", "operational-calibration"))
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(
            os.environ.get("CALIBRATION_STATE_ROOT", "/data-1/tmp/verl_agent_scratch/experiment_workflow/state")
        ),
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=float(os.environ.get("CALIBRATION_DEADLINE_SECONDS", "1800"))
    )
    parser.add_argument("--command-json", default=os.environ.get("CALIBRATION_CHILD_COMMAND_JSON"))
    parser.add_argument("--recovery-policy", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args, legacy = parser.parse_known_args()
    if args.mode in {"batch-validate", "batch-run"}:
        if args.manifest is None:
            print(
                json.dumps(
                    {"ok": False, "failure": failure("missing_batch_manifest", "--manifest is required")},
                    sort_keys=True,
                )
            )
            return 2
        if args.resume or args.recovery_policy is not None:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "failure": failure(
                            "batch_recovery_forbidden", "batch mode forbids --resume and --recovery-policy"
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 2
        try:
            static_items = (
                active_item_ids_for_restart(args.manifest, args.state_root) if args.mode == "batch-run" else set()
            )
            manifest = load_batch_manifest(args.manifest, args.repo_root, static_after_item_start=static_items)
            if args.mode == "batch-validate":
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "batch_id": manifest.batch_id,
                            "batch_manifest_sha256": manifest.batch_manifest_sha256,
                            "items": [item.item_id for item in manifest.items],
                        },
                        sort_keys=True,
                    )
                )
                return 0
            state = BatchExecutor(manifest, args.state_root, SubprocessAdapter()).run()
            completed = state.get("status") in {"completed", "completed_with_failures"}
            print(json.dumps({"ok": completed, "state": state}, sort_keys=True))
            return 0 if completed else 1
        except (OSError, ValueError, json.JSONDecodeError, BatchValidationError) as exc:
            print(json.dumps({"ok": False, "failure": failure("invalid_batch_request", str(exc))}, sort_keys=True))
            return 2
    command_json = args.command_json
    if command_json is None:
        result = failure(
            "missing_child_command", "CALIBRATION_CHILD_COMMAND_JSON or --command-json is required", legacy_args=legacy
        )
        print(json.dumps({"ok": False, "failure": result}, sort_keys=True))
        return 2
    try:
        max_attempts, resumable_codes = load_recovery_policy(args.recovery_policy)
        spec = RunSpec(
            args.run_id,
            parse_command(command_json),
            args.timeout_seconds,
            max_attempts=max_attempts,
            resumable_failure_codes=resumable_codes,
        )
        core = ExecutionCore(args.state_root, SubprocessAdapter())
        state = core.resume(spec) if args.resume else core.run(spec)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "failure": failure("invalid_execution_request", str(exc))}, sort_keys=True))
        return 2
    print(json.dumps({"ok": state.status == "succeeded", "state": asdict(state)}, sort_keys=True))
    return 0 if state.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
