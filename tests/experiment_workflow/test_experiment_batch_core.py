from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "scripts/experiment_execution_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("experiment_batch_core", CORE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        self.value += 1.0
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeAdapter:
    def __init__(self, outcomes: list[int], on_start=None, on_poll=None) -> None:
        self.outcomes = list(outcomes)
        self.on_start = on_start
        self.on_poll = on_poll
        self.started: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self.results: dict[str, int] = {}

    def start(self, command: list[str], env: dict[str, str]) -> str:
        child_id = str(len(self.started) + 1)
        self.started.append(command)
        self.environments.append(dict(env))
        self.results[child_id] = self.outcomes.pop(0)
        if self.on_start is not None:
            self.on_start()
        return child_id

    def poll(self, child_id: str) -> int | None:
        if self.on_poll is not None:
            callback, self.on_poll = self.on_poll, None
            callback()
        return self.results[child_id]

    def terminate(self, child_id: str, grace_seconds: float) -> dict[str, object]:
        return {"resources_released": True, "term_sent": True, "kill_sent": False}


def make_item(tool, item_id: str, run_id: str, phases: int = 1):
    run_ids = tuple(run_id if index == 0 else f"{run_id}-phase-{index + 1}" for index in range(phases))
    commands = tuple(("fixture", item_id, str(index + 1)) for index in range(phases))
    return tool.BatchItemSpec(
        item_id=item_id,
        goal_id=f"goal-{item_id}",
        plan_sha256="1" * 64,
        admission_bundle_path=Path(f"/{item_id}.json"),
        admission_bundle_sha256="2" * 64,
        adapter_type="cpu_fixture_v1",
        commands=commands,
        command_sha256=tool.sha256_json([list(command) for command in commands]),
        expected_run_ids=run_ids,
        input_hashes={},
        implementation_tree_sha256="3" * 64,
        evidence_commit="4" * 40,
        recipe_gitlink="5" * 40,
        timeout_seconds=30,
        poll_seconds=0,
        cleanup_grace_seconds=0,
    )


def make_manifest(tool, tmp_path: Path, items):
    return tool.BatchManifest(
        batch_id="batch-fixture",
        authorization_id="auth-fixture",
        created_at="2026-07-14T00:00:00Z",
        failure_policy_id="batch-fallback-v1",
        operator_control_path=tmp_path / "controls.jsonl",
        items=tuple(items),
        batch_manifest_sha256="6" * 64,
    )


def write_valid_manifest(tool, tmp_path: Path) -> Path:
    commands = [
        ["/data-1/verl07/run_train.sh", "python", "/workspace/verl/scripts/stage123_phase_adapter.py", "--manifest", "/workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml", "--run-id", run_id]
        for run_id in ("frac25-stage1-control", "frac25-stage2", "frac25-stage3")
    ]
    implementation_paths = ["scripts/experiment_execution_core.py", "scripts/stage123_manifest_monitor.py", "scripts/stage123_phase_adapter.py"]
    recipe_head = subprocess.check_output(["git", "-C", str(ROOT / "recipe"), "rev-parse", "HEAD"], text=True).strip()
    evidence_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    bundle = {
        "schema_version": 1,
        "bundle_type": "experiment_batch_admission",
        "adapter_type": "stage123_queue_v1",
        "canonical_commands": commands,
        "command_sha256": tool.sha256_json(commands),
        "implementation_paths": implementation_paths,
        "bindings": {
            "implementation_tree_sha256": tool.implementation_tree_sha256(ROOT, implementation_paths),
            "evidence_commit": evidence_commit,
            "recipe_gitlink": recipe_head,
            "input_hashes": {},
            "protected_asset_hashes": {
                ".claude/skills/experiment-registry": tool.protected_asset_sha256(ROOT / ".claude/skills/experiment-registry"),
                "docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md": tool.protected_asset_sha256(ROOT / "docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md"),
                "test_data": tool.protected_asset_sha256(ROOT / "test_data"),
            },
        },
    }
    bundle["bundle_sha256"] = tool.sha256_json(bundle)
    bundle_path = tmp_path / "admission.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))
    item = {
        "item_id": "stage123-primary",
        "goal_id": "stage123-primary-chain-execution",
        "plan_sha256": "1" * 64,
        "admission_bundle_path": str(bundle_path),
        "admission_bundle_sha256": tool.file_sha256(bundle_path),
        "adapter_type": "stage123_queue_v1",
        "command_sha256": tool.sha256_json(commands),
        "implementation_tree_sha256": bundle["bindings"]["implementation_tree_sha256"],
        "expected_run_ids": ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"],
        "timeout_seconds": 30,
    }
    manifest = {
        "schema_version": 1,
        "batch_id": "stage123-batch",
        "authorization_id": "auth-stage123",
        "created_at": "2026-07-14T00:00:00Z",
        "failure_policy_id": "batch-fallback-v1",
        "operator_control_path": str(tmp_path / "controls.jsonl"),
        "items": [item],
    }
    manifest["batch_manifest_sha256"] = tool.sha256_json(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path


def control(tool, manifest, seq: int, revision: int, action: str):
    value = {
        "schema_version": 1,
        "batch_id": manifest.batch_id,
        "batch_manifest_sha256": manifest.batch_manifest_sha256,
        "control_seq": seq,
        "expected_batch_revision": revision,
        "action": action,
        "authorization_id": manifest.authorization_id,
        "issued_at": "2026-07-14T00:00:00Z",
    }
    value["control_sha256"] = tool.sha256_json(value)
    return value


def test_manifest_validation_binds_admission_command_and_implementation(tmp_path: Path) -> None:
    tool = load_core()
    path = write_valid_manifest(tool, tmp_path)
    manifest = tool.load_batch_manifest(path, ROOT)
    assert manifest.batch_id == "stage123-batch"
    assert manifest.items[0].adapter_type == "stage123_queue_v1"
    raw = json.loads(path.read_text())
    raw["items"][0]["command_sha256"] = "0" * 64
    raw["batch_manifest_sha256"] = tool.sha256_json({key: value for key, value in raw.items() if key != "batch_manifest_sha256"})
    path.write_text(json.dumps(raw))
    try:
        tool.load_batch_manifest(path, ROOT)
    except tool.BatchValidationError as exc:
        assert "command hash mismatch" in str(exc)
    else:
        raise AssertionError("mutated item command was accepted")


def test_successful_items_advance_once_in_manifest_order(tmp_path: Path) -> None:
    tool = load_core()
    items = [make_item(tool, "one", "run-one"), make_item(tool, "two", "run-two")]
    adapter = FakeAdapter([0, 0])
    state = tool.BatchExecutor(make_manifest(tool, tmp_path, items), tmp_path / "state", adapter, FakeClock()).run()
    assert state["status"] == "completed"
    assert [item["item_id"] for item in state["items"]] == ["one", "two"]
    assert adapter.started == [["fixture", "one", "1"], ["fixture", "two", "1"]]


def test_core_owns_ordered_phases_inside_an_item(tmp_path: Path) -> None:
    tool = load_core()
    item = make_item(tool, "one", "run-one", phases=2)
    adapter = FakeAdapter([0, 0])
    state = tool.BatchExecutor(make_manifest(tool, tmp_path, [item]), tmp_path / "state", adapter, FakeClock()).run()
    assert state["status"] == "completed"
    assert [phase["run_id"] for phase in state["phases"]] == ["run-one", "run-one-phase-2"]
    assert adapter.started == [["fixture", "one", "1"], ["fixture", "one", "2"]]


def test_local_failure_is_inconclusive_and_falls_forward_without_retry(tmp_path: Path) -> None:
    tool = load_core()
    items = [make_item(tool, "one", "run-one", phases=2), make_item(tool, "two", "run-two")]
    adapter = FakeAdapter([7, 0])
    state = tool.BatchExecutor(make_manifest(tool, tmp_path, items), tmp_path / "state", adapter, FakeClock()).run()
    assert state["status"] == "completed_with_failures"
    assert state["items"][0]["status"] == "inconclusive_operational_failure"
    assert state["items"][0]["attempt"] == 1
    assert state["items"][0]["skipped_phases"] == ["run-one-phase-2"]
    assert len(adapter.started) == 2


def test_two_equal_normalized_failures_stop_batch(tmp_path: Path) -> None:
    tool = load_core()
    items = [make_item(tool, "one", "run-one"), make_item(tool, "two", "run-two"), make_item(tool, "three", "run-three")]
    adapter = FakeAdapter([7, 9, 0])
    state = tool.BatchExecutor(make_manifest(tool, tmp_path, items), tmp_path / "state", adapter, FakeClock()).run()
    assert state["status"] == "shared_failure"
    assert len(adapter.started) == 2


def test_pause_continue_and_replay_controls_are_revision_bound(tmp_path: Path) -> None:
    tool = load_core()
    manifest = make_manifest(tool, tmp_path, [make_item(tool, "one", "run-one")])
    executor = tool.BatchExecutor(manifest, tmp_path / "state", FakeAdapter([0]), FakeClock())
    manifest.operator_control_path.write_text(json.dumps(control(tool, manifest, 1, 0, "pause_after_current")) + "\n")
    paused = executor.run()
    assert paused["status"] == "paused_after_current"
    with manifest.operator_control_path.open("a") as handle:
        handle.write(json.dumps(control(tool, manifest, 2, 1, "continue_remaining")) + "\n")
    executor = tool.BatchExecutor(manifest, tmp_path / "state", FakeAdapter([0]), FakeClock())
    completed = executor.run()
    assert completed["status"] == "completed"
    with manifest.operator_control_path.open("a") as handle:
        handle.write(json.dumps(control(tool, manifest, 2, completed["batch_revision"], "stop_now")) + "\n")
    executor._read_controls()
    assert executor.control_rejection and "replay" in executor.control_rejection["message"]


def test_stop_now_during_active_item_terminates_and_stops_batch(tmp_path: Path) -> None:
    tool = load_core()
    manifest = make_manifest(tool, tmp_path, [make_item(tool, "one", "run-one"), make_item(tool, "two", "run-two")])
    holder = {}

    def issue_stop() -> None:
        executor = holder["executor"]
        with manifest.operator_control_path.open("a") as handle:
            handle.write(json.dumps(control(tool, manifest, 1, 2, "stop_now")) + "\n")

    adapter = FakeAdapter([0, 0], on_start=issue_stop)
    executor = tool.BatchExecutor(manifest, tmp_path / "state", adapter, FakeClock())
    holder["executor"] = executor
    state = executor.run()
    assert state["status"] == "stopped"
    assert len(adapter.started) == 1


def test_stop_now_at_success_cleanup_boundary_records_terminal_item(tmp_path: Path) -> None:
    tool = load_core()
    manifest = make_manifest(tool, tmp_path, [make_item(tool, "one", "run-one"), make_item(tool, "two", "run-two")])

    def issue_stop_after_control_poll() -> None:
        with manifest.operator_control_path.open("a") as handle:
            handle.write(json.dumps(control(tool, manifest, 1, 2, "stop_now")) + "\n")

    adapter = FakeAdapter([0, 0], on_poll=issue_stop_after_control_poll)
    state = tool.BatchExecutor(manifest, tmp_path / "state", adapter, FakeClock()).run()
    assert state["status"] == "stopped"
    assert [item["item_id"] for item in state["items"]] == ["one"]
    assert state["items"][0]["status"] == "succeeded"
    assert len(adapter.started) == 1


def test_batch_cli_rejects_resume_and_recovery_policy(tmp_path: Path) -> None:
    tool = load_core()
    manifest = write_valid_manifest(tool, tmp_path)
    result = subprocess.run(
        [sys.executable, str(CORE), "batch-validate", "--manifest", str(manifest), "--resume"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "batch_recovery_forbidden" in result.stdout


def test_committed_batch_fixture_validates_without_starting_child() -> None:
    result = subprocess.run(
        [sys.executable, str(CORE), "batch-validate", "--manifest", str(ROOT / "tests/experiment_workflow/fixtures/experiment_batch_v1.json")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["ok"] is True


def test_stage123_accepted_bundle_maps_to_frozen_phase_adapter_commands(tmp_path: Path, monkeypatch) -> None:
    tool = load_core()
    import execution_results

    freshness_modes = []
    monkeypatch.setattr(execution_results, "validate_admission_bundle", lambda bundle, require_accepted: SimpleNamespace(authorized=True, code="accepted", message="ok"))
    monkeypatch.setattr(
        execution_results,
        "validate_current_checkout",
        lambda bundle, repo_root, protected_baseline, require_accepted, **kwargs: freshness_modes.append(kwargs["enforce_result_freshness"])
        or SimpleNamespace(authorized=True, code="authorized", message="ok"),
    )
    input_path = tmp_path / "input.json"; input_path.write_text("{}")
    report_path = tmp_path / "acceptance.json"; report_path.write_text("{}")
    bundle = {
        "schema_version": 1,
        "bundle_type": "stage123_admission_bundle",
        "inputs": {"manifest": str(input_path), "protected_baseline": str(input_path)},
        "acceptance_report_path": str(report_path),
        "bindings": {
            "implementation_tree_sha256": "1" * 64,
            "readiness_evidence_commit": "2" * 40,
            "recipe_gitlink": "3" * 40,
        },
    }
    validated = tool.validate_admission_bundle(bundle, tmp_path / "bundle.json", ROOT)
    tool.validate_admission_bundle(bundle, tmp_path / "bundle.json", ROOT, static_after_item_start=True)
    assert validated["adapter_type"] == "stage123_queue_v1"
    assert [command[-1] for command in validated["commands"]] == ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]
    assert all(command[2] == "/workspace/verl/scripts/stage123_phase_adapter.py" for command in validated["commands"])
    assert freshness_modes == [True, False]


def test_batch_item_persists_live_admission_and_passes_it_to_all_phases(tmp_path: Path) -> None:
    tool = load_core()
    item = make_item(tool, "one", "run-one", phases=2)
    manifest = make_manifest(tool, tmp_path, [item])
    adapter = FakeAdapter([0, 0])
    state = tool.BatchExecutor(manifest, tmp_path / "state", adapter, FakeClock()).run()
    assert state["status"] == "completed"
    assert len(adapter.environments) == 2
    record_path = Path(adapter.environments[0]["STAGE123_BATCH_ADMISSION_RECORD"])
    record = json.loads(record_path.read_text())
    assert record["status"] == "active"
    assert record["expected_run_ids"] == list(item.expected_run_ids)
    assert all(environment["STAGE123_BATCH_ADMISSION_RECORD"] == str(record_path) for environment in adapter.environments)
    assert all(environment["STAGE123_BATCH_ADMISSION_RECORD_SHA256"] == record["record_sha256"] for environment in adapter.environments)
    assert all(environment["STAGE123_BATCH_COMMAND_SHA256"] == item.command_sha256 for environment in adapter.environments)
