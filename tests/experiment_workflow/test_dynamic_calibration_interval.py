from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_BASE = Path(tempfile.mkdtemp(prefix="calibration-history-fixtures-"))


def load():
    path = ROOT / "scripts/check_calibration_prediction_contract.py"
    spec = importlib.util.spec_from_file_location("prediction_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def manifest() -> dict:
    workload = lambda phase: {"phase": phase, "outcome_schema_version": 2, "descriptor": phase}
    return {
        "validation_dataset_hashes": {"HumanEval+": "h", "MBPP+": "m", "LiveCodeBench": "l"},
        "sampled_decoding_semantic_hash": "sem",
        "resource_profile_hash": "profile",
        "phase_topology_hash": "topology",
        "scorer_hash": "scorer",
        "timeout_policy_hash": "timeout",
        "max_response_length": 8192,
        "calibration_workloads": {phase: workload(phase) for phase in ("stage1", "stage2", "stage3")},
    }


def run(run_id: str, phase: str, day: int, elapsed: float, rss: float, idle: float, **overrides) -> dict:
    module = load()
    workload = manifest()["calibration_workloads"][phase]
    rep = int(run_id.rsplit("-", 1)[-1]) if run_id.rsplit("-", 1)[-1].isdigit() else day - 1
    artifact_root = ARTIFACT_BASE / f"cohort_{rep // 6}" / "bootstrap" / phase / f"rep_{rep % 6}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    bindings = {}
    for name in ("status", "resources", "metrics", "generation", "timeline"):
        path = artifact_root / f"{phase}.{name}.json"
        path.write_text(json.dumps({"run_id": run_id, "kind": name}) + "\n")
        bindings[name] = {"path": str(path), "sha256": module.file_sha256(path)}
    result = {
        "run_id": run_id,
        "phase": phase,
        "evidence_role": "bootstrap_history",
        "artifact_root": str(artifact_root),
        "artifact_bindings": bindings,
        "completed_at": f"2026-07-{day:02d}T00:00:00Z",
        "release_gate_passed": True,
        "content_addressed": True,
        "artifacts_readable": True,
        "validation_dataset_hashes": {"HumanEval+": "h", "MBPP+": "m", "LiveCodeBench": "l"},
        "sampled_decoding_semantic_hash": "sem",
        "resource_profile_hash": "profile",
        "phase_topology_hash": "topology",
        "scorer_hash": "scorer",
        "timeout_policy_hash": "timeout",
        "max_response_length": 8192,
        "workload_descriptor_sha256": module.canonical_json_sha256(workload),
        "outcome_schema_version": 2,
        "metrics": {
            "validation_elapsed_seconds": elapsed,
            "peak_rss_gib": rss,
            "all_gpu_idle_fraction_during_validation": idle,
            "response_length_p50_tokens": 100,
            "response_length_p95_tokens": 200,
            "truncation_rate": 0,
            "scorer_latency_p50_seconds": 1,
            "scorer_latency_p95_seconds": 2,
            "scorer_timeout_rate": 0,
            "submitted_item_count": 1422,
        },
    }
    result.update(overrides)
    return result


def history(runs: list[dict], **extra) -> dict:
    phase_scope = [phase for phase in ("stage1", "stage2", "stage3") if any(row.get("phase") == phase for row in runs)]
    result = {"cutoff_utc": "2026-07-20T00:00:00Z", "phase_scope": phase_scope, "runs": runs}
    result.update(extra)
    return result


def six_phase_runs(elapsed=None, rss=None, idle=None) -> list[dict]:
    elapsed = elapsed or [100, 100, 100, 100, 100, 100]
    rss = rss or [50, 50, 50, 50, 50, 50]
    idle = idle or [0.08, 0.20, 0.31, 0.44, 0.58, 0.70]
    rows = []
    for phase in ("stage1", "stage2", "stage3"):
        for index in range(6):
            rows.append(run(f"{phase}-{index}", phase, index + 1, elapsed[index], rss[index], idle[index]))
    return rows


def test_n6_interval_uses_required_rank_and_loo_residuals() -> None:
    module = load()
    result = module.conformal_interval([100, 101, 102, 103, 104, 105])
    assert result["point"] == 102.5
    assert result["finite_sample_rank"] == 6
    assert result["q"] == 3.0
    assert result["interval"] == [99.5, 105.5]
    assert result["loo_residuals"] == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_ties_keep_duplicate_residual_ranks() -> None:
    module = load()
    result = module.conformal_interval([7, 7, 7, 7, 7, 7])
    assert result["loo_residuals"] == [0, 0, 0, 0, 0, 0]
    assert result["interval"] == [7, 7]


def test_outward_rounding_is_six_decimal() -> None:
    module = load()
    assert module._round_interval(1.123456789, 2.123456111) == [1.123456, 2.123457]


def test_canonical_json_bytes_are_stable_for_same_input_semantics() -> None:
    module = load()
    left = {"b": [2, {"d": 4, "c": 3}], "a": 1}
    right = {"a": 1, "b": [2, {"c": 3, "d": 4}]}
    assert module.canonical_json_bytes(left) == module.canonical_json_bytes(right)
    assert module.canonical_json_bytes(left).endswith(b"\n")
    assert module.canonical_json_sha256(left) == module.canonical_json_sha256(right)


def test_latest_twelve_selection_respects_order_and_cutoff() -> None:
    module = load()
    rows = [run(f"r{i:02d}", "stage1", i + 1, 100 + i, 50 + i * 0.1, 0.2) for i in range(14)]
    rows.append(run("future", "stage1", 21, 1, 1, 0.1))
    selected, excluded = module.select_cohort(history(rows), manifest(), "stage1")
    assert [item["run_id"] for item in selected] == [f"r{i:02d}" for i in range(2, 14)]
    assert {"run_id": "future", "reason": "after_cutoff"} in excluded


def test_fewer_than_six_canonical_bootstrap_runs_is_rejected() -> None:
    module = load()
    rows = []
    for phase in ("stage1", "stage2", "stage3"):
        for index in range(5):
            rows.append(run(f"{phase}-{index}", phase, index + 1, 100, 50, 0.2))
    with pytest.raises(module.ContractError, match="exactly 6 bootstrap runs for stage1"):
        module.build_prediction_contract(manifest(), history(rows))


@pytest.mark.parametrize("phase_scope", [["stage1"], ["stage2"], ["stage1", "stage3"]])
def test_contract_rejects_unapproved_phase_subsets(phase_scope) -> None:
    module = load()
    rows = [row for row in six_phase_runs() if row["phase"] in phase_scope]
    with pytest.raises(module.ContractError, match="history phase_scope mismatch"):
        module.build_prediction_contract(manifest(), history(rows))


def test_stage12_scope_rejects_history_containing_stage3() -> None:
    module = load()
    rows = six_phase_runs()
    scoped_history = history(rows)
    scoped_history["phase_scope"] = ["stage1", "stage2"]
    with pytest.raises(module.ContractError, match="history run phase set mismatch"):
        module.build_prediction_contract(
            manifest(), scoped_history, authorization_scope="stage12_producer"
        )


def test_full_scope_rejects_stage12_history() -> None:
    module = load()
    rows = [row for row in six_phase_runs() if row["phase"] != "stage3"]
    with pytest.raises(module.ContractError, match="history phase_scope mismatch"):
        module.build_prediction_contract(manifest(), history(rows))


def test_stage12_scope_has_exact_phase_key_set() -> None:
    module = load()
    rows = [row for row in six_phase_runs() if row["phase"] != "stage3"]
    contract = module.build_prediction_contract(
        manifest(), history(rows), authorization_scope="stage12_producer"
    )
    assert contract["authorization_scope"] == "stage12_producer"
    assert [phase["phase"] for phase in contract["phases"]] == ["stage1", "stage2"]


def test_current_acceptance_run_leakage_is_rejected() -> None:
    module = load()
    rows = six_phase_runs()
    with pytest.raises(module.ContractError, match="leaked into history"):
        module.build_prediction_contract(manifest(), history(rows, current_run_ids=["stage1-0"]))


def test_gpu_idle_008_to_070_fixture_is_informative() -> None:
    module = load()
    contract = module.build_prediction_contract(manifest(), history(six_phase_runs()))
    assert contract["decision"] == "deployable"
    idle = contract["phases"][0]["predictions"]["all_gpu_idle_fraction_during_validation"]["interval"]
    assert idle == [0.06, 0.72]


def test_gpu_idle_002_to_087_fixture_is_rejected_as_noninformative() -> None:
    module = load()
    rows = six_phase_runs(idle=[0.02, 0.20, 0.31, 0.44, 0.58, 0.87])
    contract = module.build_prediction_contract(manifest(), history(rows))
    assert contract["decision"] == "inconclusive"
    assert all(phase["status"] == "noninformative" for phase in contract["phases"])


def test_elapsed_upper_at_or_above_1800_is_blocked() -> None:
    module = load()
    rows = six_phase_runs(elapsed=[1700, 1700, 1700, 1700, 1700, 1900])
    contract = module.build_prediction_contract(manifest(), history(rows))
    assert contract["decision"] == "blocked"
    assert all(phase["status"] == "runtime_risk" for phase in contract["phases"])


def test_rate_interval_uses_one_over_submitted_count_margin() -> None:
    module = load()
    result = module.rate_interval([0, 1 / 100, 2 / 100], 100)
    assert result["interval"] == [0, 0.03]
    assert result["margin"] == 0.01


def test_workload_descriptor_mismatch_excludes_history() -> None:
    module = load()
    rows = six_phase_runs()
    rows[0]["workload_descriptor_sha256"] = "changed"
    selected, excluded = module.select_cohort(history(rows), manifest(), "stage1")
    assert len(selected) == 5
    assert {"run_id": "stage1-0", "reason": "workload_descriptor_sha256_mismatch"} in excluded


def test_missing_outcome_schema_v2_metric_fails_closed() -> None:
    module = load()
    rows = six_phase_runs()
    del rows[0]["metrics"]["response_length_p95_tokens"]
    with pytest.raises(module.ContractError, match="response_length_p95_tokens"):
        module.build_prediction_contract(manifest(), history(rows))


@pytest.mark.parametrize("role", [None, "acceptance", "bootstrap"])
def test_history_requires_exact_bootstrap_role(role) -> None:
    module = load()
    rows = six_phase_runs()
    if role is None:
        rows[0].pop("evidence_role")
    else:
        rows[0]["evidence_role"] = role
    with pytest.raises(module.ContractError, match="evidence_role"):
        module.build_prediction_contract(manifest(), history(rows))


def test_history_rejects_acceptance_root_even_with_forged_bootstrap_role() -> None:
    module = load()
    rows = six_phase_runs()
    rows[0]["artifact_root"] = rows[0]["artifact_root"].replace("/bootstrap/", "/acceptance/")
    with pytest.raises(module.ContractError, match="artifact_root"):
        module.build_prediction_contract(manifest(), history(rows))


def test_history_rejects_artifact_hash_mismatch_and_duplicate_root() -> None:
    module = load()
    rows = six_phase_runs()
    rows[0]["artifact_bindings"]["metrics"]["sha256"] = "bad"
    with pytest.raises(module.ContractError, match="hash mismatch"):
        module.build_prediction_contract(manifest(), history(rows))
    rows = six_phase_runs()
    rows[1]["artifact_root"] = rows[0]["artifact_root"]
    rows[1]["artifact_bindings"] = rows[0]["artifact_bindings"]
    with pytest.raises(module.ContractError, match="duplicate bootstrap artifact_root"):
        module.build_prediction_contract(manifest(), history(rows))


def test_posthoc_prediction_contract_mismatch_is_blocked() -> None:
    module = load()
    rows = six_phase_runs()
    contract = module.build_prediction_contract(manifest(), history(rows))
    tampered = copy.deepcopy(contract)
    tampered["phases"][0]["predictions"]["validation_elapsed_seconds"]["interval"][1] += 1
    result = module.verify_prediction_contract(tampered, manifest(), history(rows))
    assert not result["ok"]
    assert result["decision"] == "blocked"
    assert "does not match" in result["failures"][0]


def test_semantic_hash_mismatch_blocks_existing_contract_validation() -> None:
    module = load()
    rows = six_phase_runs()
    contract = module.build_prediction_contract(manifest(), history(rows))
    changed_manifest = manifest()
    changed_manifest["sampled_decoding_semantic_hash"] = "changed"
    result = module.verify_prediction_contract(contract, changed_manifest, history(rows))
    assert not result["ok"]
    assert result["decision"] == "blocked"


def test_cli_generates_and_validates_contract_with_plan_arguments(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    history_path = tmp_path / "trusted_history.json"
    contract_path = tmp_path / "prediction_contract.json"
    manifest_path.write_text(json.dumps(manifest()))
    history_path.write_text(json.dumps(history(six_phase_runs())))

    write_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_calibration_prediction_contract.py"),
            "--contract",
            str(contract_path),
            "--manifest",
            str(manifest_path),
            "--history-index",
            str(history_path),
            "--write",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert write_result.returncode == 0, write_result.stderr + write_result.stdout
    assert contract_path.exists()

    check_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_calibration_prediction_contract.py"),
            "--contract",
            str(contract_path),
            "--manifest",
            str(manifest_path),
            "--history-index",
            str(history_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert check_result.returncode == 0, check_result.stderr + check_result.stdout
    assert json.loads(check_result.stdout)["ok"] is True


def test_cli_does_not_expose_arbitrary_phase_selection(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_calibration_prediction_contract.py"),
            "--contract",
            str(tmp_path / "contract.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--history-index",
            str(tmp_path / "history.json"),
            "--phases",
            "stage1",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --phases stage1" in result.stderr
