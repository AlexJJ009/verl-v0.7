from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def canonical(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verifier_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checker = load("stage12_checker_fixture", ROOT / "scripts/check_code_task_operational_calibration.py")
    verifier = load(
        "stage12_receipt_verifier",
        ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_stage12_producer_receipt.py",
    )
    manifest = checker.load_manifest(MANIFEST)
    normalized = tmp_path / "normalized.json"
    canonical(normalized, manifest)
    report = tmp_path / "report.json"; canonical(report, {"phases": [{"phase": "stage1"}, {"phase": "stage2"}]})
    policy = tmp_path / "policy.json"; canonical(policy, {"policy": "reviewed"})
    history = tmp_path / "history.json"; canonical(history, {"phases": ["stage1", "stage2"]})
    preflight_report = tmp_path / "preflight-report.json"; canonical(preflight_report, {"machine": "l40s"})
    preflight_policy = tmp_path / "preflight-policy.json"; canonical(preflight_policy, {"policy": "reviewed"})
    budget = tmp_path / "budget.json"; canonical(budget, {"ok": True, "decision": "pass"})
    preflight_tool = load("stage12_preflight_fixture", ROOT / "scripts/stage123_preflight_receipt.py")
    preflight = tmp_path / "preflight.json"
    canonical(preflight, preflight_tool.issue(SimpleNamespace(
        normalized_manifest=normalized, report=preflight_report, policy=preflight_policy,
        budget_result=budget,
    )))
    contract_doc = {
        "phases": [
            {"phase": phase, "eligible_run_ids": [f"bootstrap-{phase}-{i}" for i in range(6)]}
            for phase in ("stage1", "stage2")
        ]
    }
    contract = tmp_path / "contract.json"; canonical(contract, contract_doc)
    hashes = {
        "manifest": digest(MANIFEST), "policy": digest(policy), "history_index": digest(history),
        "prediction_contract": digest(contract), "preflight_receipt": digest(preflight),
    }
    result = {"decision": "stage12_calibrated", "failures": []}
    receipt_doc = checker.build_stage12_receipt(result, report, MANIFEST, hashes, {"queue_identity": "stage123-formal"}, manifest, contract_doc)
    receipt = tmp_path / "receipt.json"; canonical(receipt, receipt_doc)
    args = SimpleNamespace(
        receipt=receipt, normalized_manifest=normalized, manifest=MANIFEST, preflight_receipt=preflight,
        preflight_report=preflight_report, preflight_policy=preflight_policy,
        report=report, policy=policy, history_index=history, prediction_contract=contract,
        queue_identity="stage123-formal", run_id="frac25-stage2", max_age_seconds=86400,
        future_skew_seconds=300,
    )
    return verifier, args, receipt_doc


def test_valid_limited_receipt_is_exact_and_passes(tmp_path: Path):
    verifier, args, receipt = verifier_fixture(tmp_path)
    assert receipt["phase_scope"] == ["stage1", "stage2"]
    assert receipt["authorized_run_ids"] == ["frac25-stage2"]
    assert receipt["authorized_final_steps"] == {"frac25-stage2": 20}
    assert verifier.verify(args)["ok"]


def test_wrong_scope_tampering_and_staleness_fail_closed(tmp_path: Path):
    verifier, args, receipt = verifier_fixture(tmp_path)
    args.run_id = "frac25-stage3"
    assert "limited_receipt_scope_mismatch" in verifier.verify(args)["failures"]
    args.run_id = "frac25-stage2"
    receipt["producer"]["final_step"] = 40
    canonical(args.receipt, receipt)
    assert "producer mismatch" in verifier.verify(args)["failures"]
    receipt["producer"]["final_step"] = 20
    receipt["issued_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    canonical(args.receipt, receipt)
    assert "receipt stale" in verifier.verify(args)["failures"]


def test_fake_or_stale_preflight_fails(tmp_path: Path):
    verifier, args, _receipt = verifier_fixture(tmp_path)
    canonical(args.preflight_receipt, {"authorized_calibration_phases": ["stage1", "stage2"]})
    assert any("preflight verification failed" in item for item in verifier.verify(args)["failures"])
    _verifier, args, _receipt = verifier_fixture(tmp_path / "stale")
    preflight = json.loads(args.preflight_receipt.read_text())
    preflight["generated_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    canonical(args.preflight_receipt, preflight)
    assert any("receipt stale" in item for item in verifier.verify(args)["failures"])


def test_extra_schema_phase_and_cohort_hash_fail(tmp_path: Path):
    verifier, args, receipt = verifier_fixture(tmp_path)
    receipt["phase_scope"].append("stage3")
    receipt["unexpected"] = True
    receipt["selected_cohort_sha256_by_phase"]["stage1"] = "0" * 64
    canonical(args.receipt, receipt)
    failures = verifier.verify(args)["failures"]
    assert "phase_scope mismatch" in failures
    assert "limited receipt schema mismatch" in failures
    assert "selected cohort hashes mismatch" in failures


def test_full_deployability_and_completion_reject_limited_receipt(tmp_path: Path):
    _verifier, args, _receipt = verifier_fixture(tmp_path)
    full = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_deployability_receipt.py"
    result = subprocess.run([
        "python3", str(full), "--receipt", str(args.receipt), "--normalized-manifest", str(args.normalized_manifest),
        "--preflight-receipt", str(args.preflight_receipt), "--report", str(args.report), "--policy", str(args.policy),
        "--history-index", str(args.history_index), "--prediction-contract", str(args.prediction_contract),
        "--queue-identity", args.queue_identity, "--profile-hash", json.loads(args.normalized_manifest.read_text())["resource_profile"]["sha256"],
    ], text=True, capture_output=True)
    assert result.returncode != 0
    assert "receipt is not deployable" in result.stdout
    completion = load("stage12_completion", ROOT / "scripts/experiment_goal_completion_state.py")
    assert completion.state(json.loads(args.receipt.read_text()), None) == "PENDING OPERATIONAL CALIBRATION"
    completion_cli = subprocess.run(["python3", str(ROOT / "scripts/experiment_goal_completion_state.py"), "--calibration", str(args.receipt)], text=True, capture_output=True)
    assert completion_cli.returncode != 0
    assert "limited_receipt_scope_mismatch" in completion_cli.stdout


def test_checker_full_receipt_round_trips_through_formal_verifier(tmp_path: Path):
    checker = load("full_receipt_builder", ROOT / "scripts/check_code_task_operational_calibration.py")
    verifier = load("full_receipt_verifier", ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_deployability_receipt.py")
    manifest = checker.load_manifest(MANIFEST)
    normalized = tmp_path / "normalized.json"; canonical(normalized, manifest)
    files = {}
    for name in ("report", "policy", "history", "contract", "preflight"):
        files[name] = tmp_path / f"{name}.json"; canonical(files[name], {"name": name})
    hashes = {"manifest": digest(MANIFEST), "policy": digest(files["policy"]), "history_index": digest(files["history"]),
              "prediction_contract": digest(files["contract"]), "preflight_receipt": digest(files["preflight"])}
    receipt_doc = checker.build_receipt({"decision": "deployable", "failures": []}, files["report"], MANIFEST, hashes,
                                        {"queue_identity": "stage123-formal", "input_bindings": {"resource_profile": {"sha256": manifest["resource_profile"]["sha256"]}}}, manifest)
    receipt = tmp_path / "full.json"; canonical(receipt, receipt_doc)
    result = verifier.verify(SimpleNamespace(receipt=receipt, normalized_manifest=normalized, preflight_receipt=files["preflight"],
        report=files["report"], policy=files["policy"], history_index=files["history"], prediction_contract=files["contract"],
        semantic_contract=None, queue_identity="stage123-formal", profile_hash=manifest["resource_profile"]["sha256"],
        max_age_seconds=86400, future_skew_seconds=300))
    assert result["ok"], result


def test_manifest_gate_rejects_limited_receipt_for_other_runs():
    gate = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh"
    for run_id in ("frac25-stage1", "frac25-stage3", "frac50-stage2"):
        command = f'source "{gate}"; DRY_RUN=0 stage123_require_stage12_producer_receipt "{run_id}"'
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        assert result.returncode != 0
        assert "limited_receipt_scope_mismatch" in result.stderr


def test_queue_limits_stage12_mode_to_exact_producer_and_stops_before_stage3():
    queue = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh").read_text()
    assert 'STAGE123_STAGE12_PRODUCER_MODE=1' in queue
    assert '[ "$stage2_id" != "frac25-stage2" ]' in queue
    assert "Stage2 producer complete; Stage3 remains pending regenerated preflight" in queue
    producer_exit = queue.index("Stage2 producer complete; Stage3 remains pending regenerated preflight")
    stage3_admission = queue.index('stage123_require_formal_admission "$stage3_id"', producer_exit)
    assert producer_exit < stage3_admission
