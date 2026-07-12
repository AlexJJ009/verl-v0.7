from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def write_canonical_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module); return module


def receipt_fixture(tmp_path: Path):
    manifest_tool = load("manifest_receipt", ROOT / "scripts/experiment_manifest.py")
    receipt_tool = load("receipt_tool", ROOT / "scripts/stage123_preflight_receipt.py")
    normalized = manifest_tool.normalize(manifest_tool.load(MANIFEST))
    normalized_path = tmp_path / "normalized.json"; normalized_path.write_text(json.dumps(normalized))
    report = tmp_path / "report.json"; report.write_text(json.dumps({"ok": True}))
    policy = tmp_path / "policy.json"; policy.write_text(json.dumps({"policy": True}))
    budget = tmp_path / "budget.json"; budget.write_text(json.dumps({"ok": True, "decision": "pass"}))
    args = type("Args", (), {"normalized_manifest": normalized_path, "report": report, "policy": policy, "budget_result": budget})
    receipt = receipt_tool.issue(args)
    receipt_path = tmp_path / "receipt.json"; receipt_path.write_text(json.dumps(receipt))
    return receipt_tool, normalized, normalized_path, report, policy, receipt_path


def deployability_fixture(tmp_path: Path):
    _tool, normalized, normalized_path, report, policy, preflight_receipt = receipt_fixture(tmp_path)
    history = tmp_path / "history.json"; history.write_text(json.dumps({"history": []}))
    contract = tmp_path / "prediction_contract.json"; contract.write_text(json.dumps({"contract": True}))
    receipt = {
        "schema_version": 1,
        "decision": "deployable",
        "issued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "queue_identity": "qwen3_1p7b_stage123_p40",
        "manifest_sha256": normalized["manifest_sha256"],
        "profile_sha256": normalized["resource_profile"]["sha256"],
        "preflight_receipt_sha256": hashlib.sha256(preflight_receipt.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "history_index_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        "prediction_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    }
    deployability_receipt = tmp_path / "deployability_receipt.json"
    write_canonical_json(deployability_receipt, receipt)
    return normalized, normalized_path, report, policy, preflight_receipt, history, contract, deployability_receipt


def verify_args(normalized, normalized_path, report, policy, receipt_path):
    return type("Args", (), {"receipt": receipt_path, "normalized_manifest": normalized_path, "report": report, "policy": policy, "run_id": "frac25-stage2", "profile_hash": normalized["resource_profile"]["sha256"], "max_age_seconds": 3600})


def test_valid_receipt_passes(tmp_path: Path):
    tool, normalized, normalized_path, report, policy, receipt = receipt_fixture(tmp_path)
    assert tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))["ok"]


def test_pending_stage3_is_not_authorized_by_preflight_receipt(tmp_path: Path):
    tool, normalized, normalized_path, report, policy, receipt = receipt_fixture(tmp_path)
    data = json.loads(receipt.read_text())
    assert "frac25-stage2" in data["authorized_run_ids"]
    assert "frac25-stage3" not in data["authorized_run_ids"]
    args = verify_args(normalized, normalized_path, report, policy, receipt)
    args.run_id = "frac25-stage3"
    result = tool.verify(args)
    assert not result["ok"]
    assert "run_id not authorized by receipt" in result["failures"]


def test_workload_identity_drift_invalidates_preflight_receipt(tmp_path: Path):
    tool, normalized, normalized_path, report, policy, receipt = receipt_fixture(tmp_path)
    normalized["calibration_workloads"]["stage1"]["model_sources"][0]["artifact_sha256"] = "0" * 64
    normalized_path.write_text(json.dumps(normalized))
    result = tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))
    assert not result["ok"]
    assert "workload_descriptor_sha256 mismatch" in result["failures"]


def test_stale_and_mismatched_receipts_fail(tmp_path: Path):
    tool, normalized, normalized_path, report, policy, receipt = receipt_fixture(tmp_path)
    data = json.loads(receipt.read_text()); data["generated_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(); receipt.write_text(json.dumps(data))
    assert "receipt stale" in tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))["failures"]
    data["generated_at"] = datetime.now(timezone.utc).isoformat(); data["report_sha256"] = "0" * 64; receipt.write_text(json.dumps(data))
    assert "report_sha256 mismatch" in tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))["failures"]


def deployability_verify_args(normalized, normalized_path, report, policy, preflight_receipt, history, contract, deployability_receipt):
    return [
        "python3",
        str(ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_deployability_receipt.py"),
        "--receipt", str(deployability_receipt),
        "--normalized-manifest", str(normalized_path),
        "--preflight-receipt", str(preflight_receipt),
        "--report", str(report),
        "--policy", str(policy),
        "--history-index", str(history),
        "--prediction-contract", str(contract),
        "--queue-identity", "qwen3_1p7b_stage123_p40",
        "--profile-hash", normalized["resource_profile"]["sha256"],
    ]


def test_deployability_receipt_passes_and_binds_preflight(tmp_path: Path):
    fixture = deployability_fixture(tmp_path)
    args = deployability_verify_args(*fixture)
    assert subprocess.run(args, text=True, capture_output=True).returncode == 0
    other_preflight = tmp_path / "other_preflight.json"; other_preflight.write_text('{"status":"pass","other":true}')
    mismatch_args = list(args)
    mismatch_args[mismatch_args.index("--preflight-receipt") + 1] = str(other_preflight)
    failed = subprocess.run(mismatch_args, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "preflight_receipt_sha256 mismatch" in failed.stdout


def test_deployability_receipt_requires_canonical_json_bytes(tmp_path: Path):
    fixture = deployability_fixture(tmp_path)
    args = deployability_verify_args(*fixture)
    *_, receipt = fixture
    receipt.write_text(json.dumps(json.loads(receipt.read_text()), indent=2, sort_keys=True) + "\n")
    failed = subprocess.run(args, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "receipt is not canonical JSON" in failed.stdout


def test_deployability_receipt_replay_boundaries_fail(tmp_path: Path):
    fixture = deployability_fixture(tmp_path)
    args = deployability_verify_args(*fixture)
    *_, deployability_receipt = fixture
    data = json.loads(deployability_receipt.read_text())
    data["queue_identity"] = "other-queue"; write_canonical_json(deployability_receipt, data)
    failed = subprocess.run(args, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "queue_identity mismatch" in failed.stdout
    data["queue_identity"] = "qwen3_1p7b_stage123_p40"
    data["issued_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_canonical_json(deployability_receipt, data)
    failed = subprocess.run(args, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "receipt stale" in failed.stdout
    data["issued_at"] = (datetime.now(timezone.utc) + timedelta(seconds=600)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_canonical_json(deployability_receipt, data)
    failed = subprocess.run(args, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "future skew" in failed.stdout


def test_deployability_receipt_future_skew_exact_boundary(tmp_path: Path):
    fixture = deployability_fixture(tmp_path)
    normalized, normalized_path, report, policy, preflight, history, contract, receipt = fixture
    module_path = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_deployability_receipt.py"
    spec = importlib.util.spec_from_file_location("deployability_receipt", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    fixed_now = datetime(2026, 7, 12, 0, 0, 0, tzinfo=timezone.utc)
    data = json.loads(receipt.read_text())
    data["issued_at"] = "2026-07-12T00:05:00Z"
    write_canonical_json(receipt, data)
    args = module.argparse.Namespace(
        receipt=receipt,
        normalized_manifest=normalized_path,
        preflight_receipt=preflight,
        report=report,
        policy=policy,
        history_index=history,
        prediction_contract=contract,
        semantic_contract=None,
        queue_identity="qwen3_1p7b_stage123_p40",
        profile_hash=normalized["resource_profile"]["sha256"],
        max_age_seconds=86400,
        future_skew_seconds=300,
    )
    assert module.verify(args, now=fixed_now)["ok"]
    data["issued_at"] = "2026-07-12T00:05:01Z"
    write_canonical_json(receipt, data)
    result = module.verify(args, now=fixed_now)
    assert not result["ok"]
    assert "receipt issued_at exceeds future skew" in result["failures"]


def test_direct_phase_missing_receipt_fails_before_base_launcher():
    script = ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh"
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True, env={"PATH": "/usr/bin:/bin", "STAGE123_RUN_ID": "frac25-stage2", "DRY_RUN": "0"})
    assert result.returncode != 0
    assert "STAGE123_MANIFEST" in result.stderr


def test_direct_phase_missing_deployability_receipt_fails_before_base_launcher(tmp_path: Path):
    normalized, normalized_path, report, policy, preflight_receipt, _history, _contract, _deployability = deployability_fixture(tmp_path)
    script = ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh"
    env = {
        **os.environ,
        "STAGE123_RUN_ID": "frac25-stage2",
        "DRY_RUN": "0",
        "STAGE123_MANIFEST": str(MANIFEST),
        "STAGE123_NORMALIZED_MANIFEST": str(normalized_path),
        "STAGE123_PREFLIGHT_REPORT": str(report),
        "STAGE123_PREFLIGHT_RECEIPT": str(preflight_receipt),
        "STAGE123_PREFLIGHT_POLICY": str(policy),
        "STAGE123_EXPECTED_PROFILE_HASH": normalized["resource_profile"]["sha256"],
        "STAGE123_RECEIPT_MAX_AGE_SECONDS": "3600",
    }
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert "STAGE123_DEPLOYABILITY_RECEIPT" in result.stderr


def test_skip_variables_are_absent_from_formal_paths():
    paths = [
        ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh",
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh",
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
    ]
    text = "\n".join(path.read_text() for path in paths)
    assert "SKIP_STAGE123_PREFLIGHT" not in text
    assert "SKIP_STAGE123_MACHINE_GATE" not in text
    assert "--force" not in text
