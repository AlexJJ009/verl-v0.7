from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
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


def verify_args(normalized, normalized_path, report, policy, receipt_path):
    return type("Args", (), {"receipt": receipt_path, "normalized_manifest": normalized_path, "report": report, "policy": policy, "run_id": "frac25-stage2", "profile_hash": normalized["resource_profile"]["sha256"], "max_age_seconds": 3600})


def test_valid_receipt_passes(tmp_path: Path):
    tool, normalized, normalized_path, report, policy, receipt = receipt_fixture(tmp_path)
    assert tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))["ok"]


def test_stale_and_mismatched_receipts_fail(tmp_path: Path):
    tool, normalized, normalized_path, report, policy, receipt = receipt_fixture(tmp_path)
    data = json.loads(receipt.read_text()); data["generated_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(); receipt.write_text(json.dumps(data))
    assert "receipt stale" in tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))["failures"]
    data["generated_at"] = datetime.now(timezone.utc).isoformat(); data["report_sha256"] = "0" * 64; receipt.write_text(json.dumps(data))
    assert "report_sha256 mismatch" in tool.verify(verify_args(normalized, normalized_path, report, policy, receipt))["failures"]


def test_direct_phase_missing_receipt_fails_before_base_launcher():
    script = ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh"
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True, env={"PATH": "/usr/bin:/bin", "STAGE123_RUN_ID": "frac25-stage2", "DRY_RUN": "0"})
    assert result.returncode != 0
    assert "STAGE123_MANIFEST" in result.stderr


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
