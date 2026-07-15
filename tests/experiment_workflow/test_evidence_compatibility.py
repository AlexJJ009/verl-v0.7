from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]


def module():
    path = ROOT / "scripts/execution_results.py"
    spec = importlib.util.spec_from_file_location("execution_results", path)
    result = importlib.util.module_from_spec(spec); assert spec.loader; sys.modules[spec.name] = result; spec.loader.exec_module(result); return result


def result(kind: str, decision: str = "passed"):
    value = {"schema_version": 1, "result_type": kind, "manifest_sha256": "a" * 64, "decision": decision}
    if kind == "calibration_result":
        policy_path = ROOT / "config/experiment_execution/calibration_policy_v1.json"
        comparison = {"history": [1.0, 2.0, 3.0], "history_count": 3, "predicted_bound": 3.0, "observed_maximum": 3.0, "decision": {"qualified": True, "code": "qualified", "context": {"ratio": 1.0}}}
        comparisons = [{**comparison, "metric": metric} for metric in ("validation_elapsed_seconds", "phase_elapsed_seconds", "peak_rss_gib", "gpu_wait_fraction")]
        value.update({
            "resource_profile_sha256": "b" * 64, "implementation_tree_sha256": "c" * 64,
            "evidence_commit": "d" * 40, "workload_identity": {"sha256": "f" * 64, "run_ids": ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]}, "policy_id": "stage123-calibration-policy-v1",
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(), "authorization_identity": {"id": "auth"}, "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z", "phase_evidence": [{"phase": "stage1", "status": "passed"}, {"phase": "stage2", "status": "passed"}, {"phase": "stage3", "status": "passed"}], "prediction_comparison": {"qualified": True, "policy_id": "stage123-calibration-policy-v1", "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(), "comparisons": comparisons},
            "cleanup": {"resources_released": True}, "failures": [],
        })
    return value


def test_only_three_result_classes_can_authorize_current_execution():
    tool = module()
    assert tool.validate_result(result("preflight_result")).authorized
    calibration = result("calibration_result")
    bindings = {"manifest_sha256": calibration["manifest_sha256"], "resource_profile_sha256": calibration["resource_profile_sha256"], "implementation_tree_sha256": calibration["implementation_tree_sha256"], "evidence_commit": calibration["evidence_commit"], "run_ids": calibration["workload_identity"]["run_ids"], "authorization_identity": calibration["authorization_identity"]}
    assert tool.validate_result(calibration, expected_bindings=bindings).authorized
    assert tool.validate_result(result("acceptance_report", "accepted")).authorized
    assert not tool.validate_result(result("acceptance_report", "passed")).authorized


def test_every_legacy_receipt_and_adoption_class_fails_closed():
    tool = module()
    legacy = [
        {"receipt_type": "code_task_operational_calibration_deployability"},
        {"receipt_type": "code_task_operational_calibration_stage12_producer"},
        {"result_type": "stage123_preflight_receipt"},
        {"result_type": "dirty_adoption"},
        {"result_type": "document_hash_adoption"},
    ]
    for value in legacy:
        decision = tool.validate_result(value)
        assert not decision.authorized
        assert decision.code == "legacy_evidence"


def test_historical_migration_is_byte_identical(tmp_path: Path):
    tool = module(); source = tmp_path / "historical.json"; source.write_bytes(b'{"old": true}\n')
    before = source.read_bytes(); destination = tmp_path / "archive" / source.name
    report = tool.archive_historical(source, destination)
    assert source.read_bytes() == destination.read_bytes() == before
    assert report["byte_identical"] is True


def test_documentation_only_change_requires_no_adoption_receipt():
    tool = module()
    assert not tool.documentation_change_requires_receipt(["docs/guide.md", ".github/workflows/docs.yml"])
    assert tool.documentation_change_requires_receipt(["docs/guide.md", "scripts/runtime.py"])


def test_stale_or_malformed_result_files_fail_closed(tmp_path: Path):
    tool = module(); malformed = tmp_path / "bad.json"; malformed.write_text("not-json")
    assert tool.load_and_validate(malformed).code == "invalid_result_file"
    legacy = tmp_path / "legacy.json"; legacy.write_text(json.dumps({"receipt_type": "code_task_operational_calibration_deployability"}))
    assert tool.load_and_validate(legacy).code == "legacy_evidence"


def test_incomplete_calibration_result_never_authorizes():
    tool = module(); decision = tool.validate_result({"schema_version": 1, "result_type": "calibration_result", "manifest_sha256": "a" * 64, "decision": "passed"})
    assert not decision.authorized and decision.code == "result_fields"


def test_calibration_result_requires_and_matches_explicit_bindings():
    tool = module(); calibration = result("calibration_result")
    assert tool.validate_result(calibration).code == "expected_bindings"
    bindings = {"manifest_sha256": calibration["manifest_sha256"], "resource_profile_sha256": calibration["resource_profile_sha256"], "implementation_tree_sha256": calibration["implementation_tree_sha256"], "evidence_commit": calibration["evidence_commit"], "run_ids": calibration["workload_identity"]["run_ids"], "authorization_identity": calibration["authorization_identity"]}
    for key in bindings:
        mutated = dict(bindings); mutated[key] = "wrong"
        assert tool.validate_result(calibration, expected_bindings=mutated).code == "result_binding"
