from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def module():
    path = ROOT / "scripts/execution_results.py"
    spec = importlib.util.spec_from_file_location("execution_results", path)
    result = importlib.util.module_from_spec(spec); assert spec.loader; sys.modules[spec.name] = result; spec.loader.exec_module(result); return result


def result(kind: str, decision: str = "passed"):
    return {"schema_version": 1, "result_type": kind, "manifest_sha256": "a" * 64, "decision": decision}


def test_only_three_result_classes_can_authorize_current_execution():
    tool = module()
    assert tool.validate_result(result("preflight_result")).authorized
    assert tool.validate_result(result("calibration_result")).authorized
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
