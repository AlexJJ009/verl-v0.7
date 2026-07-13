from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check_code_task_operational_calibration.py"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def module():
    spec = importlib.util.spec_from_file_location("operational_checker", CHECKER)
    result = importlib.util.module_from_spec(spec); assert spec.loader; sys.modules[spec.name] = result; spec.loader.exec_module(result); return result


def manifest():
    return json.loads(subprocess.check_output(["python3", str(ROOT / "scripts/experiment_manifest.py"), "render", str(MANIFEST), "--format", "json"], text=True))


def repetition(status="passed", *, timed_out=False, cleanup=True):
    return {
        "status": status,
        "timed_out": timed_out,
        "metrics": {"validation_elapsed_seconds": 10.0},
        "resources": {"peak_rss_gib": 2.0},
        "cleanup": {"resources_released": cleanup},
        "score_complete": True,
        "truncated_count": 0,
    }


def report(data):
    phases = [item["phase"] for item in data["runs"]]
    return {
        "authorization_scope": "full",
        "evidence_class": "infrastructure_calibration",
        "decision": "candidate",
        "manifest_sha256": data["manifest_sha256"],
        "contract": {"validation_deadline_seconds": data["calibration_policy"]["validation_deadline_seconds"]},
        "phases": [{"phase": phase, "profile_hash": data["resource_profile"]["sha256"], "repetitions": [repetition()]} for phase in phases],
    }


def failure_codes(result):
    return [item["code"] for item in result["failures"]]


def test_valid_candidate_passes_with_structured_empty_failures():
    checker = module(); data = manifest(); result = checker.check(report(data), data)
    assert result == {"ok": True, "decision": "passed", "failures": [], "diagnostics": {}}


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(authorization_scope="wrong"), "authorization_scope"),
        (lambda value: value.update(evidence_class="wrong"), "evidence_class"),
        (lambda value: value.update(decision="blocked"), "candidate_decision"),
        (lambda value: value.update(manifest_sha256="wrong"), "manifest_hash"),
        (lambda value: value["phases"].reverse(), "phase_order"),
        (lambda value: value["phases"][0].update(profile_hash="wrong"), "profile_hash"),
        (lambda value: value["contract"].update(validation_deadline_seconds=1), "validation_deadline"),
    ],
)
def test_invalid_inputs_have_stable_failure_codes(mutation, code):
    checker = module(); data = manifest(); candidate = report(data); mutation(candidate)
    result = checker.check(candidate, data)
    assert not result["ok"] and result["decision"] == "blocked"
    assert code in failure_codes(result)
    assert all(set(item) == {"code", "message", "context"} for item in result["failures"])


def test_repetition_failure_codes_cover_timeout_exit_metrics_and_cleanup():
    checker = module(); data = manifest(); candidate = report(data)
    candidate["phases"][0]["repetitions"] = [
        {"status": "failed", "timed_out": True, "metrics": {}, "resources": None, "cleanup": {"resources_released": False}}
    ]
    assert set(failure_codes(checker.check(candidate, data))) >= {
        "repetition_timeout", "repetition_status", "metrics_incomplete", "resources_missing", "cleanup_failed"
    }


def test_repetition_failure_context_uses_one_based_evidence_identity():
    checker = module(); data = manifest(); candidate = report(data)
    candidate["phases"][1]["repetitions"] = [
        {**repetition("passed"), "repetition": 1},
        {**repetition("passed"), "repetition": 2},
        {**repetition("failed", timed_out=True), "repetition": 3},
    ]
    result = checker.check(candidate, data)
    failures = [item for item in result["failures"] if item["code"] in {"repetition_timeout", "repetition_status"}]
    assert failures and all(item["context"]["phase"] == "stage3" for item in failures)
    assert all(item["context"]["repetition"] == 3 for item in failures)


def test_message_formatting_does_not_change_decision_semantics():
    checker = module()
    first = checker.ValidationResult(); first.add("child_exit", "child failed", returncode=9)
    second = checker.ValidationResult(); second.add("child_exit", "Child failed.", returncode=9)
    assert first.as_dict()["decision"] == second.as_dict()["decision"] == "blocked"
    assert first.failures[0].code == second.failures[0].code == "child_exit"
    assert first.failures[0].context == second.failures[0].context == {"returncode": 9}


def test_cli_exit_semantics_and_receipt_compatibility_warning(tmp_path: Path):
    data = manifest(); manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(data))
    report_path = tmp_path / "report.json"; report_path.write_text(json.dumps(report(data)))
    receipt = tmp_path / "legacy-receipt.json"
    passed = subprocess.run(["python3", str(CHECKER), "--report", str(report_path), "--manifest", str(manifest_path), "--receipt", str(receipt)], text=True, capture_output=True)
    assert passed.returncode == 0
    output = json.loads(passed.stdout)
    assert output["ok"] and "no longer issued" in output["compatibility_warning"]
    assert not receipt.exists()
    candidate = report(data); candidate["decision"] = "blocked"; report_path.write_text(json.dumps(candidate))
    failed = subprocess.run(["python3", str(CHECKER), "--report", str(report_path), "--manifest", str(manifest_path)], text=True, capture_output=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["failures"][0]["code"] == "candidate_decision"
