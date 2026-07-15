from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_freshness_accepts_current_and_rejects_stale() -> None:
    tool = load_module("stage123_execution_results", ROOT / "scripts/execution_results.py")
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    current = {"completed_at": (now - timedelta(seconds=30)).isoformat()}
    stale = {"completed_at": (now - timedelta(seconds=61)).isoformat()}
    tool.enforce_freshness(current, 60, "preflight result", now=now)
    try:
        tool.enforce_freshness(stale, 60, "preflight result", now=now)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale preflight result was accepted")


def test_preflight_primary_inputs_exclude_frac50() -> None:
    preflight = load_module(
        "stage123_preflight_admission",
        ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_preflight.py",
    )
    assert preflight.PRIMARY_RUN_IDS == ("frac25-stage1-control", "frac25-stage2", "frac25-stage3")
    assert all("frac25" in name for name in preflight.PRIMARY_RUN_IDS)
    assert not any("frac50" in name for name in preflight.PRIMARY_RUN_IDS)


def test_accepted_bundle_requires_complete_report_bindings() -> None:
    tool = load_module("stage123_execution_results_acceptance", ROOT / "scripts/execution_results.py")
    bundle = {
        "schema_version": 1,
        "bundle_type": "stage123_admission_bundle",
        "run_ids": ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"],
        "bindings": {
            "manifest_sha256": "1" * 64,
            "resource_profile_sha256": "2" * 64,
            "implementation_tree_sha256": "3" * 64,
            "calibration_result_sha256": "4" * 64,
            "preflight_result_sha256": "5" * 64,
            "readiness_evidence_commit": "6" * 40,
        },
        "bundle_sha256": "7" * 64,
        "acceptance": {"decision": "accepted", "bundle_sha256": "7" * 64},
    }
    decision = tool.validate_admission_bundle(bundle, require_accepted=True)
    assert decision.code == "acceptance_binding"


def test_accepted_bundle_round_trip_uses_unsigned_bundle_hash() -> None:
    tool = load_module("stage123_execution_results_round_trip", ROOT / "scripts/execution_results.py")
    bundle = {
        "schema_version": 1,
        "bundle_type": "stage123_admission_bundle",
        "bundle_path": "/tmp/admission_bundle.json",
        "run_ids": ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"],
        "bindings": {
            "manifest_sha256": "1" * 64,
            "resource_profile_sha256": "2" * 64,
            "implementation_tree_sha256": "3" * 64,
            "calibration_result_sha256": "4" * 64,
            "preflight_result_sha256": "5" * 64,
            "readiness_evidence_commit": "6" * 40,
        },
    }
    bundle["bundle_sha256"] = tool.result_sha256(bundle)
    bundle["acceptance"] = {
        "result_type": "acceptance_report",
        "decision": "accepted",
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_sha256": bundle["bindings"]["manifest_sha256"],
        "resource_profile_sha256": bundle["bindings"]["resource_profile_sha256"],
        "implementation_tree_sha256": bundle["bindings"]["implementation_tree_sha256"],
        "calibration_result_sha256": bundle["bindings"]["calibration_result_sha256"],
        "preflight_result_sha256": bundle["bindings"]["preflight_result_sha256"],
        "readiness_evidence_commit": bundle["bindings"]["readiness_evidence_commit"],
        "run_ids": bundle["run_ids"],
    }
    unsigned = {key: value for key, value in bundle.items() if key not in {"bundle_sha256", "acceptance"}}
    assert bundle["bundle_sha256"] == tool.result_sha256(unsigned)
    assert tool.validate_admission_bundle(bundle, require_accepted=True).authorized
    bundle["acceptance"]["run_ids"] = ["frac50-stage2", "frac25-stage3"]
    assert tool.validate_admission_bundle(bundle, require_accepted=True).code == "acceptance_binding"
