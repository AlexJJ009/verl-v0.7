from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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


def test_live_batch_phase_validation_skips_freshness_but_requires_live_state(tmp_path: Path, monkeypatch) -> None:
    tool = load_module("stage123_execution_results_phase", ROOT / "scripts/execution_results.py")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}")
    bundle_sha = tool.file_sha256(bundle_path)
    state_path = tmp_path / "batch.json"
    record_path = tmp_path / "record.json"
    record = {
        "schema_version": 1,
        "status": "active",
        "batch_id": "batch-1",
        "batch_manifest_sha256": "1" * 64,
        "batch_state_path": str(state_path),
        "item_id": "item-1",
        "expected_run_ids": ["frac25-stage2", "frac25-stage3"],
        "command_sha256": "2" * 64,
        "admission_bundle_sha256": bundle_sha,
    }
    record["record_sha256"] = tool.result_sha256(record)
    record_path.write_text(tool.json.dumps(record))
    state_path.write_text(
        tool.json.dumps(
            {
                "status": "running",
                "current_item_id": "item-1",
                "current_item_admission": {"path": str(record_path), "sha256": record["record_sha256"]},
            }
        )
    )
    freshness_modes = []
    monkeypatch.setattr(
        tool, "validate_admission_bundle", lambda bundle, require_accepted: SimpleNamespace(authorized=True)
    )
    monkeypatch.setattr(
        tool,
        "validate_current_checkout",
        lambda bundle,
        repo_root,
        protected_baseline,
        require_accepted,
        enforce_result_freshness: freshness_modes.append(enforce_result_freshness) or SimpleNamespace(authorized=True),
    )
    bundle = {"inputs": {"protected_baseline": str(tmp_path / "protected.jsonl")}}
    decision = tool.validate_batch_phase_admission(
        bundle,
        bundle_path,
        record_path,
        ROOT,
        run_id="frac25-stage2",
        batch_id="batch-1",
        batch_manifest_sha256="1" * 64,
        item_id="item-1",
        admission_bundle_sha256=bundle_sha,
        command_sha256="2" * 64,
        record_sha256=record["record_sha256"],
    )
    assert decision.authorized
    assert freshness_modes == [False]
    state_path.write_text(tool.json.dumps({"status": "completed", "current_item_id": "item-1"}))
    rejected = tool.validate_batch_phase_admission(
        bundle,
        bundle_path,
        record_path,
        ROOT,
        run_id="frac25-stage3",
        batch_id="batch-1",
        batch_manifest_sha256="1" * 64,
        item_id="item-1",
        admission_bundle_sha256=bundle_sha,
        command_sha256="2" * 64,
        record_sha256=record["record_sha256"],
    )
    assert rejected.code == "batch_admission_not_live"
