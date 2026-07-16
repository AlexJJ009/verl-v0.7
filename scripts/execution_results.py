#!/usr/bin/env python3
"""Authoritative result classes and fail-closed legacy evidence migration."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Any


RESULT_TYPES = {
    "preflight_result": "preflight_result.json",
    "calibration_result": "calibration_result.json",
    "acceptance_report": "acceptance_report.json",
}
LEGACY_AUTHORITY_TYPES = {
    "code_task_operational_calibration_deployability",
    "code_task_operational_calibration_stage12_producer",
    "stage123_preflight_receipt",
    "dirty_adoption",
    "document_hash_adoption",
}


@dataclass(frozen=True)
class EvidenceDecision:
    authorized: bool
    code: str
    message: str
    context: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def result_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_result(value: dict[str, Any], expected_type: str | None = None, expected_bindings: dict[str, Any] | None = None) -> EvidenceDecision:
    result_type = value.get("result_type")
    if result_type in LEGACY_AUTHORITY_TYPES or value.get("receipt_type") in LEGACY_AUTHORITY_TYPES:
        return EvidenceDecision(False, "legacy_evidence", "legacy receipt or adoption evidence is not current authority", {"result_type": result_type, "receipt_type": value.get("receipt_type")})
    if result_type not in RESULT_TYPES:
        return EvidenceDecision(False, "unsupported_result_type", "unsupported execution result type", {"result_type": result_type})
    if expected_type is not None and result_type != expected_type:
        return EvidenceDecision(False, "result_type_mismatch", "execution result type mismatch", {"expected": expected_type, "actual": result_type})
    if value.get("schema_version") != 1:
        return EvidenceDecision(False, "schema_version", "unsupported execution result schema", {"schema_version": value.get("schema_version")})
    if result_type == "calibration_result":
        module_path = Path(__file__).with_name("calibration_result.py")
        spec = importlib.util.spec_from_file_location("_calibration_result_validator", module_path)
        if spec is None or spec.loader is None:
            return EvidenceDecision(False, "result_validator", "calibration result validator cannot be loaded", {})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema_path = Path(__file__).resolve().parents[1] / "config/experiment_execution/calibration_result_schema_v1.json"
        outcome = module.validate(value, load_object(schema_path))
        if not outcome["ok"]:
            first = outcome["failures"][0]
            return EvidenceDecision(False, first["code"], first["message"], first["context"])
        if expected_bindings is None:
            return EvidenceDecision(False, "expected_bindings", "calibration result requires explicit expected bindings", {})
        actual_bindings = {
            "manifest_sha256": value.get("manifest_sha256"),
            "resource_profile_sha256": value.get("resource_profile_sha256"),
            "implementation_tree_sha256": value.get("implementation_tree_sha256"),
            "evidence_commit": value.get("evidence_commit"),
            "run_ids": value.get("workload_identity", {}).get("run_ids"),
            "authorization_identity": value.get("authorization_identity"),
        }
        if actual_bindings != expected_bindings:
            return EvidenceDecision(False, "result_binding", "calibration result binding mismatch", {"expected": expected_bindings, "actual": actual_bindings})
    if not isinstance(value.get("manifest_sha256"), str) or len(value["manifest_sha256"]) != 64:
        return EvidenceDecision(False, "manifest_binding", "execution result lacks manifest binding", {})
    decision = value.get("decision")
    if decision not in {"passed", "blocked", "accepted", "rejected"}:
        return EvidenceDecision(False, "decision", "execution result has invalid decision", {"decision": decision})
    authorized = (result_type != "acceptance_report" and decision == "passed") or (result_type == "acceptance_report" and decision == "accepted")
    code = "authorized" if authorized else "result_blocked"
    message = "execution result authorizes this stage" if authorized else "execution result does not authorize this stage"
    return EvidenceDecision(authorized, code, message, {"result_type": result_type, "decision": decision})


def load_and_validate(path: Path, expected_type: str | None = None, expected_bindings: dict[str, Any] | None = None) -> EvidenceDecision:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return EvidenceDecision(False, "invalid_result_file", "execution result cannot be read", {"path": str(path), "error": str(exc)})
    if not isinstance(value, dict):
        return EvidenceDecision(False, "invalid_result_file", "execution result must be a JSON object", {"path": str(path)})
    return validate_result(value, expected_type, expected_bindings)


def archive_historical(source: Path, destination: Path) -> dict[str, Any]:
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    after = hashlib.sha256(destination.read_bytes()).hexdigest()
    if before != after:
        raise OSError("historical artifact changed during migration")
    return {"source": str(source), "destination": str(destination), "sha256": before, "byte_identical": True}


def documentation_change_requires_receipt(changed_paths: list[str]) -> bool:
    return any(not path.startswith(("docs/", ".github/")) for path in changed_paths)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_manifest_object(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        root = repo_root or Path(__file__).resolve().parents[1]
        value = json.loads(subprocess.check_output([sys.executable, str(root / "scripts/experiment_manifest.py"), "render", str(path), "--format", "json"], text=True))
        if not isinstance(value, dict):
            raise ValueError("rendered manifest is not an object")
        return value
    return load_object(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("evidence timestamp is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def enforce_freshness(value: dict[str, Any], max_age_seconds: int, label: str, *, now: datetime | None = None) -> None:
    completed = parse_timestamp(value.get("completed_at"))
    current = now or datetime.now(timezone.utc)
    age = (current - completed).total_seconds()
    if age < -300:
        raise ValueError(f"{label} timestamp is too far in the future")
    if age > max_age_seconds:
        raise ValueError(f"{label} is stale: age={age:.0f}s max={max_age_seconds}s")


def manifest_resource_profile_sha256(manifest: dict[str, Any]) -> str:
    profile = manifest.get("resource_profile")
    value = profile.get("sha256") if isinstance(profile, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("manifest lacks resource profile identity")
    return value


def rendered_resource_profile_sha256(profile_path: Path) -> str:
    result = subprocess.run(
        ["bash", "-lc", f"source {shlex.quote(str(profile_path))}; stage123_profile_snapshot"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("resource profile snapshot failed")
    return hashlib.sha256(result.stdout.encode()).hexdigest()


def current_implementation_tree_sha256(repo_root: Path) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/implementation_tree_identity.py"),
            "--repo-root",
            str(repo_root),
            "--boundary-manifest",
            str(repo_root / "config/experiment_execution/stage123_implementation_boundary_v1.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("current implementation identity cannot be recomputed")
    try:
        payload = json.loads(result.stderr)
    except json.JSONDecodeError as exc:
        raise ValueError("current implementation identity output is invalid") from exc
    value = payload.get("implementation_tree_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("current implementation identity is missing")
    return value


def protected_baseline_sha256(path: Path) -> str:
    return file_sha256(path)


def compare_protected_baseline(repo_root: Path, baseline: Path) -> None:
    tool = repo_root / "docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py"
    result = subprocess.run(
        [sys.executable, str(tool), "compare", "--repo-root", str(repo_root), "--baseline", str(baseline)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("protected asset baseline comparison failed")


def acceptance_report_sha256(report: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in report.items() if key != "acceptance_report_sha256"}
    return result_sha256(unsigned)


def validate_acceptance_report(report: dict[str, Any], bundle: dict[str, Any], *, plan_path: Path | None = None, candidate_commit: str | None = None) -> EvidenceDecision:
    required = {
        "result_type", "schema_version", "decision", "goal_id", "plan_id", "plan_version", "plan_sha256",
        "reviewer", "candidate_commit", "readiness_evidence_commit", "run_ids", "bundle_sha256",
        "acceptance_report_sha256", "ac_verdicts", "input_hashes", "protected_baseline_sha256",
    }
    missing = sorted(required - report.keys())
    if missing:
        return EvidenceDecision(False, "acceptance_schema", "acceptance report is incomplete", {"missing": missing})
    reviewer = report.get("reviewer")
    if not isinstance(reviewer, dict) or reviewer.get("model") != "GPT-5.5" or reviewer.get("reasoning_effort") != "medium":
        return EvidenceDecision(False, "acceptance_reviewer", "acceptance reviewer identity is not GPT-5.5 medium", {})
    if report.get("result_type") != "acceptance_report" or report.get("schema_version") != 1 or report.get("decision") != "accepted":
        return EvidenceDecision(False, "acceptance_schema", "acceptance report schema or decision is invalid", {})
    if report.get("bundle_sha256") != bundle.get("bundle_sha256"):
        return EvidenceDecision(False, "acceptance_binding", "acceptance report does not bind candidate bundle", {})
    if report.get("run_ids") != bundle.get("run_ids"):
        return EvidenceDecision(False, "acceptance_binding", "acceptance report run set does not match bundle", {})
    if candidate_commit and report.get("candidate_commit") != candidate_commit:
        return EvidenceDecision(False, "acceptance_binding", "acceptance candidate commit does not match current HEAD", {})
    if plan_path is not None:
        if report.get("plan_id") != "stage123-execution-readiness" or report.get("plan_version") != 9 or report.get("plan_sha256") != file_sha256(plan_path):
            return EvidenceDecision(False, "acceptance_plan_binding", "acceptance report Plan binding mismatch", {})
    verdicts = report.get("ac_verdicts")
    if not isinstance(verdicts, dict) or any(verdicts.get(f"AC-{index:02d}") != "PASS" for index in range(1, 9)):
        return EvidenceDecision(False, "acceptance_verdicts", "AC-01 through AC-08 must all be PASS", {})
    expected_inputs = bundle.get("bindings", {})
    input_hashes = report.get("input_hashes")
    if not isinstance(input_hashes, dict) or any(input_hashes.get(key) != expected_inputs.get(key) for key in ("manifest_sha256", "resource_profile_sha256", "calibration_result_sha256", "preflight_result_sha256")):
        return EvidenceDecision(False, "acceptance_input_binding", "acceptance input hashes do not match bundle", {})
    if report.get("protected_baseline_sha256") != expected_inputs.get("protected_baseline_sha256"):
        return EvidenceDecision(False, "acceptance_protected_binding", "acceptance protected baseline does not match bundle", {})
    if report.get("acceptance_report_sha256") != acceptance_report_sha256(report):
        return EvidenceDecision(False, "acceptance_report_hash", "acceptance report hash mismatch", {})
    return EvidenceDecision(True, "accepted", "acceptance report authorizes readiness", {})


def validate_current_checkout(
    bundle: dict[str, Any],
    repo_root: Path,
    protected_baseline: Path,
    *,
    require_accepted: bool = False,
    enforce_result_freshness: bool = True,
) -> EvidenceDecision:
    inputs = bundle.get("inputs")
    if not isinstance(inputs, dict):
        return EvidenceDecision(False, "admission_inputs", "admission bundle lacks canonical input paths", {})
    try:
        if file_sha256(protected_baseline) != bundle["bindings"]["protected_baseline_sha256"]:
            return EvidenceDecision(False, "protected_baseline_hash", "protected baseline hash mismatch", {})
        compare_protected_baseline(repo_root, protected_baseline)
        if subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip() != bundle["bindings"]["readiness_evidence_commit"]:
            return EvidenceDecision(False, "evidence_commit", "current HEAD does not match readiness evidence commit", {})
        if current_implementation_tree_sha256(repo_root) != bundle["bindings"]["implementation_tree_sha256"]:
            return EvidenceDecision(False, "implementation_tree", "current implementation tree does not match admission bundle", {})
        manifest = json.loads(subprocess.check_output([sys.executable, str(repo_root / "scripts/experiment_manifest.py"), "render", inputs["manifest"], "--format", "json"], text=True))
        calibration = load_object(Path(inputs["calibration_result"]))
        preflight = load_object(Path(inputs["preflight_result"]))
        profile_path = Path(inputs["resource_profile"])
        expected = {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "resource_profile_sha256": manifest_resource_profile_sha256(manifest),
            "calibration_result_sha256": file_sha256(Path(inputs["calibration_result"])),
            "preflight_result_sha256": file_sha256(Path(inputs["preflight_result"])),
        }
        for key, actual in expected.items():
            if bundle["bindings"].get(key) != actual:
                return EvidenceDecision(False, "input_hash", f"{key} does not match admission bundle", {})
        if rendered_resource_profile_sha256(profile_path) != expected["resource_profile_sha256"]:
            return EvidenceDecision(False, "resource_profile", "resource profile has changed", {})
        if enforce_result_freshness:
            enforce_freshness(calibration, int(manifest["calibration_policy"]["calibration_result_max_age_seconds"]), "calibration result")
            enforce_freshness(preflight, int(manifest["preflight"]["result_max_age_seconds"]), "preflight result")
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        return EvidenceDecision(False, "current_checkout", str(exc), {})
    if require_accepted:
        report_path = Path(bundle.get("acceptance_report_path", ""))
        report = load_object(report_path)
        if file_sha256(report_path) != bundle.get("acceptance_report_sha256"):
            return EvidenceDecision(False, "acceptance_report_file_hash", "acceptance report file hash mismatch", {})
        decision = validate_acceptance_report(report, bundle, plan_path=repo_root / "docs/joint_training/goals/stage123-execution-readiness/plan.md", candidate_commit=bundle["bindings"]["readiness_evidence_commit"])
        if not decision.authorized:
            return decision
    return EvidenceDecision(True, "authorized", "current checkout matches admission bundle", {})


def validate_batch_phase_admission(
    bundle: dict[str, Any],
    bundle_path: Path,
    record_path: Path,
    repo_root: Path,
    *,
    run_id: str,
    batch_id: str,
    batch_manifest_sha256: str,
    item_id: str,
    admission_bundle_sha256: str,
    command_sha256: str,
    record_sha256: str,
) -> EvidenceDecision:
    try:
        record = load_object(record_path)
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != result_sha256(unsigned) or record.get("record_sha256") != record_sha256:
            return EvidenceDecision(False, "batch_admission_record_hash", "batch admission record hash mismatch", {})
        expected = {
            "schema_version": 1,
            "status": "active",
            "batch_id": batch_id,
            "batch_manifest_sha256": batch_manifest_sha256,
            "item_id": item_id,
            "admission_bundle_sha256": admission_bundle_sha256,
            "command_sha256": command_sha256,
        }
        mismatched = [key for key, value in expected.items() if record.get(key) != value]
        if mismatched:
            return EvidenceDecision(False, "batch_admission_record_binding", "batch admission record binding mismatch", {"mismatched": mismatched})
        run_ids = record.get("expected_run_ids")
        if not isinstance(run_ids, list) or run_id not in run_ids:
            return EvidenceDecision(False, "batch_admission_run_id", "run id is not admitted for the active batch item", {"run_id": run_id})
        if file_sha256(bundle_path) != admission_bundle_sha256:
            return EvidenceDecision(False, "batch_admission_bundle_hash", "admission bundle file hash mismatch", {})
        state_path = Path(record["batch_state_path"])
        state = load_object(state_path)
        active = state.get("current_item_admission")
        if (
            state.get("status") != "running"
            or state.get("current_item_id") != item_id
            or not isinstance(active, dict)
            or active.get("path") != str(record_path)
            or active.get("sha256") != record_sha256
        ):
            return EvidenceDecision(False, "batch_admission_not_live", "batch item admission is not live", {})
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return EvidenceDecision(False, "batch_admission_record", str(exc), {})
    structural = validate_admission_bundle(bundle, require_accepted=True)
    if not structural.authorized:
        return structural
    inputs = bundle.get("inputs", {})
    baseline = inputs.get("protected_baseline")
    if not isinstance(baseline, str) or not baseline:
        return EvidenceDecision(False, "protected_baseline", "protected baseline is required", {})
    return validate_current_checkout(
        bundle,
        repo_root,
        Path(baseline),
        require_accepted=True,
        enforce_result_freshness=False,
    )


def validate_admission_bundle(bundle: dict[str, Any], *, require_accepted: bool = False) -> EvidenceDecision:
    if bundle.get("schema_version") != 1 or bundle.get("bundle_type") != "stage123_admission_bundle":
        return EvidenceDecision(False, "admission_schema", "unsupported admission bundle schema", {})
    if bundle.get("run_ids") != ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]:
        return EvidenceDecision(False, "admission_run_set", "admission bundle run set is not the matched primary matrix", {"run_ids": bundle.get("run_ids")})
    bindings = bundle.get("bindings")
    if not isinstance(bindings, dict):
        return EvidenceDecision(False, "admission_bindings", "admission bundle lacks bindings", {})
    required = (
        "manifest_sha256",
        "resource_profile_sha256",
        "implementation_tree_sha256",
        "calibration_result_sha256",
        "preflight_result_sha256",
        "readiness_evidence_commit",
    )
    missing = [name for name in required if not isinstance(bindings.get(name), str) or not bindings[name]]
    if missing:
        return EvidenceDecision(False, "admission_bindings", "admission bundle has incomplete bindings", {"missing": missing})
    if require_accepted:
        acceptance = bundle.get("acceptance")
        if not isinstance(acceptance, dict) or acceptance.get("decision") != "accepted":
            return EvidenceDecision(False, "admission_not_accepted", "admission bundle lacks independent acceptance", {})
        if acceptance.get("bundle_sha256") != bundle.get("bundle_sha256"):
            return EvidenceDecision(False, "acceptance_binding", "acceptance does not bind the bundle hash", {})
        required_acceptance = (
            "manifest_sha256",
            "resource_profile_sha256",
            "implementation_tree_sha256",
            "calibration_result_sha256",
            "preflight_result_sha256",
            "readiness_evidence_commit",
            "run_ids",
        )
        if bindings.get("protected_baseline_sha256"):
            required_acceptance += ("protected_baseline_sha256",)
        missing_acceptance = [name for name in required_acceptance if name not in acceptance]
        if missing_acceptance:
            return EvidenceDecision(False, "acceptance_binding", "acceptance report has incomplete bindings", {"missing": missing_acceptance})
        expected_acceptance = {
            "manifest_sha256": bindings["manifest_sha256"],
            "resource_profile_sha256": bindings["resource_profile_sha256"],
            "implementation_tree_sha256": bindings["implementation_tree_sha256"],
            "calibration_result_sha256": bindings["calibration_result_sha256"],
            "preflight_result_sha256": bindings["preflight_result_sha256"],
            "readiness_evidence_commit": bindings["readiness_evidence_commit"],
            "run_ids": bundle["run_ids"],
        }
        mismatched = [name for name, expected in expected_acceptance.items() if acceptance.get(name) != expected]
        if mismatched:
            return EvidenceDecision(False, "acceptance_binding", "acceptance report bindings do not match bundle", {"mismatched": mismatched})
    return EvidenceDecision(True, "authorized", "admission bundle authorizes the primary queue", {"run_ids": bundle["run_ids"]})


def build_admission_bundle(
    manifest_path: Path,
    profile_path: Path,
    calibration_path: Path,
    preflight_path: Path,
    evidence_commit: str,
    output: Path,
    acceptance_report_path: Path | None = None,
    repo_root: Path | None = None,
    protected_baseline_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest_object(manifest_path, repo_root)
    resource_profile_sha256 = manifest_resource_profile_sha256(manifest)
    calibration = load_object(calibration_path)
    preflight = load_object(preflight_path)
    run_ids = [item.get("id") for item in manifest.get("runs", [])]
    expected_tree = preflight.get("implementation_tree_sha256")
    expected_calibration_commit = preflight.get("calibration_evidence_commit")
    expected_authorization = preflight.get("calibration_authorization_identity")
    if not expected_tree or not expected_calibration_commit or not isinstance(expected_authorization, dict):
        raise ValueError("preflight result lacks calibration expected bindings")
    actual_profile_sha256 = rendered_resource_profile_sha256(profile_path)
    if actual_profile_sha256 != resource_profile_sha256:
        raise ValueError("resource profile file does not match manifest identity")
    if repo_root is not None and current_implementation_tree_sha256(repo_root) != expected_tree:
        raise ValueError("current implementation tree does not match preflight identity")
    calibration_decision = validate_result(
        calibration,
        "calibration_result",
        {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "resource_profile_sha256": resource_profile_sha256,
            "implementation_tree_sha256": expected_tree,
            "evidence_commit": expected_calibration_commit,
            "run_ids": run_ids,
            "authorization_identity": expected_authorization,
        },
    )
    preflight_decision = validate_result(preflight, "preflight_result")
    if not calibration_decision.authorized or not preflight_decision.authorized:
        raise ValueError("calibration and preflight results must both authorize")
    manifest_sha = manifest.get("manifest_sha256")
    if calibration.get("manifest_sha256") != manifest_sha or preflight.get("manifest_sha256") != manifest_sha:
        raise ValueError("result manifest bindings do not match")
    if preflight.get("resource_profile_sha256") != resource_profile_sha256:
        raise ValueError("preflight resource profile binding does not match")
    if preflight.get("implementation_tree_sha256") != calibration.get("implementation_tree_sha256"):
        raise ValueError("preflight implementation tree binding does not match")
    if preflight.get("run_ids") != run_ids:
        raise ValueError("preflight run set does not match manifest")
    enforce_freshness(calibration, int(manifest["calibration_policy"]["calibration_result_max_age_seconds"]), "calibration result")
    enforce_freshness(preflight, int(manifest["preflight"]["result_max_age_seconds"]), "preflight result")
    value = {
        "schema_version": 1,
        "bundle_type": "stage123_admission_bundle",
        "adapter_type": "stage123_queue_v1",
        "bundle_path": str(output),
        "run_ids": run_ids,
        "bindings": {
            "manifest_sha256": manifest_sha,
            "resource_profile_sha256": resource_profile_sha256,
            "implementation_tree_sha256": calibration.get("implementation_tree_sha256"),
            "calibration_result_sha256": file_sha256(calibration_path),
            "preflight_result_sha256": file_sha256(preflight_path),
            "readiness_evidence_commit": evidence_commit,
            "recipe_gitlink": subprocess.check_output(["git", "-C", str((repo_root or ROOT) / "recipe"), "rev-parse", "HEAD"], text=True).strip(),
            "protected_baseline_sha256": file_sha256(protected_baseline_path) if protected_baseline_path else "",
            "calibration_completed_at": calibration.get("completed_at"),
            "preflight_completed_at": preflight.get("completed_at"),
        },
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "resource_profile": str(profile_path.resolve()),
            "calibration_result": str(calibration_path.resolve()),
            "preflight_result": str(preflight_path.resolve()),
            "protected_baseline": str(protected_baseline_path.resolve()) if protected_baseline_path else "",
        },
    }
    value["bundle_sha256"] = result_sha256(value)
    if acceptance_report_path is not None:
        acceptance = load_object(acceptance_report_path)
        if acceptance.get("result_type") != "acceptance_report" or acceptance.get("decision") != "accepted":
            raise ValueError("acceptance report is not accepted")
        expected_acceptance = {
            "manifest_sha256": manifest_sha,
            "resource_profile_sha256": resource_profile_sha256,
            "implementation_tree_sha256": calibration.get("implementation_tree_sha256"),
            "calibration_result_sha256": file_sha256(calibration_path),
            "preflight_result_sha256": file_sha256(preflight_path),
            "readiness_evidence_commit": evidence_commit,
            "run_ids": run_ids,
            "bundle_sha256": value["bundle_sha256"],
        }
        if protected_baseline_path is not None:
            expected_acceptance["protected_baseline_sha256"] = file_sha256(protected_baseline_path)
        if any(acceptance.get(key) != value for key, value in expected_acceptance.items()):
            raise ValueError("acceptance report bindings do not match admission inputs")
        value["acceptance"] = acceptance
        value["acceptance_report_sha256"] = file_sha256(acceptance_report_path)
        value["acceptance_report_path"] = str(acceptance_report_path.resolve())
    decision = validate_admission_bundle(value)
    if not decision.authorized:
        raise ValueError(decision.message)
    return value


def admission_launch_command(bundle: dict[str, Any], repo_host: Path) -> list[str]:
    bindings = bundle["bindings"]
    bundle_path = bundle.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        raise ValueError("admission bundle lacks canonical bundle_path")
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "stage123_primary_chain",
        "env",
        "REPO_HOST=" + str(repo_host),
        "ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1",
        "STAGE123_ADMISSION_BUNDLE=" + bundle_path,
        "STAGE123_IMPLEMENTATION_TREE_SHA256=" + bindings["implementation_tree_sha256"],
        "STAGE123_BUNDLE_SHA256=" + bundle["bundle_sha256"],
        "EXPERIMENT_BATCH_MANIFEST=" + str(repo_host / "docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json"),
        "bash",
        str(repo_host / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh"),
    ]


def admission_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--bundle", type=Path)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--resource-profile", type=Path)
    validate.add_argument("--calibration-result", type=Path)
    validate.add_argument("--preflight-result", type=Path)
    validate.add_argument("--readiness-evidence-commit")
    validate.add_argument("--output", type=Path)
    validate.add_argument("--require-accepted", action="store_true")
    validate.add_argument("--acceptance-report", type=Path)
    validate.add_argument("--protected-baseline", type=Path)
    validate.add_argument("--repo-root", type=Path, required=True)
    validate_phase = sub.add_parser("validate-phase")
    validate_phase.add_argument("--bundle", type=Path, required=True)
    validate_phase.add_argument("--record", type=Path, required=True)
    validate_phase.add_argument("--record-sha256", required=True)
    validate_phase.add_argument("--run-id", required=True)
    validate_phase.add_argument("--batch-id", required=True)
    validate_phase.add_argument("--batch-manifest-sha256", required=True)
    validate_phase.add_argument("--item-id", required=True)
    validate_phase.add_argument("--admission-bundle-sha256", required=True)
    validate_phase.add_argument("--command-sha256", required=True)
    validate_phase.add_argument("--repo-root", type=Path, required=True)
    render = sub.add_parser("render-launch")
    render.add_argument("--bundle", type=Path, required=True)
    render.add_argument("--repo-host", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "validate" and args.bundle is None:
            required = (args.manifest, args.resource_profile, args.calibration_result, args.preflight_result, args.protected_baseline, args.readiness_evidence_commit, args.output)
            if not all(required):
                raise ValueError("candidate validation requires manifest, resource profile, calibration result, preflight result, protected baseline, evidence commit, and output")
            bundle = build_admission_bundle(args.manifest, args.resource_profile, args.calibration_result, args.preflight_result, args.readiness_evidence_commit, args.output, args.acceptance_report, args.repo_root, args.protected_baseline)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        else:
            bundle = load_object(args.bundle)
        expected_hash = bundle.get("bundle_sha256")
        unsigned = {key: value for key, value in bundle.items() if key not in {"bundle_sha256", "acceptance", "acceptance_report_sha256", "acceptance_report_path"}}
        actual_hash = result_sha256(unsigned)
        if expected_hash != actual_hash:
            decision = EvidenceDecision(False, "bundle_hash", "admission bundle hash mismatch", {"expected": expected_hash, "actual": actual_hash})
        elif args.action == "validate-phase":
            if expected_hash != actual_hash:
                decision = EvidenceDecision(False, "bundle_hash", "admission bundle hash mismatch", {"expected": expected_hash, "actual": actual_hash})
            else:
                decision = validate_batch_phase_admission(
                    bundle,
                    args.bundle,
                    args.record,
                    args.repo_root,
                    run_id=args.run_id,
                    batch_id=args.batch_id,
                    batch_manifest_sha256=args.batch_manifest_sha256,
                    item_id=args.item_id,
                    admission_bundle_sha256=args.admission_bundle_sha256,
                    command_sha256=args.command_sha256,
                    record_sha256=args.record_sha256,
                )
        else:
            require_accepted = args.action == "render-launch" or getattr(args, "require_accepted", False)
            decision = validate_admission_bundle(bundle, require_accepted=require_accepted)
            if decision.authorized:
                validation_root = getattr(args, "repo_root", None) if args.action == "validate" else args.repo_host
                if not bundle.get("inputs"):
                    decision = EvidenceDecision(False, "admission_inputs", "admission bundle lacks canonical input paths", {})
                else:
                    baseline_value = getattr(args, "protected_baseline", None) or bundle.get("inputs", {}).get("protected_baseline", "")
                    if not baseline_value:
                        decision = EvidenceDecision(False, "protected_baseline", "protected baseline is required", {})
                    else:
                        decision = validate_current_checkout(bundle, validation_root, Path(baseline_value), require_accepted=require_accepted)
        if args.action in {"validate", "validate-phase"}:
            print(json.dumps(decision.as_dict(), sort_keys=True))
            return 0 if decision.authorized else 1
        if not decision.authorized:
            print(json.dumps(decision.as_dict(), sort_keys=True))
            return 1
        print(shlex.join(admission_launch_command(bundle, args.repo_host)))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps(EvidenceDecision(False, "invalid_admission_bundle", str(exc), {}).as_dict(), sort_keys=True))
        return 2


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "admission":
        return admission_main(sys.argv[2:])
    raise SystemExit("usage: execution_results.py admission {validate,validate-phase,render-launch} ...")


if __name__ == "__main__":
    raise SystemExit(main())
