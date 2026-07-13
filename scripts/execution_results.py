#!/usr/bin/env python3
"""Authoritative result classes and fail-closed legacy evidence migration."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_admission_bundle(bundle: dict[str, Any], *, require_accepted: bool = False) -> EvidenceDecision:
    if bundle.get("schema_version") != 1 or bundle.get("bundle_type") != "stage123_admission_bundle":
        return EvidenceDecision(False, "admission_schema", "unsupported admission bundle schema", {})
    if bundle.get("run_ids") != ["frac25-stage2", "frac25-stage3"]:
        return EvidenceDecision(False, "admission_run_set", "admission bundle run set is not the primary pair", {"run_ids": bundle.get("run_ids")})
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
    return EvidenceDecision(True, "authorized", "admission bundle authorizes the primary queue", {"run_ids": bundle["run_ids"]})


def build_admission_bundle(manifest_path: Path, profile_path: Path, calibration_path: Path, preflight_path: Path, evidence_commit: str, output: Path) -> dict[str, Any]:
    manifest = load_object(manifest_path)
    calibration = load_object(calibration_path)
    preflight = load_object(preflight_path)
    run_ids = [item.get("id") for item in manifest.get("runs", [])]
    expected_tree = preflight.get("implementation_tree_sha256")
    expected_calibration_commit = preflight.get("calibration_evidence_commit")
    expected_authorization = preflight.get("calibration_authorization_identity")
    if not expected_tree or not expected_calibration_commit or not isinstance(expected_authorization, dict):
        raise ValueError("preflight result lacks calibration expected bindings")
    calibration_decision = validate_result(
        calibration,
        "calibration_result",
        {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "resource_profile_sha256": file_sha256(profile_path),
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
    value = {
        "schema_version": 1,
        "bundle_type": "stage123_admission_bundle",
        "bundle_path": str(output),
        "run_ids": run_ids,
        "bindings": {
            "manifest_sha256": manifest_sha,
            "resource_profile_sha256": file_sha256(profile_path),
            "implementation_tree_sha256": calibration.get("implementation_tree_sha256"),
            "calibration_result_sha256": file_sha256(calibration_path),
            "preflight_result_sha256": file_sha256(preflight_path),
            "readiness_evidence_commit": evidence_commit,
        },
    }
    decision = validate_admission_bundle(value)
    if not decision.authorized:
        raise ValueError(decision.message)
    value["bundle_sha256"] = result_sha256(value)
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
    validate.add_argument("--repo-root", type=Path, required=True)
    render = sub.add_parser("render-launch")
    render.add_argument("--bundle", type=Path, required=True)
    render.add_argument("--repo-host", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "validate" and args.bundle is None:
            required = (args.manifest, args.resource_profile, args.calibration_result, args.preflight_result, args.readiness_evidence_commit, args.output)
            if not all(required):
                raise ValueError("candidate validation requires manifest, resource profile, calibration result, preflight result, evidence commit, and output")
            bundle = build_admission_bundle(args.manifest, args.resource_profile, args.calibration_result, args.preflight_result, args.readiness_evidence_commit, args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        else:
            bundle = load_object(args.bundle)
        expected_hash = bundle.get("bundle_sha256")
        unsigned = {key: value for key, value in bundle.items() if key not in {"bundle_sha256", "acceptance"}}
        actual_hash = result_sha256(unsigned)
        if expected_hash != actual_hash:
            decision = EvidenceDecision(False, "bundle_hash", "admission bundle hash mismatch", {"expected": expected_hash, "actual": actual_hash})
        else:
            decision = validate_admission_bundle(bundle, require_accepted=args.action == "render-launch" or getattr(args, "require_accepted", False))
        if args.action == "validate":
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
    raise SystemExit("usage: execution_results.py admission {validate,render-launch} ...")


if __name__ == "__main__":
    raise SystemExit(main())
