#!/usr/bin/env python3
"""Structured validator for experiment execution results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationFailure:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    failures: list[ValidationFailure] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def decision(self) -> str:
        return "passed" if self.ok else "blocked"

    def add(self, code: str, message: str, **context: Any) -> None:
        self.failures.append(ValidationFailure(code, message, context))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "decision": self.decision,
            "failures": [asdict(item) for item in self.failures],
            "diagnostics": self.diagnostics,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return load_json(path)
    output = subprocess.check_output(
        ["python3", str(Path(__file__).with_name("experiment_manifest.py")), "render", str(path), "--format", "json"],
        text=True,
    )
    return json.loads(output)


def expected_phases(manifest: dict[str, Any], authorization_scope: str) -> list[str]:
    phases = [item["phase"] for item in manifest["runs"]]
    if authorization_scope == "full":
        return phases
    if authorization_scope == "stage12_producer":
        return ["stage1", "stage2"]
    if authorization_scope == "treatment_only":
        return ["stage2", "stage3"]
    raise ValueError(f"unsupported authorization scope: {authorization_scope}")


def validate_repetition(result: ValidationResult, phase: str, index: int, repetition: dict[str, Any]) -> None:
    context = {"phase": phase, "repetition": index}
    if repetition.get("timed_out"):
        result.add("repetition_timeout", "calibration repetition timed out", **context)
    if repetition.get("status") not in {"passed", "succeeded"}:
        result.add(
            "repetition_status", "calibration repetition did not pass", status=repetition.get("status"), **context
        )
    metrics = repetition.get("metrics")
    if not isinstance(metrics, dict) or "validation_elapsed_seconds" not in metrics:
        result.add("metrics_incomplete", "validation metrics are incomplete", **context)
    resources = repetition.get("resources")
    if not isinstance(resources, dict):
        result.add("resources_missing", "resource evidence is missing", **context)
    cleanup = repetition.get("cleanup")
    if not isinstance(cleanup, dict) or not cleanup.get("resources_released", False):
        result.add("cleanup_failed", "runtime resources were not released", **context)
    if repetition.get("truncated_count", 0):
        result.add(
            "response_truncation",
            "calibration responses were truncated",
            count=repetition.get("truncated_count"),
            **context,
        )
    if repetition.get("score_complete") is not True:
        result.add("score_incomplete", "calibration score evidence is incomplete", **context)


def check(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    contract: dict[str, Any] | None = None,
    history_index: dict[str, Any] | None = None,
    hashes: dict[str, str] | None = None,
    authorization_scope: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    result = ValidationResult()
    hashes = hashes or {}
    scope = authorization_scope or report.get("authorization_scope", "full")
    try:
        phases = expected_phases(manifest, scope)
    except ValueError as exc:
        result.add("authorization_scope", str(exc), authorization_scope=scope)
        return result.as_dict()
    if report.get("authorization_scope") != scope:
        result.add(
            "authorization_scope",
            "report authorization scope mismatch",
            expected=scope,
            actual=report.get("authorization_scope"),
        )
    if report.get("evidence_class") != "infrastructure_calibration":
        result.add("evidence_class", "wrong evidence class", actual=report.get("evidence_class"))
    if report.get("decision") != "candidate":
        result.add("candidate_decision", "assembler report decision must be candidate", actual=report.get("decision"))
    if report.get("manifest_sha256") != manifest.get("manifest_sha256"):
        result.add(
            "manifest_hash",
            "manifest hash mismatch",
            expected=manifest.get("manifest_sha256"),
            actual=report.get("manifest_sha256"),
        )
    bindings = report.get("input_bindings", {})
    for name, expected_hash in hashes.items():
        actual = bindings.get(name, {}).get("sha256")
        if actual not in (None, expected_hash):
            result.add("input_hash", "input binding hash mismatch", input=name, expected=expected_hash, actual=actual)
    phase_reports = report.get("phases", [])
    actual_phases = [item.get("phase") for item in phase_reports if isinstance(item, dict)]
    if actual_phases != phases:
        result.add("phase_order", "phase order mismatch", expected=phases, actual=actual_phases)
    profile_hash = manifest.get("resource_profile", {}).get("sha256")
    for phase_report in phase_reports:
        phase = phase_report.get("phase")
        if phase_report.get("profile_hash") != profile_hash:
            result.add(
                "profile_hash",
                "resource profile hash mismatch",
                phase=phase,
                expected=profile_hash,
                actual=phase_report.get("profile_hash"),
            )
        repetitions = phase_report.get("repetitions", [])
        if not repetitions:
            result.add("repetitions_missing", "phase has no calibration repetitions", phase=phase)
        for index, repetition in enumerate(repetitions):
            validate_repetition(result, phase, repetition.get("repetition", index + 1), repetition)
    expected_deadline = manifest.get("calibration_policy", {}).get("validation_deadline_seconds")
    actual_deadline = report.get("contract", {}).get("validation_deadline_seconds")
    if actual_deadline != expected_deadline:
        result.add(
            "validation_deadline",
            "validation deadline contract mismatch",
            expected=expected_deadline,
            actual=actual_deadline,
        )
    if contract is not None and contract.get("authorization_scope") not in (None, scope):
        result.add(
            "prediction_scope",
            "prediction contract authorization scope mismatch",
            expected=scope,
            actual=contract.get("authorization_scope"),
        )
    if history_index is not None:
        result.diagnostics["history_run_count"] = len(history_index.get("runs", []))
    return result.as_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--history-index", type=Path)
    parser.add_argument("--authorization-scope", default="full")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--semantic-contract", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        report = load_json(args.report)
        manifest = load_manifest(args.manifest)
        contract = load_json(args.contract) if args.contract else None
        history = load_json(args.history_index) if args.history_index else None
        output = check(
            report, manifest, contract=contract, history_index=history, authorization_scope=args.authorization_scope
        )
        if args.receipt:
            output["compatibility_warning"] = "deployability receipts are no longer issued; use acceptance_report.json"
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        output = ValidationResult([ValidationFailure("invalid_input", str(exc), {})]).as_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
