#!/usr/bin/env python3
"""Authoritative result classes and fail-closed legacy evidence migration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
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


def validate_result(value: dict[str, Any], expected_type: str | None = None) -> EvidenceDecision:
    result_type = value.get("result_type")
    if result_type in LEGACY_AUTHORITY_TYPES or value.get("receipt_type") in LEGACY_AUTHORITY_TYPES:
        return EvidenceDecision(False, "legacy_evidence", "legacy receipt or adoption evidence is not current authority", {"result_type": result_type, "receipt_type": value.get("receipt_type")})
    if result_type not in RESULT_TYPES:
        return EvidenceDecision(False, "unsupported_result_type", "unsupported execution result type", {"result_type": result_type})
    if expected_type is not None and result_type != expected_type:
        return EvidenceDecision(False, "result_type_mismatch", "execution result type mismatch", {"expected": expected_type, "actual": result_type})
    if value.get("schema_version") != 1:
        return EvidenceDecision(False, "schema_version", "unsupported execution result schema", {"schema_version": value.get("schema_version")})
    if not isinstance(value.get("manifest_sha256"), str) or len(value["manifest_sha256"]) != 64:
        return EvidenceDecision(False, "manifest_binding", "execution result lacks manifest binding", {})
    decision = value.get("decision")
    if decision not in {"passed", "blocked", "accepted", "rejected"}:
        return EvidenceDecision(False, "decision", "execution result has invalid decision", {"decision": decision})
    authorized = (result_type != "acceptance_report" and decision == "passed") or (result_type == "acceptance_report" and decision == "accepted")
    code = "authorized" if authorized else "result_blocked"
    message = "execution result authorizes this stage" if authorized else "execution result does not authorize this stage"
    return EvidenceDecision(authorized, code, message, {"result_type": result_type, "decision": decision})


def load_and_validate(path: Path, expected_type: str | None = None) -> EvidenceDecision:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return EvidenceDecision(False, "invalid_result_file", "execution result cannot be read", {"path": str(path), "error": str(exc)})
    if not isinstance(value, dict):
        return EvidenceDecision(False, "invalid_result_file", "execution result must be a JSON object", {"path": str(path)})
    return validate_result(value, expected_type)


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
