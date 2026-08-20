#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

REQUIRED = (
    "decision",
    "manifest_sha256",
    "resource_profile_sha256",
    "implementation_tree_sha256",
    "evidence_commit",
    "workload_identity",
    "policy_id",
    "policy_sha256",
    "authorization_identity",
    "started_at",
    "completed_at",
    "phase_evidence",
    "prediction_comparison",
    "cleanup",
    "failures",
)
ALLOWED_PHASE_SETS = (["stage1", "stage2", "stage3"], ["stage2", "stage3"])


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(ch in "0123456789abcdef" for ch in value)


def _time(value: Any):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if value.get("schema_version") != 1 or value.get("result_type") != "calibration_result":
        failures.append({"code": "result_schema", "message": "unsupported calibration result", "context": {}})
    missing = [key for key in REQUIRED if key not in value]
    if missing:
        failures.append(
            {"code": "result_fields", "message": "calibration result is incomplete", "context": {"missing": missing}}
        )
    if value.get("decision") not in schema["decisions"]:
        failures.append(
            {
                "code": "result_decision",
                "message": "invalid calibration decision",
                "context": {"decision": value.get("decision")},
            }
        )
    if value.get("decision") == "passed":
        for key in ("manifest_sha256", "resource_profile_sha256", "implementation_tree_sha256", "policy_sha256"):
            if not _hex(value.get(key), 64):
                failures.append(
                    {"code": "identity_hash", "message": "invalid identity hash", "context": {"field": key}}
                )
        if not _hex(value.get("evidence_commit"), 40):
            failures.append({"code": "evidence_commit", "message": "invalid evidence commit", "context": {}})
        policy_path = Path(__file__).resolve().parents[1] / "config/experiment_execution/calibration_policy_v1.json"
        policy = json.loads(policy_path.read_text())
        expected_policy_sha = policy_sha256(policy_path)
        if value.get("policy_id") != policy.get("policy_id") or value.get("policy_sha256") != expected_policy_sha:
            failures.append({"code": "policy_binding", "message": "calibration policy binding mismatch", "context": {}})
        if not isinstance(value.get("workload_identity"), dict) or not value["workload_identity"]:
            failures.append({"code": "workload_identity", "message": "workload identity is incomplete", "context": {}})
        phases = value.get("phase_evidence")
        phase_names = (
            [item.get("phase") for item in phases if isinstance(item, dict)] if isinstance(phases, list) else []
        )
        if phase_names not in ALLOWED_PHASE_SETS or any(
            item.get("status") != "passed" for item in phases if isinstance(item, dict)
        ):
            failures.append(
                {"code": "phase_evidence", "message": "phase evidence is incomplete or blocked", "context": {}}
            )
        prediction = value.get("prediction_comparison")
        cleanup = value.get("cleanup")
        comparisons = prediction.get("comparisons") if isinstance(prediction, dict) else None
        metrics = (
            {item.get("metric") for item in comparisons if isinstance(item, dict)}
            if isinstance(comparisons, list)
            else set()
        )
        required_metrics = {"validation_elapsed_seconds", "phase_elapsed_seconds", "peak_rss_gib", "gpu_wait_fraction"}

        def comparison_valid(item):
            history = item.get("history", [])
            predicted = item.get("predicted_bound")
            observed = item.get("observed_maximum")
            decision = item.get("decision", {})
            if (
                item.get("history_count") != len(history)
                or item.get("history_count", 0) < policy["prediction"]["minimum_history_count"]
                or not all(isinstance(value, int | float) for value in history)
                or not isinstance(predicted, int | float)
                or not isinstance(observed, int | float)
                or predicted <= 0
            ):
                return False
            ratio = observed / predicted
            return (
                predicted == max(history)
                and ratio <= policy["prediction"]["maximum_observed_to_predicted_ratio"]
                and math.isclose(decision.get("context", {}).get("ratio", -1), ratio, rel_tol=1e-12)
                and decision.get("qualified") is True
                and decision.get("code") == "qualified"
            )

        complete_prediction = (
            isinstance(prediction, dict)
            and prediction.get("qualified") is True
            and prediction.get("policy_id") == policy.get("policy_id")
            and prediction.get("policy_sha256") == expected_policy_sha
            and metrics == required_metrics
            and all(comparison_valid(item) for item in comparisons)
        )
        if not complete_prediction:
            failures.append(
                {
                    "code": "prediction_qualification",
                    "message": "prediction comparison is incomplete or not qualified",
                    "context": {},
                }
            )
        if not isinstance(cleanup, dict) or cleanup.get("resources_released") is not True:
            failures.append({"code": "cleanup", "message": "owned resources were not released", "context": {}})
        started, completed = _time(value.get("started_at")), _time(value.get("completed_at"))
        if started is None or completed is None or completed < started:
            failures.append({"code": "timestamps", "message": "calibration timestamps are invalid", "context": {}})
        if value.get("failures") != []:
            failures.append(
                {"code": "passed_with_failures", "message": "passed result cannot contain failures", "context": {}}
            )
    return {"ok": not failures, "decision": value.get("decision") if not failures else "blocked", "failures": failures}


def policy_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
