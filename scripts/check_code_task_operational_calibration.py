#!/usr/bin/env python3
"""Validate real local L40S calibration and issue the only deployability receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_calibration_prediction_contract import (
    canonical_json_bytes,
    verify_prediction_contract,
)


PHASES = ("stage1", "stage2", "stage3")
METRICS = (
    "validation_elapsed_seconds",
    "peak_rss_gib",
    "all_gpu_idle_fraction_during_validation",
)
POINT_ERROR_METRICS = ("validation_elapsed_seconds", "peak_rss_gib")
VALIDATION_DATASETS = {
    "HumanEval+": {"rows": 164, "sha256": "e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab"},
    "MBPP+": {"rows": 378, "sha256": "3221e7f53c88bfbd91d788fb7bcb37168fb088fa504fddf12b9126c2147312d2"},
    "LiveCodeBench": {"rows": 880, "sha256": "fe7d2bfe2779bcf106492347ca173e30b9220c15c1b8783949d35edcd93a43d1"},
}
RECEIPT_TTL_SECONDS = 86400
ROUND_QUANT = Decimal("0.000001")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def load_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return load_json(path)
    raw = subprocess.check_output(
        ["python3", str(Path(__file__).with_name("experiment_manifest.py")), "render", str(path), "--format", "json"],
        text=True,
    )
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: rendered manifest must be an object")
    return data


def _floor6(value: float) -> float:
    return float(Decimal(str(value)).quantize(ROUND_QUANT, rounding=ROUND_FLOOR))


def _ceil6(value: float) -> float:
    return float(Decimal(str(value)).quantize(ROUND_QUANT, rounding=ROUND_CEILING))


def measured_idle_interval(values: list[float]) -> list[float]:
    return [_floor6(max(0.0, min(values) - 0.02)), _ceil6(min(1.0, max(values) + 0.02))]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def metric_value(rep: dict[str, Any], metric: str) -> float | None:
    if metric == "validation_elapsed_seconds":
        value = rep.get("metrics", {}).get(metric)
    elif metric == "peak_rss_gib":
        value = rep.get("resources", {}).get(metric)
    else:
        value = rep.get("resources", {}).get("gpu_wait_fraction")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def phase_contract(contract: dict[str, Any] | None, phase: str) -> dict[str, Any] | None:
    if not contract:
        return None
    for item in contract.get("phases", []):
        if item.get("phase") == phase:
            return item
    return None


def interval_contains(interval: Any, value: float) -> bool:
    return isinstance(interval, list) and len(interval) == 2 and interval[0] <= value <= interval[1]


def intervals_overlap(left: list[float], right: list[float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def binding(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get("input_bindings", {}).get(key)
    return value if isinstance(value, dict) else {}


def check_file_binding(report: dict[str, Any], key: str, path: Path | None, failures: list[str], *, optional: bool = False) -> str | None:
    bound = binding(report, key)
    if path is None:
        if not optional and bound:
            return bound.get("sha256")
        return bound.get("sha256")
    observed = file_sha256(path)
    if bound and bound.get("sha256") != observed:
        failures.append(f"{key} hash mismatch")
    if bound and bound.get("path") and bound.get("path") != str(path):
        failures.append(f"{key} path mismatch")
    return observed


def validate_repetition(rep: dict[str, Any], name: str, index: int, failures: list[str]) -> None:
    status = rep.get("status", {})
    metrics = rep.get("metrics", {})
    resources = rep.get("resources", {})
    if status.get("timed_out") is not False:
        failures.append(f"{name}: repetition {index} timed out")
    if status.get("returncode") != 0:
        failures.append(f"{name}: repetition {index} did not pass")
    if metrics.get("complete_validation_metrics") is not True:
        failures.append(f"{name}: repetition {index} incomplete metrics")
    if metrics.get("validation_elapsed_seconds", 10**9) > 1800:
        failures.append(f"{name}: repetition {index} validation deadline exceeded")
    if resources.get("peak_rss_gib") is None or resources.get("gpu_wait_fraction") is None:
        failures.append(f"{name}: repetition {index} missing resources")
    if resources.get("measurement_started") is not True:
        failures.append(f"{name}: repetition {index} validation measurement never started")
    if resources.get("measurement_window") != "validation_rollout_readiness_to_completion":
        failures.append(f"{name}: repetition {index} invalid resource measurement window")
    interval = resources.get("gpu_sample_interval_seconds")
    if not isinstance(interval, (int, float)) or interval <= 0 or interval > 0.25:
        failures.append(f"{name}: repetition {index} GPU sampling interval is too coarse")
    sample_count = resources.get("gpu_sample_count")
    elapsed = metrics.get("validation_elapsed_seconds")
    if isinstance(interval, (int, float)) and interval > 0 and isinstance(sample_count, int) and isinstance(elapsed, (int, float)):
        expected = elapsed / interval
        if sample_count < expected * 0.75:
            failures.append(f"{name}: repetition {index} GPU sampling coverage is incomplete")
    scorer = rep.get("scorer", {})
    timeline = rep.get("timeline", {})
    if timeline.get("status") != "unavailable_pre_timeline_instrumentation":
        rollout_elapsed = timeline.get("rollout_elapsed_seconds")
        post_elapsed = timeline.get("post_generation_elapsed_seconds")
        total_elapsed = timeline.get("timeline_elapsed_seconds")
        if not all(isinstance(value, (int, float)) and value >= 0 for value in (rollout_elapsed, post_elapsed, total_elapsed)):
            failures.append(f"{name}: repetition {index} invalid validation timeline")
        elif abs(rollout_elapsed + post_elapsed - total_elapsed) > 1e-6:
            failures.append(f"{name}: repetition {index} validation timeline accounting mismatch")
    required_scorer = {
        "sample_count",
        "scorer_latency_seconds_p50",
        "scorer_latency_seconds_p95",
        "timeout_count",
        "timeout_rate",
        "invalid_score_count",
        "invalid_score_rate",
        "valid_score_count",
        "valid_score_rate",
        "valid_scores_per_minute",
        "score_distribution",
        "status_distribution",
    }
    if not required_scorer.issubset(scorer):
        failures.append(f"{name}: repetition {index} incomplete scorer evidence")
    scorer_sample_count = scorer.get("sample_count")
    if not isinstance(scorer_sample_count, int) or scorer_sample_count <= 0:
        failures.append(f"{name}: repetition {index} scorer sample count is invalid")
    if isinstance(scorer_sample_count, int) and scorer.get("valid_score_count", -1) + scorer.get("invalid_score_count", -1) != scorer_sample_count:
        failures.append(f"{name}: repetition {index} scorer sample accounting mismatch")
    if scorer.get("timeout_rate", 1) > 0.1:
        failures.append(f"{name}: repetition {index} scorer timeout rate exceeds 10%")
    if scorer.get("invalid_score_rate", 1) > 0.1:
        failures.append(f"{name}: repetition {index} invalid score rate exceeds 10%")
    valid_rate = scorer.get("valid_score_rate")
    invalid_rate = scorer.get("invalid_score_rate")
    if not isinstance(valid_rate, (int, float)) or not isinstance(invalid_rate, (int, float)) or abs(valid_rate + invalid_rate - 1) > 1e-9:
        failures.append(f"{name}: repetition {index} scorer rates are inconsistent")
    p50 = scorer.get("scorer_latency_seconds_p50")
    p95 = scorer.get("scorer_latency_seconds_p95")
    if not isinstance(p50, (int, float)) or not isinstance(p95, (int, float)) or p50 < 0 or p95 < p50:
        failures.append(f"{name}: repetition {index} invalid scorer latency percentiles")
    if not isinstance(scorer.get("valid_scores_per_minute"), (int, float)) or scorer.get("valid_scores_per_minute", 0) <= 0:
        failures.append(f"{name}: repetition {index} invalid scorer throughput")
    if not isinstance(scorer.get("score_distribution"), dict) or sum(scorer.get("score_distribution", {}).values()) != scorer.get("valid_score_count"):
        failures.append(f"{name}: repetition {index} score distribution mismatch")
    if not isinstance(scorer.get("status_distribution"), dict) or sum(scorer.get("status_distribution", {}).values()) != scorer.get("sample_count"):
        failures.append(f"{name}: repetition {index} status distribution mismatch")
    artifact_names = set(rep.get("artifacts", {}))
    if artifact_names not in ({"status", "resources", "metrics", "generation"}, {"status", "resources", "metrics", "generation", "timeline"}):
        failures.append(f"{name}: repetition {index} artifact set incomplete")
    if timeline.get("status") == "unavailable_pre_timeline_instrumentation" and "timeline" in artifact_names:
        failures.append(f"{name}: repetition {index} contradictory timeline evidence")
    for artifact_name, artifact in rep.get("artifacts", {}).items():
        path = Path(artifact.get("path", ""))
        if not path.is_file() or file_sha256(path) != artifact.get("sha256"):
            failures.append(f"{name}: repetition {index} artifact mismatch: {artifact_name}")


def check(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    contract: dict[str, Any] | None = None,
    history_index: dict[str, Any] | None = None,
    hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    inconclusive: list[str] = []
    hashes = hashes or {}
    if report.get("evidence_class") != "infrastructure_calibration":
        failures.append("wrong evidence class")
    if report.get("decision") != "candidate":
        failures.append("assembler report decision must be candidate")
    if report.get("manifest_sha256") != manifest.get("manifest_sha256"):
        failures.append("manifest hash mismatch")
    if hashes.get("manifest") and binding(report, "manifest").get("sha256") not in (None, hashes["manifest"]):
        failures.append("manifest file hash mismatch")
    if hashes.get("prediction_contract"):
        if report.get("prediction_contract_sha256") != hashes["prediction_contract"]:
            failures.append("prediction contract hash mismatch")
        if binding(report, "prediction_contract").get("sha256") != hashes["prediction_contract"]:
            failures.append("bound prediction contract hash mismatch")
    if hashes.get("history_index") and binding(report, "history_index").get("sha256") not in (None, hashes["history_index"]):
        failures.append("history index hash mismatch")
    if hashes.get("policy") and binding(report, "policy").get("sha256") not in (None, hashes["policy"]):
        failures.append("policy hash mismatch")
    if hashes.get("preflight_receipt") and binding(report, "preflight_receipt").get("sha256") not in (None, hashes["preflight_receipt"]):
        failures.append("preflight receipt hash mismatch")
    if hashes.get("semantic_contract") and binding(report, "semantic_contract").get("sha256") not in (None, hashes["semantic_contract"]):
        failures.append("semantic contract hash mismatch")

    if contract is not None:
        if report.get("prediction_contract_decision") != contract.get("decision"):
            failures.append("prediction contract decision mismatch")
        if history_index is not None:
            verification = verify_prediction_contract(
                contract,
                manifest,
                history_index,
                manifest_sha256=hashes.get("manifest"),
                history_index_sha256=hashes.get("history_index"),
            )
            if verification.get("failures"):
                failures.extend(verification["failures"])
            if verification.get("decision") == "blocked":
                failures.append("prediction contract decision is blocked")
            elif verification.get("decision") == "inconclusive":
                inconclusive.append("prediction contract decision is inconclusive")
        elif contract.get("decision") == "blocked":
            failures.append("prediction contract decision is blocked")
        elif contract.get("decision") == "inconclusive":
            inconclusive.append("prediction contract decision is inconclusive")

    phases = report.get("phases", [])
    if [p.get("phase") for p in phases] != list(PHASES):
        failures.append("phase order mismatch")
    hashes_seen = {p.get("profile_hash") for p in phases}
    if hashes_seen != {manifest.get("resource_profile", {}).get("sha256")}:
        failures.append("profile hash mismatch")
    if binding(report, "resource_profile").get("sha256") not in (None, manifest.get("resource_profile", {}).get("sha256")):
        failures.append("bound resource profile hash mismatch")
    contract_doc = report.get("contract", {})
    if contract_doc.get("val_max_samples") != -1:
        failures.append("VAL_MAX_SAMPLES must be -1")
    if contract_doc.get("validation_deadline_seconds") != 1800:
        failures.append("validation deadline contract must be 1800 seconds")
    ttl = manifest.get("calibration_policy", {}).get("calibration_receipt_max_age_seconds", RECEIPT_TTL_SECONDS)
    if ttl != RECEIPT_TTL_SECONDS:
        failures.append("calibration receipt TTL must be 86400 seconds")
    validation = report.get("validation_data", {})
    if validation.get("scope") != "full":
        failures.append("validation scope must be full")
    datasets = validation.get("datasets", [])
    by_name = {item.get("name"): item for item in datasets if isinstance(item, dict)}
    if len(datasets) != len(VALIDATION_DATASETS) or set(by_name) != set(VALIDATION_DATASETS):
        failures.append("validation datasets must contain exactly HumanEval+, MBPP+, and LiveCodeBench")
    for name, expected in VALIDATION_DATASETS.items():
        item = by_name.get(name)
        if item is None:
            continue
        if item.get("rows") != expected["rows"]:
            failures.append(f"{name}: row count mismatch")
        if item.get("sha256") != expected["sha256"]:
            failures.append(f"{name}: hash mismatch")
        path = Path(item.get("path", ""))
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            failures.append(f"{name}: validation artifact mismatch")
    if validation.get("total_rows") != sum(item["rows"] for item in VALIDATION_DATASETS.values()):
        failures.append("validation total row count mismatch")

    for phase in phases:
        name = phase.get("phase", "unknown")
        observed = phase.get("observed", {})
        predicted = phase.get("predicted", {})
        predictors = phase.get("predictor_repetitions", [])
        reps = phase.get("repetitions", [])
        expected_predictors = report.get("contract", {}).get("predictor_repetitions", {}).get(name)
        if len(predictors) != expected_predictors or expected_predictors not in (1, 3):
            failures.append(f"{name}: predictor repetition count mismatch")
        if any(r.get("warmup") is not True for r in predictors):
            failures.append(f"{name}: predictor marker mismatch")
        if len(reps) != 3:
            failures.append(f"{name}: expected three measured repetitions")
        elif any(r.get("warmup") is not False for r in reps):
            failures.append(f"{name}: measured repetition marker mismatch")
        for index, rep in enumerate([*predictors, *reps]):
            validate_repetition(rep, name, index, failures)
        if observed.get("complete_validation_metrics") is not True:
            failures.append(f"{name}: incomplete validation metrics")
        if observed.get("maximum_validation_elapsed_seconds", observed.get("validation_elapsed_seconds", 10**9)) > 1800:
            failures.append(f"{name}: validation deadline exceeded")
        for section, label in ((predicted, "prediction"), (observed, "observation")):
            idle = section.get("all_gpu_idle_fraction_during_validation")
            if not isinstance(idle, (int, float)) or idle < 0 or idle > 1:
                failures.append(f"{name}: invalid all-GPU-idle {label}")
        values_by_metric: dict[str, list[float]] = {}
        if len(reps) == 3:
            for metric in METRICS:
                values = [metric_value(rep, metric) for rep in reps]
                if any(value is None for value in values):
                    failures.append(f"{name}: missing raw {metric} acceptance values")
                else:
                    values_by_metric[metric] = [float(value) for value in values if value is not None]
        cphase = phase_contract(contract, str(name))
        if cphase is None and contract is not None:
            failures.append(f"{name}: missing prediction contract phase")
        elif cphase is not None:
            if cphase.get("decision") == "blocked":
                failures.append(f"{name}: prediction contract phase is blocked")
            elif cphase.get("decision") == "inconclusive":
                inconclusive.append(f"{name}: prediction contract phase is inconclusive")
            predictions = cphase.get("predictions", {})
            for metric in POINT_ERROR_METRICS:
                pred_doc = predictions.get(metric, {})
                estimate = pred_doc.get("point")
                interval = pred_doc.get("interval")
                raw_values = values_by_metric.get(metric)
                if not isinstance(estimate, (int, float)) or estimate == 0 or not isinstance(interval, list):
                    failures.append(f"{name}: missing {metric} prediction contract evidence")
                    continue
                if pred_doc.get("loo_coverage", 1.0) < 0.80:
                    inconclusive.append(f"{name}: {metric} leave-one-out coverage below 0.80")
                if metric == "validation_elapsed_seconds" and interval[1] >= 1800:
                    failures.append(f"{name}: elapsed prediction interval reaches 1800 seconds")
                if raw_values:
                    med = median(raw_values)
                    if abs(med - float(estimate)) / abs(float(estimate)) > 0.20:
                        failures.append(f"{name}: {metric} prediction error exceeds 20%")
                    for value in [*raw_values, med]:
                        if not interval_contains(interval, value):
                            failures.append(f"{name}: {metric} acceptance value outside prediction interval")
            idle_doc = predictions.get("all_gpu_idle_fraction_during_validation", {})
            pred_interval = idle_doc.get("interval")
            idle_values = values_by_metric.get("all_gpu_idle_fraction_during_validation")
            if not isinstance(pred_interval, list) or len(pred_interval) != 2:
                failures.append(f"{name}: missing GPU idle prediction interval")
            elif idle_values:
                measured_interval = measured_idle_interval(idle_values)
                if not intervals_overlap(pred_interval, measured_interval):
                    failures.append(f"{name}: GPU idle measured interval does not overlap prediction interval")
        if phase.get("optimized") is not True:
            failures.append(f"{name}: optimization/safety budget not met")

    decision = "blocked" if failures else "inconclusive" if inconclusive else "deployable"
    return {"ok": decision == "deployable", "decision": decision, "failures": failures, "inconclusive_reasons": inconclusive}


def build_receipt(
    result: dict[str, Any],
    report_path: Path,
    manifest_path: Path,
    hashes: dict[str, str],
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": 1,
        "receipt_type": "code_task_operational_calibration_deployability",
        "issued_at": now,
        "decision": result["decision"],
        "ttl_seconds": RECEIPT_TTL_SECONDS,
        "queue_identity": report.get("queue_identity"),
        "profile": report.get("input_bindings", {}).get("resource_profile", {}),
        "hashes": {
            "report_sha256": file_sha256(report_path),
            "manifest_sha256": hashes.get("manifest") or file_sha256(manifest_path),
            "rendered_manifest_sha256": manifest.get("manifest_sha256"),
            "policy_sha256": hashes.get("policy"),
            "history_index_sha256": hashes.get("history_index"),
            "prediction_contract_sha256": hashes.get("prediction_contract"),
            "semantic_contract_sha256": hashes.get("semantic_contract"),
            "preflight_receipt_sha256": hashes.get("preflight_receipt"),
        },
        "failures": result.get("failures", []),
        "inconclusive_reasons": result.get("inconclusive_reasons", []),
    }


def write_receipt(path: Path, receipt: dict[str, Any]) -> str:
    data = canonical_json_bytes(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--contract", type=Path, help="frozen prediction contract")
    p.add_argument("--history-index", type=Path, help="immutable trusted history snapshot")
    p.add_argument("--policy", type=Path, help="reviewed calibration/admission policy")
    p.add_argument("--preflight-receipt", type=Path, help="AC-24 preflight receipt")
    p.add_argument("--semantic-contract", type=Path, help="optional sampled-decoding semantic contract")
    p.add_argument("--receipt", type=Path, help="write canonical checker receipt")
    a = p.parse_args()

    report = load_json(a.report)
    manifest = load_manifest(a.manifest)
    failures: list[str] = []
    contract = load_json(a.contract) if a.contract else None
    history_index = load_json(a.history_index) if a.history_index else None
    hashes = {
        "report": file_sha256(a.report),
        "manifest": file_sha256(a.manifest),
    }
    for key, path in (
        ("prediction_contract", a.contract),
        ("history_index", a.history_index),
        ("policy", a.policy),
        ("preflight_receipt", a.preflight_receipt),
        ("semantic_contract", a.semantic_contract),
    ):
        observed = check_file_binding(report, key, path, failures, optional=key == "semantic_contract")
        if observed:
            hashes[key] = observed
    if a.receipt:
        for key, path in (("contract", a.contract), ("history-index", a.history_index), ("policy", a.policy), ("preflight-receipt", a.preflight_receipt)):
            if path is None:
                failures.append(f"--{key} is required when writing --receipt")
    result = check(report, manifest, contract=contract, history_index=history_index, hashes=hashes)
    result["failures"] = failures + result["failures"]
    if failures:
        result["ok"] = False
        result["decision"] = "blocked"
    if a.receipt:
        receipt = build_receipt(result, a.report, a.manifest, hashes, report, manifest)
        result["receipt_sha256"] = write_receipt(a.receipt, receipt)
        result["receipt"] = str(a.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
