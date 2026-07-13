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
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_calibration_prediction_contract import (
    canonical_json_bytes,
    canonical_json_sha256,
    verify_prediction_contract,
)
from calibration_outcomes import CONTINUOUS_OUTCOMES, RATE_OUTCOMES


PHASES = ("stage1", "stage2", "stage3")
STAGE12_PHASES = ("stage1", "stage2")
STAGE12_PRODUCER = {
    "run_id": "frac25-stage2",
    "run_prefix": "CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V1",
    "final_step": 20,
    "train_file_sha256": "160be1866e6c1dc439dcfbd594b54324f000f1f48db1f6a0fc88cf227c628dab",
    "expected_output_path": "/data-2/model_weights/code_task/qwen3_1p7b_stage123/frac25_p40_s220_s340/stage2_final_model2",
    "expected_provenance_path": "/data-2/model_weights/code_task/qwen3_1p7b_stage123/frac25_p40_s220_s340/frac25-stage3.provenance.json",
}
METRICS = (*CONTINUOUS_OUTCOMES, *RATE_OUTCOMES, "all_gpu_idle_fraction_during_validation")
POINT_ERROR_METRICS = CONTINUOUS_OUTCOMES
VALIDATION_DATASETS = {
    "HumanEval+": {"rows": 164, "sha256": "e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab"},
    "MBPP+": {"rows": 378, "sha256": "3221e7f53c88bfbd91d788fb7bcb37168fb088fa504fddf12b9126c2147312d2"},
    "LiveCodeBench": {"rows": 880, "sha256": "fe7d2bfe2779bcf106492347ca173e30b9220c15c1b8783949d35edcd93a43d1"},
}
RECEIPT_TTL_SECONDS = 86400
ROUND_QUANT = Decimal("0.000001")
WILSON_Z = 1.959963984540054


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
    elif metric == "all_gpu_idle_fraction_during_validation":
        value = rep.get("resources", {}).get("gpu_wait_fraction")
    else:
        value = rep.get("metrics", {}).get(metric)
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


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if not isinstance(successes, int) or not isinstance(total, int) or total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson count")
    p = successes / total
    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = WILSON_Z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def outward(value: float, *, lower: bool) -> float:
    rounding = ROUND_FLOOR if lower else ROUND_CEILING
    return float(Decimal(str(value)).quantize(ROUND_QUANT, rounding=rounding))


def half_even(value: float) -> float:
    return float(Decimal(str(value)).quantize(ROUND_QUANT, rounding=ROUND_HALF_EVEN))


def truncation_counts(phase: dict[str, Any], workload: dict[str, Any]) -> dict[str, tuple[int, int]]:
    eligible_counts = workload.get("validation_eligibility", {}).get("per_dataset_eligible_counts")
    if not isinstance(eligible_counts, dict) or set(eligible_counts) != set(VALIDATION_DATASETS):
        raise ValueError("missing manifest eligible truncation counts")
    if sum(eligible_counts.values()) != 1379:
        raise ValueError("manifest eligible truncation count must total 1379")
    totals = {"aggregate": [0, 0], **{name: [0, 0] for name in eligible_counts}}
    reps = phase.get("repetitions", [])
    if len(reps) != 3:
        raise ValueError("expected three truncation repetitions")
    for rep in reps:
        metrics = rep.get("metrics", {})
        truncated = metrics.get("truncated_item_count")
        submitted = metrics.get("submitted_item_count")
        if not isinstance(truncated, int) or not isinstance(submitted, int) or submitted != 1379 or not 0 <= truncated <= submitted:
            raise ValueError("invalid aggregate truncation counts")
        totals["aggregate"][0] += truncated
        totals["aggregate"][1] += submitted
        by_dataset = metrics.get("truncation_by_dataset")
        if not isinstance(by_dataset, dict) or set(by_dataset) != set(eligible_counts):
            raise ValueError("missing per-dataset truncation counts")
        for name, expected_total in eligible_counts.items():
            item = by_dataset[name]
            dataset_total = item.get("submitted_item_count")
            dataset_truncated = item.get("truncated_item_count")
            if dataset_total != expected_total or not isinstance(dataset_truncated, int) or not 0 <= dataset_truncated <= dataset_total:
                raise ValueError(f"invalid {name} truncation counts")
            if abs(item.get("truncation_rate", -1) - dataset_truncated / dataset_total) > 1e-12:
                raise ValueError(f"inconsistent {name} truncation rate")
            totals[name][0] += dataset_truncated
            totals[name][1] += dataset_total
    return {name: (values[0], values[1]) for name, values in totals.items()}


def truncation_comparison(
    prior: dict[str, Any], later: dict[str, Any], prior_workload: dict[str, Any], later_workload: dict[str, Any]
) -> dict[str, Any]:
    prior_counts = truncation_counts(prior, prior_workload)
    later_counts = truncation_counts(later, later_workload)
    if {name: total for name, (_, total) in prior_counts.items()} != {name: total for name, (_, total) in later_counts.items()}:
        raise ValueError("phase truncation denominators differ")
    comparisons = {}
    for name in prior_counts:
        prior_truncated, prior_total = prior_counts[name]
        later_truncated, later_total = later_counts[name]
        prior_interval = wilson_interval(prior_truncated, prior_total)
        later_interval = wilson_interval(later_truncated, later_total)
        prior_rate = prior_truncated / prior_total
        later_rate = later_truncated / later_total
        delta = later_rate - prior_rate
        interval = (later_interval[0] - prior_interval[1], later_interval[1] - prior_interval[0])
        if interval[0] > 0 and delta >= 0.02:
            classification = "meaningful_worsening"
        elif interval[1] < 0 and delta <= -0.02:
            classification = "meaningful_improvement"
        else:
            classification = "not_statistically_meaningful"
        comparisons[name] = {
            "prior": {"truncated": prior_truncated, "submitted": prior_total, "rate": half_even(prior_rate)},
            "later": {"truncated": later_truncated, "submitted": later_total, "rate": half_even(later_rate)},
            "delta": half_even(delta),
            "delta_interval_95": [outward(interval[0], lower=True), outward(interval[1], lower=False)],
            "classification": classification,
        }
    return comparisons


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
    expected_phases: tuple[str, ...] = PHASES,
) -> dict[str, Any]:
    failures: list[str] = []
    inconclusive: list[str] = []
    diagnostics: dict[str, Any] = {}
    hashes = hashes or {}
    expected_authorization_scope = "stage12_producer" if expected_phases == STAGE12_PHASES else "full"
    if report.get("authorization_scope") != expected_authorization_scope:
        failures.append("report authorization_scope mismatch")
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
        if contract.get("authorization_scope") != expected_authorization_scope:
            failures.append("prediction contract authorization_scope mismatch")
        if report.get("prediction_contract_decision") != contract.get("decision"):
            failures.append("prediction contract decision mismatch")
        if history_index is not None:
            verification = verify_prediction_contract(
                contract,
                manifest,
                history_index,
                manifest_sha256=hashes.get("manifest"),
                history_index_sha256=hashes.get("history_index"),
                authorization_scope=expected_authorization_scope,
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
    if [p.get("phase") for p in phases] != list(expected_phases):
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
        workload = manifest.get("calibration_workloads", {}).get(name)
        if not isinstance(workload, dict):
            failures.append(f"{name}: missing calibration workload descriptor")
        else:
            expected_workload_hash = canonical_json_sha256(workload)
            if phase.get("workload_descriptor_sha256") != expected_workload_hash:
                failures.append(f"{name}: workload descriptor hash mismatch")
            if phase.get("outcome_schema_version") != workload.get("outcome_schema_version"):
                failures.append(f"{name}: outcome schema version mismatch")
        observed = phase.get("observed", {})
        predicted = phase.get("predicted", {})
        predictors = phase.get("predictor_repetitions", [])
        reps = phase.get("repetitions", [])
        expected_predictors = report.get("contract", {}).get("predictor_repetitions", {}).get(name)
        if len(predictors) != expected_predictors or expected_predictors != 0:
            failures.append(f"{name}: predictor repetition count mismatch")
        if len(reps) != 3:
            failures.append(f"{name}: expected three measured repetitions")
        elif any(r.get("warmup") is not False for r in reps):
            failures.append(f"{name}: measured repetition marker mismatch")
        for index, rep in enumerate(reps):
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
            if workload is not None:
                contract_workload = cphase.get("features")
                if not isinstance(contract_workload, dict) or canonical_json_sha256(contract_workload) != canonical_json_sha256(workload):
                    failures.append(f"{name}: prediction contract workload descriptor mismatch")
                if not isinstance(contract_workload, dict) or contract_workload.get("outcome_schema_version") != workload.get("outcome_schema_version"):
                    failures.append(f"{name}: prediction contract outcome schema mismatch")
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
            for metric in RATE_OUTCOMES:
                pred_interval = predictions.get(metric, {}).get("interval")
                raw_values = values_by_metric.get(metric)
                if not isinstance(pred_interval, list) or len(pred_interval) != 2:
                    failures.append(f"{name}: missing {metric} prediction interval")
                    continue
                if raw_values:
                    aggregate = sum(raw_values) / len(raw_values)
                    for value in [*raw_values, aggregate]:
                        if not interval_contains(pred_interval, value):
                            failures.append(f"{name}: {metric} acceptance value outside prediction interval")
                    if metric == "scorer_timeout_rate" and aggregate > 0.10:
                        failures.append(f"{name}: scorer timeout rate exceeds 10%")
        if phase.get("optimized") is not True:
            failures.append(f"{name}: optimization/safety budget not met")

    phase_by_name = {phase.get("phase"): phase for phase in phases}
    for prior_name, later_name in (("stage1", "stage2"), ("stage2", "stage3")):
        if prior_name not in phase_by_name or later_name not in phase_by_name:
            continue
        try:
            diagnostics[f"{prior_name}_to_{later_name}_truncation"] = truncation_comparison(
                phase_by_name[prior_name],
                phase_by_name[later_name],
                manifest["calibration_workloads"][prior_name],
                manifest["calibration_workloads"][later_name],
            )
        except ValueError as exc:
            failures.append(f"{prior_name}->{later_name}: {exc}")

    decision = "blocked" if failures else "inconclusive" if inconclusive else "deployable"
    return {
        "ok": decision == "deployable",
        "decision": decision,
        "failures": failures,
        "inconclusive_reasons": inconclusive,
        "diagnostics": diagnostics,
    }


def stage12_producer_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    producer_runs = [item for item in manifest.get("runs", []) if item.get("id") == STAGE12_PRODUCER["run_id"]]
    if len(producer_runs) != 1:
        raise ValueError("stage12 producer run missing from manifest")
    run = producer_runs[0]
    if sum(item.get("run_prefix") == STAGE12_PRODUCER["run_prefix"] for item in manifest.get("runs", [])) != 1:
        raise ValueError("stage12 producer run prefix must be unique")
    dependencies = [
        item for item in manifest.get("runs", [])
        if item.get("phase") == "stage3" and item.get("source", {}).get("run_id") == run.get("id")
    ]
    if len(dependencies) != 1:
        raise ValueError("stage12 producer must have exactly one Stage3 dependency")
    observed = {
        "run_id": run.get("id"),
        "run_prefix": run.get("run_prefix"),
        "final_step": run.get("final_step"),
        "train_file_sha256": run.get("train_file_sha256"),
        "expected_output_path": STAGE12_PRODUCER["expected_output_path"],
        "expected_provenance_path": dependencies[0].get("provenance_file"),
    }
    if observed != STAGE12_PRODUCER:
        raise ValueError("stage12 producer identity mismatch")
    return observed


def build_stage12_receipt(result, report_path, manifest_path, hashes, report, manifest, contract):
    producer = stage12_producer_from_manifest(manifest)
    cohort_hashes = {}
    for phase in STAGE12_PHASES:
        phase_doc = phase_contract(contract, phase) or {}
        cohort_hashes[phase] = canonical_json_sha256(phase_doc.get("eligible_run_ids", []))
    now = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": 1,
        "receipt_type": "code_task_operational_calibration_stage12_producer",
        "issued_at": now,
        "decision": result["decision"],
        "ttl_seconds": RECEIPT_TTL_SECONDS,
        "queue_identity": report.get("queue_identity"),
        "phase_scope": list(STAGE12_PHASES),
        "authorized_run_ids": [producer["run_id"]],
        "authorized_final_steps": {producer["run_id"]: producer["final_step"]},
        "producer": producer,
        "selected_cohort_sha256_by_phase": cohort_hashes,
        "workload_descriptor_sha256_by_phase": {
            phase: canonical_json_sha256(manifest["calibration_workloads"][phase]) for phase in STAGE12_PHASES
        },
        "hashes": {
            "report_sha256": file_sha256(report_path),
            "manifest_sha256": hashes.get("manifest") or file_sha256(manifest_path),
            "rendered_manifest_sha256": manifest.get("manifest_sha256"),
            "policy_sha256": hashes.get("policy"),
            "history_index_sha256": hashes.get("history_index"),
            "prediction_contract_sha256": hashes.get("prediction_contract"),
            "preflight_receipt_sha256": hashes.get("preflight_receipt"),
        },
        "failures": result.get("failures", []),
    }


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
    p.add_argument("--normalized-manifest", type=Path, help="normalized manifest bound by preflight")
    p.add_argument("--contract", type=Path, help="frozen prediction contract")
    p.add_argument("--history-index", type=Path, required=True, help="immutable trusted history snapshot")
    p.add_argument("--policy", type=Path, help="reviewed calibration/admission policy")
    p.add_argument("--preflight-receipt", type=Path, help="AC-24 preflight receipt")
    p.add_argument("--preflight-report", type=Path, help="machine report bound by the AC-24 receipt")
    p.add_argument("--preflight-policy", type=Path, help="policy bound by the AC-24 receipt")
    p.add_argument("--semantic-contract", type=Path, help="optional sampled-decoding semantic contract")
    p.add_argument("--receipt", type=Path, help="write canonical checker receipt")
    p.add_argument("--authorization-scope", choices=("full", "stage12_producer"), default="full")
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
    expected_phases = STAGE12_PHASES if a.authorization_scope == "stage12_producer" else PHASES
    result = check(report, manifest, contract=contract, history_index=history_index, hashes=hashes, expected_phases=expected_phases)
    result["failures"] = failures + result["failures"]
    if failures:
        result["ok"] = False
        result["decision"] = "blocked"
    if a.authorization_scope == "stage12_producer" and result["ok"]:
        try:
            stage12_producer_from_manifest(manifest)
            if contract is None or [item.get("phase") for item in contract.get("phases", [])] != list(STAGE12_PHASES):
                raise ValueError("stage12 prediction contract must contain exactly stage1 and stage2")
            if set(manifest.get("calibration_workloads", {})) != set(PHASES):
                raise ValueError("stage123 manifest workload graph mismatch")
            history_phases = {item.get("phase") for item in (history_index or {}).get("runs", [])}
            if history_phases != set(STAGE12_PHASES):
                raise ValueError("stage12 history index must contain exactly stage1 and stage2")
            if a.normalized_manifest is None or a.preflight_report is None or a.preflight_receipt is None or a.preflight_policy is None:
                raise ValueError("stage12 issuance requires normalized manifest and preflight report, receipt, and policy")
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import stage123_preflight_receipt as preflight_tool
            preflight_args = argparse.Namespace(
                receipt=a.preflight_receipt, normalized_manifest=a.normalized_manifest, report=a.preflight_report,
                policy=a.preflight_policy, run_id=STAGE12_PRODUCER["run_id"], calibration_phase=None,
                profile_hash=manifest.get("resource_profile", {}).get("sha256"), max_age_seconds=RECEIPT_TTL_SECONDS,
            )
            try:
                preflight_result = preflight_tool.verify(preflight_args)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"stage12 preflight verification failed: {exc}") from exc
            if not preflight_result["ok"]:
                raise ValueError("stage12 preflight verification failed: " + "; ".join(preflight_result["failures"]))
            preflight = load_json(a.preflight_receipt)
            if preflight.get("authorized_calibration_phases") != list(STAGE12_PHASES):
                raise ValueError("stage12 preflight must authorize exactly stage1 and stage2")
        except ValueError as exc:
            result["ok"] = False
            result["decision"] = "blocked"
            result["failures"].append(str(exc))
        else:
            result["decision"] = "stage12_calibrated"
    if a.receipt and result["ok"]:
        receipt = (
            build_stage12_receipt(result, a.report, a.manifest, hashes, report, manifest, contract or {})
            if a.authorization_scope == "stage12_producer"
            else build_receipt(result, a.report, a.manifest, hashes, report, manifest)
        )
        result["receipt_sha256"] = write_receipt(a.receipt, receipt)
        result["receipt"] = str(a.receipt)
    elif a.receipt:
        result["receipt"] = None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
