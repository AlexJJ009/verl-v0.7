#!/usr/bin/env python3
"""Outcome-schema-v2 extraction for calibration generation evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


CONTINUOUS_OUTCOMES = (
    "validation_elapsed_seconds",
    "peak_rss_gib",
    "response_length_p50_tokens",
    "response_length_p95_tokens",
    "scorer_latency_p50_seconds",
    "scorer_latency_p95_seconds",
)
RATE_OUTCOMES = ("truncation_rate", "scorer_timeout_rate")


class OutcomeValidationError(ValueError):
    def __init__(self, code: str, message: str, **context: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context


def invalid(code: str, message: str, **context: object) -> None:
    raise OutcomeValidationError(code, message, **context)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0 < quantile <= 1:
        invalid("nearest_rank", "nearest-rank requires values and 0 < quantile <= 1", value_count=len(values), quantile=quantile)
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def load_generation_outcomes(path: Path, workload: dict) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    eligibility = workload["validation_eligibility"]
    expected = int(eligibility["submitted_prompt_count"])
    if len(rows) != expected:
        invalid("submitted_row_count", "submitted row count mismatch", path=str(path), expected=expected, actual=len(rows))
    uids = [row.get("uid") for row in rows]
    if any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != len(uids):
        invalid("stable_uid", "missing or duplicate stable UIDs", path=str(path))
    expected_names = [item["name"] for item in workload["datasets"]]
    by_dataset = {name: [] for name in expected_names}
    rows_by_dataset = {name: [] for name in expected_names}
    for row in rows:
        name = row.get("data_source")
        if name not in by_dataset:
            invalid("data_source", "unknown validation data source", path=str(path), data_source=name)
        by_dataset[name].append(row["uid"])
        rows_by_dataset[name].append(row)
    actual_counts = {name: len(by_dataset[name]) for name in expected_names}
    if actual_counts != eligibility["per_dataset_eligible_counts"]:
        invalid("dataset_counts", "eligible UID dataset counts mismatch", path=str(path), expected=eligibility["per_dataset_eligible_counts"], actual=actual_counts)
    uid_doc = {
        "schema_version": 1,
        "datasets": [
            {"name": name, "source_index": source_index, "ordered_uids": by_dataset[name]}
            for source_index, name in enumerate(expected_names)
        ],
    }
    actual_uid_sha256 = hashlib.sha256(canonical_json(uid_doc)).hexdigest()
    if actual_uid_sha256 != eligibility["ordered_eligible_uid_sha256"]:
        invalid("uid_order_hash", "ordered eligible UID hash mismatch", path=str(path), expected=eligibility["ordered_eligible_uid_sha256"], actual=actual_uid_sha256)

    counts = [row.get("response_token_count") for row in rows]
    eos = [row.get("response_eos_present") for row in rows]
    reasons = [row.get("response_finish_reason") for row in rows]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        invalid("response_token_count", "invalid response token count", path=str(path))
    if any(not isinstance(value, bool) for value in eos):
        invalid("response_eos", "invalid response EOS evidence", path=str(path))
    if any(value not in {"stop", "length"} for value in reasons):
        invalid("finish_reason", "incomplete response finish reason telemetry", path=str(path))
    for count, has_eos, reason in zip(counts, eos, reasons, strict=True):
        if reason == "stop" and not has_eos:
            invalid("stop_without_eos", "stop finish reason without EOS", path=str(path))
        if reason == "length" and (has_eos or count != 8192):
            invalid("length_finish", "invalid length finish evidence", path=str(path), token_count=count, has_eos=has_eos)

    latencies = [row.get("code_reward_latency_seconds") for row in rows]
    timeouts = [row.get("code_reward_timeout") for row in rows]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in latencies):
        invalid("scorer_latency", "invalid scorer latency", path=str(path))
    if any(value not in (0, 1, False, True) for value in timeouts):
        invalid("scorer_timeout", "invalid scorer timeout", path=str(path))

    truncation_by_dataset = {}
    for name in expected_names:
        dataset_rows = rows_by_dataset[name]
        truncated = sum(row["response_finish_reason"] == "length" for row in dataset_rows)
        truncation_by_dataset[name] = {
            "submitted_item_count": len(dataset_rows),
            "truncated_item_count": truncated,
            "truncation_rate": truncated / len(dataset_rows),
        }
    truncated_count = sum(reason == "length" for reason in reasons)
    return {
        "submitted_item_count": expected,
        "response_length_p50_tokens": float(nearest_rank(counts, 0.50)),
        "response_length_p95_tokens": float(nearest_rank(counts, 0.95)),
        "truncated_item_count": truncated_count,
        "truncation_rate": truncated_count / expected,
        "truncation_by_dataset": truncation_by_dataset,
        "scorer_latency_p50_seconds": float(nearest_rank(latencies, 0.50)),
        "scorer_latency_p95_seconds": float(nearest_rank(latencies, 0.95)),
        "scorer_timeout_rate": sum(bool(value) for value in timeouts) / expected,
    }
