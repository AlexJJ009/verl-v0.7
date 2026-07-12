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


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0 < quantile <= 1:
        raise ValueError("nearest-rank requires values and 0 < quantile <= 1")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def load_generation_outcomes(path: Path, workload: dict) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    eligibility = workload["validation_eligibility"]
    expected = int(eligibility["submitted_prompt_count"])
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} submitted rows, found {len(rows)}")
    uids = [row.get("uid") for row in rows]
    if any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError(f"{path}: missing or duplicate stable UIDs")
    expected_names = [item["name"] for item in workload["datasets"]]
    by_dataset = {name: [] for name in expected_names}
    for row in rows:
        name = row.get("data_source")
        if name not in by_dataset:
            raise ValueError(f"{path}: unknown validation data_source: {name!r}")
        by_dataset[name].append(row["uid"])
    actual_counts = {name: len(by_dataset[name]) for name in expected_names}
    if actual_counts != eligibility["per_dataset_eligible_counts"]:
        raise ValueError(f"{path}: eligible UID dataset counts mismatch")
    uid_doc = {
        "schema_version": 1,
        "datasets": [
            {"name": name, "source_index": source_index, "ordered_uids": by_dataset[name]}
            for source_index, name in enumerate(expected_names)
        ],
    }
    actual_uid_sha256 = hashlib.sha256(canonical_json(uid_doc)).hexdigest()
    if actual_uid_sha256 != eligibility["ordered_eligible_uid_sha256"]:
        raise ValueError(f"{path}: ordered eligible UID hash mismatch")

    counts = [row.get("response_token_count") for row in rows]
    eos = [row.get("response_eos_present") for row in rows]
    reasons = [row.get("response_finish_reason") for row in rows]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise ValueError(f"{path}: invalid response_token_count")
    if any(not isinstance(value, bool) for value in eos):
        raise ValueError(f"{path}: invalid response_eos_present")
    if any(value not in {"stop", "length"} for value in reasons):
        raise ValueError(f"{path}: incomplete response_finish_reason telemetry")
    for count, has_eos, reason in zip(counts, eos, reasons, strict=True):
        if reason == "stop" and not has_eos:
            raise ValueError(f"{path}: stop finish reason without EOS")
        if reason == "length" and (has_eos or count != 8192):
            raise ValueError(f"{path}: invalid length finish evidence")

    latencies = [row.get("code_reward_latency_seconds") for row in rows]
    timeouts = [row.get("code_reward_timeout") for row in rows]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError(f"{path}: invalid scorer latency")
    if any(value not in (0, 1, False, True) for value in timeouts):
        raise ValueError(f"{path}: invalid scorer timeout")

    return {
        "submitted_item_count": expected,
        "response_length_p50_tokens": float(nearest_rank(counts, 0.50)),
        "response_length_p95_tokens": float(nearest_rank(counts, 0.95)),
        "truncation_rate": sum(reason == "length" for reason in reasons) / expected,
        "scorer_latency_p50_seconds": float(nearest_rank(latencies, 0.50)),
        "scorer_latency_p95_seconds": float(nearest_rank(latencies, 0.95)),
        "scorer_timeout_rate": sum(bool(value) for value in timeouts) / expected,
    }
