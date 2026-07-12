#!/usr/bin/env python3
"""Outcome-schema-v2 extraction for calibration generation evidence."""

from __future__ import annotations

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


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0 < quantile <= 1:
        raise ValueError("nearest-rank requires values and 0 < quantile <= 1")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def load_generation_outcomes(path: Path, workload: dict) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = sum(int(item["row_count"]) for item in workload["datasets"])
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} submitted rows, found {len(rows)}")
    uids = [row.get("uid") for row in rows]
    if any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError(f"{path}: missing or duplicate stable UIDs")

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
