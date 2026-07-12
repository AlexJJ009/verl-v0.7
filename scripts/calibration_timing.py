#!/usr/bin/env python3
"""Canonical validation timing semantics for operational calibration evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path


EVENTS = ("validation_ready", "generation_complete", "metrics_complete")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_validation_timing(path: Path, trainer_elapsed_seconds: object) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [row.get("event") for row in rows]
    if events != list(EVENTS):
        raise ValueError(f"{path}: invalid validation timeline event order: {events}")

    timestamps = [row.get("monotonic_seconds") for row in rows]
    if any(not _finite_number(value) for value in timestamps):
        raise ValueError(f"{path}: validation timeline timestamps must be finite numbers")
    if timestamps != sorted(timestamps):
        raise ValueError(f"{path}: validation timeline is not monotonic")
    if not _finite_number(trainer_elapsed_seconds) or trainer_elapsed_seconds < 0:
        raise ValueError(f"{path}: timing_s/testing must be a finite non-negative number")

    rollout_elapsed = timestamps[1] - timestamps[0]
    post_generation_elapsed = timestamps[2] - timestamps[1]
    canonical_elapsed = timestamps[2] - timestamps[0]
    if trainer_elapsed_seconds < canonical_elapsed:
        raise ValueError(
            f"{path}: timing_s/testing ({trainer_elapsed_seconds}) does not contain canonical "
            f"validation interval ({canonical_elapsed})"
        )

    return {
        "rollout_elapsed_seconds": rollout_elapsed,
        "post_generation_elapsed_seconds": post_generation_elapsed,
        "timeline_elapsed_seconds": canonical_elapsed,
        "validation_elapsed_seconds": canonical_elapsed,
        "trainer_validation_elapsed_seconds": float(trainer_elapsed_seconds),
        "pre_readiness_elapsed_seconds": float(trainer_elapsed_seconds) - canonical_elapsed,
    }
