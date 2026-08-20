#!/usr/bin/env python3
"""Compare preflight reports without allowing semantic downscope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(before: dict, after: dict) -> dict:
    if before["semantic_hash"] != after["semantic_hash"]:
        return {"ok": False, "decision": "rejected_semantic_change", "reason": "semantic_hash mismatch"}
    phase_results = []
    for old, new in zip(before["phases"], after["phases"], strict=True):
        a, b = old["metrics"], new["metrics"]
        throughput_gain = b["valid_scores_per_minute"] / a["valid_scores_per_minute"] - 1
        timeout_delta = b["timeout_rate"] - a["timeout_rate"]
        rss_gain = b["peak_rss_gib"] / a["peak_rss_gib"] - 1
        hard_wall = b["validation_elapsed_seconds"] <= 1800 and b["complete_validation_metrics"] is True
        optimized = throughput_gain >= 0.10 and timeout_delta <= 0.01 and rss_gain <= 0.10 and hard_wall
        phase_results.append(
            {
                "phase": old["phase"],
                "throughput_gain": round(throughput_gain, 6),
                "timeout_rate_delta": round(timeout_delta, 6),
                "peak_rss_change": round(rss_gain, 6),
                "hard_wall_pass": hard_wall,
                "optimized": optimized,
            }
        )
    return {
        "ok": all(item["optimized"] for item in phase_results),
        "decision": "optimized" if all(item["optimized"] for item in phase_results) else "neutral_or_regressed",
        "phases": phase_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
