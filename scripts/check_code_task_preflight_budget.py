#!/usr/bin/env python3
"""Block launch when a preflight report violates hard or soft policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(report: dict, policy: dict) -> dict:
    failures = []
    contract = report.get("contract", {})
    hard = policy["hard"]
    if contract.get("max_response_length") != hard["max_response_length"]:
        failures.append(
            {
                "tier": "hard",
                "metric": "max_response_length",
                "actual": contract.get("max_response_length"),
                "limit": hard["max_response_length"],
            }
        )
    if contract.get("validation_datasets") != hard["required_datasets"]:
        failures.append(
            {
                "tier": "hard",
                "metric": "validation_datasets",
                "actual": contract.get("validation_datasets"),
                "limit": hard["required_datasets"],
            }
        )
    for phase in report.get("phases", []):
        metrics = phase.get("metrics", {})
        prefix = phase.get("phase", "unknown")
        if phase.get("profile_hash") != hard["required_profile_hash"]:
            failures.append(
                {
                    "tier": "hard",
                    "metric": f"{prefix}.profile_hash",
                    "actual": phase.get("profile_hash"),
                    "limit": hard["required_profile_hash"],
                }
            )
        if hard["require_complete_validation_metrics"] and metrics.get("complete_validation_metrics") is not True:
            failures.append(
                {
                    "tier": "hard",
                    "metric": f"{prefix}.complete_validation_metrics",
                    "actual": metrics.get("complete_validation_metrics"),
                    "limit": True,
                }
            )
        for metric, limit_key in (("validation_elapsed_seconds", "max_validation_elapsed_seconds"),):
            value = metrics.get(metric)
            if value is None or value > hard[limit_key]:
                failures.append(
                    {"tier": "hard", "metric": f"{prefix}.{metric}", "actual": value, "limit": hard[limit_key]}
                )
        for metric, limit_key in (
            ("timeout_rate", "max_timeout_rate"),
            ("invalid_score_rate", "max_invalid_score_rate"),
            ("peak_rss_gib", "max_peak_rss_gib"),
            ("gpu_wait_fraction", "max_gpu_wait_fraction"),
        ):
            value = metrics.get(metric)
            if value is None or value > policy["soft"][limit_key]:
                failures.append(
                    {
                        "tier": "soft",
                        "metric": f"{prefix}.{metric}",
                        "actual": value,
                        "limit": policy["soft"][limit_key],
                    }
                )
    return {
        "schema_version": 1,
        "ok": not failures,
        "decision": "pass"
        if not failures
        else "user_decision_required"
        if all(item["tier"] == "soft" for item in failures)
        else "blocked",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(
            json.dumps({"ok": False, "decision": "blocked", "error": "unknown arguments", "arguments": unknown}),
            file=sys.stderr,
        )
        return 2
    result = check(json.loads(args.report.read_text()), json.loads(args.policy.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
