#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic workflow baseline from structured local evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_classifier():
    path = Path(__file__).with_name("experiment_failure_classifier.py")
    spec = importlib.util.spec_from_file_location("experiment_failure_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load failure classifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify


classify = _load_classifier()


SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"(?:WXPUSHER_APP_TOKEN|WANDB_API_KEY|HF_TOKEN)\s*[=:]\s*\S+", re.I),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unknown_if_none(value: Any) -> Any:
    return "unknown" if value is None else value


def interval_seconds(samples: list[dict[str, Any]], predicate) -> float | None:
    if len(samples) < 2:
        return None
    total = 0.0
    for left, right in zip(samples, samples[1:], strict=False):
        if predicate(left):
            total += float(right["timestamp_s"]) - float(left["timestamp_s"])
    return round(total, 3)


def score_bin(score: float) -> str:
    if score < 0:
        return "negative"
    if score == 0:
        return "zero"
    if score < 1:
        return "partial"
    return "positive"


def collect(fixture: Path) -> dict[str, Any]:
    metadata = load_json(fixture / "metadata.json")
    events = load_jsonl(fixture / "events.jsonl")
    processes = load_jsonl(fixture / "process_samples.jsonl")
    gpus = load_jsonl(fixture / "gpu_samples.jsonl")
    submitted = [event for event in events if event.get("event") == "score_submitted"]
    completed = [event for event in events if event.get("event") == "score_completed"]
    timeouts = [event for event in events if event.get("event") == "score_timeout"]
    scores = [float(event["score"]) for event in completed if isinstance(event.get("score"), int | float)]
    scorer_start = [float(event["timestamp_s"]) for event in events if event.get("event") == "scorer_start"]
    scorer_end = [float(event["timestamp_s"]) for event in events if event.get("event") == "scorer_end"]
    elapsed = max(scorer_end) - min(scorer_start) if scorer_start and scorer_end else None
    peak_rss = max(
        (float(sample["rss_gib"]) for sample in processes if sample.get("process_type") == "RewardLoopWorker"),
        default=None,
    )
    gpu_idle = interval_seconds(
        gpus,
        lambda sample: sample.get("phase_active") is True
        and all(float(value) <= 2 for value in sample.get("utilization_pct", [])),
    )
    phase_wall = interval_seconds(gpus, lambda sample: sample.get("phase_active") is True)
    timeout_rate = len(timeouts) / len(submitted) if submitted else None
    distribution = Counter(score_bin(score) for score in scores)
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in fixture.iterdir() if path.is_file()
    )
    classifier_signals = [source_text]
    if timeouts:
        classifier_signals.append("Reward computation timed out")
    secret_hits = [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(source_text)]
    return {
        "schema_version": 1,
        "run": metadata.get("run", "unknown"),
        "phase": metadata.get("phase", "unknown"),
        "profile_hash": metadata.get("profile_hash", "unknown"),
        "docker_image_id": metadata.get("docker_image_id", "unknown"),
        "evidence_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "scorer": {
            "submitted_count": len(submitted),
            "completed_count": len(completed),
            "timeout_count": len(timeouts),
            "timeout_rate": unknown_if_none(round(timeout_rate, 6) if timeout_rate is not None else None),
            "elapsed_seconds": unknown_if_none(round(elapsed, 3) if elapsed is not None else None),
            "score_count": len(scores),
            "score_distribution": dict(sorted(distribution.items())),
        },
        "resources": {
            "reward_worker_peak_rss_gib": unknown_if_none(round(peak_rss, 3) if peak_rss is not None else None),
            "gpu_idle_seconds": unknown_if_none(gpu_idle),
            "phase_wall_seconds": unknown_if_none(phase_wall),
            "gpu_idle_fraction": unknown_if_none(
                round(gpu_idle / phase_wall, 6) if gpu_idle is not None and phase_wall else None
            ),
        },
        "failure": classify("\n".join(classifier_signals)),
        "secret_scan": {"ok": not secret_hits, "matches": secret_hits},
    }


def markdown(report: dict[str, Any]) -> str:
    scorer = report["scorer"]
    resources = report["resources"]
    return "\n".join(
        (
            "# Experiment Workflow Baseline",
            "",
            f"- Run: `{report['run']}`",
            f"- Phase: `{report['phase']}`",
            f"- Profile hash: `{report['profile_hash']}`",
            f"- Docker image: `{report['docker_image_id']}`",
            f"- Failure class: `{report['failure']['primary_reason']}`",
            "",
            "## Scorer",
            "",
            f"- Submitted: {scorer['submitted_count']}",
            f"- Completed: {scorer['completed_count']}",
            f"- Timeouts: {scorer['timeout_count']}",
            f"- Timeout rate: {scorer['timeout_rate']}",
            f"- Elapsed seconds: {scorer['elapsed_seconds']}",
            f"- Score distribution: `{json.dumps(scorer['score_distribution'], sort_keys=True)}`",
            "",
            "## Resources",
            "",
            f"- RewardLoopWorker peak RSS GiB: {resources['reward_worker_peak_rss_gib']}",
            f"- GPU idle seconds: {resources['gpu_idle_seconds']}",
            f"- GPU idle fraction: {resources['gpu_idle_fraction']}",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.fixture)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"ok": report["secret_scan"]["ok"], "json": str(args.json_out), "markdown": str(args.md_out)},
            sort_keys=True,
        )
    )
    return 0 if report["secret_scan"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
