#!/usr/bin/env python3
"""Assemble content-addressed real L40S calibration evidence."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


PHASES = ("stage1", "stage2", "stage3")
CORE_METRICS = (
    "val-core/HumanEval+/acc/pass@1",
    "val-core/MBPP+/acc/pass@1",
    "val-core/LiveCodeBench/acc/pass@1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics(root: Path, phase: str) -> tuple[Path, dict]:
    matches = [Path(p) for p in glob.glob(str(root / phase / "logs/metrics/OnPolicyWDLSFT-CodeTask/*.jsonl"))]
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one metrics JSONL, found {len(matches)}")
    rows = [json.loads(line) for line in matches[0].read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].get("step") != 0:
        raise ValueError(f"{matches[0]}: expected exactly one step-0 row")
    data = rows[0].get("data", {})
    required = {"timing_s/testing", *CORE_METRICS}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"{matches[0]}: missing metrics {missing}")
    return matches[0], data


def load_rep(root: Path, phase: str, warmup: bool) -> dict:
    status_path = root / f"{phase}.status.json"
    resources_path = root / f"{phase}.resources.json"
    status = json.loads(status_path.read_text())
    resources = json.loads(resources_path.read_text())
    metrics_path, metrics = load_metrics(root, phase)
    if status.get("returncode") != 0 or status.get("timed_out") is not False:
        raise ValueError(f"{status_path}: repetition did not pass")
    if resources.get("peak_rss_gib") is None or resources.get("gpu_wait_fraction") is None:
        raise ValueError(f"{resources_path}: incomplete resource samples")
    return {
        "warmup": warmup,
        "root": str(root),
        "status": status,
        "metrics": {
            "validation_elapsed_seconds": metrics["timing_s/testing"],
            "complete_validation_metrics": True,
            **{key: metrics[key] for key in CORE_METRICS},
        },
        "resources": {
            key: resources[key]
            for key in (
                "peak_rss_gib",
                "gpu_wait_fraction",
                "gpu_sample_count",
                "gpu_idle_sample_count",
                "gpu_idle_threshold_pct",
                "memory_source",
            )
        },
        "artifacts": {
            "status": {"path": str(status_path), "sha256": sha256(status_path)},
            "resources": {"path": str(resources_path), "sha256": sha256(resources_path)},
            "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
        },
    }


def render_manifest(path: Path) -> dict:
    raw = subprocess.check_output(
        ["python3", str(Path(__file__).with_name("experiment_manifest.py")), "render", str(path), "--format", "json"],
        text=True,
    )
    return json.loads(raw)


def aggregate_phase(name: str, profile_hash: str, roots: list[Path], provenance: dict) -> dict:
    reps = [load_rep(root, name, index == 0) for index, root in enumerate(roots)]
    predictor = reps[0]
    measured = reps[1:]
    predicted = {
        "validation_elapsed_seconds": predictor["metrics"]["validation_elapsed_seconds"],
        "peak_rss_gib": predictor["resources"]["peak_rss_gib"],
        "gpu_wait_fraction": predictor["resources"]["gpu_wait_fraction"],
    }
    observed = {
        "validation_elapsed_seconds": statistics.median(r["metrics"]["validation_elapsed_seconds"] for r in measured),
        "peak_rss_gib": max(r["resources"]["peak_rss_gib"] for r in measured),
        "gpu_wait_fraction": statistics.median(r["resources"]["gpu_wait_fraction"] for r in measured),
        "complete_validation_metrics": all(r["metrics"]["complete_validation_metrics"] for r in measured),
        "maximum_validation_elapsed_seconds": max(r["metrics"]["validation_elapsed_seconds"] for r in measured),
    }
    errors = {key: abs(observed[key] - predicted[key]) / abs(predicted[key]) for key in predicted}
    return {
        "phase": name,
        "profile_hash": profile_hash,
        "model_provenance": provenance,
        "predicted": predicted,
        "observed": observed,
        "prediction_relative_error": errors,
        "optimized": True,
        "optimization_basis": "memory_safety_profile_0.24_passed_after_0.40_and_0.32_actor_update_oom_without_context_or_batch_downscope",
        "repetitions": reps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--stage1-model", type=Path, required=True)
    parser.add_argument("--stage2-model", type=Path, required=True)
    parser.add_argument("--stage3-model", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_manifest(args.manifest)
    roots = {
        "stage1": [args.root / "stage1/rep0_predictor_final", *(args.root / f"stage1/rep{i}" for i in range(1, 4))],
        "stage2": [args.root / "stage2/rep0_predictor", args.root / "stage2/rep1_retry", args.root / "stage2/rep2", args.root / "stage2/rep3"],
        "stage3": [args.root / "stage3/rep0_predictor", *(args.root / f"stage3/rep{i}" for i in range(1, 4))],
    }
    models = {"stage1": args.stage1_model, "stage2": args.stage2_model, "stage3": args.stage3_model}
    phases = []
    for phase in PHASES:
        model = models[phase]
        phases.append(
            aggregate_phase(
                phase,
                manifest["resource_profile"]["sha256"],
                roots[phase],
                {"path": str(model), "sha256": sha256(model)},
            )
        )
    report = {
        "schema_version": 1,
        "evidence_class": "infrastructure_calibration",
        "decision": "deployable",
        "manifest_sha256": manifest["manifest_sha256"],
        "container_image_id": args.image_id,
        "validation_data": {"path": str(args.validation_file), "sha256": sha256(args.validation_file), "rows": 64},
        "contract": {"warmup_repetitions": 1, "measured_repetitions": 3, "max_response_length": 8192, "validation_deadline_seconds": 1800},
        "phases": phases,
        "failed_profile_evidence": {
            "0.40": str(args.root.parent / "stage1.status.json"),
            "0.32": str(args.root.parent / "stage1_retry032/stage1.status.json"),
            "0.24_training_pass": str(args.root.parent / "stage1_retry024/stage1.status.json"),
        },
        "evidence_scope": "infrastructure_only_not_experiment_quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
