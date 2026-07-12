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
VALIDATION_DATASETS = {
    "HumanEval+": {
        "rows": 164,
        "sha256": "e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab",
    },
    "MBPP+": {
        "rows": 378,
        "sha256": "3221e7f53c88bfbd91d788fb7bcb37168fb088fa504fddf12b9126c2147312d2",
    },
    "LiveCodeBench": {
        "rows": 880,
        "sha256": "fe7d2bfe2779bcf106492347ca173e30b9220c15c1b8783949d35edcd93a43d1",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(path: Path) -> str:
    """Hash a file or a directory tree without depending on directory mtimes."""
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise ValueError(f"model artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory has no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def validation_provenance(paths: dict[str, Path]) -> dict:
    datasets = []
    for name, expected in VALIDATION_DATASETS.items():
        path = paths[name]
        observed = {"path": str(path), "rows": parquet_rows(path), "sha256": sha256(path)}
        if observed["rows"] != expected["rows"] or observed["sha256"] != expected["sha256"]:
            raise ValueError(
                f"{name}: expected full validation rows={expected['rows']} sha256={expected['sha256']}, "
                f"got rows={observed['rows']} sha256={observed['sha256']}"
            )
        datasets.append({"name": name, **observed})
    return {"scope": "full", "datasets": datasets, "total_rows": sum(item["rows"] for item in datasets)}


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


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_scorer_evidence(root: Path, phase: str, validation_elapsed_seconds: float) -> tuple[Path, dict]:
    matches = [Path(p) for p in glob.glob(str(root / phase / "logs/validation/*/*.jsonl"))]
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one validation generation JSONL, found {len(matches)}")
    rows = [json.loads(line) for line in matches[0].read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{matches[0]}: validation generation JSONL is empty")
    latencies = [row.get("code_reward_latency_seconds") for row in rows]
    if any(not isinstance(value, (int, float)) or value < 0 for value in latencies):
        raise ValueError(f"{matches[0]}: missing or invalid code_reward_latency_seconds")
    scores = [row.get("score") for row in rows]
    valid_scores = [float(score) for score in scores if isinstance(score, (int, float))]
    invalid_count = len(rows) - len(valid_scores)
    timeout_count = sum(bool(row.get("code_reward_timeout")) for row in rows)
    status_counts: dict[str, int] = {}
    score_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("code_reward_status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
    for score in valid_scores:
        key = format(score, ".17g")
        score_counts[key] = score_counts.get(key, 0) + 1
    total = len(rows)
    return matches[0], {
        "sample_count": total,
        "scorer_latency_seconds_p50": percentile(latencies, 0.50),
        "scorer_latency_seconds_p95": percentile(latencies, 0.95),
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / total,
        "invalid_score_count": invalid_count,
        "invalid_score_rate": invalid_count / total,
        "valid_score_count": len(valid_scores),
        "valid_score_rate": len(valid_scores) / total,
        "valid_scores_per_minute": len(valid_scores) * 60 / validation_elapsed_seconds,
        "score_distribution": dict(sorted(score_counts.items())),
        "status_distribution": dict(sorted(status_counts.items())),
    }


def load_validation_timeline(root: Path, phase: str) -> tuple[Path, dict]:
    path = root / f"{phase}.validation_timeline.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [row.get("event") for row in rows]
    if events != ["validation_ready", "generation_complete", "metrics_complete"]:
        raise ValueError(f"{path}: invalid validation timeline event order: {events}")
    times = [row.get("monotonic_seconds") for row in rows]
    if any(not isinstance(value, (int, float)) for value in times) or times != sorted(times):
        raise ValueError(f"{path}: validation timeline is not monotonic")
    return path, {
        "rollout_elapsed_seconds": times[1] - times[0],
        "post_generation_elapsed_seconds": times[2] - times[1],
        "timeline_elapsed_seconds": times[2] - times[0],
    }


def load_rep(root: Path, phase: str, warmup: bool) -> dict:
    status_path = root / f"{phase}.status.json"
    resources_path = root / f"{phase}.resources.json"
    status = json.loads(status_path.read_text())
    resources = json.loads(resources_path.read_text())
    metrics_path, metrics = load_metrics(root, phase)
    generation_path, scorer = load_scorer_evidence(root, phase, metrics["timing_s/testing"])
    timeline_path = root / f"{phase}.validation_timeline.jsonl"
    if timeline_path.is_file():
        timeline_path, timeline = load_validation_timeline(root, phase)
    else:
        timeline = {"status": "unavailable_pre_timeline_instrumentation"}
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
        "scorer": scorer,
        "timeline": timeline,
        "resources": {
            key: resources[key]
            for key in (
                "peak_rss_gib",
                "gpu_wait_fraction",
                "gpu_sample_count",
                "gpu_idle_sample_count",
                "gpu_idle_threshold_pct",
                "memory_source",
                "measurement_started",
                "measurement_window",
                "gpu_sample_interval_seconds",
                "readiness_wait_seconds",
            )
        },
        "artifacts": {
            "status": {"path": str(status_path), "sha256": sha256(status_path)},
            "resources": {"path": str(resources_path), "sha256": sha256(resources_path)},
            "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
            "generation": {"path": str(generation_path), "sha256": sha256(generation_path)},
            **(
                {"timeline": {"path": str(timeline_path), "sha256": sha256(timeline_path)}}
                if timeline_path.is_file()
                else {}
            ),
        },
    }


def render_manifest(path: Path) -> dict:
    raw = subprocess.check_output(
        ["python3", str(Path(__file__).with_name("experiment_manifest.py")), "render", str(path), "--format", "json"],
        text=True,
    )
    return json.loads(raw)


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def optional_file_binding(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": sha256(path)}


def phase_prediction(contract: dict, phase: str) -> dict:
    for item in contract.get("phases", []):
        if item.get("phase") == phase:
            return item
    raise ValueError(f"prediction contract missing phase: {phase}")


def median_metric(reps: list[dict], metric: str) -> float:
    if metric == "peak_rss_gib":
        return statistics.median(rep["resources"][metric] for rep in reps)
    if metric == "all_gpu_idle_fraction_during_validation":
        return statistics.median(rep["resources"]["gpu_wait_fraction"] for rep in reps)
    return statistics.median(rep["metrics"][metric] for rep in reps)


def raw_metric_values(reps: list[dict], metric: str) -> list[float]:
    if metric == "peak_rss_gib":
        return [rep["resources"][metric] for rep in reps]
    if metric == "all_gpu_idle_fraction_during_validation":
        return [rep["resources"]["gpu_wait_fraction"] for rep in reps]
    return [rep["metrics"][metric] for rep in reps]


def aggregate_phase(
    name: str,
    profile_hash: str,
    predictor_roots: list[Path],
    measured_roots: list[Path],
    provenance: dict,
    prediction_contract_phase: dict,
) -> dict:
    predictors = [load_rep(root, name, True) for root in predictor_roots]
    measured = [load_rep(root, name, False) for root in measured_roots]
    predictions = prediction_contract_phase.get("predictions", {})
    predicted = {
        metric: predictions.get(metric, {}).get("point")
        for metric in (
            "validation_elapsed_seconds",
            "peak_rss_gib",
            "all_gpu_idle_fraction_during_validation",
        )
    }
    observed = {
        "validation_elapsed_seconds": median_metric(measured, "validation_elapsed_seconds"),
        "peak_rss_gib": median_metric(measured, "peak_rss_gib"),
        "all_gpu_idle_fraction_during_validation": median_metric(measured, "all_gpu_idle_fraction_during_validation"),
        "complete_validation_metrics": all(r["metrics"]["complete_validation_metrics"] for r in measured),
        "maximum_validation_elapsed_seconds": max(r["metrics"]["validation_elapsed_seconds"] for r in measured),
    }
    errors = {
        key: abs(observed[key] - predicted[key]) / abs(predicted[key])
        for key in (
            "validation_elapsed_seconds",
            "peak_rss_gib",
        )
        if predicted.get(key) not in (None, 0)
    }
    raw_acceptance = {
        metric: raw_metric_values(measured, metric)
        for metric in (
            "validation_elapsed_seconds",
            "peak_rss_gib",
            "all_gpu_idle_fraction_during_validation",
        )
    }
    return {
        "phase": name,
        "profile_hash": profile_hash,
        "model_provenance": provenance,
        "prediction_contract_phase": {
            "status": prediction_contract_phase.get("status"),
            "decision": prediction_contract_phase.get("decision"),
            "cohort_size": prediction_contract_phase.get("cohort_size"),
            "eligible_run_ids": prediction_contract_phase.get("eligible_run_ids", []),
            "failures": prediction_contract_phase.get("failures", []),
        },
        "predicted": predicted,
        "observed": observed,
        "acceptance_repetition_values": raw_acceptance,
        "acceptance_aggregate": {
            "method": "median_of_three_valid_repetitions",
            "validation_elapsed_seconds": observed["validation_elapsed_seconds"],
            "peak_rss_gib": observed["peak_rss_gib"],
            "all_gpu_idle_fraction_during_validation": observed["all_gpu_idle_fraction_during_validation"],
        },
        "prediction_intervals": {
            metric: predictions.get(metric, {}).get("interval")
            for metric in predictions
        },
        "prediction_relative_error": errors,
        "optimized": True,
        "optimization_basis": "shared_ctx8k_profile_uses_indexed_lcb_process_isolation_bounded_reward_concurrency_and_validation_window_nvml_sampling",
        "predictor_repetitions": predictors,
        "repetitions": measured,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--humaneval-plus-file", type=Path, required=True)
    parser.add_argument("--mbpp-plus-file", type=Path, required=True)
    parser.add_argument("--livecodebench-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True, help="frozen prediction contract")
    parser.add_argument("--history-index", type=Path, required=True, help="immutable trusted history snapshot")
    parser.add_argument("--policy", type=Path, required=True, help="reviewed calibration/admission policy")
    parser.add_argument("--preflight-receipt", type=Path, required=True, help="AC-24 preflight receipt")
    parser.add_argument("--semantic-contract", type=Path, help="optional sampled-decoding semantic contract")
    parser.add_argument("--queue-identity", required=True)
    parser.add_argument("--profile", required=True, help="canonical resource profile identity")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--stage1-model", type=Path, required=True)
    parser.add_argument("--stage2-model", type=Path, required=True)
    parser.add_argument("--stage3-model", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_manifest(args.manifest)
    prediction_contract = load_json(args.contract)
    validation_data = validation_provenance(
        {
            "HumanEval+": args.humaneval_plus_file,
            "MBPP+": args.mbpp_plus_file,
            "LiveCodeBench": args.livecodebench_file,
        }
    )
    predictor_roots = {
        "stage1": [args.root / "stage1/rep0_predictor"],
        "stage2": [args.root / "stage2/rep0_predictor", args.root / "stage2/rep0_predictor_2", args.root / "stage2/rep0_predictor_3"],
        "stage3": [args.root / "stage3/rep0_predictor", args.root / "stage3/rep0_predictor_2", args.root / "stage3/rep0_predictor_3"],
    }
    measured_roots = {phase: [args.root / phase / f"rep{i}" for i in range(1, 4)] for phase in PHASES}
    models = {"stage1": args.stage1_model, "stage2": args.stage2_model, "stage3": args.stage3_model}
    phases = []
    for phase in PHASES:
        model = models[phase]
        phases.append(
            aggregate_phase(
                phase,
                manifest["resource_profile"]["sha256"],
                predictor_roots[phase],
                measured_roots[phase],
                {"path": str(model), "sha256": content_sha256(model), "hash_scheme": "tree-v1" if model.is_dir() else "file-v1"},
                phase_prediction(prediction_contract, phase),
            )
        )
    semantic_binding = optional_file_binding(args.semantic_contract)
    input_bindings = {
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "rendered_manifest_sha256": manifest["manifest_sha256"],
        "prediction_contract": {"path": str(args.contract), "sha256": sha256(args.contract)},
        "history_index": {"path": str(args.history_index), "sha256": sha256(args.history_index)},
        "policy": {"path": str(args.policy), "sha256": sha256(args.policy)},
        "preflight_receipt": {"path": str(args.preflight_receipt), "sha256": sha256(args.preflight_receipt)},
        "resource_profile": {"identity": args.profile, "sha256": manifest["resource_profile"]["sha256"]},
    }
    if semantic_binding is not None:
        input_bindings["semantic_contract"] = semantic_binding
    report = {
        "schema_version": 1,
        "evidence_class": "infrastructure_calibration",
        "decision": "candidate",
        "manifest_sha256": manifest["manifest_sha256"],
        "input_bindings": input_bindings,
        "prediction_contract_sha256": sha256(args.contract),
        "prediction_contract_decision": prediction_contract.get("decision"),
        "queue_identity": args.queue_identity,
        "container_image_id": args.image_id,
        "validation_data": validation_data,
        "contract": {
            "warmup_repetitions": 1,
            "predictor_repetitions": {phase: len(predictor_roots[phase]) for phase in PHASES},
            "measured_repetitions": 3,
            "max_response_length": 8192,
            "validation_deadline_seconds": 1800,
            "val_max_samples": -1,
        },
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
