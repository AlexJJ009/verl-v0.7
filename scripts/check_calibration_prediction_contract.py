#!/usr/bin/env python3
"""Generate and verify dynamic calibration prediction contracts.

The module is intentionally CPU-only and importable by later assemblers/checkers.
It implements the ``stage123_history_conformal_v1`` contract from the experiment
execution reliability Goal plan.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_outcomes import CONTINUOUS_OUTCOMES, RATE_OUTCOMES

ALGORITHM_VERSION = "stage123_history_conformal_v1"
PHASES = ("stage1", "stage2", "stage3")
AUTHORIZATION_SCOPES = {
    "full": PHASES,
    "stage12_producer": ("stage1", "stage2"),
}
MIN_COHORT_SIZE = 6
MAX_COHORT_SIZE = 12
COVERAGE = 0.90
LOO_COVERAGE_MIN = 0.80
ELAPSED_WIDTH_LIMIT = 0.50
RSS_WIDTH_LIMIT = 0.25
GPU_IDLE_WIDTH_LIMIT = 0.75
ELAPSED_HARD_UPPER_SECONDS = 1800.0
ROUND_QUANT = Decimal("0.000001")

EXACT_MATCH_FIELDS = (
    "validation_dataset_hashes",
    "sampled_decoding_semantic_hash",
    "resource_profile_hash",
    "phase_topology_hash",
    "scorer_hash",
    "timeout_policy_hash",
    "max_response_length",
    "workload_descriptor_sha256",
    "outcome_schema_version",
)
METRIC_FIELDS = (*CONTINUOUS_OUTCOMES, *RATE_OUTCOMES, "all_gpu_idle_fraction_during_validation")


class ContractError(ValueError):
    """Raised when a contract cannot be generated or verified."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes with exactly one trailing newline."""
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        data = load_json(path)
        if not isinstance(data, dict):
            raise ContractError(f"{path}: manifest must be a JSON object")
        return data
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    renderer = Path(__file__).with_name("experiment_manifest.py")
    raw = subprocess.check_output(
        ["python3", str(renderer), "render", str(path), "--format", "json"],
        text=True,
    )
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ContractError(f"{path}: rendered manifest must be an object")
    return data


def _utc_second(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError(f"timestamp must be UTC RFC3339 whole seconds with Z: {value!r}")
    if "." in value:
        raise ContractError(f"timestamp must not include fractional seconds: {value!r}")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _number(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(f"{name}: expected finite number")
    return float(value)


def _floor6(value: float) -> float:
    return float(Decimal(str(value)).quantize(ROUND_QUANT, rounding=ROUND_FLOOR))


def _ceil6(value: float) -> float:
    return float(Decimal(str(value)).quantize(ROUND_QUANT, rounding=ROUND_CEILING))


def _round_interval(lower: float, upper: float) -> list[float]:
    return [_floor6(max(0.0, lower)), _ceil6(upper)]


def _metric_value(run: dict[str, Any], metric: str) -> float:
    metrics = run.get("metrics", run)
    return _number(metrics.get(metric), f"{run.get('run_id', '<unknown>')}:{metric}")


def _phase_manifest(manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    workloads = manifest.get("calibration_workloads")
    if isinstance(workloads, dict) and isinstance(workloads.get(phase), dict):
        return workloads[phase]
    phases = manifest.get("phases")
    if isinstance(phases, dict) and isinstance(phases.get(phase), dict):
        return phases[phase]
    if isinstance(phases, list):
        for item in phases:
            if isinstance(item, dict) and item.get("phase") == phase:
                return item
    return {}


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _expected_match_values(manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_doc = _phase_manifest(manifest, phase)
    semantic = _first_dict(
        manifest.get("semantics"), manifest.get("semantic_contract"), phase_doc.get("semantic_contract")
    )
    validation = manifest.get("validation_dataset_hashes") or semantic.get("validation_dataset_hashes")
    if validation is None and isinstance(manifest.get("validation_data"), dict):
        datasets = manifest["validation_data"].get("datasets")
        if isinstance(datasets, list):
            validation = {item.get("name"): item.get("sha256") for item in datasets if isinstance(item, dict)}
    resource_profile = _first_dict(manifest.get("resource_profile"), phase_doc.get("resource_profile"))
    timeout_policy = _first_dict(manifest.get("timeout_policy"), phase_doc.get("timeout_policy"))
    values = {
        "validation_dataset_hashes": validation,
        "sampled_decoding_semantic_hash": manifest.get("sampled_decoding_semantic_hash")
        or semantic.get("sampled_decoding_semantic_hash")
        or semantic.get("sha256"),
        "resource_profile_hash": manifest.get("resource_profile_hash") or resource_profile.get("sha256"),
        "phase_topology_hash": _first_dict(semantic.get("phase_topology_hashes")).get(phase)
        or phase_doc.get("phase_topology_hash")
        or phase_doc.get("topology_hash")
        or manifest.get("phase_topology_hash"),
        "scorer_hash": manifest.get("scorer_hash") or semantic.get("scorer_hash"),
        "timeout_policy_hash": manifest.get("timeout_policy_hash")
        or semantic.get("timeout_policy_hash")
        or timeout_policy.get("sha256"),
        "max_response_length": manifest.get("max_response_length")
        or phase_doc.get("max_response_length")
        or resource_profile.get("max_response_length")
        or manifest.get("MAX_RESPONSE_LENGTH"),
        "workload_descriptor_sha256": canonical_json_sha256(phase_doc),
        "outcome_schema_version": phase_doc.get("outcome_schema_version"),
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ContractError(f"{phase}: manifest missing exact-match fields: {', '.join(missing)}")
    return values


def _run_match_values(run: dict[str, Any]) -> dict[str, Any]:
    semantic = _first_dict(run.get("semantic_contract"))
    values = {key: run.get(key) for key in EXACT_MATCH_FIELDS}
    if values.get("sampled_decoding_semantic_hash") is None:
        values["sampled_decoding_semantic_hash"] = semantic.get("sampled_decoding_semantic_hash") or semantic.get(
            "sha256"
        )
    if values.get("scorer_hash") is None:
        values["scorer_hash"] = semantic.get("scorer_hash")
    return values


def _current_run_ids(manifest: dict[str, Any], history_index: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for source in (manifest, history_index):
        for key in ("current_run_ids", "current_calibration_run_ids", "acceptance_run_ids", "predictor_run_ids"):
            values = source.get(key)
            if isinstance(values, list):
                result.update(str(value) for value in values)
    return result


def phases_for_authorization_scope(authorization_scope: str) -> tuple[str, ...]:
    try:
        return AUTHORIZATION_SCOPES[authorization_scope]
    except KeyError as exc:
        raise ContractError(f"unsupported authorization scope: {authorization_scope}") from exc


def validate_history_snapshot(
    history_index: dict[str, Any],
    *,
    expected_phases: tuple[str, ...] | None = None,
) -> None:
    cutoff = history_index.get("cutoff_utc")
    _utc_second(cutoff)
    runs = history_index.get("runs")
    if not isinstance(runs, list):
        raise ContractError("history index must contain a runs list")
    if expected_phases is not None:
        phase_scope = history_index.get("phase_scope")
        if phase_scope != list(expected_phases):
            raise ContractError(f"history phase_scope mismatch: expected {list(expected_phases)}, got {phase_scope!r}")
        run_phases = [run.get("phase") for run in runs if isinstance(run, dict)]
        if set(run_phases) != set(expected_phases):
            raise ContractError(
                f"history run phase set mismatch: expected {list(expected_phases)}, got {sorted(set(run_phases))}"
            )
        for phase in expected_phases:
            count = run_phases.count(phase)
            if count != MIN_COHORT_SIZE:
                raise ContractError(
                    f"history must contain exactly {MIN_COHORT_SIZE} bootstrap runs for {phase}; got {count}"
                )
    seen: set[str] = set()
    seen_roots: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ContractError("history run entries must be objects")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ContractError("history run missing run_id")
        if run_id in seen:
            raise ContractError(f"duplicate history run_id: {run_id}")
        seen.add(run_id)
        _utc_second(run.get("completed_at"))
        if run.get("evidence_role") != "bootstrap_history":
            raise ContractError(f"{run_id}: history evidence_role must be bootstrap_history")
        phase = run.get("phase")
        artifact_root = Path(str(run.get("artifact_root", "")))
        parts = artifact_root.parts
        if len(parts) < 3 or parts[-3:-1] != ("bootstrap", phase) or not parts[-1].startswith("rep_"):
            raise ContractError(f"{run_id}: invalid bootstrap artifact_root")
        try:
            rep = int(parts[-1].removeprefix("rep_"))
        except ValueError as exc:
            raise ContractError(f"{run_id}: invalid bootstrap repetition root") from exc
        if rep not in range(6):
            raise ContractError(f"{run_id}: bootstrap root/phase/repetition mismatch")
        root_text = str(artifact_root.resolve())
        if root_text in seen_roots:
            raise ContractError(f"{run_id}: duplicate bootstrap artifact_root")
        seen_roots.add(root_text)
        bindings = run.get("artifact_bindings")
        if not isinstance(bindings, dict):
            raise ContractError(f"{run_id}: missing artifact bindings")
        expected_names = {"status", "resources", "metrics", "generation", "timeline"}
        if set(bindings) != expected_names:
            raise ContractError(f"{run_id}: incomplete artifact bindings")
        for name, binding in bindings.items():
            if not isinstance(binding, dict):
                raise ContractError(f"{run_id}: invalid {name} artifact binding")
            path = Path(str(binding.get("path", "")))
            try:
                path.resolve().relative_to(artifact_root.resolve())
            except ValueError as exc:
                raise ContractError(f"{run_id}: {name} artifact escapes artifact_root") from exc
            if not path.is_file() or file_sha256(path) != binding.get("sha256"):
                raise ContractError(f"{run_id}: {name} artifact hash mismatch")


def select_cohort(
    history_index: dict[str, Any],
    manifest: dict[str, Any],
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    validate_history_snapshot(history_index)
    cutoff = _utc_second(history_index["cutoff_utc"])
    expected = _expected_match_values(manifest, phase)
    current_ids = _current_run_ids(manifest, history_index)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for run in history_index["runs"]:
        run_id = str(run["run_id"])
        reason = ""
        if run_id in current_ids:
            raise ContractError(f"{run_id}: current acceptance/calibration run leaked into history")
        if run.get("phase") != phase:
            reason = "phase_mismatch"
        elif _utc_second(run["completed_at"]) > cutoff:
            reason = "after_cutoff"
        elif run.get("release_gate_passed") is not True:
            reason = "release_gate_not_passed"
        elif run.get("artifacts_readable", True) is not True or run.get("content_addressed", True) is not True:
            reason = "artifacts_not_content_addressed_readable"
        else:
            observed = _run_match_values(run)
            for key, value in expected.items():
                if observed.get(key) != value:
                    reason = f"{key}_mismatch"
                    break
        if reason:
            excluded.append({"run_id": run_id, "reason": reason})
        else:
            eligible.append(run)
    eligible.sort(key=lambda item: (item["completed_at"], item["run_id"]))
    return eligible[-MAX_COHORT_SIZE:], excluded


def conformal_interval(values: list[float]) -> dict[str, Any]:
    if len(values) < MIN_COHORT_SIZE:
        raise ContractError("at least six values are required")
    point = float(median(values))
    residuals: list[float] = []
    for index, value in enumerate(values):
        others = values[:index] + values[index + 1 :]
        residuals.append(abs(value - float(median(others))))
    residuals.sort()
    k = min(len(values), math.ceil((len(values) + 1) * COVERAGE))
    q = residuals[k - 1]
    lower, upper = _round_interval(point - q, point + q)
    return {
        "point": _floor6(point),
        "interval": [lower, upper],
        "loo_residuals": [_ceil6(value) for value in residuals],
        "finite_sample_rank": k,
        "q": _ceil6(q),
    }


def _loo_interval_for_training(values: list[float]) -> list[float]:
    point = float(median(values))
    residuals = []
    for index, value in enumerate(values):
        others = values[:index] + values[index + 1 :]
        residuals.append(abs(value - float(median(others))))
    residuals.sort()
    k = min(len(values), math.ceil((len(values) + 1) * COVERAGE))
    q = residuals[k - 1]
    return [max(0.0, point - q), point + q]


def leave_one_out_coverage(values: list[float]) -> float:
    covered = 0
    for index, value in enumerate(values):
        training = values[:index] + values[index + 1 :]
        lower, upper = _loo_interval_for_training(training)
        if lower <= value <= upper:
            covered += 1
    return covered / len(values)


def gpu_idle_interval(values: list[float]) -> dict[str, Any]:
    if any(value < 0 or value > 1 for value in values):
        raise ContractError("GPU idle values must be in [0, 1]")
    lower, upper = _round_interval(min(values) - 0.02, min(1.0, max(values) + 0.02))
    return {"interval": [lower, upper], "raw_min": _floor6(min(values)), "raw_max": _ceil6(max(values))}


def rate_interval(values: list[float], submitted_count: int) -> dict[str, Any]:
    if submitted_count <= 0 or any(value < 0 or value > 1 for value in values):
        raise ContractError("rate values/count are invalid")
    margin = 1.0 / submitted_count
    lower, upper = _round_interval(min(values) - margin, min(1.0, max(values) + margin))
    return {
        "interval": [lower, upper],
        "raw_min": _floor6(min(values)),
        "raw_max": _ceil6(max(values)),
        "margin": _ceil6(margin),
    }


def _width_ratio(interval: list[float]) -> float:
    midpoint = (interval[0] + interval[1]) / 2.0
    if midpoint <= 0:
        return math.inf if interval[1] > interval[0] else 0.0
    return (interval[1] - interval[0]) / midpoint


def predict_for_cohort(cohort: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cohort) < MIN_COHORT_SIZE:
        return {
            "status": "out_of_domain",
            "decision": "inconclusive",
            "failures": [f"fewer than {MIN_COHORT_SIZE} eligible prior runs"],
            "predictions": {},
        }
    continuous_values = {metric: [_metric_value(run, metric) for run in cohort] for metric in CONTINUOUS_OUTCOMES}
    rate_values = {metric: [_metric_value(run, metric) for run in cohort] for metric in RATE_OUTCOMES}
    idle_values = [_metric_value(run, "all_gpu_idle_fraction_during_validation") for run in cohort]
    submitted_counts = {int(_metric_value(run, "submitted_item_count")) for run in cohort}
    if len(submitted_counts) != 1:
        raise ContractError("cohort submitted item counts differ")
    submitted_count = submitted_counts.pop()
    continuous = {metric: conformal_interval(values) for metric, values in continuous_values.items()}
    rates = {metric: rate_interval(values, submitted_count) for metric, values in rate_values.items()}
    idle = gpu_idle_interval(idle_values)
    coverage = {metric: _floor6(leave_one_out_coverage(values)) for metric, values in continuous_values.items()}
    predictions = {metric: {**doc, "loo_coverage": coverage[metric]} for metric, doc in continuous.items()}
    predictions.update(rates)
    predictions["all_gpu_idle_fraction_during_validation"] = idle
    failures: list[str] = []
    status = "deployable"
    decision = "deployable"
    if continuous["validation_elapsed_seconds"]["interval"][1] >= ELAPSED_HARD_UPPER_SECONDS:
        failures.append("elapsed upper bound is at or above 1800 seconds")
        status = "runtime_risk"
        decision = "blocked"
    elif (
        any(
            _width_ratio(continuous[metric]["interval"])
            > (RSS_WIDTH_LIMIT if metric == "peak_rss_gib" else ELAPSED_WIDTH_LIMIT)
            for metric in CONTINUOUS_OUTCOMES
        )
        or any(doc["interval"][1] - doc["interval"][0] > 0.25 for doc in rates.values())
        or idle["interval"][1] - idle["interval"][0] > GPU_IDLE_WIDTH_LIMIT
    ):
        failures.append("noninformative prediction interval")
        status = "noninformative"
        decision = "inconclusive"
    elif min(coverage.values()) < LOO_COVERAGE_MIN:
        failures.append("leave-one-run-out empirical coverage below 0.80")
        status = "insufficient_coverage"
        decision = "inconclusive"
    return {"status": status, "decision": decision, "failures": failures, "predictions": predictions}


def build_prediction_contract(
    manifest: dict[str, Any],
    history_index: dict[str, Any],
    *,
    manifest_sha256: str | None = None,
    history_index_sha256: str | None = None,
    authorization_scope: str = "full",
) -> dict[str, Any]:
    phases = phases_for_authorization_scope(authorization_scope)
    validate_history_snapshot(history_index, expected_phases=phases)
    phase_results = []
    for phase in phases:
        cohort, excluded = select_cohort(history_index, manifest, phase)
        prediction = predict_for_cohort(cohort)
        phase_results.append(
            {
                "phase": phase,
                "status": prediction["status"],
                "decision": prediction["decision"],
                "failures": prediction["failures"],
                "eligible_run_ids": [run["run_id"] for run in cohort],
                "excluded_run_ids": excluded,
                "cohort_size": len(cohort),
                "features": _phase_manifest(manifest, phase),
                "predictions": prediction["predictions"],
            }
        )
    decisions = {phase["decision"] for phase in phase_results}
    decision = "blocked" if "blocked" in decisions else "inconclusive" if "inconclusive" in decisions else "deployable"
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "authorization_scope": authorization_scope,
        "decision": decision,
        "history_query": {
            "cutoff_utc": history_index["cutoff_utc"],
            "selection_order": ["completed_at_ascending", "run_id_ascending", "latest_12"],
            "min_cohort_size": MIN_COHORT_SIZE,
            "max_cohort_size": MAX_COHORT_SIZE,
            "current_run_ids": sorted(_current_run_ids(manifest, history_index)),
        },
        "hashes": {
            "manifest_sha256": manifest_sha256 or canonical_json_sha256(manifest),
            "history_index_sha256": history_index_sha256 or canonical_json_sha256(history_index),
        },
        "parameters": {
            "coverage": COVERAGE,
            "finite_sample_rank": "min(n,ceil((n+1)*0.90))",
            "rounding": "outward_6_decimals",
            "elapsed_width_limit_midpoint_fraction": ELAPSED_WIDTH_LIMIT,
            "rss_width_limit_midpoint_fraction": RSS_WIDTH_LIMIT,
            "gpu_idle_width_limit": GPU_IDLE_WIDTH_LIMIT,
            "loo_coverage_min": LOO_COVERAGE_MIN,
            "elapsed_hard_upper_seconds": ELAPSED_HARD_UPPER_SECONDS,
        },
        "phases": phase_results,
    }


def verify_prediction_contract(
    contract: dict[str, Any],
    manifest: dict[str, Any],
    history_index: dict[str, Any],
    *,
    manifest_sha256: str | None = None,
    history_index_sha256: str | None = None,
    authorization_scope: str = "full",
) -> dict[str, Any]:
    failures: list[str] = []
    decision = "deployable"
    try:
        expected = build_prediction_contract(
            manifest,
            history_index,
            manifest_sha256=manifest_sha256,
            history_index_sha256=history_index_sha256,
            authorization_scope=authorization_scope,
        )
    except ContractError as exc:
        return {"ok": False, "decision": "blocked", "failures": [str(exc)]}
    if contract != expected:
        failures.append("prediction contract does not match manifest/history recomputation")
        decision = "blocked"
    for phase in expected["phases"]:
        if phase["decision"] == "blocked":
            decision = "blocked"
        elif decision != "blocked" and phase["decision"] == "inconclusive":
            decision = "inconclusive"
    if contract.get("algorithm_version") != ALGORITHM_VERSION:
        failures.append("algorithm version mismatch")
        decision = "blocked"
    return {
        "ok": decision == "deployable" and not failures,
        "decision": decision,
        "failures": failures,
        "expected": expected,
    }


def write_contract(path: Path, contract: dict[str, Any]) -> str:
    data = canonical_json_bytes(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history-index", type=Path, required=True)
    parser.add_argument(
        "--authorization-scope",
        choices=tuple(AUTHORIZATION_SCOPES),
        default="full",
    )
    parser.add_argument("--write", action="store_true", help="write a missing or stale contract")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    history_index = load_json(args.history_index)
    manifest_hash = file_sha256(args.manifest)
    history_hash = file_sha256(args.history_index)
    try:
        generated = build_prediction_contract(
            manifest,
            history_index,
            manifest_sha256=manifest_hash,
            history_index_sha256=history_hash,
            authorization_scope=args.authorization_scope,
        )
        if args.write or not args.contract.exists():
            contract_hash = write_contract(args.contract, generated)
            print(
                json.dumps(
                    {"ok": True, "decision": generated["decision"], "contract_sha256": contract_hash},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if generated["decision"] == "deployable" else 1
        contract = load_json(args.contract)
        result = verify_prediction_contract(
            contract,
            manifest,
            history_index,
            manifest_sha256=manifest_hash,
            history_index_sha256=history_hash,
            authorization_scope=args.authorization_scope,
        )
        result = copy.deepcopy(result)
        result.pop("expected", None)
        result["contract_sha256"] = file_sha256(args.contract)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    except ContractError as exc:
        print(json.dumps({"ok": False, "decision": "blocked", "failures": [str(exc)]}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
