#!/usr/bin/env python3
"""Rebind accepted Stage123 calibration when the capacity plane is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RUN_IDS = ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]
CAPACITY_PROFILE_FIELDS = (
    "MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH",
    "ROLLOUT_MAX_MODEL_LEN",
    "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
    "LOG_PROB_MAX_TOKEN_LEN_PER_GPU",
    "ACTOR_PPO_MAX_TOKEN_LEN",
    "GENERATION_MICRO_BATCH_SIZE",
    "LOG_PROB_MICRO_BATCH_SIZE",
    "ROLLOUT_GPU_MEMORY_UTILIZATION",
    "TRAIN_PROMPT_BSZ",
    "ROLLOUT_N",
    "TRAIN_PROMPT_MINI_BSZ",
    "VAL_N",
    "CODE_REWARD_NUM_WORKERS",
    "CODE_REWARD_MAX_CONCURRENCY_PER_WORKER",
    "ROLLOUT_AGENT_NUM_WORKERS",
    "CODE_REWARD_TIMEOUT",
    "CODE_REWARD_MANAGER_TIMEOUT",
    "CODE_REWARD_STDIN_CASE_TIMEOUT",
    "CODE_REWARD_EXEC_MAX_AS_MB",
    "BIGCODEBENCH_MAX_AS_LIMIT",
    "BIGCODEBENCH_MAX_DATA_LIMIT",
    "BIGCODEBENCH_MAX_STACK_LIMIT",
    "LCB_SUBPROCESS_TIMEOUT",
    "RAY_memory_usage_threshold",
    "RAY_memory_monitor_refresh_ms",
    "OMP_NUM_THREADS",
    "NGPUS_PER_NODE",
    "ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def render_manifest(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", str(path), "--format", "json"],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def profile_snapshot(path: Path) -> tuple[dict[str, str], str]:
    output = subprocess.check_output(
        [
            "bash",
            "-lc",
            f"source {path}; stage123_profile_snapshot; printf '__HASH__=%s\\n' \"$(stage123_profile_hash)\"",
        ],
        cwd=ROOT,
        text=True,
    )
    values: dict[str, str] = {}
    full_hash = ""
    for line in output.splitlines():
        key, value = line.split("=", 1)
        if key == "__HASH__":
            full_hash = value
        else:
            values[key] = value
    if not full_hash:
        raise ValueError("profile hash missing")
    return values, full_hash


def git_file(repo: Path, commit: str, relative_path: str, output: Path) -> None:
    content = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{relative_path}"])
    output.write_bytes(content)


def workload_shape(manifest: dict[str, Any]) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for phase, workload in manifest["calibration_workloads"].items():
        shapes[phase] = {
            "phase": workload["phase"],
            "rollout_model_parameter_count_sum": workload["rollout_model_parameter_count_sum"],
            "rollout_model_parameter_counts": workload["rollout_model_parameter_counts"],
            "tokenizer": workload["tokenizer"],
            "validation_submitted_prompt_count": workload["validation_eligibility"]["submitted_prompt_count"],
            "validation_dataset_counts": {item["name"]: item["row_count"] for item in workload["datasets"]},
            "calibration_proxy_parameter_count": workload.get("calibration_proxy", {}).get(
                "rollout_model_parameter_count"
            ),
        }
    return shapes


def capacity_snapshot(profile: dict[str, str], manifest: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in CAPACITY_PROFILE_FIELDS if field not in profile]
    if missing:
        raise ValueError(f"capacity profile fields missing: {missing}")
    return {
        "profile": {field: profile[field] for field in CAPACITY_PROFILE_FIELDS},
        "models": {
            "base_model": manifest["paths"]["base_model"],
            "stage1_init_model": manifest["paths"]["stage1_init_model"],
        },
        "phase_topology_hashes": manifest["semantics"]["phase_topology_hashes"],
        "stage2_topology": manifest["semantics"]["stage2_topology"],
        "workload_shapes": workload_shape(manifest),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            result.update(flatten(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def diff_snapshots(source: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    source_flat = flatten(source)
    candidate_flat = flatten(candidate)
    return [
        {"field": field, "source": source_flat.get(field), "candidate": candidate_flat.get(field)}
        for field in sorted(set(source_flat) | set(candidate_flat))
        if source_flat.get(field) != candidate_flat.get(field)
    ]


def validate_source_calibration(calibration: dict[str, Any]) -> None:
    if calibration.get("decision") != "passed":
        raise ValueError("source calibration is not passed")
    if calibration.get("workload_identity", {}).get("run_ids") != PRIMARY_RUN_IDS:
        raise ValueError("source calibration run set mismatch")
    phases = calibration.get("phase_evidence")
    if not isinstance(phases, list) or [item.get("phase") for item in phases] != ["stage1", "stage2", "stage3"]:
        raise ValueError("source calibration phase set mismatch")
    for phase in phases:
        if phase.get("status") != "passed" or not phase.get("repetitions"):
            raise ValueError(f"source calibration phase is not passed: {phase.get('phase')}")
        for repetition in phase["repetitions"]:
            if repetition.get("training_steps") != 0 or repetition.get("status") != "passed":
                raise ValueError("source calibration contains training or failed repetition")
            if repetition.get("cleanup", {}).get("resources_released") is not True:
                raise ValueError("source calibration did not release resources")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--source-recipe-gitlink", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--implementation-tree-sha256", required=True)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    args = parser.parse_args()

    source_calibration = load_json(args.source_calibration)
    validate_source_calibration(source_calibration)
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary = Path(temporary_dir)
        source_profile_path = temporary / "source_profile.sh"
        source_manifest_path = temporary / "source_manifest.yaml"
        git_file(
            ROOT / "recipe",
            args.source_recipe_gitlink,
            "on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh",
            source_profile_path,
        )
        git_file(
            ROOT / "recipe",
            args.source_recipe_gitlink,
            "on_policy_wdl_sft/experiment_manifest/stage123.yaml",
            source_manifest_path,
        )
        source_profile, source_profile_hash = profile_snapshot(source_profile_path)
        source_manifest = render_manifest(source_manifest_path)

    if source_profile_hash != source_calibration.get("resource_profile_sha256"):
        raise ValueError("source calibration profile binding mismatch")
    if source_manifest["manifest_sha256"] != source_calibration.get("manifest_sha256"):
        raise ValueError("source calibration manifest binding mismatch")

    candidate_profile, candidate_profile_hash = profile_snapshot(args.candidate_profile)
    candidate_manifest = render_manifest(args.candidate_manifest)
    if candidate_profile_hash != candidate_manifest["resource_profile"]["sha256"]:
        raise ValueError("candidate profile does not match candidate manifest")

    source_capacity = capacity_snapshot(source_profile, source_manifest)
    candidate_capacity = capacity_snapshot(candidate_profile, candidate_manifest)
    differences = diff_snapshots(source_capacity, candidate_capacity)
    decision = "applicable" if not differences else "requires_gpu_requalification"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "schema_version": 1,
        "result_type": "stage123_calibration_applicability",
        "decision": decision,
        "source_calibration_result_sha256": file_sha256(args.source_calibration),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "source_resource_profile_sha256": source_profile_hash,
        "candidate_resource_profile_sha256": candidate_profile_hash,
        "source_capacity_sha256": canonical_sha256(source_capacity),
        "candidate_capacity_sha256": canonical_sha256(candidate_capacity),
        "capacity_differences": differences,
        "implementation_tree_sha256": args.implementation_tree_sha256,
        "evidence_commit": args.evidence_commit,
        "plan_sha256": args.plan_sha256,
        "authorization_decision_id": args.decision_id,
        "completed_at": now,
    }
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision != "applicable":
        print(json.dumps(report, sort_keys=True))
        return 2

    derived = dict(source_calibration)
    derived.update(
        {
            "manifest_sha256": candidate_manifest["manifest_sha256"],
            "resource_profile_sha256": candidate_profile_hash,
            "implementation_tree_sha256": args.implementation_tree_sha256,
            "evidence_commit": args.evidence_commit,
            "workload_identity": {
                "run_ids": PRIMARY_RUN_IDS,
                "source_calibration_result_sha256": report["source_calibration_result_sha256"],
                "calibration_applicability_sha256": file_sha256(args.report_output),
            },
            "authorization_identity": {
                "decision_id": args.decision_id,
                "plan_version": 18,
                "plan_sha256": args.plan_sha256,
                "mode": "resource_neutral_calibration_applicability",
            },
            "started_at": now,
            "completed_at": now,
            "calibration_applicability": report,
        }
    )
    args.calibration_output.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"ok": True, "decision": decision, "report_sha256": file_sha256(args.report_output)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
