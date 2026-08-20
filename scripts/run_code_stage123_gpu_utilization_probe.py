#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Probe the admitted Code Stage123 matrix at 8K context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55)
PROFILE_SCRIPT = ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh"
VALIDATION_FILES = {
    "HumanEval+": Path("/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet"),
    "MBPP+": Path("/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet"),
    "LiveCodeBench": Path(
        "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidates(raw: str) -> list[float]:
    candidates = [float(value) for value in raw.split(",")]
    if candidates != sorted(set(candidates)) or any(value not in DEFAULT_CANDIDATES for value in candidates):
        raise SystemExit("candidates must be an ordered subset of 0.35,0.40,0.45,0.50,0.55")
    return candidates


def profile_hash(utilization: float) -> str:
    command = (
        f"export ROLLOUT_GPU_MEMORY_UTILIZATION={utilization:.2f}; "
        "export ACTOR_CALCULATE_ENTROPY=False CALCULATE_ENTROPY=False; "
        f"source {PROFILE_SCRIPT} >/dev/null; stage123_profile_hash"
    )
    return subprocess.check_output(["bash", "-lc", command], text=True).strip()


def assert_tmux_and_idle_gpus(max_utilization: int, max_memory_used_mib: int) -> dict[str, Any]:
    if not os.environ.get("TMUX"):
        raise SystemExit("non-dry-run GPU probe must run inside tmux")
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if compute:
        raise SystemExit(f"GPU compute processes are active: {compute}")
    rows = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    samples = []
    for row in rows:
        index, utilization, memory_used = (int(value.strip()) for value in row.split(","))
        samples.append({"index": index, "utilization_gpu_percent": utilization, "memory_used_mib": memory_used})
    if len(samples) != 8:
        raise SystemExit("GPU probe requires exactly 8 visible GPUs")
    if any(item["utilization_gpu_percent"] > max_utilization for item in samples):
        raise SystemExit("GPU utilization is not idle enough for probing")
    if any(item["memory_used_mib"] > max_memory_used_mib for item in samples):
        raise SystemExit("GPU memory is not idle enough for probing")
    return {"compute_processes": [], "gpus": samples}


def dry_run_plan(args: argparse.Namespace, candidates: list[float]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "mode": "dry_run",
        "candidates": candidates,
        "validation_scope": "full_code3",
        "validation_n": 3,
        "zero_step_phases": ["stage1"],
        "throughput_arms": ["stage1-beta01", "stage2-m2kl-beta01"],
        "optimizer_steps_per_throughput_arm": 1,
        "max_response_length": 8192,
        "entropy": {"actor": False, "joint": False},
        "tmux_required": True,
        "idle_gpu_guard": {
            "max_utilization_percent": args.idle_max_utilization,
            "max_memory_used_mib": args.idle_max_memory_used_mib,
        },
        "scratch_root": str(args.scratch_root),
        "real_run_manifest_requirement": "admitted step20 manifest plus frozen Model1 and dataset receipts",
        "selection_policy": "highest requested candidate passing one full validation and positive-reward one-step single/joint paths",
    }


def assert_cot_v3_probe_inputs(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if manifest.get("task") != "code" or manifest.get("model1_selection_policy", {}).get("selected_step") != 20:
        raise SystemExit("GPU probe requires the Code step-20 CoT-v3 manifest")
    selection_path = Path(manifest["paths"]["model1_selection"])
    receipt_path = Path(manifest["paths"]["dataset_receipt"])
    selection = json.loads(selection_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    model = Path(selection["identity"]["model_path"])
    if selection.get("selected_step") != 20 or "cold_start_cotmask_v3" not in str(model) or not model.is_dir():
        raise SystemExit("GPU probe refuses legacy answer-only Model1")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("overlap_policy", {}).get("cold_start_vs_stage1_stage2_stage3") != "pairwise_disjoint"
    ):
        raise SystemExit("GPU probe requires a Cold Start-disjoint Stage123 dataset receipt")
    if any(not path.is_file() for path in VALIDATION_FILES.values()):
        raise SystemExit("full Code-3 validation files are incomplete")
    return model, selection, receipt


def run_phase(
    manifest: Path,
    run_id: str,
    utilization: float,
    output_root: Path,
    mode: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True)
    result_path = output_root / "result.json"
    command = [
        "/data-1/verl07/run_train.sh",
        "python3",
        "/workspace/verl/scripts/code_stage123_probe_phase.py",
        "--manifest",
        str(manifest).replace(str(ROOT), "/workspace/verl"),
        "--run-id",
        run_id,
        "--mode",
        mode,
        "--output-root",
        str(output_root),
        "--utilization",
        f"{utilization:.2f}",
        "--result",
        str(result_path),
    ]
    environment = dict(os.environ)
    environment["REPO_HOST"] = str(ROOT)
    started = time.monotonic()
    timed_out = False
    with (output_root / "driver.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait(timeout=60)
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    result.update(
        {
            "run_id": run_id,
            "mode": mode,
            "driver_returncode": returncode,
            "driver_timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "driver_log": str(output_root / "driver.log"),
        }
    )
    result["status"] = "passed" if returncode == 0 and not timed_out and result.get("status") == "passed" else "failed"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def reused_validation_result(path: Path, utilization: float, manifest_path: Path) -> dict[str, Any]:
    reused = json.loads(path.read_text())
    if reused.get("status") != "passed" or reused.get("validation_complete") is not True:
        raise SystemExit("reused validation result is not passing")
    source_candidate = next((part for part in path.parts if re.fullmatch(r"mem\d{2}", part)), None)
    if source_candidate is None:
        raise SystemExit("reused validation result has no candidate utilization in its path")
    source_utilization = int(source_candidate.removeprefix("mem")) / 100
    if source_utilization < utilization:
        raise SystemExit("reused validation result must come from an equal or higher utilization")
    source_root = next((parent for parent in path.parents if (parent / "input-manifest.yaml").is_file()), None)
    if source_root is None:
        raise SystemExit("reused validation result is not bound to a source manifest")
    source_manifest = yaml.safe_load((source_root / "input-manifest.yaml").read_text())
    current_manifest = yaml.safe_load(manifest_path.read_text())
    source_manifest["resources"]["rollout_gpu_memory_utilization"] = utilization
    if source_manifest != current_manifest:
        raise SystemExit("reused validation source manifest differs beyond utilization")
    reused["reused_from"] = str(path)
    reused["reuse_validation_source_manifest"] = str(source_root / "input-manifest.yaml")
    reused["reuse_validation_source_manifest_sha256"] = sha256(source_root / "input-manifest.yaml")
    reused["reuse_validation_source_utilization"] = source_utilization
    reused["reused_for_utilization"] = utilization
    return reused


def reused_throughput_result(
    probe_root: Path,
    utilization: float,
    phase: str,
    run_id: str,
    manifest_path: Path,
) -> dict[str, Any]:
    source_manifest = probe_root / "input-manifest.yaml"
    if not source_manifest.is_file() or source_manifest.read_bytes() != manifest_path.read_bytes():
        raise SystemExit("reused throughput probe is not bound to the current manifest")
    result_path = probe_root / f"mem{int(utilization * 100):02d}" / "one-step" / phase / "result.json"
    if not result_path.is_file():
        raise SystemExit(f"reused throughput result is missing: {result_path}")
    reused = json.loads(result_path.read_text())
    observed = reused.get("observed_training_metrics", {})
    positive_loss = float(observed.get("actor/wdl_sft_loss_positive", 0.0))
    grad_norm = float(observed.get("actor/grad_norm", 0.0))
    if (
        reused.get("status") != "passed"
        or reused.get("run_id") != run_id
        or reused.get("mode") != "train"
        or reused.get("training_contract_complete") is not True
        or reused.get("runtime_contract_complete") is not True
        or int(reused.get("optimizer_steps", 0)) < 1
        or int(observed.get("wdl_sft/n_correct", 0)) <= 0
        or not math.isfinite(positive_loss)
        or positive_loss <= 0
        or not math.isfinite(grad_norm)
        or grad_norm <= 0
        or reused.get("formal_checkpoint_files")
    ):
        raise SystemExit(f"reused throughput result is not admissible: {result_path}")
    reused["reused_from"] = str(result_path)
    return reused


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--candidates", default=",".join(f"{value:.2f}" for value in DEFAULT_CANDIDATES))
    parser.add_argument("--validation-timeout-seconds", type=int, default=10800)
    parser.add_argument("--throughput-timeout-seconds", type=int, default=7200)
    parser.add_argument("--minimum-headroom-mib", type=int, default=512)
    parser.add_argument("--reuse-validation-result", type=Path)
    parser.add_argument("--reuse-throughput-root", type=Path)
    parser.add_argument("--idle-max-utilization", type=int, default=5)
    parser.add_argument("--idle-max-memory-used-mib", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    candidates = parse_candidates(args.candidates)
    if not str(args.scratch_root).startswith("/data-1/tmp/verl_agent_scratch/"):
        raise SystemExit("scratch root must be under /data-1/tmp/verl_agent_scratch")
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, candidates), indent=2, sort_keys=True))
        return 0
    if not args.manifest.is_file():
        raise SystemExit("real GPU probe requires the step-20 manifest")
    manifest = yaml.safe_load(args.manifest.read_text())
    assert_cot_v3_probe_inputs(manifest)
    idle_snapshot = assert_tmux_and_idle_gpus(args.idle_max_utilization, args.idle_max_memory_used_mib)
    run_root = args.scratch_root / ("probe-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_root.mkdir(parents=True)
    manifest_snapshot = run_root / "input-manifest.yaml"
    manifest_snapshot.write_bytes(args.manifest.read_bytes())
    validation_run = ("stage1", "b0-stage1")
    throughput_runs = {
        "stage1-beta01": "b01-stage1",
        "stage2-m2kl-beta01": "b01-stage2-m2kl",
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "result_type": "code_stage123_gpu_utilization_probe",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "manifest_snapshot": str(manifest_snapshot),
        "manifest_snapshot_sha256": sha256(manifest_snapshot),
        "validation_scope": "full_code3",
        "validation_n": 3,
        "max_response_length": 8192,
        "idle_snapshot": idle_snapshot,
        "candidates": [],
        "selected_rollout_gpu_memory_utilization": None,
        "selection_policy": "highest_requested_candidate_with_full_validation_and_positive_reward_single_joint_one_step",
        "minimum_required_gpu_headroom_mib": args.minimum_headroom_mib,
        "run_root": str(run_root),
    }
    for utilization in sorted(candidates, reverse=True):
        candidate_root = run_root / f"mem{int(utilization * 100):02d}"
        candidate_root.mkdir()
        phase, run_id = validation_run
        if args.reuse_validation_result:
            phase_results = [reused_validation_result(args.reuse_validation_result, utilization, args.manifest)]
        else:
            phase_results = [
                run_phase(
                    args.manifest,
                    run_id,
                    utilization,
                    candidate_root / "zero-step" / phase,
                    "validation",
                    args.validation_timeout_seconds,
                )
            ]
        throughput_results = []
        if phase_results[0]["status"] == "passed":
            for phase, run_id in throughput_runs.items():
                result = (
                    reused_throughput_result(
                        args.reuse_throughput_root,
                        utilization,
                        phase,
                        run_id,
                        args.manifest,
                    )
                    if args.reuse_throughput_root
                    else run_phase(
                        args.manifest,
                        run_id,
                        utilization,
                        candidate_root / "one-step" / phase,
                        "train",
                        args.throughput_timeout_seconds,
                    )
                )
                throughput_results.append(result)
                if result["status"] != "passed":
                    break
        headrooms = [
            int(result.get("resources", {}).get("minimum_gpu_headroom_mib") or 0)
            for result in [*phase_results, *throughput_results]
        ]
        passed = (
            len(phase_results) == 1
            and len(throughput_results) == 2
            and all(item["status"] == "passed" for item in [*phase_results, *throughput_results])
            and min(headrooms, default=0) >= args.minimum_headroom_mib
        )
        candidate = {
            "rollout_gpu_memory_utilization": utilization,
            "resource_profile_sha256": profile_hash(utilization),
            "zero_step": phase_results,
            "one_step": throughput_results,
            "minimum_observed_gpu_headroom_mib": min(headrooms, default=0),
            "status": "passed" if passed else "failed",
        }
        (candidate_root / "candidate-report.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
        report["candidates"].append(candidate)
        if passed:
            report["selected_rollout_gpu_memory_utilization"] = utilization
            break
        else:
            report["first_failed_candidate"] = utilization
    report["status"] = "passed" if report["selected_rollout_gpu_memory_utilization"] is not None else "failed"
    report_path = run_root / "gpu-utilization-probe-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    latest = args.scratch_root / "latest-gpu-utilization-probe.json"
    latest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "report": str(report_path),
                "report_sha256": sha256(report_path),
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"ok": report["status"] == "passed", "report": str(report_path)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
