#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE_SCRIPT = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh"
WORKLOAD = Path("/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/code_validation_16_16_32.parquet")
WORKLOAD_MANIFEST = WORKLOAD.with_suffix(".manifest.json")
REQUIRED_METRICS = {
    "timing_s/testing",
    "val-core/HumanEval+/acc/pass@1",
    "val-core/MBPP+/acc/pass@1",
    "val-core/LiveCodeBench/acc/pass@1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_manifest(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", str(path), "--format", "json"],
        text=True,
    )
    return json.loads(output)


def split_workload(root: Path) -> dict[str, Path]:
    import pandas as pd

    expected = {"HumanEval+": 16, "MBPP+": 16, "LiveCodeBench": 32}
    frame = pd.read_parquet(WORKLOAD)
    actual = frame["data_source"].value_counts().to_dict()
    if actual != expected or sha256(WORKLOAD) != "c3eaf3374661fba71d1132f0de7a8dbdbd3d90295d4fabeb77b5e9dd7c221608":
        raise RuntimeError(f"calibration workload mismatch: {actual}")
    outputs: dict[str, Path] = {}
    for name in expected:
        path = root / "workload" / f"{name.lower().replace('+', '_plus')}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[frame["data_source"] == name].to_parquet(path, index=False)
        outputs[name] = path
    return outputs


def phase_environment(rendered: dict[str, Any], phase: str, repetition: int, output: Path, splits: dict[str, Path]) -> dict[str, str]:
    runs = {item["phase"]: item for item in rendered["runs"]}
    stage2_sources = {item["role"]: item for item in rendered["calibration_workloads"]["stage2"]["model_sources"]}
    stage1_source = json.loads((Path(stage2_sources["model2"]["path"]) / "stage1_source.json").read_text())
    stage3_workload = rendered["calibration_workloads"]["stage3"]
    stage3_source = stage3_workload["model_sources"][0]
    stage3_model = Path(stage3_source["path"])
    proxy_kind = "materialized_manifest_source"
    if not stage3_model.is_dir():
        proxy = stage3_workload.get("calibration_proxy")
        if stage3_source.get("state") != "pending" or not proxy:
            raise RuntimeError("stage3 pending source has no explicit calibration proxy")
        stage3_model = Path(proxy["path"])
        if not stage3_model.is_dir() or proxy["rollout_model_parameter_count"] != stage3_workload["rollout_model_parameter_count_sum"]:
            raise RuntimeError("stage3 calibration proxy identity mismatch")
        proxy_kind = proxy["purpose"]
    offset = ((0 if phase == "stage2" else 3) + repetition - 1) * 100
    env = {
        "QWEN3_1P7B_MODEL_PATH": rendered["paths"]["base_model"],
        "CALIBRATION_HUMANEVAL_PLUS_FILE": str(splits["HumanEval+"]),
        "CALIBRATION_MBPP_PLUS_FILE": str(splits["MBPP+"]),
        "CALIBRATION_LIVE_CODE_BENCH_FILE": str(splits["LiveCodeBench"]),
        "CALIBRATION_OUTPUT_ROOT": str(output),
        "CALIBRATION_TOTAL_TRAINING_STEPS": "0",
        "CALIBRATION_OPTIMIZER_ENABLED": "false",
        "CALIBRATION_RAY_WORKER_PORT_MIN": str(21000 + offset),
        "CALIBRATION_RAY_WORKER_PORT_MAX": str(21099 + offset),
        "CALIBRATION_RAY_HEAD_PORT": str(22000 + offset),
        "CALIBRATION_TCPSTORE_PORT_MIN": str(35000 + offset),
        "CALIBRATION_TCPSTORE_PORT_MAX": str(35099 + offset),
        "CALIBRATION_STAGE1_CKPT_DIR": stage1_source["source_checkpoint"],
        "CALIBRATION_STAGE1_MODEL2": stage2_sources["model2"]["path"],
        "CALIBRATION_STAGE1_RUN_PREFIX": stage1_source["stage1_run_prefix"],
        "CALIBRATION_STAGE1_HANDOFF_STEP": str(stage1_source["handoff_step"]),
        "CALIBRATION_TRAIN_FILE": runs["stage2"]["train_file"],
        "CALIBRATION_STAGE3_MODEL_PATH": str(stage3_model),
        "CALIBRATION_STAGE3_PROXY_KIND": proxy_kind,
    }
    return env


def sample_resources(process: subprocess.Popen[str], output: Path, interval: float = 1.0) -> None:
    samples = idle = peak = 0
    while process.poll() is None:
        try:
            values = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
            ).splitlines()
            values = [int(value.strip()) for value in values]
            if len(values) == 8:
                samples += 1
                idle += int(all(value <= 2 for value in values))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        for candidate in (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.peak")):
            try:
                peak = max(peak, int(candidate.read_text().strip()))
            except (OSError, ValueError):
                pass
        time.sleep(interval)
    write_json(output, {
        "schema_version": 1,
        "sample_interval_seconds": interval,
        "gpu_idle_threshold_pct": 2,
        "gpu_sample_count": samples,
        "gpu_idle_sample_count": idle,
        "gpu_wait_fraction": idle / samples if samples else None,
        "peak_rss_gib": peak / (1024 ** 3) if peak else None,
        "memory_source": "calibration_container_cgroup_v2",
    })


def read_metrics(output: Path) -> tuple[dict[str, Any], list[str]]:
    files = sorted((output / "logs" / "metrics").glob("**/*.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            try:
                data = json.loads(line).get("data", {})
            except json.JSONDecodeError:
                continue
            if REQUIRED_METRICS <= data.keys():
                return data, [str(item) for item in files]
    return {}, [str(item) for item in files]


def generation_summary(paths: list[str]) -> tuple[int, int]:
    count = truncated = 0
    for path in paths:
        for line in Path(path).read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            status = str(value.get("code_reward_status", "")).lower()
            truncated += int(status in {"truncated", "max_length"} or value.get("finish_reason") == "length")
    return count, truncated


def owned_cleanup(output: Path, process: subprocess.Popen[str]) -> dict[str, Any]:
    ray_root = output / "ray"
    matches = []
    result = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False)
    for line in result.stdout.splitlines():
        if str(ray_root) in line:
            matches.append(line.strip())
    return {
        "resources_released": process.poll() is not None and not matches,
        "child_process_released": process.poll() is not None,
        "owned_ray_root": str(ray_root),
        "owned_ray_processes": matches,
    }


def run_repetition(rendered: dict[str, Any], phase: str, repetition: int, root: Path, splits: dict[str, Path], timeout: int) -> dict[str, Any]:
    rep_root = root / "runs" / phase / f"rep{repetition}"
    output = rep_root / "output"
    log_path = rep_root / "host.log"
    resources_path = rep_root / "resources.json"
    env_delta = phase_environment(rendered, phase, repetition, output, splits)
    write_json(rep_root / "launch.json", {"phase": phase, "repetition": repetition, "environment": env_delta})
    start = time.time()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["bash", str(PHASE_SCRIPT), phase],
            cwd=ROOT,
            env={**os.environ, **env_delta},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        sampler = threading.Thread(target=sample_resources, args=(process, resources_path), daemon=True)
        sampler.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        sampler.join(timeout=5)
    metrics, metric_files = read_metrics(output)
    checkpoint_files = [str(path) for path in (output / "checkpoints").glob("**/*") if path.is_file()]
    generations = [str(path) for path in (output / "logs" / "validation").glob("**/*.jsonl")]
    generation_count, truncated_count = generation_summary(generations)
    cleanup = owned_cleanup(output, process)
    elapsed = time.time() - start
    status = "passed" if returncode == 0 and REQUIRED_METRICS <= metrics.keys() and not checkpoint_files and cleanup["resources_released"] else "failed"
    value = {
        "schema_version": 1,
        "phase": phase,
        "repetition": repetition,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "training_steps": 0,
        "optimizer_enabled": False,
        "formal_checkpoint_files": checkpoint_files,
        "metrics_files": metric_files,
        "validation_generation_files": generations,
        "metrics": {
            "validation_elapsed_seconds": metrics.get("timing_s/testing"),
            **{key: metrics[key] for key in sorted(REQUIRED_METRICS & metrics.keys())},
        },
        "resources": json.loads(resources_path.read_text()) if resources_path.is_file() else None,
        "cleanup": cleanup,
        "generation_count": generation_count,
        "truncated_count": truncated_count,
        "score_complete": REQUIRED_METRICS <= metrics.keys(),
        "stage3_proxy_kind": env_delta.get("CALIBRATION_STAGE3_PROXY_KIND") if phase == "stage3" else None,
    }
    write_json(rep_root / "status.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    parser.add_argument("--optimizer-enabled", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--resource-profile-sha256", required=True)
    args = parser.parse_args()
    phases = args.phases.split(",")
    if phases != ["stage2", "stage3"]:
        raise SystemExit("phase_set")
    if args.training_steps != 0 or args.optimizer_enabled.lower() != "false":
        raise SystemExit("training_disabled")
    if not 1 <= args.repetitions <= 3:
        raise SystemExit("repetitions")
    if not str(args.scratch_root).startswith("/data-1/tmp/verl_agent_scratch/"):
        raise SystemExit("scratch_root")
    rendered = load_manifest(args.manifest)
    if rendered["manifest_sha256"] != args.manifest_sha256 or rendered["resource_profile"]["sha256"] != args.resource_profile_sha256:
        raise SystemExit("identity_mismatch")
    if [item["id"] for item in rendered["runs"]] != ["frac25-stage2", "frac25-stage3"]:
        raise SystemExit("primary_run_set")
    if not WORKLOAD.is_file() or not WORKLOAD_MANIFEST.is_file():
        raise SystemExit("workload_missing")
    run_root = args.scratch_root / ("probe-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_root.mkdir(parents=True)
    splits = split_workload(run_root)
    spec = {
        "schema_version": 1,
        "driver": "stage123-zero-step-calibration-v2",
        "manifest": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "resource_profile_sha256": args.resource_profile_sha256,
        "workload": str(WORKLOAD),
        "workload_sha256": sha256(WORKLOAD),
        "phases": phases,
        "repetitions": args.repetitions,
        "training_steps": 0,
        "optimizer_enabled": False,
        "run_root": str(run_root),
    }
    write_json(run_root / "probe-spec.json", spec)
    started = time.time()
    phase_reports = []
    deadline = int(rendered["calibration_policy"]["validation_deadline_seconds"])
    for phase in phases:
        repetitions = []
        for repetition in range(1, args.repetitions + 1):
            result = run_repetition(rendered, phase, repetition, run_root, splits, deadline)
            repetitions.append(result)
            if result["status"] != "passed":
                break
        phase_reports.append({"phase": phase, "profile_hash": args.resource_profile_sha256, "repetitions": repetitions})
        if repetitions[-1]["status"] != "passed":
            break
    candidate = {
        "schema_version": 1,
        "result_type": "bounded_calibration_probe",
        "authorization_scope": "full",
        "evidence_class": "infrastructure_calibration",
        "decision": "candidate",
        "manifest_sha256": args.manifest_sha256,
        "contract": {"validation_deadline_seconds": deadline},
        "phases": phase_reports,
        "started_at_epoch": started,
        "completed_at_epoch": time.time(),
        "elapsed_seconds": time.time() - started,
        "probe_spec": spec,
    }
    write_json(run_root / "probe-candidate.json", candidate)
    checked = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_code_task_operational_calibration.py"), "--report", str(run_root / "probe-candidate.json"), "--manifest", str(args.manifest)],
        text=True,
        capture_output=True,
        check=False,
    )
    verification = json.loads(checked.stdout)
    report = {**candidate, "status": "passed" if verification["ok"] else "failed", "verification": verification, "failures": verification["failures"]}
    write_json(run_root / "probe-report.json", report)
    write_json(args.scratch_root / "latest-probe.json", {"run_root": str(run_root), "report": str(run_root / "probe-report.json"), "status": report["status"]})
    print(json.dumps({"ok": verification["ok"], "probe_report": str(run_root / "probe-report.json")}, sort_keys=True))
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
