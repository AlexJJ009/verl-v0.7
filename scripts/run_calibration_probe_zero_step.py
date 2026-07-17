#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import tokenize
from io import StringIO
from typing import Any
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.calibration_prediction import qualify

PHASE_SCRIPT = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh"
REQUIRED_PHASES = ["stage1", "stage2", "stage3"]
TREATMENT_ONLY_PHASES = ["stage2", "stage3"]
ALLOWED_PHASE_SETS = (REQUIRED_PHASES, TREATMENT_ONLY_PHASES)
PRIMARY_RUN_IDS = ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]
WORKLOAD = Path("/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/code_validation_16_16_32.parquet")
WORKLOAD_MANIFEST = WORKLOAD.with_suffix(".manifest.json")
REQUIRED_METRICS = {
    "timing_s/testing",
    "val-core/HumanEval+/acc/pass@1",
    "val-core/MBPP+/acc/pass@1",
    "val-core/LiveCodeBench/acc/pass@1",
}
PR_SET_CHILD_SUBREAPER = 36
PREDICTION_METRICS = (
    "validation_elapsed_seconds",
    "phase_elapsed_seconds",
    "peak_rss_gib",
    "gpu_wait_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authorization_scope(phases: list[str]) -> str:
    if phases == REQUIRED_PHASES:
        return "full"
    if phases == TREATMENT_ONLY_PHASES:
        return "treatment_only"
    raise ValueError(f"unsupported calibration phase set: {phases}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_prediction_comparison(history_result_path: Path, phase_reports: list[dict[str, Any]]) -> dict[str, Any]:
    history_result = json.loads(history_result_path.read_text())
    policy_path = ROOT / "config/experiment_execution/calibration_policy_v1.json"
    policy = json.loads(policy_path.read_text())
    source = history_result.get("prediction_comparison", {})
    if history_result.get("decision") != "passed" or source.get("qualified") is not True:
        raise ValueError("prediction history result is not accepted")
    history_policy_sha256 = history_result.get("policy_sha256")
    source_policy_sha256 = source.get("policy_sha256", history_policy_sha256)
    if history_result.get("policy_id") != policy.get("policy_id") or source.get("policy_id", history_result.get("policy_id")) != policy.get("policy_id"):
        raise ValueError("prediction history policy id mismatch")
    if not isinstance(history_policy_sha256, str) or len(history_policy_sha256) != 64 or source_policy_sha256 != history_policy_sha256:
        raise ValueError("prediction history policy binding mismatch")
    source_comparisons = source.get("comparisons")
    if not isinstance(source_comparisons, list):
        raise ValueError("prediction history comparisons are missing")
    by_metric = {item.get("metric"): item for item in source_comparisons if isinstance(item, dict)}
    if set(by_metric) != set(PREDICTION_METRICS):
        raise ValueError("prediction history metric set mismatch")
    repetitions = [item for phase in phase_reports for item in phase.get("repetitions", [])]
    observed = {
        "validation_elapsed_seconds": max(item["metrics"]["validation_elapsed_seconds"] for item in repetitions),
        "phase_elapsed_seconds": max(item["elapsed_seconds"] for item in repetitions),
        "peak_rss_gib": max(item["resources"]["peak_rss_gib"] for item in repetitions),
        "gpu_wait_fraction": max(item["resources"]["gpu_wait_fraction"] for item in repetitions),
    }
    comparisons = []
    for metric in PREDICTION_METRICS:
        source_item = by_metric[metric]
        history = source_item.get("history")
        predicted = source_item.get("predicted_bound")
        if not isinstance(history, list) or not all(isinstance(value, (int, float)) for value in history) or not isinstance(predicted, (int, float)):
            raise ValueError(f"invalid prediction history for {metric}")
        decision = qualify([float(value) for value in history], float(predicted), float(observed[metric]), policy)
        comparisons.append({
            "metric": metric,
            "history": history,
            "history_count": len(history),
            "predicted_bound": predicted,
            "observed_maximum": observed[metric],
            "decision": decision.as_dict(),
        })
    return {
        "qualified": all(item["decision"]["qualified"] for item in comparisons),
        "comparisons": comparisons,
        "policy_id": policy["policy_id"],
        "policy_sha256": sha256(policy_path),
        "predecessor_result_path": str(history_result_path),
        "predecessor_result_sha256": sha256(history_result_path),
        "historical_evidence_role": "accepted_predecessor_history_only_not_current_authority",
    }


def enable_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def reap_adopted_children() -> int:
    reaped = 0
    while True:
        try:
            child, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        except OSError as error:
            if error.errno == errno.ECHILD:
                return reaped
            raise
        if child == 0:
            return reaped
        reaped += 1


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
    phase_code = {"stage1": "1", "stage2": "2", "stage3": "3"}[phase]
    ray_tmpdir = Path("/data-1/tmp/verl_agent_scratch/r") / f"{phase_code}{repetition}"
    shutil.rmtree(ray_tmpdir, ignore_errors=True)
    env = {
        "QWEN3_1P7B_MODEL_PATH": rendered["paths"]["base_model"],
        "CALIBRATION_HUMANEVAL_PLUS_FILE": str(splits["HumanEval+"]),
        "CALIBRATION_MBPP_PLUS_FILE": str(splits["MBPP+"]),
        "CALIBRATION_LIVE_CODE_BENCH_FILE": str(splits["LiveCodeBench"]),
        "CALIBRATION_OUTPUT_ROOT": str(output),
        "CALIBRATION_RAY_TMPDIR": str(ray_tmpdir),
        "CALIBRATION_TOTAL_TRAINING_STEPS": "0",
        "CALIBRATION_OPTIMIZER_ENABLED": "false",
        "CALIBRATION_RAY_WORKER_PORT_MIN": "21000",
        "CALIBRATION_RAY_WORKER_PORT_MAX": "21999",
        "CALIBRATION_RAY_HEAD_PORT": "22000",
        "CALIBRATION_TCPSTORE_PORT_MIN": "35000",
        "CALIBRATION_TCPSTORE_PORT_MAX": "35999",
        "CALIBRATION_STAGE1_CKPT_DIR": stage1_source["source_checkpoint"],
        "CALIBRATION_STAGE1_MODEL2": stage2_sources["model2"]["path"],
        "CALIBRATION_STAGE1_RUN_PREFIX": stage1_source["stage1_run_prefix"],
        "CALIBRATION_STAGE1_HANDOFF_STEP": str(stage1_source["handoff_step"]),
        "CALIBRATION_TRAIN_FILE": runs[phase]["train_file"],
        "STAGE1_INIT_MODEL_PATH": rendered["paths"]["stage1_init_model"],
        "STAGE1_INIT_PROVENANCE_PATH": rendered["paths"]["stage1_init_provenance"],
        "CALIBRATION_STAGE3_MODEL_PATH": str(stage3_model),
        "CALIBRATION_STAGE3_PROXY_KIND": proxy_kind,
    }
    return env


def sample_resources(process: subprocess.Popen[str], output: Path, interval: float = 1.0) -> None:
    samples = idle = peak = 0
    peak_gpu_memory_mib: dict[int, int] = {}
    gpu_total_memory_mib: dict[int, int] = {}
    while process.poll() is None:
        try:
            rows = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
            parsed = [tuple(int(value.strip()) for value in row.split(",")) for row in rows]
            if len(parsed) == 8:
                samples += 1
                idle += int(all(utilization <= 2 for _, utilization, _, _ in parsed))
                for index, _, used_mib, total_mib in parsed:
                    peak_gpu_memory_mib[index] = max(peak_gpu_memory_mib.get(index, 0), used_mib)
                    gpu_total_memory_mib[index] = total_mib
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        for candidate in (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.peak")):
            try:
                peak = max(peak, int(candidate.read_text().strip()))
            except (OSError, ValueError):
                pass
        time.sleep(interval)
    per_gpu = [
        {
            "index": index,
            "peak_memory_used_mib": peak_gpu_memory_mib[index],
            "total_memory_mib": gpu_total_memory_mib.get(index),
            "peak_memory_fraction": peak_gpu_memory_mib[index] / gpu_total_memory_mib[index]
            if gpu_total_memory_mib.get(index)
            else None,
        }
        for index in sorted(peak_gpu_memory_mib)
    ]
    write_json(output, {
        "schema_version": 1,
        "sample_interval_seconds": interval,
        "gpu_idle_threshold_pct": 2,
        "gpu_sample_count": samples,
        "gpu_idle_sample_count": idle,
        "gpu_wait_fraction": idle / samples if samples else None,
        "peak_rss_gib": peak / (1024 ** 3) if peak else None,
        "memory_source": "calibration_container_cgroup_v2",
        "gpu_memory_source": "nvidia_smi_device_memory_used",
        "per_gpu_memory": per_gpu,
        "peak_gpu_memory_used_mib": max(peak_gpu_memory_mib.values(), default=None),
        "peak_gpu_memory_fraction": max(
            (item["peak_memory_fraction"] for item in per_gpu if item["peak_memory_fraction"] is not None),
            default=None,
        ),
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
    log_files = sorted((output / "logs").glob("CALIBRATION-*.log"))
    for path in log_files:
        text = path.read_text(errors="replace")
        fragments = []
        for line in text.splitlines():
            payload = re.sub(r"\x1b\[[0-9;]*m", "", line)
            payload = re.sub(r"^\([^)]*\)\s*", "", payload).strip()
            try:
                tokens = tokenize.generate_tokens(StringIO(payload).readline)
                strings = [token.string for token in tokens if token.type == tokenize.STRING]
            except (tokenize.TokenError, IndentationError):
                continue
            for string in strings:
                try:
                    value = ast.literal_eval(string)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(value, str):
                    fragments.append(value)
        normalized = "".join(fragments)
        data: dict[str, Any] = {}
        for key in REQUIRED_METRICS:
            matches = re.findall(rf"['\"]{re.escape(key)}['\"]:\s*(-?[0-9]+(?:\.[0-9]+)?)", normalized)
            if matches:
                data[key] = float(matches[-1])
        if REQUIRED_METRICS <= data.keys():
            return data, [str(path)]
    return {}, [str(item) for item in (*files, *log_files)]


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


def owned_cleanup(ray_root: Path, process: subprocess.Popen[str]) -> dict[str, Any]:
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


def run_repetition(
    rendered: dict[str, Any],
    phase: str,
    repetition: int,
    root: Path,
    splits: dict[str, Path],
    timeout: int,
    *,
    environment_overrides: dict[str, str] | None = None,
    repetition_label: str | None = None,
) -> dict[str, Any]:
    rep_root = root / "runs" / (repetition_label or phase) / f"rep{repetition}"
    output = rep_root / "output"
    log_path = rep_root / "host.log"
    resources_path = rep_root / "resources.json"
    env_delta = phase_environment(rendered, phase, repetition, output, splits)
    if environment_overrides:
        env_delta.update(environment_overrides)
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
    reaped_descendants = reap_adopted_children()
    cleanup = owned_cleanup(Path(env_delta["CALIBRATION_RAY_TMPDIR"]), process)
    cleanup["reaped_descendant_count"] = reaped_descendants
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
    parser.add_argument("--prediction-history-result", type=Path, required=True)
    parser.add_argument("--execution-run-id", default="stage123-readiness-requalification")
    parser.add_argument("--authorization-decision-id", default="unspecified")
    args = parser.parse_args()
    enable_child_subreaper()
    phases = args.phases.split(",")
    if phases not in ALLOWED_PHASE_SETS:
        raise SystemExit("phase_set")
    scope = authorization_scope(phases)
    if args.training_steps != 0 or args.optimizer_enabled.lower() != "false":
        raise SystemExit("training_disabled")
    if not 1 <= args.repetitions <= 3:
        raise SystemExit("repetitions")
    if not str(args.scratch_root).startswith("/data-1/tmp/verl_agent_scratch/"):
        raise SystemExit("scratch_root")
    rendered = load_manifest(args.manifest)
    if rendered["manifest_sha256"] != args.manifest_sha256 or rendered["resource_profile"]["sha256"] != args.resource_profile_sha256:
        raise SystemExit("identity_mismatch")
    if [item["id"] for item in rendered["runs"]] != PRIMARY_RUN_IDS:
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
        "run_id": args.execution_run_id,
        "authorization_decision_id": args.authorization_decision_id,
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
        "authorization_scope": scope,
        "evidence_class": "infrastructure_calibration",
        "decision": "candidate",
        "manifest_sha256": args.manifest_sha256,
        "run_id": args.execution_run_id,
        "authorization_decision_id": args.authorization_decision_id,
        "contract": {"validation_deadline_seconds": deadline},
        "phases": phase_reports,
        "started_at_epoch": started,
        "completed_at_epoch": time.time(),
        "elapsed_seconds": time.time() - started,
        "probe_spec": spec,
    }
    write_json(run_root / "probe-candidate.json", candidate)
    checked = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_code_task_operational_calibration.py"), "--report", str(run_root / "probe-candidate.json"), "--manifest", str(args.manifest), "--authorization-scope", scope],
        text=True,
        capture_output=True,
        check=False,
    )
    verification = json.loads(checked.stdout)
    prediction_comparison = build_prediction_comparison(args.prediction_history_result, phase_reports)
    if not prediction_comparison["qualified"]:
        verification["failures"].append({"code": "prediction_exceeded", "message": "fresh probe exceeds the accepted prediction policy", "context": {}})
        verification["ok"] = False
        verification["decision"] = "blocked"
    report = {**candidate, "status": "passed" if verification["ok"] else "failed", "verification": verification, "failures": verification["failures"]}
    report["optimizer_steps"] = 0
    report["formal_checkpoints"] = []
    report["prediction_comparison"] = prediction_comparison
    report["cleanup"] = {"resources_released": all(item.get("cleanup", {}).get("resources_released") is True for phase in phase_reports for item in phase.get("repetitions", []))}
    report_started = datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z")
    report_completed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report["report_started_at_utc"] = report_started
    report["report_completed_at_utc"] = report_completed
    write_json(run_root / "probe-report.json", report)
    report_path = run_root / "probe-report.json"
    write_json(args.scratch_root / "latest-probe.json", {
        "schema_version": 2,
        "run_id": args.execution_run_id,
        "authorization_decision_id": args.authorization_decision_id,
        "report_sha256": sha256(report_path),
        "generated_at_utc": report_completed,
        "report_started_at_utc": report_started,
        "report_completed_at_utc": report_completed,
        "run_root": str(run_root),
        "report": str(report_path),
        "status": report["status"],
    })
    print(json.dumps({"ok": verification["ok"], "probe_report": str(run_root / "probe-report.json")}, sort_keys=True))
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
