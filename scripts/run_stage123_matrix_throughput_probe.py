#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGE2_IDS = ("frac25-stage2-nokl", "frac25-stage2-m2kl")


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def render_manifest(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/stage123_matrix_manifest.py"), "render", str(path)], text=True
    )
    return json.loads(output)


def model1_identity(run: dict[str, Any]) -> dict[str, str]:
    source = run["source"]
    fields = (
        "model1_path",
        "model1_config_sha256",
        "model1_tokenizer_config_sha256",
        "model1_chat_template_sha256",
        "model1_provenance_path",
        "model1_provenance_sha256",
    )
    return {field: source[field] for field in fields}


def gpu_sample() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        {
            "index": int(index.strip()),
            "memory_used_mib": int(used.strip()),
            "memory_total_mib": int(total.strip()),
            "utilization_gpu_percent": int(utilization.strip()),
        }
        for index, used, total, utilization in (line.split(",") for line in output.splitlines())
    ]


def monitor_gpus(stop: threading.Event, samples: list[list[dict[str, int]]]) -> None:
    while not stop.wait(2):
        try:
            samples.append(gpu_sample())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass


def latest_training_metrics(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(root.glob("**/*.jsonl"), key=lambda path: path.stat().st_mtime)
    selected_path = None
    selected_metrics = None
    for path in candidates:
        for line in path.read_text(errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = payload.get("data", payload)
            if int(metrics.get("training/global_step", 0)) >= 1:
                selected_path = path
                selected_metrics = metrics
    return selected_path, selected_metrics


def resources_released(container_name: str) -> dict[str, Any]:
    running = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.ID}}"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    samples = gpu_sample()
    released = not running and all(item["memory_used_mib"] <= 256 for item in samples)
    return {"resources_released": released, "container_running": bool(running), "final_gpu_sample": samples}


def peak_resources(samples: list[list[dict[str, int]]]) -> dict[str, Any]:
    flattened = [item for sample in samples for item in sample]
    peak = max((item["memory_used_mib"] for item in flattened), default=0)
    peak_utilization = max((item["utilization_gpu_percent"] for item in flattened), default=0)
    totals = [item["memory_total_mib"] for item in flattened]
    return {
        "peak_gpu_memory_used_mib": peak,
        "peak_gpu_utilization_percent": peak_utilization,
        "minimum_gpu_headroom_mib": min(totals, default=0) - peak if totals else 0,
        "sample_count": len(samples),
    }


def qualify_result(result: dict[str, Any], run: dict[str, Any], minimum_headroom_mib: int = 1024) -> dict[str, Any]:
    expected_sources = {
        "model1": str(Path(run["source"]["model1_path"]).resolve()),
        "model2": str(Path(run["source"]["model2_path"]).resolve()),
    }
    joint_sources = result.get("joint_model_sources", {})
    observed_sources = {
        key: str(Path(joint_sources.get(key, "")).resolve()) for key in expected_sources if joint_sources.get(key)
    }
    log_path = Path(result["log"])
    passed = (
        result.get("returncode") == 0
        and result.get("timed_out") is False
        and int(result.get("optimizer_steps", 0)) >= 1
        and float(result.get("metrics", {}).get("step_time_seconds", 0.0)) > 0
        and float(result.get("metrics", {}).get("rollout_tokens_per_second", 0.0)) > 0
        and float(result.get("metrics", {}).get("actor_grad_norm", 0.0)) > 0
        and int(result.get("resources", {}).get("peak_gpu_memory_used_mib", 0)) > 0
        and int(result.get("resources", {}).get("minimum_gpu_headroom_mib", 0)) >= minimum_headroom_mib
        and result.get("cleanup", {}).get("resources_released") is True
        and not result.get("formal_checkpoint_files")
        and observed_sources == expected_sources
        and log_path.is_file()
        and "out of memory" not in log_path.read_text(errors="replace").lower()
    )
    result = dict(result)
    result["status"] = "passed" if passed else "failed"
    return result


def run_environment(
    run: dict[str, Any], output_root: Path, profile: dict[str, Any], container_name: str
) -> dict[str, str]:
    source = run["source"]
    kl = run["submodel_kl"]
    return {
        "REPO_HOST": str(ROOT),
        "DOCKER_CONTAINER_NAME": container_name,
        "THROUGHPUT_OUTPUT_ROOT": str(output_root),
        "THROUGHPUT_RUN_ID": run["id"],
        "THROUGHPUT_TRAIN_FILE": run["train_file"],
        "THROUGHPUT_STAGE1_CKPT_DIR": source["checkpoint_root"],
        "THROUGHPUT_STAGE1_RUN_PREFIX": source["run_prefix"],
        "THROUGHPUT_STAGE1_HANDOFF_STEP": str(source["handoff_step"]),
        "THROUGHPUT_MODEL2_PATH": source["model2_path"],
        "BASE_MODEL_PATH": source["model1_path"],
        "EXPECTED_MODEL1_PATH": source["model1_path"],
        "EXPECTED_MODEL1_CONFIG_SHA256": source["model1_config_sha256"],
        "EXPECTED_MODEL1_TOKENIZER_CONFIG_SHA256": source["model1_tokenizer_config_sha256"],
        "EXPECTED_MODEL1_CHAT_TEMPLATE_SHA256": source["model1_chat_template_sha256"],
        "EXPECTED_MODEL1_PROVENANCE_PATH": source["model1_provenance_path"],
        "EXPECTED_MODEL1_PROVENANCE_SHA256": source["model1_provenance_sha256"],
        "STAGE123_EXPECTED_PROFILE_HASH": profile["sha256"],
        "STAGE123_EXPECTED_VAL_N": "3",
        "VAL_N": "3",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": f"{float(profile['rollout_gpu_memory_utilization']):.2f}",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(profile["rollout_max_num_batched_tokens"]),
        "REF_LOG_PROB_MICRO_BATCH_SIZE": str(profile["ref_log_prob_micro_batch_size"]),
        "REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": str(profile["ref_log_prob_max_token_len_per_gpu"]),
        "SUBMODEL_KL_ENABLED": str(bool(kl["enabled"])).lower(),
        "SUBMODEL_KL_MODEL1_ENABLED": str(bool(kl["model1_enabled"])).lower(),
        "SUBMODEL_KL_MODEL1_COEF": str(kl["model1_coef"]),
        "SUBMODEL_KL_MODEL2_ENABLED": str(bool(kl["model2_enabled"])).lower(),
        "SUBMODEL_KL_MODEL2_COEF": str(kl["model2_coef"]),
        "SUBMODEL_KL_MODEL2_REF_PATH": kl.get("model2_ref_path", source["model2_path"]),
        "THROUGHPUT_RAY_HEAD_PORT": "22300",
        "THROUGHPUT_RAY_WORKER_PORT_MIN": "22400",
        "THROUGHPUT_RAY_WORKER_PORT_MAX": "22999",
    }


def run_arm(run: dict[str, Any], run_root: Path, profile: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    output_root = run_root / run["id"]
    output_root.mkdir(parents=True)
    log_path = output_root / "throughput-probe.log"
    container_name = f"stage123-throughput-{run['id']}-{int(time.time())}"
    environment = os.environ.copy()
    environment.update(run_environment(run, output_root, profile, container_name))
    command = [
        "/data-1/verl07/run_train.sh",
        "bash",
        "/workspace/verl/recipe/on_policy_wdl_sft/code_task/run_stage123_matrix_throughput_probe_phase.sh",
    ]
    samples: list[list[dict[str, int]]] = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpus, args=(stop, samples), daemon=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
        monitor.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(
                ["docker", "kill", container_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            returncode = process.wait(timeout=60)
        finally:
            stop.set()
            monitor.join(timeout=5)
    elapsed = time.monotonic() - started
    metrics_path, metrics = latest_training_metrics(output_root / "metrics")
    metrics = metrics or {}
    total_tokens = float(metrics.get("perf/total_num_tokens", 0.0))
    generation_seconds = float(metrics.get("timing_s/gen", 0.0))
    formal_checkpoints = [str(path) for path in (output_root / "checkpoints").glob("**/global_step_*")]
    joint_config_path = output_root / "joint_model" / "config.json"
    joint_config = json.loads(joint_config_path.read_text()) if joint_config_path.is_file() else {}
    joint_sources = joint_config.get("joint_model_sources", {})
    cleanup = resources_released(container_name)
    resource_summary = peak_resources(samples)
    optimizer_steps = int(metrics.get("training/global_step", 0))
    result = {
        "run_id": run["id"],
        "status": "failed",
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "metrics_file": str(metrics_path) if metrics_path else None,
        "metrics_file_sha256": sha256(metrics_path) if metrics_path else None,
        "optimizer_steps": optimizer_steps,
        "metrics": {
            "step_time_seconds": float(metrics.get("timing_s/step", 0.0)),
            "generation_time_seconds": generation_seconds,
            "rollout_tokens_per_second": total_tokens / generation_seconds if generation_seconds > 0 else 0.0,
            "total_tokens": total_tokens,
            "actor_grad_norm": float(metrics.get("actor/grad_norm", 0.0)),
            "model1_grad_norm": float(metrics.get("jointTraining/model1_grad_norm", 0.0)),
            "model2_grad_norm": float(metrics.get("jointTraining/model2_grad_norm", 0.0)),
        },
        "resources": resource_summary,
        "cleanup": cleanup,
        "formal_checkpoint_files": formal_checkpoints,
        "joint_model_config_sha256": sha256(joint_config_path) if joint_config_path.is_file() else None,
        "joint_model_sources": joint_sources,
    }
    result = qualify_result(result, run, int(profile["minimum_gpu_headroom_mib"]))
    write_json(output_root / "result.json", result)
    shutil.rmtree(output_root / "joint_model", ignore_errors=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--run-ids", default=",".join(STAGE2_IDS))
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--reuse-run", action="append", default=[])
    args = parser.parse_args()
    if not str(args.scratch_root).startswith("/data-1/tmp/verl_agent_scratch/"):
        raise SystemExit("scratch root must be under /data-1/tmp/verl_agent_scratch")
    manifest = render_manifest(args.manifest)
    profile = manifest["resource_profile"]
    if float(profile["rollout_gpu_memory_utilization"]) < 0.4:
        raise SystemExit("throughput probe rejects safety-only GPU utilization")
    by_id = {run["id"]: run for run in manifest["runs"]}
    run_ids = args.run_ids.split(",")
    runs = [by_id[run_id] for run_id in run_ids]
    if any(run["phase"] != "stage2" for run in runs):
        raise SystemExit("throughput probe supports Stage2 arms only")
    identities = [model1_identity(run) for run in runs]
    if any(identity != identities[0] for identity in identities[1:]):
        raise SystemExit("Stage2 Model1 identities differ")
    run_root = args.scratch_root / ("throughput-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_root.mkdir(parents=True)
    reused = {}
    for value in args.reuse_run:
        run_id, separator, path = value.partition("=")
        if not separator:
            raise SystemExit("reuse run must use RUN_ID=RESULT_JSON")
        reused[run_id] = Path(path)
    results = []
    for run in runs:
        if run["id"] in reused:
            result = qualify_result(
                json.loads(reused[run["id"]].read_text()), run, int(profile["minimum_gpu_headroom_mib"])
            )
        else:
            result = run_arm(run, run_root, profile, args.timeout_seconds)
        results.append(result)
        if result["status"] != "passed":
            break
    report = {
        "schema_version": 1,
        "result_type": "stage123_matrix_throughput_probe",
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "training_steps": 1,
        "optimizer_enabled": True,
        "rollout_gpu_memory_utilization": profile["rollout_gpu_memory_utilization"],
        "rollout_max_num_batched_tokens": profile["rollout_max_num_batched_tokens"],
        "rollout_free_cache_engine": profile["rollout_free_cache_engine"],
        "rollout_enable_sleep_mode": profile["rollout_enable_sleep_mode"],
        "ref_fsdp_offload": profile["ref_fsdp_offload"],
        "actor_optimizer_offload": profile["actor_optimizer_offload"],
        "actor_param_offload": profile["actor_param_offload"],
        "minimum_gpu_headroom_mib": profile["minimum_gpu_headroom_mib"],
        "ref_log_prob_micro_batch_size": profile["ref_log_prob_micro_batch_size"],
        "ref_log_prob_max_token_len_per_gpu": profile["ref_log_prob_max_token_len_per_gpu"],
        "model1_identity": identities[0],
        "run_root": str(run_root),
        "runs": results,
        "status": "passed"
        if len(results) == len(runs) and all(result["status"] == "passed" for result in results)
        else "failed",
    }
    report_path = run_root / "matrix-throughput-probe-report.json"
    write_json(report_path, report)
    write_json(
        args.scratch_root / "latest-matrix-throughput-probe.json",
        {
            "schema_version": 1,
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "status": report["status"],
            "run_root": str(run_root),
        },
    )
    print(json.dumps({"ok": report["status"] == "passed", "report": str(report_path)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
