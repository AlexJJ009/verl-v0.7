#!/usr/bin/env python3
"""Run fresh one-step GPU probes for Code A, D0, and C."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARM_ORDER = (
    "arm-a-stage1-continuation",
    "arm-d0-matched-scale-no-weak",
    "arm-c-mixture",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0


def validate_arm_results(arms: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for arm in ARM_ORDER:
        result = arms.get(arm, {})
        checks[f"{arm}_optimizer_step"] = int(result.get("optimizer_steps", 0)) == 1
        checks[f"{arm}_positive_samples"] = int(result.get("n_correct", 0)) > 0
        checks[f"{arm}_positive_loss"] = _finite_positive(result.get("positive_loss"))
        checks[f"{arm}_actor_gradient"] = _finite_positive(result.get("actor_grad_norm"))
        checks[f"{arm}_no_formal_checkpoint"] = not result.get("formal_checkpoint_files", [])
    d0 = arms.get("arm-d0-matched-scale-no-weak", {})
    c = arms.get("arm-c-mixture", {})
    checks["d0_model1_gradient_zero"] = abs(float(d0.get("model1_grad_norm", math.inf))) <= 1e-12
    checks["d0_model2_gradient_nonzero"] = _finite_positive(d0.get("model2_grad_norm"))
    checks["c_model1_gradient_nonzero"] = _finite_positive(c.get("model1_grad_norm"))
    checks["c_model2_gradient_nonzero"] = _finite_positive(c.get("model2_grad_norm"))
    return checks


def build_receipt(
    manifest: Path,
    arms: dict[str, dict[str, Any]],
    checks: dict[str, bool],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_type": "code_wdl_acd0_one_step_gpu_admission",
        "status": "pass" if checks and all(checks.values()) else "fail",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "preflight": preflight,
        "checks": checks,
        "arms": arms,
    }


def gpu_sample() -> list[dict[str, int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [
        {
            "index": int(index),
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_gpu_percent": int(utilization),
        }
        for index, used, total, utilization in (line.split(",") for line in output.splitlines())
    ]


def assert_tmux_and_idle() -> dict[str, Any]:
    if not os.environ.get("TMUX"):
        raise RuntimeError("real A/D0/C GPU probe must run inside tmux")
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if compute:
        raise RuntimeError(f"GPU compute processes are active: {compute}")
    samples = gpu_sample()
    if len(samples) != 8 or any(item["memory_used_mib"] > 256 or item["utilization_gpu_percent"] > 5 for item in samples):
        raise RuntimeError("eight idle GPUs are required")
    return {"compute_processes": [], "gpus": samples}


def _monitor(stop: threading.Event, samples: list[list[dict[str, int]]]) -> None:
    while not stop.wait(2):
        try:
            samples.append(gpu_sample())
        except Exception:
            pass


def _last_metrics(root: Path) -> tuple[Path | None, dict[str, Any]]:
    path_seen = None
    selected: dict[str, Any] = {}
    for path in sorted(root.glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = payload.get("data", payload)
            if int(data.get("training/global_step", payload.get("step", 0)) or 0) >= 1:
                path_seen, selected = path, data
    return path_seen, selected


def _queue_module():
    path = ROOT / "scripts/code_wdl_acd0_queue.py"
    spec = importlib.util.spec_from_file_location("code_wdl_acd0_queue_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_arm(manifest: dict[str, Any], run: dict[str, Any], output_root: Path) -> dict[str, Any]:
    queue = _queue_module()
    arm_root = output_root / run["id"]
    arm_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        {
            "CODE_WDL_ACD0_GPU_PROBE_ADMITTED": "1",
            "RUN_PREFIX": f"PROBE-{run['run_prefix']}",
            "WANDB_RUN_NAME": f"PROBE-{run['run_prefix']}-{int(time.time())}",
            "INIT_MODEL_PATH": manifest["paths"]["model2"],
            "BASE_MODEL_PATH": manifest["paths"]["model1"],
            "EXPECTED_MODEL1_PATH": manifest["paths"]["model1"],
            "MODEL2_PATH": manifest["paths"]["model2"],
            "STAGE1_MODEL2_PROVENANCE_FILE": manifest["paths"]["model2_provenance"],
            "TRAIN_FILE": manifest["paths"]["train_file"],
            "BASE_CKPT_DIR": str(arm_root / "checkpoints"),
            "MODEL_PATH": str(arm_root / "joint_model"),
            "LOG_DIR": str(arm_root / "logs"),
            "VERL_FILE_LOGGER_ROOT": str(arm_root / "metrics"),
            "VALIDATION_DATA_DIR": str(arm_root / "validation"),
            "WANDB_DIR": str(arm_root / "wandb"),
            "WANDB_MODE": "disabled",
            "ROLLOUT_DATA_DIR": str(arm_root / "rollout_data"),
            "FUSION_LAMBDA": str(run.get("fusion_lambda") or ""),
            "FUSION_MODE": str(run.get("fusion_mode") or ""),
            "MIN_FREE_GB_FOR_CKPT": "20",
        }
    )
    command = ["bash", str(queue.WRAPPERS[run["id"]]), 'trainer.logger=["file"]']
    samples: list[list[dict[str, int]]] = []
    stop = threading.Event()
    watcher = threading.Thread(target=_monitor, args=(stop, samples), daemon=True)
    log_path = arm_root / "probe.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        watcher.start()
        returncode = process.wait()
        stop.set()
        watcher.join(timeout=5)
    metrics_path, observed = _last_metrics(arm_root / "metrics")
    flattened = [item for sample in samples for item in sample]
    peak = max((item["memory_used_mib"] for item in flattened), default=0)
    total = min((item["memory_total_mib"] for item in flattened), default=0)
    checkpoint_files = [str(path) for path in (arm_root / "checkpoints").glob("**/global_step_*")]
    result = {
        "returncode": returncode,
        "optimizer_steps": int(observed.get("training/global_step", 0) or 0),
        "positive_loss": observed.get("actor/wdl_sft_loss_positive"),
        "n_correct": observed.get("wdl_sft/n_correct"),
        "actor_grad_norm": observed.get("actor/grad_norm"),
        "model1_grad_norm": observed.get("jointTraining/model1_grad_norm"),
        "model2_grad_norm": observed.get("jointTraining/model2_grad_norm"),
        "metrics_file": str(metrics_path) if metrics_path else None,
        "log": str(log_path),
        "formal_checkpoint_files": checkpoint_files,
        "resources": {
            "peak_gpu_memory_used_mib": peak,
            "minimum_gpu_headroom_mib": total - peak if total else 0,
            "sample_count": len(samples),
        },
    }
    if returncode != 0:
        raise RuntimeError(f"one-step probe failed for {run['id']}; see {log_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = assert_tmux_and_idle()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    queue = _queue_module()
    queue.validate_manifest(manifest, require_launch=False)
    for name in ("stage1_reuse_receipt", "eos_regression_receipt", "evaluator_receipt"):
        payload = json.loads(Path(manifest["paths"][name]).read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise RuntimeError(f"prerequisite receipt failed: {name}")
    run_root = args.scratch_root / ("probe-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_root.mkdir(parents=True, exist_ok=False)
    run_map = {run["id"]: run for run in manifest["runs"]}
    arms = {arm: run_arm(manifest, run_map[arm], run_root) for arm in ARM_ORDER}
    checks = validate_arm_results(arms)
    receipt = build_receipt(args.manifest, arms, checks, preflight)
    receipt["scratch_root"] = str(run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "scratch": str(run_root)}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
