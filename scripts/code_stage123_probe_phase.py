#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run one Code Stage123 probe phase without producing formal checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "stage1": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "stage2": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "stage3": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def monitor(stop: threading.Event, samples: list[list[dict[str, int]]]) -> None:
    while not stop.wait(2):
        try:
            samples.append(gpu_sample())
        except Exception:
            pass


def metrics(root: Path) -> tuple[Path | None, dict]:
    selected_path = None
    selected = {}
    for path in sorted(root.glob("**/*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = payload.get("data", payload)
            if data:
                selected_path, selected = path, data
    return selected_path, selected


def validation_contract(observed: dict, *, joint: bool) -> tuple[bool, list[str]]:
    views = ("model1", "model2") if joint else (None,)
    required = []
    for view in views:
        prefix = f"val-core/{view}/" if view else "val-core/"
        required.extend(
            f"{prefix}{dataset}/acc/{metric}"
            for dataset in ("HumanEval+", "MBPP+", "LiveCodeBench")
            for metric in ("mean@3", "pass@3")
        )
    missing = [key for key in required if key not in observed]
    return not missing, missing


def runtime_contract(log_text: str, *, joint: bool) -> tuple[bool, list[str]]:
    required = {
        "actor_param_offload": "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_optimizer_offload": "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "reference_param_offload": "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    }
    missing = [name for name, marker in required.items() if marker not in log_text]
    return not missing, missing


def training_contract(observed: dict) -> tuple[bool, list[str]]:
    learning_rate = float(observed.get("actor/lr", 0.0))
    positive_loss = float(observed.get("actor/wdl_sft_loss_positive", 0.0))
    grad_norm = float(observed.get("actor/grad_norm", 0.0))
    update_time = float(observed.get("timing_s/update_actor", 0.0))
    required = {
        "optimizer_step": int(observed.get("training/global_step", 0)) >= 1,
        "learning_rate": math.isfinite(learning_rate) and learning_rate > 0,
        "positive_samples": int(observed.get("wdl_sft/n_correct", 0)) > 0,
        "positive_loss": math.isfinite(positive_loss) and positive_loss > 0,
        "nonzero_gradient": math.isfinite(grad_norm) and grad_norm > 0,
        "actor_update_time": math.isfinite(update_time) and update_time > 0,
    }
    missing = [name for name, satisfied in required.items() if not satisfied]
    return not missing, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("validation", "train"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--utilization", type=float, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    run = next(item for item in manifest["runs"] if item["id"] == args.run_id)
    selection_path = Path(manifest["paths"]["model1_selection"])
    selection = json.loads(selection_path.read_text())
    receipt = json.loads(Path(manifest["paths"]["dataset_receipt"]).read_text())
    model1 = selection["identity"]["model_path"]
    train_file = receipt["shards"][run["train_shard"]]["path"]
    validation_files = [
        "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet",
        "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet",
        "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet",
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    identity = selection["identity"]
    env.update(
        {
            "DRY_RUN": "0",
            "CODE_STAGE123_GPU_PROBE_ADMITTED": "1",
            "CODE_STAGE123_GPU_PROBE_OUTPUT_ROOT": "/data-1/tmp/verl_agent_scratch/code_stage123_gpu_utilization_probe",
            "CODE_STAGE123_MANIFEST": str(args.manifest),
            "CODE_STAGE123_MANIFEST_SHA256": sha256(args.manifest),
            "CODE_STAGE123_MODEL1_SELECTION_SHA256": sha256(selection_path),
            "CODE_STAGE123_DATASET_RECEIPT_SHA256": sha256(Path(manifest["paths"]["dataset_receipt"])),
            "STAGE123_RUN_ID": run["id"],
            "RUN_PREFIX": f"PROBE-{run['id'].upper().replace('-', '_')}-{int(args.utilization * 100)}",
            "CODE_TRAIN_FILE": train_file,
            "TRAIN_FILE": train_file,
            "DATA_SEED": str(manifest["seed"]),
            "DATA_SHUFFLE": "False",
            "WDL_SFT_BETA": str(run["beta"]),
            "LR": "1e-6",
            "LR_WARMUP_STEPS": "0",
            "ROLLOUT_GPU_MEMORY_UTILIZATION": f"{args.utilization:.2f}",
            "ACTOR_CALCULATE_ENTROPY": "False",
            "CALCULATE_ENTROPY": "False",
            "CODE_VAL_FILES": str(validation_files),
            "TEST_FILES": str(validation_files),
            "CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE": validation_files[2],
            "BASE_CKPT_DIR": str(args.output_root / "checkpoints"),
            "LOG_DIR": str(args.output_root / "logs"),
            "VERL_FILE_LOGGER_ROOT": str(args.output_root / "metrics"),
            "VALIDATION_DATA_DIR": str(args.output_root / "validation"),
            "WANDB_DIR": str(args.output_root / "wandb"),
            "WANDB_MODE": "disabled",
            "KEEP_BEST_CKPT": "False",
            "MAX_ACTOR_CKPTS_TO_KEEP": "0",
            "MAX_CRITIC_CKPTS_TO_KEEP": "0",
            "EXPECTED_MODEL1_PATH": model1,
            "EXPECTED_MODEL1_CONFIG_SHA256": identity["config_sha256"],
            "EXPECTED_MODEL1_TOKENIZER_CONFIG_SHA256": identity["tokenizer_config_sha256"],
            "EXPECTED_MODEL1_CHAT_TEMPLATE_SHA256": identity["chat_template_sha256"],
            "EXPECTED_MODEL1_PROVENANCE_PATH": str(selection_path),
            "EXPECTED_MODEL1_PROVENANCE_SHA256": sha256(selection_path),
            "BASE_MODEL_PATH": model1,
            "FUSION_LAMBDA": str(manifest["matrix"]["stage2_fusion_lambda"]),
            "SUBMODEL_KL_ENABLED": "true" if run.get("kl") == "m2kl" else "false",
            "SUBMODEL_KL_MODEL1_ENABLED": "false",
            "SUBMODEL_KL_MODEL2_ENABLED": "true" if run.get("kl") == "m2kl" else "false",
            "SUBMODEL_KL_MODEL2_COEF": str(manifest["matrix"]["model2_kl_coef"] if run.get("kl") == "m2kl" else 0.0),
        }
    )
    if args.mode == "validation":
        env.update(
            {
                "TOTAL_TRAINING_STEPS": "0",
                "VAL_ONLY": "True",
                "VAL_BEFORE_TRAIN": "True",
                "TEST_FREQ": "1",
                "SAVE_FREQ": "1000",
            }
        )
    else:
        env.update(
            {
                "TOTAL_TRAINING_STEPS": "1",
                "VAL_BEFORE_TRAIN": "False",
                "TEST_FREQ": "-1",
                "SAVE_FREQ": "1000",
                "TRAIN_MAX_SAMPLES": "-1",
                "ROLLOUT_DATA_DIR": str(args.output_root / "rollout_data"),
            }
        )
    source_run = next((item for item in manifest["runs"] if item["id"] == run.get("source_run")), None)
    proxy_model = model1
    wrapper_phase = run["phase"]
    if wrapper_phase == "stage1_control":
        wrapper_phase = "stage1"
    if wrapper_phase == "stage1":
        env["INIT_MODEL_PATH"] = model1
    elif wrapper_phase == "stage2":
        stage1_proxy_provenance = args.output_root / "stage1-proxy.json"
        stage1_proxy_provenance.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "release_eligible": True,
                    "run": source_run,
                    "outputs": {"model": proxy_model},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        env.update(
            {
                "MODEL2_PATH": proxy_model,
                "ALLOW_EXTERNAL_MODEL2": "1",
                "STAGE1_MODEL2_PROVENANCE_FILE": str(stage1_proxy_provenance),
                "STAGE1_RUN_PREFIX": source_run["id"],
                "EXPECTED_STAGE1_RUN_PREFIX": source_run["id"],
                "STAGE1_STEP": str(source_run["final_step"]),
                "STAGE2_HANDOFF_STEP": str(source_run["final_step"]),
                "EXPECTED_STAGE1_BETA": str(run["beta"]),
                "MODEL_PATH": str(args.output_root / "joint"),
                "SUBMODEL_KL_MODEL2_REF_PATH": proxy_model,
            }
        )
    else:
        env["STAGE2_MODEL_PATH"] = model1
        env["STAGE2_SUBMODEL"] = run["submodel"]
        stage2_proxy_provenance = args.output_root / "stage2-proxy.json"
        stage2_proxy_provenance.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "release_eligible": True,
                    "source": {
                        "extracted_model1": model1,
                        "extracted_model2": model1,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        env["STAGE2_PROVENANCE_FILE"] = str(stage2_proxy_provenance)
    command = ["bash", str(WRAPPERS[wrapper_phase])]
    command.extend(["trainer.save_freq=-1", 'trainer.logger=["file"]'])
    if args.mode == "validation":
        command.extend(["trainer.val_only=true", "trainer.val_before_train=true"])
    else:
        command.extend(
            [
                "trainer.val_before_train=false",
                "trainer.test_freq=-1",
                f"trainer.rollout_data_dir={env['ROLLOUT_DATA_DIR']}",
            ]
        )
    samples: list[list[dict[str, int]]] = []
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
    with (args.output_root / "phase.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        watcher.start()
        returncode = process.wait()
        stop.set()
        watcher.join(timeout=5)
    metrics_path, observed = metrics(args.output_root / "metrics")
    flattened = [item for sample in samples for item in sample]
    peak = max((item["memory_used_mib"] for item in flattened), default=0)
    total = min((item["memory_total_mib"] for item in flattened), default=0)
    log_text = (args.output_root / "phase.log").read_text(errors="replace")
    validation_ok, missing_validation_metrics = validation_contract(
        observed,
        joint=wrapper_phase == "stage2",
    )
    runtime_ok, missing_runtime_contract = runtime_contract(log_text, joint=wrapper_phase == "stage2")
    training_ok, missing_training_contract = training_contract(observed)
    optimizer_steps = int(observed.get("training/global_step", 0))
    passed = (
        returncode == 0
        and "out of memory" not in log_text.lower()
        and runtime_ok
        and ((args.mode == "validation" and validation_ok) or (args.mode == "train" and training_ok))
        and not list((args.output_root / "checkpoints").glob("**/global_step_*"))
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "returncode": returncode,
        "metrics_file": str(metrics_path) if metrics_path else None,
        "optimizer_steps": optimizer_steps,
        "validation_complete": validation_ok,
        "missing_validation_metrics": missing_validation_metrics,
        "runtime_contract_complete": runtime_ok,
        "missing_runtime_contract": missing_runtime_contract,
        "training_contract_complete": training_ok,
        "missing_training_contract": missing_training_contract,
        "observed_training_metrics": {
            key: observed.get(key)
            for key in (
                "training/global_step",
                "actor/lr",
                "actor/grad_norm",
                "actor/pg_loss",
                "actor/wdl_sft_loss_positive",
                "actor/wdl_sft_loss_negative",
                "actor/wdl_sft_loss_total",
                "wdl_sft/n_correct",
                "wdl_sft/n_incorrect",
                "timing_s/update_actor",
            )
            if key in observed
        },
        "resources": {
            "peak_gpu_memory_used_mib": peak,
            "minimum_gpu_headroom_mib": total - peak if total else 0,
            "sample_count": len(samples),
        },
        "formal_checkpoint_files": [str(path) for path in (args.output_root / "checkpoints").glob("**/global_step_*")],
        "log": str(args.output_root / "phase.log"),
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
