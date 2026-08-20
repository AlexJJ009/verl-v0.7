#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Execute an admitted Qwen3-1.7B Math or Code Stage1/2/3 matrix sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATH_WRAPPERS = {
    "stage1": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s1_math_qwen3_1p7b_stage123_common.sh",
    "stage1_control": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s1_math_qwen3_1p7b_stage123_common.sh",
    "stage2": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s2_math_qwen3_1p7b_stage123_common.sh",
    "stage3": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s3_math_qwen3_1p7b_stage123_common.sh",
}
CODE_WRAPPERS = {
    "stage1": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "stage1_control": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "stage2": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "stage3": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
}

TRANSIENT_PORT_COLLISION_PATTERNS = (
    "EADDRINUSE",
    "address already in use",
)


def emit_event(event: str, **payload: object) -> None:
    event_log = os.environ.get("STAGE123_EVENT_LOG") or os.environ.get("MATH_EVENT_LOG")
    if not event_log:
        return
    path = Path(event_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_prefix(run: dict, task: str = "math") -> str:
    suffix = "COTMASK-V3-AUTHOR-SIGNATURE-V2-STEP20" if task == "code" else "COTMASK-V3"
    return f"{task.upper()}-{run['id'].upper().replace('-', '_')}-QWEN3-1P7B-{suffix}"


def stage2_joint_cache_path(artifact_root: Path, run_id: str, task: str = "math") -> Path:
    run_key = re.sub(r"[^a-z0-9]+", "-", run_id.lower()).strip("-")
    launch_key = hashlib.sha256(str(artifact_root).encode()).hexdigest()[:12]
    basename = f"{task}-s2-{run_key}-{launch_key}"
    if len(basename.replace("-", "_hyphen_")) > 180:
        raise RuntimeError(f"generated Stage2 joint cache basename is too long: {basename}")
    return Path("/data-1/.cache/huggingface") / basename


def load_completed_outputs(artifact_root: Path, run_spec: dict) -> dict[str, str]:
    provenance_path = artifact_root / run_spec["id"] / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"missing completed-run provenance for continuation: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("run") != run_spec:
        raise RuntimeError(f"completed-run provenance does not match manifest for {run_spec['id']}")
    outputs = provenance.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"completed-run provenance has no outputs for {run_spec['id']}")
    for name, output_path in outputs.items():
        if not Path(output_path).exists():
            raise FileNotFoundError(f"completed output {name} is missing for {run_spec['id']}: {output_path}")
    return {str(name): str(output_path) for name, output_path in outputs.items()}


def continuation_state(
    manifest: dict, artifact_root: Path, start_run: str | None
) -> tuple[dict[str, dict], list[dict]]:
    runs = manifest["runs"]
    if start_run is None:
        return {}, runs
    run_ids = [run["id"] for run in runs]
    if start_run not in run_ids:
        raise RuntimeError(f"unknown continuation run {start_run}; expected one of {run_ids}")
    start_index = run_ids.index(start_run)
    outputs = {run_spec["id"]: load_completed_outputs(artifact_root, run_spec) for run_spec in runs[:start_index]}
    return outputs, runs[start_index:]


def checkpoint_after(prefix: str, started_at: float, final_step: int) -> Path:
    checkpoint_roots = [
        path for path in Path("/data-1/checkpoints").glob(f"{prefix}_*") if path.stat().st_mtime >= started_at - 5
    ]
    candidates = []
    for path in checkpoint_roots:
        actor = path / f"global_step_{final_step}" / "actor"
        config_path = actor / "fsdp_config.json"
        if not config_path.is_file() or not (actor / "huggingface" / "config.json").is_file():
            continue
        try:
            world_size = int(json.loads(config_path.read_text())["world_size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if world_size <= 0:
            continue
        if all((actor / f"model_world_size_{world_size}_rank_{rank}.pt").is_file() for rank in range(world_size)):
            candidates.append(actor)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one complete new checkpoint root for {prefix} at global_step_{final_step}, "
            f"found actors={candidates}, roots={checkpoint_roots}"
        )
    return candidates[0]


def attempt_log_dir() -> Path:
    configured = os.environ.get("MATH_RUN_ATTEMPT_LOG_DIR")
    if configured:
        return Path(configured)
    event_log = os.environ.get("STAGE123_EVENT_LOG") or os.environ.get("MATH_EVENT_LOG")
    if event_log:
        return Path(event_log).parent / "run_attempt_logs"
    return Path("/data-1/tmp/verl_agent_scratch/math_stage123_queue_attempt_logs") / str(os.getpid())


def is_transient_port_collision(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(errors="replace")
    return (
        any(pattern in text for pattern in TRANSIENT_PORT_COLLISION_PATTERNS)
        and ("torch.distributed.DistNetworkError" in text or "TCPStore" in text)
        and ("vLLMHttpServer" in text or "Engine core initialization failed" in text)
    )


def execute(command: list[str], env: dict[str, str], dry_run: bool, run_id: str) -> None:
    printable = {
        key: env[key]
        for key in sorted(env)
        if key
        in {
            "RUN_PREFIX",
            "INIT_MODEL_PATH",
            "BASE_MODEL_PATH",
            "MODEL2_PATH",
            "STAGE2_MODEL_PATH",
            "MODEL_PATH",
            "TRAIN_FILE",
            "CODE_TRAIN_FILE",
            "CODE_VAL_FILES",
            "TEST_FILES",
            "TOTAL_TRAINING_STEPS",
            "WDL_SFT_BETA",
            "SUBMODEL_KL_ENABLED",
            "SUBMODEL_KL_MODEL2_ENABLED",
            "SUBMODEL_KL_MODEL2_COEF",
            "SUBMODEL_KL_MODEL2_REF_PATH",
            "LR",
            "LR_WARMUP_STEPS",
            "ROLLOUT_GPU_MEMORY_UTILIZATION",
            "ACTOR_CALCULATE_ENTROPY",
            "CALCULATE_ENTROPY",
            "EXPECTED_MODEL1_PATH",
            "STAGE1_MODEL2_PROVENANCE_FILE",
            "FUSION_LAMBDA",
        }
    }
    print(json.dumps({"command": command, "environment": printable}, sort_keys=True), flush=True)
    if dry_run:
        return

    retry_count = int(os.environ.get("MATH_TRANSIENT_PORT_RETRIES", "2"))
    retry_delay_sec = float(os.environ.get("MATH_TRANSIENT_PORT_RETRY_DELAY_SEC", "15"))
    log_dir = attempt_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retry_count + 2):
        log_path = log_dir / f"{run_id}.attempt-{attempt}.log"
        with log_path.open("w") as attempt_log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                attempt_log.write(line)
            returncode = process.wait()
        if returncode == 0:
            return
        if attempt > retry_count or not is_transient_port_collision(log_path):
            raise subprocess.CalledProcessError(returncode, command)
        emit_event(
            "stage_run_retrying",
            run_id=run_id,
            attempt=attempt,
            next_attempt=attempt + 1,
            reason="transient_vllm_tcpstore_port_collision",
            log_path=str(log_path),
        )
        print(
            f"[math-stage123] transient vLLM TCPStore port collision for {run_id}; "
            f"retrying attempt {attempt + 1}/{retry_count + 1} after {retry_delay_sec:g}s",
            flush=True,
        )
        time.sleep(retry_delay_sec)


def emit_queue_terminal(event: str, **payload: object) -> None:
    emit_event(event, **payload)


def verify_dataset_receipt(receipt_path: Path, source_path: Path, seed: int, dry_run: bool, task: str = "math") -> None:
    if receipt_path.name != "dataset_receipt.json":
        raise ValueError(f"unexpected dataset receipt filename: {receipt_path}")
    prepare_script = (
        ROOT / "recipe/on_policy_wdl_sft/code_task/prepare_qwen3_1p7b_code_stage123_data.py"
        if task == "code"
        else ROOT / "recipe/on_policy_wdl_sft/math_task/prepare_qwen3_1p7b_math_stage123_data.py"
    )
    command = [
        sys.executable,
        str(prepare_script),
        "--source",
        str(source_path),
        "--output-root",
        str(receipt_path.parent),
        "--seed",
        str(seed),
        "--verify-only",
    ]
    if task == "code":
        cold_start_file = receipt_path.parent / "cold_start_source.parquet"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        if receipt.get("cold_start_file"):
            cold_start_file = Path(receipt["cold_start_file"])
        command.extend(["--cold-start-file", str(cold_start_file)])
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def merge_single(actor: Path, target: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(target),
        ],
        cwd=ROOT,
        check=True,
    )


def merge_stage2(actor: Path, artifact_dir: Path) -> dict[str, str]:
    joint = artifact_dir / "stage2_final_joint"
    model1 = artifact_dir / "stage2_final_model1"
    model2 = artifact_dir / "stage2_final_model2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(joint),
            "--trust-remote-code",
        ],
        cwd=ROOT,
        check=True,
    )
    for index, target in ((0, model1), (1, model2)):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "recipe/joint_training/extract_sub_model.py"),
                "--joint_model_path",
                str(joint),
                "--output_path",
                str(target),
                "--sub_model_index",
                str(index),
            ],
            cwd=ROOT,
            check=True,
        )
    return {"joint": str(joint), "model1": str(model1), "model2": str(model2)}


def model1_selection_policy(manifest: dict) -> dict:
    policy = manifest.get("model1_selection_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("admitted Stage123 manifest must bind model1_selection_policy")
    if "selected_step" not in policy or "allow_below_format_threshold" not in policy:
        raise RuntimeError("model1_selection_policy must bind selected_step and allow_below_format_threshold")
    try:
        selected_step = int(policy["selected_step"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("model1_selection_policy.selected_step must be an integer") from exc
    expected_step = 20
    if selected_step != expected_step:
        raise RuntimeError(f"model1_selection_policy.selected_step must be {expected_step}, got {selected_step}")
    if policy["allow_below_format_threshold"] is not True:
        raise RuntimeError("model1_selection_policy.allow_below_format_threshold must be true")
    return policy


def verify_model_identity(selection: dict) -> Path:
    identity = selection.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Model1 selection has no identity")
    model_path = Path(identity["model_path"])
    files = {
        "config_sha256": model_path / "config.json",
        "tokenizer_config_sha256": model_path / "tokenizer_config.json",
        "chat_template_sha256": model_path / "chat_template.jinja",
    }
    for field, path in files.items():
        if field in identity and (
            not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != identity[field]
        ):
            raise RuntimeError(f"Model1 identity mismatch: {path}")
    for weight in identity.get("weight_files", []):
        path = Path(weight["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != weight["sha256"]:
            raise RuntimeError(f"Model1 weight identity mismatch: {path}")
    return model_path


def selected_model_from_receipt(manifest: dict, selection_path: Path) -> Path:
    selection = json.loads(selection_path.read_text())
    policy = model1_selection_policy(manifest)
    expected_step = int(policy["selected_step"])
    if int(selection["selected_step"]) != expected_step:
        raise RuntimeError(
            f"Model1 selection step mismatch: receipt={selection['selected_step']} manifest={expected_step}"
        )
    format_gate_override = bool(selection.get("format_gate_override", False))
    if format_gate_override and policy.get("allow_below_format_threshold") is not True:
        raise RuntimeError("Model1 selection uses a format-gate override that the admitted manifest does not allow")
    if not format_gate_override and selection.get("candidate", {}).get("passed_format_gate") is not True:
        raise RuntimeError("Model1 selection receipt is internally inconsistent")
    selected_model = verify_model_identity(selection)
    if not selected_model.is_dir():
        raise FileNotFoundError(f"selected Model1 path does not exist: {selected_model}")
    return selected_model


def model1_identity_environment(selection_path: Path) -> dict[str, str]:
    selection = json.loads(selection_path.read_text())
    identity = selection.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Model1 selection has no identity")
    required = (
        "model_path",
        "config_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
    )
    missing = [field for field in required if not identity.get(field)]
    if missing:
        raise RuntimeError(f"Model1 selection identity is incomplete: {missing}")
    return {
        "EXPECTED_MODEL1_PATH": str(identity["model_path"]),
        "EXPECTED_MODEL1_CONFIG_SHA256": str(identity["config_sha256"]),
        "EXPECTED_MODEL1_TOKENIZER_CONFIG_SHA256": str(identity["tokenizer_config_sha256"]),
        "EXPECTED_MODEL1_CHAT_TEMPLATE_SHA256": str(identity["chat_template_sha256"]),
        "EXPECTED_MODEL1_PROVENANCE_PATH": str(selection_path),
        "EXPECTED_MODEL1_PROVENANCE_SHA256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_stage123_cotmask_v3.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-run", help="Continue from this run after verifying all earlier provenance outputs")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    task = str(manifest.get("task", "math"))
    if task not in {"math", "code"}:
        raise RuntimeError(f"unsupported Stage123 task: {task}")
    wrappers = CODE_WRAPPERS if task == "code" else MATH_WRAPPERS
    if not args.dry_run and manifest["launch_allowed"] is not True:
        raise RuntimeError("manifest launch_allowed is false; selection review and GPU probe are still required")
    if not args.dry_run and not os.environ.get("TMUX"):
        raise RuntimeError("Stage123 queue must run inside tmux")
    if not args.dry_run:
        processes = gpu_processes()
        if processes:
            raise RuntimeError(f"GPU compute processes are active; refusing to launch: {processes}")

    selection_path = Path(manifest["paths"]["model1_selection"])
    receipt_path = Path(manifest["paths"]["dataset_receipt"])
    source_path = Path(manifest["paths"]["source_train_file"])
    selection_policy = model1_selection_policy(manifest)
    verify_dataset_receipt(receipt_path, source_path, manifest["seed"], args.dry_run, task)
    if args.dry_run and not selection_path.exists():
        selected_model = Path(f"/SELECTED_MODEL1_PENDING_STEP_{int(selection_policy['selected_step'])}")
    else:
        selected_model = selected_model_from_receipt(manifest, selection_path)
    model1_identity_env = (
        model1_identity_environment(selection_path) if task == "code" and selection_path.exists() else {}
    )
    if args.dry_run and not receipt_path.exists():
        shards = {
            name: {"path": f"/DATASET_PENDING/{name}.parquet"}
            for name in ("stage1", "stage2", "stage3", "stage1_control")
        }
    else:
        shards = json.loads(receipt_path.read_text())["shards"]

    artifact_root = Path(manifest["paths"]["artifact_root"])
    outputs, runs_to_execute = continuation_state(manifest, artifact_root, args.start_run)
    try:
        for run_spec in runs_to_execute:
            source = outputs.get(run_spec.get("source_run", ""), {})
            artifact_dir = artifact_root / run_spec["id"]
            env = dict(os.environ)
            env.update(
                {
                    "RUN_PREFIX": run_prefix(run_spec),
                    "TRAIN_FILE": shards[run_spec["train_shard"]]["path"],
                    "TOTAL_TRAINING_STEPS": str(run_spec["final_step"]),
                    "WDL_SFT_BETA": str(run_spec["beta"]),
                    "DATA_SEED": str(manifest["seed"]),
                    "DATA_SHUFFLE": "False",
                    "LR": "1e-6",
                    "LR_WARMUP_STEPS": "0",
                    "ROLLOUT_GPU_MEMORY_UTILIZATION": str(manifest["resources"]["rollout_gpu_memory_utilization"]),
                    "ACTOR_CALCULATE_ENTROPY": "False",
                    "CALCULATE_ENTROPY": "False",
                    "CODE_STAGE123_QUEUE_ADMITTED": "1",
                    "CODE_STAGE123_MANIFEST": str(args.manifest),
                    "CODE_STAGE123_MANIFEST_SHA256": os.environ.get("CODE_STAGE123_MANIFEST_SHA256", "dry-run"),
                    "CODE_STAGE123_MODEL1_SELECTION_SHA256": os.environ.get(
                        "CODE_STAGE123_MODEL1_SELECTION_SHA256", "dry-run"
                    ),
                    "CODE_STAGE123_DATASET_RECEIPT_SHA256": os.environ.get(
                        "CODE_STAGE123_DATASET_RECEIPT_SHA256", "dry-run"
                    ),
                    "STAGE123_RUN_ID": run_spec["id"],
                }
            )
            if task == "code":
                code_validation_files = [
                    "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet",
                    "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet",
                    "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet",
                ]
                env.update(model1_identity_env)
                env.update(
                    {
                        "CODE_TRAIN_FILE": env["TRAIN_FILE"],
                        "FUSION_LAMBDA": str(manifest["matrix"]["stage2_fusion_lambda"]),
                        "CODE_ONLINE_HUMANEVAL_PLUS_VAL_FILE": "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet",
                        "CODE_ONLINE_MBPP_PLUS_VAL_FILE": "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet",
                        "CODE_ONLINE_LCB_V5_SUBSET_VAL_FILE": "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet",
                        "CODE_VAL_FILES": str(code_validation_files),
                        "TEST_FILES": str(code_validation_files),
                    }
                )
            if run_spec["phase"] == "stage1":
                env["INIT_MODEL_PATH"] = str(selected_model)
            elif run_spec["phase"] == "stage1_control":
                env["INIT_MODEL_PATH"] = source.get("model", f"/SOURCE_PENDING/{run_spec['source_run']}")
            elif run_spec["phase"] == "stage2":
                stage1_model = source.get("model", f"/SOURCE_PENDING/{run_spec['source_run']}")
                env.update(
                    {
                        "BASE_MODEL_PATH": str(selected_model),
                        "MODEL2_PATH": stage1_model,
                        "STAGE1_MODEL2_PROVENANCE_FILE": str(
                            artifact_root / run_spec["source_run"] / "provenance.json"
                        ),
                        "ALLOW_EXTERNAL_MODEL2": "1" if not args.dry_run else "0",
                        "ALLOW_EXTERNAL_MODEL2_FOR_DRY_RUN": "1" if args.dry_run else "0",
                        "MODEL_PATH": str(stage2_joint_cache_path(artifact_root, run_spec["id"], task)),
                        "STAGE1_RUN_PREFIX": run_prefix(
                            next(item for item in manifest["runs"] if item["id"] == run_spec["source_run"]), task
                        ),
                        "EXPECTED_STAGE1_RUN_PREFIX": run_prefix(
                            next(item for item in manifest["runs"] if item["id"] == run_spec["source_run"]), task
                        ),
                        "STAGE1_STEP": str(
                            next(
                                item["final_step"] for item in manifest["runs"] if item["id"] == run_spec["source_run"]
                            )
                        ),
                        "STAGE2_HANDOFF_STEP": str(
                            next(
                                item["final_step"] for item in manifest["runs"] if item["id"] == run_spec["source_run"]
                            )
                        ),
                        "EXPECTED_STAGE1_BETA": str(run_spec["beta"]),
                        "SUBMODEL_KL_ENABLED": "true" if run_spec["kl"] == "m2kl" else "false",
                        "SUBMODEL_KL_MODEL1_ENABLED": "false",
                        "SUBMODEL_KL_MODEL2_ENABLED": "true" if run_spec["kl"] == "m2kl" else "false",
                        "SUBMODEL_KL_MODEL2_COEF": str(
                            manifest["matrix"]["model2_kl_coef"] if run_spec["kl"] == "m2kl" else 0.0
                        ),
                        "SUBMODEL_KL_MODEL2_REF_PATH": stage1_model,
                    }
                )
            else:
                submodel = run_spec["submodel"]
                env["STAGE2_MODEL_PATH"] = source.get(submodel, f"/SOURCE_PENDING/{run_spec['source_run']}/{submodel}")
                env["STAGE2_SUBMODEL"] = submodel
                env["STAGE2_PROVENANCE_FILE"] = str(artifact_root / run_spec["source_run"] / "provenance.json")

            started_at = time.time()
            if not args.dry_run:
                emit_event("stage_run_started", run_id=run_spec["id"], phase=run_spec["phase"])
            env["RUN_PREFIX"] = run_prefix(run_spec, task)
            execute(["bash", str(wrappers[run_spec["phase"]])], env, args.dry_run, run_spec["id"])
            if args.dry_run:
                outputs[run_spec["id"]] = {
                    "model": f"/DRY_RUN/{run_spec['id']}/model",
                    "model1": f"/DRY_RUN/{run_spec['id']}/model1",
                    "model2": f"/DRY_RUN/{run_spec['id']}/model2",
                }
                continue
            actor = checkpoint_after(env["RUN_PREFIX"], started_at, run_spec["final_step"])
            artifact_dir.mkdir(parents=True, exist_ok=False)
            if run_spec["phase"] == "stage2":
                outputs[run_spec["id"]] = merge_stage2(actor, artifact_dir)
            else:
                model = artifact_dir / "final_model"
                merge_single(actor, model)
                outputs[run_spec["id"]] = {"model": str(model)}
            provenance = {
                "schema_version": 2,
                "run": run_spec,
                "outputs": outputs[run_spec["id"]],
                "release_eligible": True,
            }
            if run_spec["phase"] == "stage2":
                provenance["source"] = {
                    "extracted_model1": outputs[run_spec["id"]]["model1"],
                    "extracted_model2": outputs[run_spec["id"]]["model2"],
                }
            (artifact_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
            emit_event(
                "stage_run_completed", run_id=run_spec["id"], phase=run_spec["phase"], outputs=outputs[run_spec["id"]]
            )
    except Exception as exc:
        if not args.dry_run:
            emit_queue_terminal("queue_failed", reason=str(exc), task=task)
        raise
    if not args.dry_run:
        emit_queue_terminal("queue_completed", task=task, run_count=len(manifest["runs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
