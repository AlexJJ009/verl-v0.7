#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "stage1": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "stage2": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "stage3": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and "manifest_sha256" in raw and "runs" in raw:
        return raw
    manifest_tool = "stage123_matrix_manifest.py" if "base_manifest" in raw else "experiment_manifest.py"
    command = "render" if manifest_tool == "stage123_matrix_manifest.py" else "render"
    rendered = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts" / manifest_tool), command, str(path)]
        + ([] if manifest_tool == "stage123_matrix_manifest.py" else ["--format", "json"]),
        text=True,
    )
    return json.loads(rendered)


def run_by_id(manifest: dict, run_id: str) -> dict:
    for run in manifest["runs"]:
        if run["id"] == run_id:
            return run
    raise ValueError(f"unknown Stage123 run id: {run_id}")


def checkpoint_for(prefix: str, final_step: int, started_at: float) -> Path:
    root = Path("/data-1/checkpoints")
    candidates = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime)
    candidates = [path for path in candidates if path.stat().st_mtime >= started_at - 5]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one new checkpoint root for {prefix}, found {len(candidates)}")
    checkpoint = candidates[0]
    latest_file = checkpoint / "latest_checkpointed_iteration.txt"
    latest = int("".join(character for character in latest_file.read_text() if character.isdigit()))
    actor = checkpoint / f"global_step_{final_step}" / "actor"
    if latest < final_step or not actor.is_dir():
        raise RuntimeError(f"{prefix} stopped before required step {final_step}: latest={latest}")
    return checkpoint


def metrics_for(checkpoint: Path, runtime_root: Path) -> Path:
    matches = list(runtime_root.glob(f"**/metrics/OnPolicyWDLSFT-CodeTask/{checkpoint.name}.jsonl"))
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(f"expected one metrics file for {checkpoint.name}, found {len(matches)}")
    return matches[0]


def write_provenance(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError(f"provenance already exists; retry/resume is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def common_environment(manifest_path: Path, manifest: dict, run: dict) -> dict[str, str]:
    environment = dict(os.environ)
    runtime_root = Path(run["artifact_dir"]) / "runtime" / run["id"]
    log_root = runtime_root / "logs"
    environment.update(
        {
            "STAGE123_RUN_ID": run["id"],
            "STAGE123_MANIFEST": str(manifest_path),
            "RUN_PREFIX": run["run_prefix"],
            "CODE_TRAIN_FILE": run["train_file"],
            "TRAIN_FILE": run["train_file"],
            "TOTAL_TRAINING_STEPS": str(run["final_step"]),
            "DATA_SHUFFLE": "False",
            "LR": "1e-6",
            "LR_WARMUP_STEPS": "0",
            "WDL_SFT_BETA": "0.1",
            "EXPECTED_STAGE1_BETA": "0.1",
            "FUSION_LAMBDA": "0.8",
            "STAGE123_EXPECTED_PROFILE_HASH": manifest["resource_profile"]["sha256"],
            "BASE_CKPT_DIR": manifest["paths"]["checkpoint_root"],
            "LOG_DIR": str(log_root),
            "VERL_FILE_LOGGER_ROOT": str(log_root / "metrics"),
            "VALIDATION_DATA_DIR": str(log_root / "validation"),
            "WANDB_DIR": str(runtime_root / "wandb"),
            "WANDB_MODE": "offline",
            "RAY_TMPDIR": str(Path(manifest["paths"]["scratch_root"]) / "ray" / run["id"]),
        }
    )
    validation = manifest.get("validation", {})
    if validation:
        primary_metric = (
            validation.get("joint_primary_metric")
            if run["phase"] == "stage2"
            else validation.get("single_primary_metric")
        ) or validation.get("primary_metric", "val-core/HumanEval+/acc/pass@1")
        environment.update(
            {
                "VAL_N": str(validation.get("n", 1)),
                "STAGE123_EXPECTED_VAL_N": str(validation.get("n", 1)),
                "VAL_TEMPERATURE": str(validation.get("temperature", 0.2)),
                "VAL_TOP_P": str(validation.get("top_p", 0.95)),
                "TOP_K": str(validation.get("top_k", -1)),
                "VAL_DO_SAMPLE": str(validation.get("do_sample", True)),
                "BEST_CKPT_METRIC_KEY": primary_metric,
            }
        )
    return environment


def stage_environment(manifest: dict, run: dict, environment: dict[str, str]) -> dict[str, str]:
    source = run["source"]
    phase = run["phase"]
    if phase == "stage1":
        environment["INIT_MODEL_PATH"] = source["model_path"]
    elif phase == "stage2":
        model1_path = source.get(
            "model1_path", manifest["paths"].get("stage1_init_model", manifest["paths"]["base_model"])
        )
        model1_provenance_path = source.get(
            "model1_provenance_path", manifest["paths"].get("stage1_init_provenance", "")
        )
        identity_files = {
            "model1_config_sha256": Path(model1_path) / "config.json",
            "model1_tokenizer_config_sha256": Path(model1_path) / "tokenizer_config.json",
            "model1_chat_template_sha256": Path(model1_path) / "chat_template.jinja",
        }
        environment.update(
            {
                "STAGE1_RUN_PREFIX": source["run_prefix"],
                "EXPECTED_STAGE1_RUN_PREFIX": source["run_prefix"],
                "STAGE1_CKPT_DIR": source["checkpoint_root"],
                "STAGE2_HANDOFF_STEP": str(source["handoff_step"]),
                "MERGED_MODEL2_DIR": source["model2_path"],
                "MODEL2_CACHE_TAG": run["chain"],
                "TRACK_JOINT_SUBMODEL_LOSSES": "true",
                "BASE_MODEL_PATH": model1_path,
                "EXPECTED_MODEL1_PATH": model1_path,
                "EXPECTED_MODEL1_CONFIG_SHA256": source.get(
                    "model1_config_sha256", file_sha256(identity_files["model1_config_sha256"])
                ),
                "EXPECTED_MODEL1_TOKENIZER_CONFIG_SHA256": source.get(
                    "model1_tokenizer_config_sha256", file_sha256(identity_files["model1_tokenizer_config_sha256"])
                ),
                "EXPECTED_MODEL1_CHAT_TEMPLATE_SHA256": source.get(
                    "model1_chat_template_sha256", file_sha256(identity_files["model1_chat_template_sha256"])
                ),
                "EXPECTED_MODEL1_PROVENANCE_PATH": model1_provenance_path,
                "EXPECTED_MODEL1_PROVENANCE_SHA256": source.get(
                    "model1_provenance_sha256",
                    file_sha256(Path(model1_provenance_path)) if model1_provenance_path else "",
                ),
            }
        )
        validation_views = run.get("validation_views")
        if validation_views:
            environment["JOINT_VALIDATION_VIEWS"] = "[" + ",".join(validation_views) + "]"
        submodel_kl = run.get("submodel_kl", {})
        if submodel_kl:
            environment.update(
                {
                    "SUBMODEL_KL_ENABLED": str(submodel_kl.get("enabled", False)).lower(),
                    "SUBMODEL_KL_MODEL1_ENABLED": str(submodel_kl.get("model1_enabled", False)).lower(),
                    "SUBMODEL_KL_MODEL1_COEF": str(submodel_kl.get("model1_coef", 0.0)),
                    "SUBMODEL_KL_MODEL2_ENABLED": str(submodel_kl.get("model2_enabled", False)).lower(),
                    "SUBMODEL_KL_MODEL2_COEF": str(submodel_kl.get("model2_coef", 0.0)),
                    "SUBMODEL_KL_MODEL2_REF_PATH": submodel_kl.get("model2_ref_path", source["model2_path"]),
                }
            )
    elif phase == "stage3":
        submodel = source.get("submodel", "model2")
        if submodel not in {"model1", "model2"}:
            raise ValueError(f"unsupported Stage3 source submodel: {submodel}")
        model_path = source.get("model_path") or source.get(f"{submodel}_path")
        provenance_file = source.get("provenance_file")
        if model_path is None or provenance_file is None:
            stage2 = run_by_id(manifest, source["run_id"])
            model_path = str(Path(stage2["artifact_dir"]) / f"stage2_final_{submodel}")
            provenance_file = stage2["provenance_file"]
        environment.update(
            {
                "STAGE2_SUBMODEL": submodel,
                "STAGE2_MODEL_PATH": model_path,
                "STAGE2_MODEL2_PATH": model_path,
                "STAGE2_PROVENANCE_FILE": provenance_file,
            }
        )
    return environment


def execute_wrapper(
    manifest_path: Path, manifest: dict, run: dict, *, dry_run: bool
) -> tuple[Path | None, Path | None]:
    environment = stage_environment(manifest, run, common_environment(manifest_path, manifest, run))
    command = ["bash", str(WRAPPERS[run["phase"]])]
    if dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        key: environment[key]
                        for key in sorted(environment)
                        if key.startswith("STAGE123_")
                        or key.startswith("SUBMODEL_KL_")
                        or key.startswith("EXPECTED_MODEL1_")
                        or key
                        in {
                            "RUN_PREFIX",
                            "INIT_MODEL_PATH",
                            "BASE_MODEL_PATH",
                            "STAGE1_CKPT_DIR",
                            "STAGE2_HANDOFF_STEP",
                            "MERGED_MODEL2_DIR",
                            "STAGE2_SUBMODEL",
                            "STAGE2_MODEL_PATH",
                            "STAGE2_MODEL2_PATH",
                            "STAGE2_PROVENANCE_FILE",
                            "JOINT_VALIDATION_VIEWS",
                            "VAL_N",
                            "VAL_TEMPERATURE",
                            "VAL_TOP_P",
                            "TOP_K",
                            "VAL_DO_SAMPLE",
                            "BEST_CKPT_METRIC_KEY",
                            "CODE_TRAIN_FILE",
                            "TOTAL_TRAINING_STEPS",
                            "LR",
                            "LR_WARMUP_STEPS",
                            "TRACK_JOINT_SUBMODEL_LOSSES",
                            "BASE_CKPT_DIR",
                            "LOG_DIR",
                            "VERL_FILE_LOGGER_ROOT",
                            "VALIDATION_DATA_DIR",
                            "WANDB_DIR",
                            "WANDB_MODE",
                            "RAY_TMPDIR",
                        }
                    },
                },
                sort_keys=True,
            )
        )
        return None, None
    existing = list(Path(manifest["paths"]["checkpoint_root"]).glob(f"{run['run_prefix']}_*"))
    if existing:
        raise RuntimeError(f"existing checkpoint root forbids automatic retry/resume for {run['id']}: {existing}")
    started_at = time.time()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    checkpoint = checkpoint_for(run["run_prefix"], int(run["final_step"]), started_at)
    return checkpoint, metrics_for(checkpoint, Path(run["artifact_dir"]) / "runtime" / run["id"])


def finalize(manifest: dict, run: dict, checkpoint: Path, metrics: Path) -> None:
    payload = {
        "schema_version": 1,
        "run_id": run["id"],
        "phase": run["phase"],
        "manifest_sha256": manifest["manifest_sha256"],
        "train_file": run["train_file"],
        "train_file_sha256": run["train_file_sha256"],
        "checkpoint": str(checkpoint),
        "final_step": run["final_step"],
        "metrics": str(metrics),
        "metrics_sha256": file_sha256(metrics),
        "release_eligible": True,
    }
    if run["phase"] == "stage2":
        joint = Path(run["artifact_dir"]) / "stage2_final_joint"
        model1 = Path(run["artifact_dir"]) / "stage2_final_model1"
        model2 = Path(run["artifact_dir"]) / "stage2_final_model2"
        if joint.exists() or model1.exists() or model2.exists():
            raise RuntimeError("Stage2 extraction target already exists; retry/resume is forbidden")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(checkpoint / f"global_step_{run['final_step']}" / "actor"),
                "--target_dir",
                str(joint),
                "--trust-remote-code",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "recipe/joint_training/extract_sub_model.py"),
                "--joint_model_path",
                str(joint),
                "--output_path",
                str(model1),
                "--sub_model_index",
                "0",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "recipe/joint_training/extract_sub_model.py"),
                "--joint_model_path",
                str(joint),
                "--output_path",
                str(model2),
                "--sub_model_index",
                "1",
            ],
            cwd=ROOT,
            check=True,
        )
        payload["source"] = {
            "type": "stage2_complete",
            "extracted_model1": str(model1),
            "extracted_model2": str(model2),
            "joint_model": str(joint),
        }
    elif run["phase"] == "stage3":
        submodel = run["source"].get("submodel", "model2")
        payload["source"] = {
            "type": "stage2_submodel",
            "run_id": run["source"]["run_id"],
            "submodel": submodel,
            "model_path": os.environ.get(
                "STAGE2_MODEL_PATH", str(Path(run["artifact_dir"]) / f"stage2_final_{submodel}")
            ),
        }
    else:
        payload["source"] = {"type": "matched_stage1_control", **run["source"]}
    write_provenance(Path(run["provenance_file"]), payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    run = run_by_id(manifest, args.run_id)
    checkpoint, metrics = execute_wrapper(args.manifest, manifest, run, dry_run=args.dry_run)
    if not args.dry_run:
        assert checkpoint is not None and metrics is not None
        finalize(manifest, run, checkpoint, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
