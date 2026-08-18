#!/usr/bin/env python3
"""Fail-closed beta=0 Code A/D0/C P60 queue."""

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
    "arm-a-stage1-continuation": ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_wdl_acd0_arm_a.sh",
    "arm-d0-matched-scale-no-weak": ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_wdl_acd0_arm_d0.sh",
    "arm-c-mixture": ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_wdl_acd0_arm_c.sh",
}


def select_runs(manifest: dict, only_run_id: str | None) -> list[dict]:
    """Return manifest-ordered runs, optionally narrowed to one exact arm."""
    run_map = {run["id"]: run for run in manifest["runs"]}
    if only_run_id is not None:
        if only_run_id not in run_map or only_run_id not in manifest["queue_order"]:
            raise RuntimeError(f"run id is not in the admitted queue: {only_run_id}")
        return [run_map[only_run_id]]
    return [run_map[run_id] for run_id in manifest["queue_order"]]


def joint_model_cache_path(run: dict, timestamp: int, env: dict[str, str]) -> Path:
    """Build a short, arm-specific, run-unique Transformers module cache path."""
    arm_tags = {
        "arm-d0-matched-scale-no-weak": "d0",
        "arm-c-mixture": "c",
    }
    if run.get("model_kind") != "joint" or run["id"] not in arm_tags:
        raise RuntimeError(f"joint cache path requested for non-joint arm: {run['id']}")
    cache_root = Path(env.get("JOINT_MODEL_CACHE_ROOT", "/data-1/.cache/huggingface"))
    path = cache_root / f"code-acd0-{arm_tags[run['id']]}-{timestamp}"
    transformed_basename = path.name.replace("-", "_hyphen_")
    if len(transformed_basename) > 180:
        raise RuntimeError(f"joint model cache basename is too long: {path.name}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict, require_launch: bool) -> None:
    if require_launch and manifest.get("launch_allowed") is not True:
        raise RuntimeError("manifest launch_allowed is false")
    paths, identity = manifest["paths"], manifest["identity"]
    checks = {
        Path(paths["model1"]) / "format_cold_start_source.json": identity["model1_source_sha256"],
        Path(paths["model1"]) / "config.json": identity["model1_config_sha256"],
        Path(paths["model1"]) / "model.safetensors": identity["model1_weights_sha256"],
        Path(paths["model2_provenance"]): identity["model2_provenance_sha256"],
        Path(paths["model2"]) / "config.json": identity["model2_config_sha256"],
        Path(paths["model2"]) / "model.safetensors": identity["model2_weights_sha256"],
        Path(paths["dataset_receipt"]): identity["dataset_receipt_sha256"],
        Path(paths["train_file"]): identity["train_sha256"],
        ROOT / manifest["evaluator_contract"]["reward_path"]: identity["reward_sha256"],
        ROOT / "verl/workers/reward_manager/dapo.py": identity["dapo_reward_manager_sha256"],
        ROOT / "verl/experimental/reward_loop/reward_manager/dapo.py": identity[
            "async_dapo_reward_manager_sha256"
        ],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"identity mismatch: {path}")
    contract = manifest["training_contract"]
    required = {
        "beta": 0.0,
        "negative_loss_gradient_contribution": 0.0,
        "loss_mode": "wdl_sft",
        "kl_enabled": False,
        "rollout_source": "model2",
        "final_step": 60,
        "lr": 1e-6,
        "lr_warmup_steps": 0,
        "data_shuffle": False,
        "train_prompt_batch_size": 64,
        "rollouts_per_prompt": 8,
        "validation_frequency": 5,
        "save_frequency": 5,
        "protected_checkpoint_steps": [20, 40, 45, 50, 60],
        "stage3": "omitted",
    }
    if contract != required:
        raise RuntimeError("training contract differs from frozen beta=0 A/C/D0 P60 contract")
    if manifest["queue_order"] != [
        "arm-a-stage1-continuation",
        "arm-d0-matched-scale-no-weak",
        "arm-c-mixture",
    ]:
        raise RuntimeError("queue order must be A -> D0 -> C")
    run_map = {run["id"]: run for run in manifest["runs"]}
    expected_prefixes = {
        "arm-a-stage1-continuation": "CODE-WDL-ACD0-P60-ARM-A-QWEN3-1P7B",
        "arm-d0-matched-scale-no-weak": "CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B",
        "arm-c-mixture": "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B",
    }
    if {run_id: run_map[run_id].get("run_prefix") for run_id in expected_prefixes} != expected_prefixes:
        raise RuntimeError("run prefixes differ from the frozen collision-safe contract")
    if (run_map["arm-c-mixture"]["fusion_lambda"], run_map["arm-c-mixture"]["fusion_mode"]) != (0.8, "mixture"):
        raise RuntimeError("Arm C must be lambda=0.8 mixture")
    if (run_map["arm-d0-matched-scale-no-weak"]["fusion_lambda"], run_map["arm-d0-matched-scale-no-weak"]["fusion_mode"]) != (0.8, "strong_scaled"):
        raise RuntimeError("Arm D0 must be lambda=0.8 strong_scaled")


def require_receipts(manifest: dict) -> None:
    receipt_names = [
        "stage1_reuse_receipt",
        "eos_regression_receipt",
        "evaluator_receipt",
        "gpu_probe_receipt",
        "review_receipt",
    ]
    for name in receipt_names:
        receipt_path = Path(manifest["paths"][name])
        if not receipt_path.is_file():
            raise RuntimeError(f"admission receipt missing: {name}: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "pass":
            raise RuntimeError(f"admission receipt did not pass: {name}")
        if name == "stage1_reuse_receipt" and receipt.get("decision") != "reuse_allowed":
            raise RuntimeError("Stage1 reuse quality gate did not allow reuse")


def require_clean_targets(manifest: dict, run: dict) -> None:
    checkpoint_root = Path(manifest["paths"]["checkpoint_root"])
    collisions = sorted(checkpoint_root.glob(f"{run['run_prefix']}_*"))
    artifact_root = Path(manifest["paths"]["artifact_root"])
    for candidate in (artifact_root / "runs" / run["id"], artifact_root / "state" / run["id"]):
        if candidate.exists():
            collisions.append(candidate)
    if collisions:
        raise RuntimeError(
            "refusing implicit overwrite/resume for "
            f"{run['id']}: " + ", ".join(str(path) for path in collisions)
        )


def _metrics_path(run_name: str, env: dict[str, str]) -> Path:
    metrics_root = Path(
        env.get("VERL_FILE_LOGGER_ROOT", ROOT / "recipe/on_policy_wdl_sft/code_task/metrics")
    )
    return metrics_root / env.get("WANDB_PROJECT", "OnPolicyWDLSFT-CodeTask") / f"{run_name}.jsonl"


def _metrics_has_step(metrics_path: Path, final_step: int) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except (TypeError, ValueError):
                continue
            data = payload.get("data", {})
            observed = payload.get("step", data.get("training/global_step"))
            try:
                if int(observed) == final_step:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _record_gate_event(
    *,
    run: dict,
    run_name: str,
    status: str,
    checkpoint: Path,
    metrics: Path,
    final_step: int,
    observed_step: int | None,
    env: dict[str, str],
    notes: str,
) -> None:
    state = Path(
        env.get("TRAINING_RELEASE_GATE_STATE", "/data-1/experiment_registry/training_release_gate.jsonl")
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/training_result_release_gate.py"),
        "--state",
        str(state),
        "record",
        "--run-name",
        run_name,
        "--family",
        run["run_prefix"],
        "--status",
        status,
        "--source",
        "code_wdl_acd0_queue",
        "--final-step",
        str(final_step),
        "--notes",
        notes,
    ]
    if checkpoint.exists():
        command.extend(["--checkpoint", str(checkpoint)])
    if metrics.is_file():
        command.extend(["--metrics", str(metrics)])
    if observed_step is not None:
        command.extend(["--observed-step", str(observed_step)])
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def record_terminal_outcome(
    manifest: dict,
    run: dict,
    run_name: str,
    *,
    completed: bool,
    env: dict[str, str],
    notes: str | None = None,
) -> None:
    """Record one fail-closed terminal event; never publish results."""
    final_step = int(manifest["training_contract"]["final_step"])
    checkpoint = Path(manifest["paths"]["checkpoint_root"]) / run_name
    latest_marker = checkpoint / "latest_checkpointed_iteration.txt"
    metrics = _metrics_path(run_name, env)
    observed_step: int | None = None
    try:
        observed_step = int(latest_marker.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        pass

    evidence_complete = (
        (checkpoint / f"global_step_{final_step}").is_dir()
        and observed_step == final_step
        and _metrics_has_step(metrics, final_step)
    )
    if completed and evidence_complete:
        _record_gate_event(
            run=run,
            run_name=run_name,
            status="success_complete",
            checkpoint=checkpoint,
            metrics=metrics,
            final_step=final_step,
            observed_step=observed_step,
            env=env,
            notes=notes or "Wrapper completed with final checkpoint and metrics evidence.",
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/training_result_release_gate.py"),
                "--state",
                env.get("TRAINING_RELEASE_GATE_STATE", "/data-1/experiment_registry/training_release_gate.jsonl"),
                "check",
                "--run-name",
                run_name,
                "--family",
                run["run_prefix"],
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        return

    failure_notes = notes or (
        "Wrapper completed but terminal success evidence incomplete."
        if completed
        else "Wrapper failed before terminal success evidence was verified."
    )
    _record_gate_event(
        run=run,
        run_name=run_name,
        status="failed",
        checkpoint=checkpoint,
        metrics=metrics,
        final_step=final_step,
        observed_step=observed_step,
        env=env,
        notes=failure_notes,
    )
    if completed:
        raise RuntimeError(f"terminal success evidence incomplete for {run_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_wdl_acd0_p60_beta0.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-run-id",
        choices=tuple(WRAPPERS),
        help="Run exactly one admitted arm without changing the frozen manifest order.",
    )
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, require_launch=not args.dry_run)
    if not args.dry_run:
        require_receipts(manifest)
        if not os.environ.get("TMUX"):
            raise RuntimeError("real A/C/D0 queue must run inside tmux")
    for run in select_runs(manifest, args.only_run_id):
        run_id = run["id"]
        if not args.dry_run:
            require_clean_targets(manifest, run)
        env = dict(os.environ)
        env.update({
            "RUN_PREFIX": run["run_prefix"],
            "INIT_MODEL_PATH": manifest["paths"]["model2"],
            "BASE_MODEL_PATH": manifest["paths"]["model1"],
            "EXPECTED_MODEL1_PATH": manifest["paths"]["model1"],
            "MODEL2_PATH": manifest["paths"]["model2"],
            "STAGE1_MODEL2_PROVENANCE_FILE": manifest["paths"]["model2_provenance"],
            "TRAIN_FILE": manifest["paths"]["train_file"],
            "BASE_CKPT_DIR": manifest["paths"]["checkpoint_root"],
            "FUSION_LAMBDA": str(run.get("fusion_lambda") or ""),
            "FUSION_MODE": str(run.get("fusion_mode") or ""),
            "DRY_RUN": "1" if args.dry_run else "0",
        })
        env.setdefault("LOG_DIR", str(Path(manifest["paths"]["artifact_root"]) / "logs"))
        env.setdefault("VERL_FILE_LOGGER_ROOT", str(Path(env["LOG_DIR"]) / "metrics"))
        launch_timestamp = int(time.time())
        run_name = f"{run['run_prefix']}_{launch_timestamp}"
        env["WANDB_RUN_NAME"] = run_name
        if run.get("model_kind") == "joint":
            env["MODEL_PATH"] = str(joint_model_cache_path(run, launch_timestamp, env))
        payload = {
            "run_id": run_id,
            "wrapper": str(WRAPPERS[run_id]),
            "dry_run": args.dry_run,
            "model_path": env.get("MODEL_PATH"),
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        try:
            subprocess.run(["bash", str(WRAPPERS[run_id])], cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError:
            if not args.dry_run:
                record_terminal_outcome(
                    manifest,
                    run,
                    run_name,
                    completed=False,
                    env=env,
                    notes="Wrapper exited nonzero before terminal success was verified.",
                )
            raise
        if not args.dry_run:
            record_terminal_outcome(manifest, run, run_name, completed=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
