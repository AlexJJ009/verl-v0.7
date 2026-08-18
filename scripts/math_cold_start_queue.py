#!/usr/bin/env python3
"""Run guarded Qwen3-1.7B math cold-start in five-step increments with full Math-7 validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def emit_event(event_log: Path, event: str, **payload: object) -> None:
    event_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with event_log.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def validation_command(manifest: dict, model_path: Path, output_dir: Path) -> list[str]:
    validation = manifest["validation"]
    return [
        sys.executable,
        str(ROOT / "recipe/joint_training/offline_eval.py"),
        "--model_path",
        str(model_path),
        "--tensor_parallel",
        "8",
        "--n",
        str(validation["n"]),
        "--temperature",
        str(validation["temperature"]),
        "--top_p",
        str(validation["top_p"]),
        "--max_tokens",
        str(validation["max_tokens"]),
        "--gpu_memory_utilization",
        str(validation["gpu_memory_utilization"]),
        "--seed",
        str(validation["seed"]),
        "--output_dir",
        str(output_dir),
        "--test_files",
        *manifest["validation"]["datasets"],
    ]


def passes_thresholds(metrics_path: Path, thresholds: dict) -> tuple[bool, dict]:
    metrics = json.loads(metrics_path.read_text())
    micro = metrics["micro_metrics"]
    checks = {
        "format_contract_success_rate": micro["format_contract_success_rate"]
        >= thresholds["format_contract_success_rate"],
    }
    return all(checks.values()), {"micro_metrics": micro, "macro_metrics": metrics["macro_metrics"], "checks": checks}


def backfill_format_contract_metric(output_dir: Path) -> None:
    metrics_path = output_dir / "eval_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if "format_contract_success_rate" in metrics.get("micro_metrics", {}):
        return
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("step-zero metric backfill must run inside the verl harness") from exc

    details = pd.read_parquet(output_dir / "eval_details.parquet")
    details["format_contract_success"] = (
        details["think_complete"].astype(bool)
        & details["answer_complete"].astype(bool)
        & details["boxed_extraction_success"].astype(bool)
        & details["reward_grader_success"].astype(bool)
        & details["has_eos"].astype(bool)
        & ~details["truncated"].astype(bool)
    )
    per_source = details.groupby("data_source")["format_contract_success"].mean().to_dict()
    for source, rate in per_source.items():
        metrics["metrics"][source]["format_contract_success_rate"] = float(rate)
    metrics["macro_metrics"]["format_contract_success_rate"] = sum(per_source.values()) / len(per_source)
    micro_metrics = {"response_count": int(len(details))}
    source_columns = {
        "think_complete_rate": "think_complete",
        "answer_complete_rate": "answer_complete",
        "boxed_extraction_success_rate": "boxed_extraction_success",
        "reward_grader_success_rate": "reward_grader_success",
        "format_contract_success_rate": "format_contract_success",
        "eos_rate": "has_eos",
        "truncation_rate": "truncated",
    }
    for rate_name, column in source_columns.items():
        micro_metrics[rate_name] = float(details[column].astype(bool).mean())
    metrics["micro_metrics"] = micro_metrics
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def merge_checkpoint(checkpoint: Path, target: Path, dry_run: bool) -> None:
    run(
        [
            "bash",
            str(ROOT / "recipe/on_policy_wdl_sft/format_cold_start/merge_sft_checkpoint.sh"),
            "--checkpoint-dir",
            str(checkpoint),
            "--target-dir",
            str(target),
        ],
        dry_run=dry_run,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_cold_start_cotmask_v3.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    event_log = paths["event_log"]

    try:
        if manifest["execution"].get("requires_whole_message_loss_mask") is True:
            run([sys.executable, str(ROOT / "scripts/check_sft_loss_mask_policy.py")], dry_run=args.dry_run)
        if manifest["execution"]["launch_allowed"] is not True and not args.dry_run:
            raise RuntimeError("manifest launch_allowed is false; GPU probe/admission must update the manifest first")
        if not args.dry_run:
            processes = gpu_processes()
            if processes:
                raise RuntimeError(f"GPU compute processes are active; refusing to launch: {processes}")
            if not os.environ.get("TMUX"):
                raise RuntimeError("cold-start queue must run inside tmux")

        artifact_root = paths["artifact_root"]
        run_name = manifest["training"]["run_name"]
        candidates: list[dict] = []
        if not args.dry_run:
            emit_event(event_log, "queue_started", phase="cold_start", manifest=str(args.manifest))

        run(
            [
                sys.executable,
                str(ROOT / "recipe/on_policy_wdl_sft/math_task/prepare_qwen3_1p7b_math_stage123_data.py"),
                "--source",
                str(paths["source_train_file"]),
                "--output-root",
                str(paths["dataset_receipt"].parent),
                "--seed",
                str(manifest["seed"]),
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                sys.executable,
                str(ROOT / "recipe/on_policy_wdl_sft/format_cold_start/prepare_math_sft_dataset.py"),
                "--input",
                str(paths["cold_start_rl_file"]),
                "--output",
                str(paths["cold_start_sft_file"]),
                "--seed",
                str(manifest["seed"]),
            ],
            dry_run=args.dry_run,
        )
        if manifest["execution"].get("requires_whole_message_loss_mask") is True:
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_sft_loss_mask.py"),
                    "--model",
                    str(paths["raw_model"]),
                    "--dataset",
                    str(paths["cold_start_sft_file"]),
                    "--output",
                    str(paths["loss_mask_preflight_receipt"]),
                    "--samples",
                    "32",
                    "--seed",
                    str(manifest["seed"]),
                    "--max-length",
                    str(manifest["training"]["max_length"]),
                ],
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                emit_event(
                    event_log,
                    "loss_mask_preflight_passed",
                    receipt=str(paths["loss_mask_preflight_receipt"]),
                )

        for step in range(0, manifest["training"]["max_steps"] + 1, manifest["training"]["step_interval"]):
            if step == 0:
                model_path = paths["raw_model"]
                output_dir = paths["step_zero_validation"]
            else:
                env = dict(os.environ)
                env.update(
                    {
                        "ALLOW_FORMAT_COLD_START_SFT": "1",
                        "RUN_NAME": run_name,
                        "RUN_PREFIX": run_name,
                        "MODEL_PATH": str(paths["raw_model"]),
                        "TRAIN_FILE": str(paths["cold_start_sft_file"]),
                        "CKPT_ROOT": str(paths["checkpoint_root"]),
                        "TOTAL_TRAINING_STEPS": str(step),
                        "SAVE_FREQ": str(manifest["training"]["step_interval"]),
                        "TRAIN_BATCH_SIZE": str(manifest["training"]["train_batch_size"]),
                        "LR": str(manifest["training"]["learning_rate"]),
                        "MAX_LENGTH": str(manifest["training"]["max_length"]),
                        "MAX_TOKEN_LEN_PER_GPU": str(manifest["training"]["max_length"]),
                    }
                )
                run(
                    ["bash", str(ROOT / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_math_qwen3_1p7b_format.sh")],
                    env=env,
                    dry_run=args.dry_run,
                )
                checkpoint = paths["checkpoint_root"] / run_name / f"global_step_{step}"
                model_path = artifact_root / "candidates" / f"step_{step}"
                if model_path.exists():
                    raise FileExistsError(f"cold-start candidate path already exists: {model_path}")
                merge_checkpoint(checkpoint, model_path, args.dry_run)
                output_dir = artifact_root / "validation" / f"step_{step}_n1"
                if output_dir.exists():
                    raise FileExistsError(f"cold-start validation path already exists: {output_dir}")

            if output_dir.exists() and not args.dry_run:
                raise FileExistsError(f"cold-start validation path already exists: {output_dir}")
            run(validation_command(manifest, model_path, output_dir), dry_run=args.dry_run)
            if args.dry_run:
                continue
            passed, evidence = passes_thresholds(output_dir / "eval_metrics.json", manifest["admission_thresholds"])
            candidate = {"step": step, "model_path": str(model_path), "passed_format_gate": passed, **evidence}
            candidates.append(candidate)
            artifact_root.mkdir(parents=True, exist_ok=True)
            (artifact_root / "cold_start_candidates.json").write_text(
                json.dumps({"schema_version": 1, "candidates": candidates}, indent=2, sort_keys=True) + "\n"
            )
            emit_event(
                event_log,
                "cold_candidate_evaluated",
                step=step,
                passed=passed,
                format_contract_success_rate=evidence["micro_metrics"]["format_contract_success_rate"],
                metrics_path=str(output_dir / "eval_metrics.json"),
            )
            if passed and manifest["execution"]["stop_after_first_passing_checkpoint"]:
                break

        if args.dry_run:
            return 0
        passing = [candidate for candidate in candidates if candidate["passed_format_gate"]]
        if not passing:
            raise RuntimeError("no cold-start checkpoint reached the 95% complete-format gate")
        selected = min(passing, key=lambda item: item["step"])
        run(
            [
                sys.executable,
                str(ROOT / "scripts/select_math_model1.py"),
                "--step",
                str(selected["step"]),
                "--artifact-root",
                str(artifact_root),
                "--review-note",
                "Automatically selected the earliest checkpoint satisfying the pre-registered 95% complete Math-7 n=1 format-contract gate.",
            ]
        )
        emit_event(
            event_log,
            "model1_selected",
            step=selected["step"],
            model_path=selected["model_path"],
            format_contract_success_rate=selected["micro_metrics"]["format_contract_success_rate"],
        )
        if manifest["execution"]["continue_to_stage123"]:
            emit_event(event_log, "stage123_handoff_started")
            env = dict(os.environ)
            env["MATH_EVENT_LOG"] = str(event_log)
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/math_stage123_queue.py"),
                    "--manifest",
                    str(Path(manifest["execution"]["stage123_manifest"])),
                ],
                env=env,
            )
        emit_event(event_log, "queue_completed")
        return 0
    except Exception as exc:
        if not args.dry_run:
            emit_event(event_log, "queue_failed", reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
