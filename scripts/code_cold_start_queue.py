#!/usr/bin/env python3
"""Run Qwen3-1.7B Code Cold Start with full Code-3 validation every five steps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def emit_event(path: Path, event: str, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


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


def benchmark_output_name(benchmark: str) -> str:
    return "livecodebench_samples_n3.json" if benchmark == "livecodebench" else f"{benchmark}_samples_n3.jsonl"


def run_benchmark(manifest: dict[str, Any], model_path: Path, step_dir: Path, benchmark: str, dry_run: bool) -> None:
    validation = manifest["validation"]
    validation_file = Path(validation["benchmarks"][benchmark])
    case_dir = step_dir / benchmark
    raw = case_dir / "raw_generations_n3.jsonl"
    converted = case_dir / benchmark_output_name(benchmark)
    generation_summary = case_dir / "generation_summary.json"
    conversion_report = case_dir / "conversion_report.json"
    official_summary = case_dir / "official_summary.json"
    official_output = case_dir / "official"
    if not dry_run:
        case_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [
            str(ROOT),
            "/data-1/code_eval_envs/official_site",
            "/data-1/code_eval_envs/LiveCodeBench",
            env.get("PYTHONPATH", ""),
        ]
    )
    env.update(
        {
            "PROJECT_CACHE_ROOT": "/data-1/.cache",
            "HF_HOME": "/data-1/.cache/huggingface",
            "HF_DATASETS_CACHE": "/data-1/.cache/huggingface/datasets",
            "HUGGINGFACE_HUB_CACHE": "/data-1/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/data-1/.cache/huggingface",
            "XDG_CACHE_HOME": "/data-1/.cache",
            "CODE_OFFICIAL_SOURCE_ROOT": "/data-1/dataset/code/official_sources",
            "BIGCODEBENCH_OVERRIDE_PATH": "/data-1/dataset/code/official_sources/bigcodebench/BigCodeBench-v0.1.4.jsonl",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )

    generation = [
        sys.executable,
        str(ROOT / "recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py"),
        "--model",
        str(model_path),
        "--validation-parquet",
        str(validation_file),
        "--output",
        str(raw),
        "--summary",
        str(generation_summary),
        "--tensor-parallel",
        str(validation["tensor_parallel"]),
        "--n",
        str(validation["n"]),
        "--temperature",
        str(validation["temperature"]),
        "--top-p",
        str(validation["top_p"]),
        "--top-k",
        str(validation["top_k"]),
        "--max-tokens",
        str(validation["max_tokens"]),
        "--gpu-memory-utilization",
        str(validation["gpu_memory_utilization"]),
        "--seed",
        str(validation["seed"]),
        "--enable-thinking",
        str(validation["enable_thinking"]),
    ]
    run(generation, env=env, dry_run=dry_run)
    run(
        [
            sys.executable,
            str(ROOT / "recipe/on_policy_wdl_sft/code_task/convert_official_outputs.py"),
            "--raw-outputs",
            str(raw),
            "--validation-parquet",
            str(validation_file),
            "--output",
            str(converted),
            "--benchmark",
            benchmark,
            "--report",
            str(conversion_report),
            "--allow-extraction-failures",
        ],
        env=env,
        dry_run=dry_run,
    )
    official = [
        sys.executable,
        str(ROOT / "recipe/on_policy_wdl_sft/code_task/eval_code_official.py"),
        "--benchmark",
        benchmark,
        "--output-dir",
        str(official_output),
        "--summary",
        str(official_summary),
        "--parallel",
        str(validation["official_parallel"]),
        "--overwrite",
    ]
    if benchmark == "livecodebench":
        official.extend(
            [
                "--custom-output",
                str(converted),
                "--lcb-python",
                "/opt/venv/bin/python",
                "--lcb-release-version",
                str(validation["livecodebench_release"]),
            ]
        )
    else:
        official.extend(["--samples", str(converted)])
    run(official, env=env, dry_run=dry_run)


def summarize_step(manifest: dict[str, Any], step_dir: Path, dry_run: bool) -> dict[str, Any]:
    metric_path = step_dir / "format_metrics.json"
    command = [
        sys.executable,
        str(ROOT / "recipe/on_policy_wdl_sft/code_task/summarize_code_format_contract.py"),
        "--output",
        str(metric_path),
        "--threshold",
        str(manifest["admission_thresholds"]["format_contract_success_rate"]),
    ]
    for benchmark in manifest["validation"]["benchmarks"]:
        command.extend(["--raw-output", str(step_dir / benchmark / "raw_generations_n3.jsonl")])
    run(command, dry_run=dry_run)
    if dry_run:
        return {
            "passed_format_gate": False,
            "micro_metrics": {"format_contract_success_rate": 0.0},
            "per_source": {},
        }
    return json.loads(metric_path.read_text())


def write_selection(manifest: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [candidate for candidate in candidates if candidate["passed_format_gate"]]
    if passing:
        selected = min(passing, key=lambda candidate: candidate["step"])
        selection_policy = "earliest checkpoint with micro format_contract_success_rate >= threshold"
    elif candidates:
        selected = min(
            candidates,
            key=lambda candidate: (-candidate["micro_metrics"]["format_contract_success_rate"], candidate["step"]),
        )
        selection_policy = "highest micro format_contract_success_rate; earliest step breaks ties"
    else:
        selected = None
        selection_policy = "no evaluated candidates"
    selection_path = Path(manifest["paths"]["selection_file"])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_policy": selection_policy,
                "threshold": manifest["admission_thresholds"]["format_contract_success_rate"],
                "selected": selected,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_cold_start_cotmask_v3.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    training = manifest["training"]
    event_log = paths["event_log"]

    try:
        if manifest["execution"]["launch_allowed"] is not True and not args.dry_run:
            raise RuntimeError("manifest launch_allowed is false")
        if not args.dry_run:
            processes = gpu_processes()
            if processes:
                raise RuntimeError(f"GPU compute processes are active: {processes}")
            if not os.environ.get("TMUX"):
                raise RuntimeError("Code Cold Start queue must run inside tmux")
            if paths["checkpoint_root"].exists() or paths["artifact_root"].exists() or paths["output_root"].exists():
                raise FileExistsError("fresh Code Cold Start roots must not already exist")
            emit_event(event_log, "queue_started", manifest=str(args.manifest))

        if manifest["execution"].get("requires_whole_message_loss_mask") is True:
            run(
                [
                    sys.executable,
                    str(ROOT / "recipe/on_policy_wdl_sft/format_cold_start/prepare_code_kodcode_sft_dataset.py"),
                    "--output",
                    str(paths["train_file"]),
                    "--verify-only",
                ],
                dry_run=args.dry_run,
            )
            run([sys.executable, str(ROOT / "scripts/check_sft_loss_mask_policy.py")], dry_run=args.dry_run)
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_sft_loss_mask.py"),
                    "--model",
                    str(paths["raw_model"]),
                    "--dataset",
                    str(paths["train_file"]),
                    "--output",
                    str(paths["artifact_root"] / "loss_mask_preflight.json"),
                    "--samples",
                    "-1",
                    "--seed",
                    str(manifest["seed"]),
                    "--max-length",
                    str(training["max_length"]),
                ],
                dry_run=args.dry_run,
            )

        run_name = training["run_name"]
        candidates: list[dict[str, Any]] = []
        for step in range(0, training["max_steps"] + 1, training["step_interval"]):
            if step == 0:
                model_path = paths["raw_model"]
            else:
                env = dict(os.environ)
                env.update(
                    {
                        "ALLOW_FORMAT_COLD_START_SFT": "1",
                        "RUN_NAME": run_name,
                        "RUN_PREFIX": run_name,
                        "MODEL_PATH": str(paths["raw_model"]),
                        "TRAIN_FILE": str(paths["train_file"]),
                        "CKPT_ROOT": str(paths["checkpoint_root"]),
                        "TOTAL_TRAINING_STEPS": str(step),
                        "SAVE_FREQ": str(training["step_interval"]),
                        "MAX_CKPT_TO_KEEP": str(training["max_steps"] // training["step_interval"]),
                        "TRAIN_BATCH_SIZE": str(training["train_batch_size"]),
                        "LR": str(training["learning_rate"]),
                        "LR_WARMUP_STEPS": str(training["warmup_steps"]),
                        "MAX_LENGTH": str(training["max_length"]),
                        "MAX_RESPONSE_LENGTH": str(training["max_response_length"]),
                        "MAX_TOKEN_LEN_PER_GPU": str(training["max_length"]),
                        "TRAIN_SEED": str(manifest["seed"]),
                        "DATA_SHUFFLE": "False",
                    }
                )
                run(
                    [
                        "bash",
                        str(
                            ROOT
                            / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh"
                        ),
                    ],
                    env=env,
                    dry_run=args.dry_run,
                )
                checkpoint = paths["checkpoint_root"] / run_name / f"global_step_{step}"
                model_path = paths["artifact_root"] / "candidates" / f"step_{step}"
                merge_checkpoint(checkpoint, model_path, args.dry_run)
            step_dir = paths["output_root"] / "validation" / f"step_{step}_n3"
            for benchmark in manifest["validation"]["benchmarks"]:
                run_benchmark(manifest, model_path, step_dir, benchmark, args.dry_run)
            metrics = summarize_step(manifest, step_dir, args.dry_run)
            candidate = {
                "step": step,
                "model_path": str(model_path),
                "validation_dir": str(step_dir),
                "passed_format_gate": metrics["passed_format_gate"],
                "micro_metrics": metrics["micro_metrics"],
                "per_source": metrics["per_source"],
            }
            candidates.append(candidate)
            if not args.dry_run:
                write_selection(manifest, candidates)
                emit_event(
                    event_log,
                    "candidate_evaluated",
                    step=step,
                    passed_format_gate=metrics["passed_format_gate"],
                    format_contract_success_rate=metrics["micro_metrics"]["format_contract_success_rate"],
                    per_source={
                        source: source_metrics["format_contract_success_rate"]
                        for source, source_metrics in metrics["per_source"].items()
                    },
                )

        if not args.dry_run:
            selected = write_selection(manifest, candidates)
            pause_marker = paths["pause_marker"]
            pause_marker.parent.mkdir(parents=True, exist_ok=True)
            pause_marker.write_text(
                json.dumps(
                    {
                        "selected": selected,
                        "reason": "Cold Start complete; regenerate/admit CoT-v3 Stage123 manifests before launch",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            emit_event(event_log, "queue_completed", selected=selected)
        return 0
    except Exception as exc:
        if not args.dry_run:
            emit_event(event_log, "queue_failed", reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
