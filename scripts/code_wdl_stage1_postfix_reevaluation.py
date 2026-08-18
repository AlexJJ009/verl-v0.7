#!/usr/bin/env python3
"""Run and finalize the frozen Code Stage1 step40 post-fix reevaluation.

The real launch is validation-only and must run inside tmux.  Finalization
creates a candidate reuse receipt next to the reevaluation artifacts; it never
overwrites the manifest-owned admission receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ADMISSION_RECEIPT = (
    "/data-2/model_weights/code_task/qwen3_1p7b_wdl_acd0_p60/"
    "admission/stage1_reuse_receipt.json"
)
HISTORICAL_BASELINE = (
    ROOT
    / "recipe/on_policy_wdl_sft/code_task/validation/"
    "CODE-B0_STAGE1-QWEN3-1P7B-COTMASK-V3-AUTHOR-SIGNATURE-V2-STEP20_1784965213/0.jsonl"
)

FROZEN_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "evaluation_kind": "post_fix_reevaluation",
    "logical_checkpoint_step": 40,
    "model": {
        "path": (
            "/data-2/model_weights/code_task/"
            "qwen3_1p7b_stage123_cotmask_v3_author_signature_v2_step20/"
            "b0-stage1/final_model"
        ),
        "config_sha256": "a4a451865e8d45a519133031f19cda7d347813159fde1756d63e2beaf67f2288",
        "weights_sha256": "a6c69262975ada9e1bc5054128d9f6f79b14167653ba817809bc771799d43c74",
    },
    "train_file": (
        "/data-1/dataset/code/verl_rl/"
        "qwen3_1p7b_code_stage123_author_signature_v2_seed20260706/stage1.parquet"
    ),
    "validation": {
        "HumanEval+": {
            "path": "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet",
            "parquet_rows": 164,
            "expected_eval_rows": 164,
            "sha256": "e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab",
        },
        "MBPP+": {
            "path": "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet",
            "parquet_rows": 378,
            "expected_eval_rows": 378,
            "sha256": "3221e7f53c88bfbd91d788fb7bcb37168fb088fa504fddf12b9126c2147312d2",
        },
        "LiveCodeBench": {
            "path": "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet",
            "parquet_rows": 880,
            "expected_eval_rows": 837,
            "sha256": "fe7d2bfe2779bcf106492347ca173e30b9220c15c1b8783949d35edcd93a43d1",
        },
    },
    "generation": {
        "n": 3,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_response_length": 8192,
    },
    "expected_response_rows": 4137,
    "runtime_hashes": {
        "recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py": (
            "2854639c4bd3e34b89b3b4d53d553406b46a800fb44ff0c3657670f2792c59a2"
        ),
        "verl/workers/reward_manager/dapo.py": (
            "4d05aaf514a199bca81d393d9d057eeb5f38b7067303e010f16a4c2b17c4829b"
        ),
        "verl/experimental/reward_loop/reward_manager/dapo.py": (
            "54c24d5df68c0c6afc86b534e2ecd0fef3842de3b6295d71dbfbac852dc701ba"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production container owns this dependency
        raise RuntimeError("pyarrow is required inside verl-harness for parquet identity checks") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required input missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"identity mismatch: {path}: expected={expected} actual={actual}")


def validate_frozen_inputs() -> dict[str, Any]:
    contract = FROZEN_CONTRACT
    model = Path(contract["model"]["path"])
    _require_hash(model / "config.json", contract["model"]["config_sha256"])
    _require_hash(model / "model.safetensors", contract["model"]["weights_sha256"])
    train_file = Path(contract["train_file"])
    if not train_file.is_file():
        raise RuntimeError(f"required Stage1 train file missing: {train_file}")
    observed_rows = 0
    for source, item in contract["validation"].items():
        path = Path(item["path"])
        _require_hash(path, item["sha256"])
        rows = parquet_rows(path)
        if rows != item["parquet_rows"]:
            raise RuntimeError(
                f"parquet row-count mismatch for {source}: expected={item['parquet_rows']} actual={rows}"
            )
        observed_rows += item["expected_eval_rows"] * contract["generation"]["n"]
    if observed_rows != contract["expected_response_rows"]:
        raise RuntimeError(
            f"Code-3 response-count mismatch: expected={contract['expected_response_rows']} actual={observed_rows}"
        )
    for relative, expected in contract["runtime_hashes"].items():
        _require_hash(ROOT / relative, expected)
    return {"status": "pass", "expected_response_rows": observed_rows}


def validate_launch_guard(
    output_root: Path,
    *,
    real_run: bool,
    environ: Mapping[str, str] | None = None,
) -> None:
    environ = os.environ if environ is None else environ
    if real_run and not environ.get("TMUX"):
        raise RuntimeError("real Stage1 post-fix reevaluation must run inside tmux")
    if real_run and output_root.exists():
        raise RuntimeError(f"output collision: refusing overwrite/resume: {output_root}")


def validate_candidate_receipt_path(path: Path) -> None:
    if path.resolve() == Path(ADMISSION_RECEIPT).resolve():
        raise RuntimeError("candidate output must not overwrite the admission receipt")


def _gpu_facts() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, name, memory = (part.strip() for part in line.split(",", 2))
        rows.append({"index": int(index), "name": name, "memory_total_mib": int(memory)})
    if len(rows) != 8 or any(row["name"] != "NVIDIA L40S" for row in rows):
        raise RuntimeError("real reevaluation requires exactly 8 visible NVIDIA L40S GPUs")
    return rows


def _evaluation_environment(output_root: Path) -> dict[str, str]:
    validation_paths = [item["path"] for item in FROZEN_CONTRACT["validation"].values()]
    return {
        "RUN_PREFIX": "CODE-WDL-STAGE1-STEP40-POSTFIX-REEVAL",
        "INIT_MODEL_PATH": FROZEN_CONTRACT["model"]["path"],
        "CODE_TRAIN_FILE": FROZEN_CONTRACT["train_file"],
        "TRAIN_FILE": FROZEN_CONTRACT["train_file"],
        "CODE_VAL_FILES": repr(validation_paths),
        "TEST_FILES": repr(validation_paths),
        "LOSS_MODE": "wdl_sft",
        "WDL_SFT_BETA": "0.0",
        "LR": "1e-6",
        "LR_WARMUP_STEPS": "0",
        "TOTAL_TRAINING_STEPS": "0",
        "VAL_N": "3",
        "STAGE123_EXPECTED_VAL_N": "3",
        "VAL_TEMPERATURE": "0.2",
        "VAL_TOP_P": "0.95",
        "VAL_DO_SAMPLE": "True",
        "MAX_RESPONSE_LENGTH": "8192",
        "VAL_BEFORE_TRAIN": "True",
        "VAL_MAX_SAMPLES": "-1",
        "DATA_SHUFFLE": "False",
        "WANDB_MODE": "disabled",
        "KEEP_BEST_CKPT": "False",
        "MAX_ACTOR_CKPTS_TO_KEEP": "0",
        "MAX_CRITIC_CKPTS_TO_KEEP": "0",
        "BASE_CKPT_DIR": str(output_root / "checkpoints"),
        "LOG_DIR": str(output_root / "logs"),
        "VERL_FILE_LOGGER_ROOT": str(output_root / "metrics"),
        "VALIDATION_DATA_DIR": str(output_root / "raw_validation"),
        "WANDB_DIR": str(output_root / "wandb"),
    }


def _evaluation_command() -> list[str]:
    return [
        "bash",
        "-lc",
        "source recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh; "
        "exec bash recipe/on_policy_wdl_sft/code_task/run_s1_code_base.sh "
        "trainer.val_only=true trainer.val_before_train=true trainer.total_training_steps=0 "
        "trainer.save_freq=-1 'trainer.logger=[\"file\"]' "
        "'+trainer.validation_macro_average_sources=[HumanEval+,MBPP+,LiveCodeBench]' "
        "+trainer.validation_macro_average_name=code3_macro "
        "+trainer.validation_macro_average_metric=acc/mean@3",
    ]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def launch(output_root: Path, *, dry_run: bool) -> int:
    validate_launch_guard(output_root, real_run=not dry_run)
    verified = validate_frozen_inputs()
    environment = _evaluation_environment(output_root)
    plan = {
        "schema_version": 1,
        "mode": "dry_run" if dry_run else "real_run",
        "contract": deepcopy(FROZEN_CONTRACT),
        "verified_inputs": verified,
        "output_root": str(output_root),
        "command": _evaluation_command(),
        "environment": environment,
        "tmux_required": True,
        "candidate_receipt": str(output_root / "stage1_reuse_receipt.candidate.json"),
        "admission_receipt_untouched": ADMISSION_RECEIPT,
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    plan["gpu_facts"] = _gpu_facts()
    output_root.mkdir(parents=True)
    for name in ("checkpoints", "logs", "metrics", "raw_validation", "wandb"):
        (output_root / name).mkdir()
    plan["created_at"] = datetime.now(timezone.utc).isoformat()
    plan["repo_head"] = _git_head()
    launch_provenance = output_root / "launch_provenance.json"
    launch_provenance.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(environment)
    subprocess.run(_evaluation_command(), cwd=ROOT, env=env, check=True)
    return 0


def _validate_fresh_rows(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"fresh validation output missing: {path}")
    required = {
        "data_source",
        "acc",
        "score",
        "format_contract_success",
        "response_finish_reason",
        "response_eos_present",
        "code_reward_dependency_error",
        "code_reward_timeout",
    }
    sources: Counter[str] = Counter()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(required - row.keys())
            if missing:
                raise RuntimeError(f"fresh validation row {line_number} missing telemetry: {missing}")
            sources[str(row["data_source"])] += 1
            rows += 1
    if rows != FROZEN_CONTRACT["expected_response_rows"]:
        raise RuntimeError(
            f"fresh response rows mismatch: expected={FROZEN_CONTRACT['expected_response_rows']} actual={rows}"
        )
    expected_sources = {
        source: item["expected_eval_rows"] * FROZEN_CONTRACT["generation"]["n"]
        for source, item in FROZEN_CONTRACT["validation"].items()
    }
    if dict(sources) != expected_sources:
        raise RuntimeError(f"fresh source cardinality mismatch: expected={expected_sources} actual={dict(sources)}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "rows": rows, "sources": dict(sources)}


def finalize(
    output_root: Path,
    *,
    baseline_file: Path,
    candidate_receipt: Path,
) -> int:
    validate_candidate_receipt_path(candidate_receipt)
    if candidate_receipt.exists():
        raise RuntimeError(f"candidate receipt collision: {candidate_receipt}")
    validate_frozen_inputs()
    fresh_file = output_root / "raw_validation" / "0.jsonl"
    fresh = _validate_fresh_rows(fresh_file)
    if not baseline_file.is_file():
        raise RuntimeError(f"historical baseline missing: {baseline_file}")
    provenance = {
        "schema_version": 1,
        "evaluation_kind": "post_fix_reevaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_head(),
        "contract": deepcopy(FROZEN_CONTRACT),
        "fresh_validation": fresh,
        "historical_baseline": {
            "path": str(baseline_file.resolve()),
            "sha256": sha256(baseline_file),
        },
        "candidate_receipt": str(candidate_receipt.resolve()),
        "admission_receipt_untouched": ADMISSION_RECEIPT,
    }
    provenance_path = output_root / "post_fix_reevaluation_provenance.json"
    if provenance_path.exists():
        raise RuntimeError(f"provenance collision: {provenance_path}")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts/code_wdl_stage1_reuse_gate.py"),
        "--current-file",
        str(fresh_file),
        "--baseline-file",
        str(baseline_file),
        "--step",
        "40",
        "--baseline-step",
        "0",
        "--evaluation-kind",
        "post_fix_reevaluation",
        "--provenance",
        str(provenance_path),
        "--output",
        str(candidate_receipt),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.add_argument("--dry-run", action="store_true")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--baseline-file", type=Path, default=HISTORICAL_BASELINE)
    finalize_parser.add_argument("--candidate-receipt", type=Path)
    args = parser.parse_args()
    if args.command == "launch":
        return launch(args.output_root, dry_run=args.dry_run)
    candidate = args.candidate_receipt or args.output_root / "stage1_reuse_receipt.candidate.json"
    return finalize(args.output_root, baseline_file=args.baseline_file, candidate_receipt=candidate)


if __name__ == "__main__":
    raise SystemExit(main())
