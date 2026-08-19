#!/usr/bin/env python3
"""Fail-closed provenance gate for strict-scorer GRPO retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_clean(repo: Path, label: str) -> str:
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise RuntimeError(f"{label} checkout is dirty; commit the exact launch candidate first")
    return git_output(repo, "rev-parse", "HEAD")


def require_hex_digest(value: str, length: int, label: str) -> str:
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a full {length}-character lowercase hex digest")
    return value


def resolve_source_identity(
    *,
    scheduler_managed: bool,
    supplied_root_commit: str | None,
    supplied_recipe_commit: str | None,
    supplied_snapshot_digest: str | None,
) -> tuple[str, str, str | None, str]:
    """Resolve Git identity locally or from an immutable scheduler snapshot.

    Slurm workspaces intentionally exclude .git. In that environment the
    scheduler receipt binds the exact snapshot digest and the snapshot manifest
    binds the root/recipe commits. Callers must pass all three values rather
    than silently weakening the clean-checkout gate.
    """
    has_git_metadata = (ROOT / ".git").exists() and (ROOT / "recipe/.git").exists()
    if scheduler_managed:
        if not supplied_root_commit or not supplied_recipe_commit or not supplied_snapshot_digest:
            raise RuntimeError(
                "scheduler-managed admission requires root commit, recipe commit, and snapshot digest"
            )
        root_commit = require_hex_digest(supplied_root_commit, 40, "root commit")
        recipe_commit = require_hex_digest(supplied_recipe_commit, 40, "recipe commit")
        snapshot_digest = require_hex_digest(supplied_snapshot_digest, 64, "snapshot digest")
        if has_git_metadata:
            observed_root = require_clean(ROOT, "root")
            observed_recipe = require_clean(ROOT / "recipe", "recipe")
            if observed_root != root_commit or observed_recipe != recipe_commit:
                raise RuntimeError("scheduler source commits differ from the checked-out launch candidate")
            gitlink_commit = git_output(ROOT, "ls-tree", "HEAD", "recipe").split()[2]
            if observed_recipe != gitlink_commit:
                raise RuntimeError(
                    f"recipe checkout does not match committed gitlink: checkout={observed_recipe} "
                    f"gitlink={gitlink_commit}"
                )
        return root_commit, recipe_commit, snapshot_digest, "scheduler_snapshot"

    if not has_git_metadata:
        raise RuntimeError("local admission requires root and recipe Git metadata")
    root_commit = require_clean(ROOT, "root")
    recipe_commit = require_clean(ROOT / "recipe", "recipe")
    gitlink_commit = git_output(ROOT, "ls-tree", "HEAD", "recipe").split()[2]
    if recipe_commit != gitlink_commit:
        raise RuntimeError(
            f"recipe checkout does not match committed gitlink: checkout={recipe_commit} gitlink={gitlink_commit}"
        )
    return root_commit, recipe_commit, None, "local_git"


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} sha256 mismatch: expected={expected} actual={actual}")
    return actual


def resolve_ordered_data_schedule(
    train_file: Path,
    train_prompt_bsz: int,
    total_training_steps: int,
    total_epochs: int,
) -> dict[str, int | bool]:
    train_rows = pq.ParquetFile(train_file).metadata.num_rows
    if train_rows <= 0:
        raise RuntimeError("train parquet must contain at least one row")
    if train_rows % train_prompt_bsz:
        raise RuntimeError(
            f"train rows ({train_rows}) are not divisible by prompt batch size "
            f"({train_prompt_bsz})"
        )
    steps_per_epoch = train_rows // train_prompt_bsz
    required_epochs = (total_training_steps + steps_per_epoch - 1) // steps_per_epoch
    if total_epochs != required_epochs:
        raise RuntimeError(
            f"total_epochs must exactly cover the ordered training schedule: "
            f"steps={total_training_steps}, steps_per_epoch={steps_per_epoch}, "
            f"expected={required_epochs}, got={total_epochs}"
        )
    full_epochs, trailing_steps = divmod(total_training_steps, steps_per_epoch)
    return {
        "shuffle": False,
        "train_rows": train_rows,
        "train_prompt_bsz": train_prompt_bsz,
        "steps_per_epoch": steps_per_epoch,
        "total_training_steps": total_training_steps,
        "total_epochs": total_epochs,
        "full_epochs": full_epochs,
        "trailing_steps_in_next_epoch": trailing_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("math", "code"), required=True)
    parser.add_argument(
        "--pipeline",
        choices=("stage1_grpo", "cold_start_grpo", "c_wdl_p60_grpo"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--reward-path", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-reward-sha256", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--expected-image-digest", required=True)
    parser.add_argument("--actor-seed", type=int, required=True)
    parser.add_argument("--rollout-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--data-shuffle", choices=("True", "False"), required=True)
    parser.add_argument("--train-prompt-bsz", type=int, required=True)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--scheduler-managed", action="store_true")
    parser.add_argument("--root-commit")
    parser.add_argument("--recipe-commit")
    parser.add_argument("--snapshot-digest")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    for label, value in (
        ("actor seed", args.actor_seed),
        ("rollout seed", args.rollout_seed),
        ("data seed", args.data_seed),
    ):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    expected_randomness = {
        "math": {"actor_seed": 42, "rollout_seed": 0, "data_seed": 20260719},
        "code": {"actor_seed": 42, "rollout_seed": 0, "data_seed": 20260706},
    }[args.task]
    observed_randomness = {
        "actor_seed": args.actor_seed,
        "rollout_seed": args.rollout_seed,
        "data_seed": args.data_seed,
    }
    if observed_randomness != expected_randomness:
        raise ValueError(
            f"{args.task} randomness does not match the frozen A/C/D0 contract: "
            f"expected={expected_randomness} observed={observed_randomness}"
        )
    if args.data_shuffle != "False":
        raise ValueError("matched A/C/D0 admission requires data shuffle=False")
    if args.train_prompt_bsz <= 0 or args.total_training_steps <= 0 or args.total_epochs <= 0:
        raise ValueError("batch size, training steps, and epochs must be positive")
    digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
    if not digest_pattern.fullmatch(args.runtime_image_digest):
        raise ValueError("runtime image digest must be a full sha256 digest")
    if args.runtime_image_digest != args.expected_image_digest:
        raise RuntimeError("runtime image digest differs from the admitted image")
    if args.receipt.exists():
        raise FileExistsError(f"admission receipt already exists: {args.receipt}")

    root_commit, recipe_commit, snapshot_digest, source_identity_mode = resolve_source_identity(
        scheduler_managed=args.scheduler_managed,
        supplied_root_commit=args.root_commit,
        supplied_recipe_commit=args.recipe_commit,
        supplied_snapshot_digest=args.snapshot_digest,
    )

    model_file = args.model_path / "model.safetensors"
    identities = {
        "model_safetensors_sha256": require_sha(
            model_file, args.expected_model_sha256, "init model.safetensors"
        ),
        "train_file_sha256": require_sha(args.train_file, args.expected_train_sha256, "train file"),
        "reward_sha256": require_sha(args.reward_path, args.expected_reward_sha256, "reward scorer"),
    }
    ordered_data_schedule = resolve_ordered_data_schedule(
        args.train_file,
        args.train_prompt_bsz,
        args.total_training_steps,
        args.total_epochs,
    )
    payload = {
        "schema_version": 2,
        "status": "admitted",
        "admitted_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "pipeline": args.pipeline,
        "randomness": {
            "actor_seed": args.actor_seed,
            "rollout_seed": args.rollout_seed,
            "data_seed": args.data_seed,
        },
        "ordered_data_schedule": ordered_data_schedule,
        "root_commit": root_commit,
        "recipe_commit": recipe_commit,
        "snapshot_digest": snapshot_digest,
        "source_identity_mode": source_identity_mode,
        "image_digest": args.runtime_image_digest,
        "paths": {
            "model": str(args.model_path),
            "train_file": str(args.train_file),
            "reward": str(args.reward_path),
        },
        "identities": identities,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.receipt)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
