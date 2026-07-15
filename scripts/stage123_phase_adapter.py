#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "stage1": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "stage2": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "stage3": ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict:
    rendered = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", str(path), "--format", "json"],
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


def metrics_for(checkpoint: Path) -> Path:
    matches = list((ROOT / "recipe/on_policy_wdl_sft").glob(f"**/metrics/OnPolicyWDLSFT-CodeTask/{checkpoint.name}.jsonl"))
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
    environment.update(
        {
            "STAGE123_RUN_ID": run["id"],
            "STAGE123_MANIFEST": str(manifest_path),
            "RUN_PREFIX": run["run_prefix"],
            "CODE_TRAIN_FILE": run["train_file"],
            "TRAIN_FILE": run["train_file"],
            "TOTAL_TRAINING_STEPS": str(run["final_step"]),
            "DATA_SHUFFLE": "False",
            "WDL_SFT_BETA": "0.1",
            "EXPECTED_STAGE1_BETA": "0.1",
            "FUSION_LAMBDA": "0.8",
            "STAGE123_EXPECTED_PROFILE_HASH": manifest["resource_profile"]["sha256"],
        }
    )
    return environment


def stage_environment(manifest: dict, run: dict, environment: dict[str, str]) -> dict[str, str]:
    source = run["source"]
    phase = run["phase"]
    if phase == "stage1":
        environment["INIT_MODEL_PATH"] = source["model_path"]
    elif phase == "stage2":
        environment.update(
            {
                "STAGE1_RUN_PREFIX": source["run_prefix"],
                "EXPECTED_STAGE1_RUN_PREFIX": source["run_prefix"],
                "STAGE1_CKPT_DIR": source["checkpoint_root"],
                "STAGE2_HANDOFF_STEP": str(source["handoff_step"]),
                "MERGED_MODEL2_DIR": source["model2_path"],
                "MODEL2_CACHE_TAG": run["chain"],
            }
        )
    elif phase == "stage3":
        stage2 = run_by_id(manifest, source["run_id"])
        environment.update(
            {
                "STAGE2_MODEL2_PATH": str(Path(stage2["artifact_dir"]) / "stage2_final_model2"),
                "STAGE2_PROVENANCE_FILE": stage2["provenance_file"],
            }
        )
    return environment


def execute_wrapper(manifest_path: Path, manifest: dict, run: dict, *, dry_run: bool) -> tuple[Path | None, Path | None]:
    environment = stage_environment(manifest, run, common_environment(manifest_path, manifest, run))
    command = ["bash", str(WRAPPERS[run["phase"]])]
    if dry_run:
        print(json.dumps({"command": command, "environment": {key: environment[key] for key in sorted(environment) if key.startswith("STAGE123_") or key in {"RUN_PREFIX", "INIT_MODEL_PATH", "STAGE1_CKPT_DIR", "STAGE2_HANDOFF_STEP", "MERGED_MODEL2_DIR", "STAGE2_MODEL2_PATH", "STAGE2_PROVENANCE_FILE", "CODE_TRAIN_FILE", "TOTAL_TRAINING_STEPS"}}}, sort_keys=True))
        return None, None
    started_at = time.time()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    checkpoint = checkpoint_for(run["run_prefix"], int(run["final_step"]), started_at)
    return checkpoint, metrics_for(checkpoint)


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
        model2 = Path(run["artifact_dir"]) / "stage2_final_model2"
        if joint.exists() or model2.exists():
            raise RuntimeError("Stage2 extraction target already exists; retry/resume is forbidden")
        subprocess.run(
            [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str(checkpoint / f"global_step_{run['final_step']}" / "actor"), "--target_dir", str(joint), "--trust-remote-code"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "recipe/joint_training/extract_sub_model.py"), "--joint_model_path", str(joint), "--output_path", str(model2), "--sub_model_index", "1"],
            cwd=ROOT,
            check=True,
        )
        payload["source"] = {"type": "stage2_complete", "extracted_model2": str(model2), "joint_model": str(joint)}
    elif run["phase"] == "stage3":
        payload["source"] = {"type": "stage2_model2", "run_id": run["source"]["run_id"], "model2": os.environ.get("STAGE2_MODEL2_PATH", str(Path(run["artifact_dir"]) / "stage2_final_model2"))}
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
