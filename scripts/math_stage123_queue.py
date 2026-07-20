#!/usr/bin/env python3
"""Execute the admitted Qwen3-1.7B Math Stage1/2/3 matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "stage1": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s1_math_qwen3_1p7b_stage123_common.sh",
    "stage1_control": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s1_math_qwen3_1p7b_stage123_common.sh",
    "stage2": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s2_math_qwen3_1p7b_stage123_common.sh",
    "stage3": ROOT / "recipe/on_policy_wdl_sft/math_task/run_s3_math_qwen3_1p7b_stage123_common.sh",
}


def emit_event(event: str, **payload: object) -> None:
    event_log = os.environ.get("MATH_EVENT_LOG")
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


def run_prefix(run: dict) -> str:
    return f"MATH-{run['id'].upper().replace('-', '_')}-QWEN3-1P7B-V1"


def checkpoint_after(prefix: str, started_at: float, final_step: int) -> Path:
    candidates = [
        path
        for path in Path("/data-1/checkpoints").glob(f"{prefix}_*")
        if path.stat().st_mtime >= started_at - 5
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one new checkpoint root for {prefix}, found {candidates}")
    actor = candidates[0] / f"global_step_{final_step}" / "actor"
    if not actor.is_dir():
        raise RuntimeError(f"missing final actor checkpoint: {actor}")
    return actor


def execute(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = {key: env[key] for key in sorted(env) if key in {
        "RUN_PREFIX", "INIT_MODEL_PATH", "BASE_MODEL_PATH", "MODEL2_PATH", "STAGE2_MODEL_PATH",
        "TRAIN_FILE", "TOTAL_TRAINING_STEPS", "WDL_SFT_BETA", "SUBMODEL_KL_ENABLED",
        "SUBMODEL_KL_MODEL2_ENABLED", "SUBMODEL_KL_MODEL2_COEF", "SUBMODEL_KL_MODEL2_REF_PATH",
    }}
    print(json.dumps({"command": command, "environment": printable}, sort_keys=True), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def verify_dataset_receipt(receipt_path: Path, source_path: Path, seed: int, dry_run: bool) -> None:
    if receipt_path.name != "dataset_receipt.json":
        raise ValueError(f"unexpected dataset receipt filename: {receipt_path}")
    command = [
        sys.executable,
        str(ROOT / "recipe/on_policy_wdl_sft/math_task/prepare_qwen3_1p7b_math_stage123_data.py"),
        "--source",
        str(source_path),
        "--output-root",
        str(receipt_path.parent),
        "--seed",
        str(seed),
        "--verify-only",
    ]
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def merge_single(actor: Path, target: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(target)],
        cwd=ROOT,
        check=True,
    )


def merge_stage2(actor: Path, artifact_dir: Path) -> dict[str, str]:
    joint = artifact_dir / "stage2_final_joint"
    model1 = artifact_dir / "stage2_final_model1"
    model2 = artifact_dir / "stage2_final_model2"
    subprocess.run(
        [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(joint), "--trust-remote-code"],
        cwd=ROOT,
        check=True,
    )
    for index, target in ((0, model1), (1, model2)):
        subprocess.run(
            [sys.executable, str(ROOT / "recipe/joint_training/extract_sub_model.py"), "--joint_model_path", str(joint), "--output_path", str(target), "--sub_model_index", str(index)],
            cwd=ROOT,
            check=True,
        )
    return {"joint": str(joint), "model1": str(model1), "model2": str(model2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_stage123.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
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
    verify_dataset_receipt(receipt_path, source_path, manifest["seed"], args.dry_run)
    if args.dry_run and not selection_path.exists():
        selected_model = Path("/SELECTED_MODEL1_PENDING")
    else:
        selected_model = Path(json.loads(selection_path.read_text())["identity"]["model_path"])
    if args.dry_run and not receipt_path.exists():
        shards = {name: {"path": f"/DATASET_PENDING/{name}.parquet"} for name in ("stage1", "stage2", "stage3", "stage1_control")}
    else:
        shards = json.loads(receipt_path.read_text())["shards"]

    artifact_root = Path(manifest["paths"]["artifact_root"])
    outputs: dict[str, dict] = {}
    for run_spec in manifest["runs"]:
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
                    "STAGE1_RUN_PREFIX": run_prefix(next(item for item in manifest["runs"] if item["id"] == run_spec["source_run"])),
                    "STAGE1_STEP": str(next(item["final_step"] for item in manifest["runs"] if item["id"] == run_spec["source_run"])),
                    "SUBMODEL_KL_ENABLED": "true" if run_spec["kl"] == "m2kl" else "false",
                    "SUBMODEL_KL_MODEL1_ENABLED": "false",
                    "SUBMODEL_KL_MODEL2_ENABLED": "true" if run_spec["kl"] == "m2kl" else "false",
                    "SUBMODEL_KL_MODEL2_COEF": str(manifest["matrix"]["model2_kl_coef"] if run_spec["kl"] == "m2kl" else 0.0),
                    "SUBMODEL_KL_MODEL2_REF_PATH": stage1_model,
                }
            )
        else:
            submodel = run_spec["submodel"]
            env["STAGE2_MODEL_PATH"] = source.get(submodel, f"/SOURCE_PENDING/{run_spec['source_run']}/{submodel}")

        started_at = time.time()
        emit_event("stage_run_started", run_id=run_spec["id"], phase=run_spec["phase"])
        execute(["bash", str(WRAPPERS[run_spec["phase"]])], env, args.dry_run)
        if args.dry_run:
            outputs[run_spec["id"]] = {"model": f"/DRY_RUN/{run_spec['id']}/model", "model1": f"/DRY_RUN/{run_spec['id']}/model1", "model2": f"/DRY_RUN/{run_spec['id']}/model2"}
            continue
        actor = checkpoint_after(env["RUN_PREFIX"], started_at, run_spec["final_step"])
        artifact_dir.mkdir(parents=True, exist_ok=False)
        if run_spec["phase"] == "stage2":
            outputs[run_spec["id"]] = merge_stage2(actor, artifact_dir)
        else:
            model = artifact_dir / "final_model"
            merge_single(actor, model)
            outputs[run_spec["id"]] = {"model": str(model)}
        (artifact_dir / "provenance.json").write_text(
            json.dumps({"schema_version": 1, "run": run_spec, "outputs": outputs[run_spec["id"]]}, indent=2, sort_keys=True) + "\n"
        )
        emit_event("stage_run_completed", run_id=run_spec["id"], phase=run_spec["phase"], outputs=outputs[run_spec["id"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
