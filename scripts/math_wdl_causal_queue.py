#!/usr/bin/env python3
"""Validate and execute required Math WDL causal-P60 arms D0 and C.

The direct-Model2 D arm is optional and is omitted by default after its
equivalence checks pass. Use ``--include-optional-d`` only for an explicit
same-wrapper replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "arm-c-mixture": ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_wdl_causal_arm_c.sh",
    "arm-d-strong-only": ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_wdl_causal_arm_d.sh",
    "arm-d0-matched-scale-no-weak": ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_wdl_causal_arm_d0.sh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict, *, require_launch: bool) -> None:
    if require_launch and manifest.get("launch_allowed") is not True:
        raise RuntimeError("manifest launch_allowed is false")
    paths = manifest["paths"]
    identity = manifest["identity"]
    required_paths = ["model1", "model2", "model2_provenance", "dataset_receipt", "train_file"]
    if require_launch:
        for name in required_paths:
            if not Path(paths[name]).exists():
                raise FileNotFoundError(f"missing {name}: {paths[name]}")

    receipt = json.loads(Path(paths["dataset_receipt"]).read_text())
    shard = receipt["shards"]["stage1_control"]
    if Path(shard["path"]).resolve() != Path(paths["train_file"]).resolve():
        raise RuntimeError("manifest train_file does not match receipt stage1_control path")
    if int(shard["rows"]) != int(identity["train_rows"]) or shard["sha256"] != identity["train_sha256"]:
        raise RuntimeError("stage1_control row/hash identity mismatch")
    if sha256(Path(paths["train_file"])) != identity["train_sha256"]:
        raise RuntimeError("train file content hash mismatch")
    if require_launch:
        model1 = Path(paths["model1"])
        if sha256(model1 / "format_cold_start_source.json") != identity["model1_source_sha256"]:
            raise RuntimeError("Model1 source receipt hash mismatch")
        if sha256(model1 / "config.json") != identity["model1_config_sha256"]:
            raise RuntimeError("Model1 config hash mismatch")
        if sha256(model1 / "model.safetensors") != identity["model1_weights_sha256"]:
            raise RuntimeError("Model1 weight hash mismatch")
        if sha256(Path(paths["model2_provenance"])) != identity["model2_provenance_sha256"]:
            raise RuntimeError("Model2 provenance hash mismatch")
        if sha256(Path(paths["model2"]) / "config.json") != identity["model2_config_sha256"]:
            raise RuntimeError("Model2 config hash mismatch")
        if sha256(Path(paths["model2"]) / "model.safetensors") != identity["model2_weights_sha256"]:
            raise RuntimeError("Model2 weight hash mismatch")

    contract = manifest["training_contract"]
    if contract != {
        "beta": 0.0,
        "loss_mode": "wdl_sft",
        "kl_enabled": False,
        "rollout_source": "model2",
        "final_step": 60,
        "lr": 1e-6,
        "lr_warmup_steps": 0,
        "data_shuffle": False,
        "validation_frequency": 5,
        "save_frequency": 5,
        "protected_checkpoint_steps": [20, 40, 45, 50, 60],
        "validation_views": ["model1", "model2"],
        "track_counterfactual_submodel_losses": True,
    }:
        raise RuntimeError("training_contract differs from the frozen causal-P60 contract")

    run_projection = {
        run["id"]: (float(run["fusion_lambda"]), run["fusion_mode"], run["execution"])
        for run in manifest["runs"]
    }
    expected = {
        "arm-c-mixture": (0.8, "mixture", "required"),
        "arm-d-strong-only": (1.0, "mixture", "optional"),
        "arm-d0-matched-scale-no-weak": (0.8, "strong_scaled", "required"),
    }
    if run_projection != expected:
        raise RuntimeError(f"unexpected C/D/D0 treatment matrix: {run_projection}")
    optional_d = next(run for run in manifest["runs"] if run["id"] == "arm-d-strong-only")
    if optional_d.get("default_in_queue") is not False:
        raise RuntimeError("optional D must be disabled in the default queue")
    if optional_d.get("omit_if_manipulation_checks") != [
        "D_is_direct_model2",
        "D_ignores_and_does_not_update_model1",
        "D_and_D0_are_model1_invariant",
    ]:
        raise RuntimeError("optional D omission checks differ from the frozen contract")

    reward_contract = manifest.get("reward_contract", {})
    expected_reward_path = "recipe/joint_training/custom_reward_function_latex_verify.py"
    if reward_contract.get("path") != expected_reward_path:
        raise RuntimeError("causal-P60 reward path is not the strict structured scorer")
    reward_path = ROOT / expected_reward_path
    if sha256(reward_path) != reward_contract.get("sha256"):
        raise RuntimeError("strict reward scorer hash differs from the frozen contract")
    if reward_contract.get("function") != "compute_score_latex_verify":
        raise RuntimeError("unexpected causal-P60 reward function")
    if float(reward_contract.get("missing_answer_tag_reward", 0.0)) != -1.0:
        raise RuntimeError("missing <answer> must receive reward -1")


def select_run_ids(manifest: dict, probe_receipt: dict, *, include_optional_d: bool) -> list[str]:
    """Return the live queue order, failing closed before omitting D."""
    if include_optional_d:
        return ["arm-d0-matched-scale-no-weak", "arm-d-strong-only", "arm-c-mixture"]

    optional_d = next(run for run in manifest["runs"] if run["id"] == "arm-d-strong-only")
    checks = probe_receipt.get("checks", {})
    missing = [name for name in optional_d["omit_if_manipulation_checks"] if checks.get(name) is not True]
    if probe_receipt.get("status") != "pass" or missing:
        raise RuntimeError(f"cannot omit optional D without passing equivalence checks: {missing}")
    return ["arm-d0-matched-scale-no-weak", "arm-c-mixture"]


def ensure_probe(manifest: dict, dry_run: bool) -> Path:
    output = Path(manifest["paths"]["manipulation_receipt"])
    command = [
        "python3",
        str(ROOT / "scripts/math_wdl_manipulation_probe.py"),
        "--output",
        str(output),
    ]
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)
        receipt = json.loads(output.read_text())
        if receipt.get("status") != "pass":
            raise RuntimeError(f"manipulation probe failed: {output}")
        admitted_status = manifest.get("admission", {}).get("manipulation_receipt_status")
        if admitted_status != "pass":
            raise RuntimeError("manifest admission does not record a passing manipulation receipt")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_causal_p60.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-optional-d",
        action="store_true",
        help="Run the redundant direct-Model2 D arm despite a passing equivalence probe.",
    )
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    validate_manifest(manifest, require_launch=not args.dry_run)
    if not args.dry_run and not os.environ.get("TMUX"):
        raise RuntimeError("causal-P60 queue must run inside tmux")
    probe_receipt = ensure_probe(manifest, args.dry_run)

    if args.dry_run and not probe_receipt.exists():
        # A dry-run does not execute the probe. Model the admitted omission;
        # the real queue still reads and verifies the live receipt fail-closed.
        probe_data = {
            "status": "pass",
            "checks": {
                name: True
                for name in next(
                    run for run in manifest["runs"] if run["id"] == "arm-d-strong-only"
                )["omit_if_manipulation_checks"]
            },
        }
    else:
        probe_data = json.loads(probe_receipt.read_text())
    ordered_ids = select_run_ids(
        manifest,
        probe_data,
        include_optional_d=args.include_optional_d,
    )
    if "arm-d-strong-only" not in ordered_ids:
        print(
            json.dumps(
                {
                    "run_id": "arm-d-strong-only",
                    "status": "omitted",
                    "reason": "direct-Model2 equivalence probe passed; historical Stage1 control A is reused",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    run_by_id = {run["id"]: run for run in manifest["runs"]}
    for run_id in ordered_ids:
        run = run_by_id[run_id]
        wrapper = WRAPPERS[run_id]
        env = dict(os.environ)
        env.update(
            {
                "BASE_MODEL_PATH": manifest["paths"]["model1"],
                "EXPECTED_MODEL1_PATH": manifest["paths"]["model1"],
                "MODEL2_PATH": manifest["paths"]["model2"],
                "STAGE1_MODEL2_PROVENANCE_FILE": manifest["paths"]["model2_provenance"],
                "TRAIN_FILE": manifest["paths"]["train_file"],
                "BASE_CKPT_DIR": manifest["paths"]["checkpoint_root"],
                "FUSION_LAMBDA": str(run["fusion_lambda"]),
                "FUSION_MODE": run["fusion_mode"],
                "WDL_MANIPULATION_RECEIPT": str(probe_receipt),
                "RUN_OPTIONAL_D": "1" if args.include_optional_d else "0",
                "CUSTOM_REWARD_FN_PATH": str(ROOT / manifest["reward_contract"]["path"]),
                "CUSTOM_REWARD_FN_NAME": manifest["reward_contract"]["function"],
            }
        )
        command = ["bash", str(wrapper)]
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "command": command,
                    "fusion_lambda": env["FUSION_LAMBDA"],
                    "fusion_mode": env["FUSION_MODE"],
                    "model2": env["MODEL2_PATH"],
                    "train_file": env["TRAIN_FILE"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
