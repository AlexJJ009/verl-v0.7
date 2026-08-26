# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATH = ROOT / "recipe/on_policy_wdl_sft/math_task"
WRAPPER = MATH / "run_math_qwen3_1p7b_wdl_lambda_matrix_p60.sh"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_lambda_matrix_p60.yaml"
SUBMITTER = MATH / "submit_math_qwen3_1p7b_wdl_lambda_matrix_p60.sh"
SBATCH = MATH / "slurm/run_math_qwen3_1p7b_wdl_lambda_matrix_p60.sbatch"
JOB_BODY = MATH / "slurm/run_math_qwen3_1p7b_wdl_lambda_matrix_p60_job.sh"


def _run(arm: str, fusion_lambda: str, lr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=os.environ | {"LAMBDA_ARM": arm, "FUSION_LAMBDA": fusion_lambda, "TRAINING_LR": lr},
        check=False,
        capture_output=True,
        text=True,
    )


def test_lambda_followup_has_exactly_six_new_runs() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert [(run["id"], run["arm"], run["fusion_lambda"], run["lr"]) for run in manifest["runs"]] == [
        ("lambda07-fixed-lr1e6", "fixed-m1", 0.7, "1e-6"),
        ("lambda09-fixed-lr1e6", "fixed-m1", 0.9, "1e-6"),
        ("lambda07-d0-lr1e6", "d0", 0.7, "1e-6"),
        ("lambda09-d0-lr1e6", "d0", 0.9, "1e-6"),
        ("lambda05-c-lr5e7", "standard-c", 0.5, "5e-7"),
        ("lambda08-c-lr5e7", "standard-c", 0.8, "5e-7"),
    ]
    assert manifest["training_contract"]["protected_checkpoint_steps"] == [20, 40, 45, 50, 55, 60]
    assert "learning-rate grid" in manifest["stop_rules"][1]


def test_lambda_matrix_pins_the_causal_p60_contract() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    for pin in (
        "export TOTAL_TRAINING_STEPS=60",
        "export WDL_SFT_BETA=0.0",
        "export LOSS_MODE=wdl_sft",
        'export LR="$TRAINING_LR"',
        "export DATA_SEED=20260719",
        "export DATA_SHUFFLE=False",
        "export JOINT_TRAINING_ROLLOUT_SOURCE=model2",
        "export TRAIN_PROMPT_BSZ=64",
        "export ROLLOUT_N=8",
        "export MAX_RESPONSE_LENGTH=4096",
        "export VAL_N=3",
        "export PROTECTED_CKPT_STEPS=\"[20,40,45,50,55,60]\"",
        "actor_rollout_ref.actor.fsdp_config.seed=42",
        "actor_rollout_ref.actor.data_loader_seed=42",
        "+actor_rollout_ref.rollout.seed=0",
    ):
        assert pin in wrapper


def test_lambda_matrix_rejects_duplicate_and_unregistered_pairs_before_io() -> None:
    duplicate = _run("standard-c", "0.5", "1e-6")
    assert duplicate.returncode == 64
    assert "unauthorized lambda follow-up triple" in duplicate.stderr

    unsupported_lambda = _run("fixed-m1", "0.5", "1e-6")
    assert unsupported_lambda.returncode == 64
    assert "unauthorized lambda follow-up triple" in unsupported_lambda.stderr

    unsupported_arm = _run("unknown", "0.7", "1e-6")
    assert unsupported_arm.returncode == 64
    assert "unauthorized lambda follow-up triple" in unsupported_arm.stderr

    unsupported_lr = _run("standard-c", "0.5", "2e-6")
    assert unsupported_lr.returncode == 64
    assert "permits only TRAINING_LR" in unsupported_lr.stderr


def test_lambda_matrix_arm_semantics_are_explicit() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "fixed-m1:0.7:1e-6|fixed-m1:0.9:1e-6" in wrapper
    assert "export FUSION_MODE=mixture" in wrapper
    assert "export FREEZE_MODEL1=true" in wrapper
    assert "d0:0.7:1e-6|d0:0.9:1e-6" in wrapper
    assert "export FUSION_MODE=strong_scaled" in wrapper
    assert "standard-c:0.5:5e-7|standard-c:0.8:5e-7" in wrapper
    assert 'export LAMBDA_EXPECTED_MODEL1_GRADIENT="$expected_model1_gradient"' in wrapper
    assert 'qwen3_1p7b_wdl_lambda_followup/${lambda_tag}/${artifact_arm}-${lr_tag}-p60' in wrapper


def test_lambda_matrix_slurm_launch_is_candidate_bound_and_exact() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "ARM_VALUES=(fixed-m1 d0 fixed-m1 d0 standard-c standard-c)" in submitter
    assert "LAMBDA_VALUES=(0.7 0.7 0.9 0.9 0.5 0.8)" in submitter
    assert "LR_VALUES=(1e-6 1e-6 1e-6 1e-6 5e-7 5e-7)" in submitter
    assert "--hold" in submitter
    assert 'scontrol release "$(IFS=,; echo "${submitted[*]}")"' in submitter
    assert "exact six-run receipt required" in submitter
    assert "exact six-run cell contract required" in submitter
    assert "rev-parse HEAD:recipe" in submitter

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:L40S:8" in sbatch
    assert "#SBATCH --exclusive" in sbatch
    assert "#SBATCH --no-requeue" in sbatch


def test_lambda_matrix_job_has_first_step_completion_and_release_gates() -> None:
    body = JOB_BODY.read_text(encoding="utf-8")
    assert 'launch_receipt="$workspace/.launch-receipt.json"' in body
    assert "LAMBDA_MATRIX_LAUNCH_RECEIPT" not in body
    assert "math_wdl_first_step_gate.py" in body
    assert '--expected-model1-gradient "$expected_gradient"' in body
    assert "FUSION_LAMBDA TRAINING_LR" in body
    assert "unauthorized arm/lambda/lr triple" in body
    assert 'int(row.get("step", -1)) == 60' in body
    assert 'latest != 60 or not (ckpt / "global_step_60").is_dir()' in body
    assert "training_result_release_gate.py record" in body
    assert "training_result_release_gate.py check" in body
    assert "local rc=$?" in body
    assert "local terminal_rc=$rc" in body
    assert "discover_run_name || true" in body
    assert "trap cleanup EXIT" in body
    assert "trap 'exit 143' TERM" in body
