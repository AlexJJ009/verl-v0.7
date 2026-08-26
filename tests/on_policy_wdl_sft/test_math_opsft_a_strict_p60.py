# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATH = ROOT / "recipe/on_policy_wdl_sft/math_task"
WRAPPER = MATH / "run_math_qwen3_1p7b_opsft_a_strict_p60.sh"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_opsft_a_strict_p60.yaml"
SUBMITTER = MATH / "submit_math_qwen3_1p7b_opsft_a_strict_p60.sh"
SBATCH = MATH / "slurm/run_math_qwen3_1p7b_opsft_a_strict_p60.sbatch"
JOB_BODY = MATH / "slurm/run_math_qwen3_1p7b_opsft_a_strict_p60_job.sh"


def test_strict_a_is_one_lambda_free_single_model_anchor() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    contract = manifest["training_contract"]
    assert contract["joint_training"] is False
    assert contract["fusion_lambda"] is None
    assert contract["final_step"] == 60
    assert contract["actor_fsdp_seed"] == 42
    assert contract["data_seed"] == 20260719
    assert manifest["identity"]["reward_sha256"] == (
        "6fc2364da021bc5d14e1e3e8788d52cd49a3036088cacbb96d4eb5535e4473e5"
    )


def test_strict_a_pins_every_common_causal_p60_variable() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    for pin in (
        "export INIT_MODEL_PATH=/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/",
        "export TOTAL_TRAINING_STEPS=60",
        "export WDL_SFT_BETA=0.0",
        "export LOSS_MODE=wdl_sft",
        "export LR=1e-6",
        "export DATA_SEED=${DATA_SEED:-20260719}",
        "export DATA_SHUFFLE=False",
        "export TRAINING_SEED=${TRAINING_SEED:-42}",
        "export ROLLOUT_SEED=${ROLLOUT_SEED:-0}",
        "export JOINT_TRAINING=False",
        "export ROLLOUT_CALCULATE_LOG_PROBS=False",
        "export TRAIN_PROMPT_BSZ=64",
        "export ROLLOUT_N=8",
        "export TRAIN_PROMPT_MINI_BSZ=512",
        "export MAX_RESPONSE_LENGTH=4096",
        "export VAL_N=3",
        'export PROTECTED_CKPT_STEPS="[20,40,45,50,55,60]"',
        "../../joint_training/custom_reward_function_latex_verify.py",
        'actor_rollout_ref.actor.fsdp_config.seed="${TRAINING_SEED}"',
        'actor_rollout_ref.actor.data_loader_seed="${TRAINING_SEED}"',
    ):
        assert pin in wrapper
    assert "FUSION_LAMBDA=" not in wrapper
    # ROLLOUT_SEED is consumed once by _common_ablation.sh. Passing the same
    # Hydra append override here a second time makes composition fail before
    # training starts.
    assert "+actor_rollout_ref.rollout.seed=0" not in wrapper


def test_strict_a_rejects_hydra_overrides_before_io() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "data.seed=1"],
        cwd=ROOT,
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 64
    assert "accepts no positional/Hydra overrides" in result.stderr


def test_strict_a_first_step_gate_is_single_model_specific() -> None:
    module_path = ROOT / "scripts/math_opsft_first_step_gate.py"
    spec = importlib.util.spec_from_file_location("math_opsft_first_step_gate", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {key: 1 for key in module.REQUIRED_METRICS}
    data.update({"wdl_sft/n_correct": 256, "wdl_sft/n_incorrect": 256})
    checks = module.validate_step_one(data)
    assert checks["all_required_metrics_present"] is True
    assert checks["full_rollout_batch_scored"] is True
    assert not any(key.startswith("jointTraining/") for key in module.REQUIRED_METRICS)


def test_strict_a_slurm_is_candidate_bound_and_release_gated() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "one shared A anchor" in submitter
    assert "--hold" in submitter
    assert 'scontrol release "$job"' in submitter
    assert "rev-parse HEAD:recipe" in submitter
    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:L40S:8" in sbatch
    assert "#SBATCH --exclusive" in sbatch
    assert "#SBATCH --no-requeue" in sbatch
    body = JOB_BODY.read_text(encoding="utf-8")
    assert 'launch_receipt="$workspace/.launch-receipt.json"' in body
    assert "check_math_reward_contract.py" in WRAPPER.read_text(encoding="utf-8")
    assert "math_opsft_first_step_gate.py" in body
    assert '"val-core/math7_macro/acc/mean@3"' in body
    assert 'latest != 60 or not (ckpt / "global_step_60").is_dir()' in body
    assert "training_result_release_gate.py record" in body
    assert "training_result_release_gate.py check" in body
    assert "local terminal_rc=$rc" in body
