from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.ray_trainer import compute_response_mask


ROOT = Path(__file__).resolve().parents[2]
GRPO_DIR = ROOT / "recipe/on_policy_wdl_sft/standard_grpo"


def config_only(wrapper: str, **extra_env: str) -> dict[str, str]:
    env = {
        **os.environ,
        "GRPO_CONFIG_ONLY": "1",
        "STAGE1_MODEL_PATH": "/models/stage1",
        **extra_env,
    }
    output = subprocess.check_output(["bash", str(GRPO_DIR / wrapper)], cwd=ROOT, env=env, text=True)
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


@pytest.mark.parametrize(
    ("wrapper", "task", "pipeline", "steps"),
    [
        ("run_math_stage1_grpo.sh", "math", "stage1_grpo", "60"),
        ("run_math_cold_start_grpo.sh", "math", "cold_start_grpo", "100"),
        ("run_code_stage1_grpo.sh", "code", "stage1_grpo", "60"),
        ("run_code_cold_start_grpo.sh", "code", "cold_start_grpo", "100"),
    ],
)
def test_friendly_entries_resolve_frozen_standard_grpo_contract(wrapper, task, pipeline, steps):
    config = config_only(wrapper)
    assert config["task"] == task
    assert config["pipeline"] == pipeline
    assert config["wandb_project"] == f"StandardGRPO-Qwen3-1P7B-{task.title()}"
    assert config["total_training_steps"] == steps
    assert config["train_prompt_bsz"] == "64"
    assert config["rollout_n"] == "8"
    assert config["responses_per_step"] == "512"
    assert config["ppo_mini_batch_size"] == "64"
    assert config["learning_rate"] == "1e-6"
    assert config["ppo_epochs"] == "1"
    assert config["training_seed"] == "1"
    assert config["resume_mode"] == "disable"
    assert config["actor_grad_clip"] == "1.0"
    assert config["loss_mode"] == "vanilla"
    assert config["loss_agg_mode"] == "seq-mean-token-mean"
    assert config["norm_adv_by_std_in_grpo"] == "True"
    assert config["use_kl_in_reward"] == "False"
    assert config["use_kl_loss"] == "True"
    assert config["kl_loss_coef"] == "0.001"
    assert config["kl_loss_type"] == "low_var_kl"
    assert config["ref_log_prob_micro_batch_size"] == "1"
    assert config["rollout_is"] == "null"
    assert config["enable_thinking"] == "True"
    assert config["data_shuffle"] == "False"
    expected_reward = {
        "math": (
            str(ROOT / "recipe/joint_training/custom_reward_function_latex_verify.py"),
            "compute_score_latex_verify",
        ),
        "code": (
            str(ROOT / "recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py"),
            "compute_score_code_official_aligned",
        ),
    }
    assert (config["custom_reward_fn_path"], config["custom_reward_fn_name"]) == expected_reward[task]
    assert config["protected_ckpt_strip_optimizer"] == "True"
    expected_best_metric = {
        "math": "val-core/math7_macro/acc/mean@3",
        "code": "val-core/code3_macro/acc/mean@3",
    }
    assert config["best_ckpt_metric_key"] == expected_best_metric[task]
    if pipeline == "stage1_grpo":
        assert config["protected_ckpt_steps"] == "[20]"
    else:
        assert config["protected_ckpt_steps"] == "[40,60]"


@pytest.mark.parametrize("learning_rate", ["5e-7", "1e-6"])
def test_learning_rate_sensitivity_uses_the_same_math_entry(learning_rate):
    config = config_only("run_math_stage1_grpo.sh", LR=learning_rate)
    assert config["learning_rate"] == learning_rate
    assert config["total_training_steps"] == "60"
    assert config["protected_ckpt_steps"] == "[20]"


def test_c_wdl_p60_then_grpo_is_one_fresh_100_step_post_stage_run():
    config = config_only(
        "run_math_c_wdl_p60_then_grpo.sh",
        INIT_MODEL_PATH="/models/c_p60_model2",
        DATASET_ROOT="/datasets/math_stage123",
    )
    assert config["pipeline"] == "c_wdl_p60_grpo"
    assert config["init_model_path"] == "/models/c_p60_model2"
    assert config["train_file"] == "/datasets/math_stage123/stage1_control_stage2_then_stage3.parquet"
    assert config["total_training_steps"] == "100"
    assert config["learning_rate"] == "1e-6"
    assert config["protected_ckpt_steps"] == "[20,40,60,80]"
    assert config["resume_mode"] == "disable"
    assert config["total_epochs"] == "2"

    wrapper = (GRPO_DIR / "run_math_c_wdl_p60_then_grpo.sh").read_text()
    assert "TOTAL_TRAINING_STEPS=98" not in wrapper
    assert wrapper.count('exec bash "${COMMON}" "$@"') == 1


def test_common_launcher_forwards_standard_grpo_actor_contract():
    launcher = (ROOT / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh").read_text()
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}" in launcher
    assert "actor_rollout_ref.actor.ppo_epochs=${ppo_epochs}" in launcher
    assert "actor_rollout_ref.actor.data_loader_seed=${training_seed}" in launcher
    assert "+actor_rollout_ref.rollout.seed=${rollout_seed}" in launcher
    assert "data.seed=${training_seed}" in launcher
    assert "trainer.resume_mode=${resume_mode}" in launcher
    assert "actor_rollout_ref.actor.grad_clip=${actor_grad_clip}" in launcher
    assert "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}" in launcher
    assert "actor_rollout_ref.actor.grad_clip=500.0" not in launcher


def test_math_stage1_entry_fails_closed_on_provenance_hashes():
    launcher = (ROOT / "recipe/on_policy_wdl_sft/standard_grpo/run_qwen3_1p7b_standard_grpo.sh").read_text()
    assert "MATH_STAGE1_MODEL_PROVENANCE_PATH" in launcher
    assert "Math S1-P0 model hash mismatch in provenance receipt" in launcher
    assert "Math S1-P0 source joint hash mismatch" in launcher
    assert 'GRPO_PREFLIGHT_ONLY:-0' in launcher


def test_math_grpo_formal_launch_is_gated_by_strict_reward_contract():
    launcher = (ROOT / "recipe/on_policy_wdl_sft/standard_grpo/run_qwen3_1p7b_standard_grpo.sh").read_text()
    assert 'joint_training/custom_reward_function_latex_verify.py' in launcher
    assert 'scripts/check_math_reward_contract.py' in launcher
    assert '--reward-path "${CUSTOM_REWARD_FN_PATH}"' in launcher
    assert 'PYTHONPATH="${REPO_PYTHONPATH_ROOT:-${SCRIPT_DIR}/../../..}:${PYTHONPATH:-}"' in launcher
    assert 'grpo_retrain_admission.py' in launcher
    assert '--runtime-image-digest "${GRPO_RUNTIME_IMAGE_DIGEST}"' in launcher
    assert '--scheduler-managed' in launcher
    assert 'GRPO_ROOT_COMMIT' in launcher
    assert 'GRPO_RECIPE_COMMIT' in launcher
    assert 'GRPO_SNAPSHOT_DIGEST' in launcher
    assert '--receipt "${GRPO_ADMISSION_RECEIPT}"' in launcher


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("LOSS_AGG_MODE", "token-mean", "LOSS_AGG_MODE=seq-mean-token-mean"),
        ("ACTOR_GRAD_CLIP", "500.0", "ACTOR_GRAD_CLIP=1.0"),
        ("ROLLOUT_IS", "token", "ROLLOUT_IS=null"),
        ("TRAIN_PROMPT_MINI_BSZ", "512", "TRAIN_PROMPT_MINI_BSZ=64"),
        ("PPO_EPOCHS", "2", "PPO_EPOCHS=1"),
        ("RESUME_MODE", "auto", "RESUME_MODE=disable"),
    ],
)
def test_config_only_rejects_noncanonical_grpo_overrides(override, value, message):
    env = {
        **os.environ,
        "GRPO_CONFIG_ONLY": "1",
        "STAGE1_MODEL_PATH": "/models/stage1",
        override: value,
    }
    result = subprocess.run(
        ["bash", str(GRPO_DIR / "run_math_stage1_grpo.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_grpo_loss_mask_includes_every_non_padding_response_token():
    # Conceptual response positions: <think>, reasoning, </think>, <answer>, pad.
    data = SimpleNamespace(
        batch={
            "responses": torch.tensor([[10, 11, 12, 13, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
        }
    )
    response_mask = compute_response_mask(data)
    assert response_mask.tolist() == [[1, 1, 1, 1, 0]]

    # Keep ratios inside the PPO clip range; otherwise clipping correctly
    # zeroes the gradient and would confound the response-mask assertion.
    log_prob = torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.0]], requires_grad=True)
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 10.0,
            "global_batch_info": {},
        }
    )
    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=torch.zeros_like(log_prob),
        log_prob=log_prob,
        advantages=torch.ones_like(log_prob),
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        config=config,
    )
    loss.backward()
    assert torch.all(log_prob.grad[0, :4] != 0)
    assert log_prob.grad[0, 4] == 0


def test_continuous_dataset_is_stage1_then_stage2_then_stage3(tmp_path):
    source_index = 0
    for name, rows in (("stage1", 2560), ("stage2", 1280), ("stage3", 2560)):
        frame = pd.DataFrame(
            {
                "stage123_source_index": range(source_index, source_index + rows),
                "stage123_order": range(rows),
                "shard": [name] * rows,
            }
        )
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
        source_index += rows
    output = tmp_path / "cold_start_grpo_stage1_stage2_stage3.parquet"
    subprocess.run(
        [
            "python",
            str(GRPO_DIR / "prepare_qwen3_1p7b_grpo_data.py"),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    merged = pd.read_parquet(output)
    assert len(merged) == 6400
    assert merged["grpo_order"].tolist() == list(range(6400))
    assert merged["shard"].iloc[[0, 2559, 2560, 3839, 3840, 6399]].tolist() == [
        "stage1",
        "stage1",
        "stage2",
        "stage2",
        "stage3",
        "stage3",
    ]
