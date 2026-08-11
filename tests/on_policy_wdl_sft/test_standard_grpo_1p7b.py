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
    assert config["ppo_mini_batch_size"] == "512"
    assert config["learning_rate"] == "5e-7"
    assert config["loss_mode"] == "vanilla"
    assert config["loss_agg_mode"] == "token-mean"
    assert config["norm_adv_by_std_in_grpo"] == "True"
    assert config["use_kl_in_reward"] == "False"
    assert config["use_kl_loss"] == "True"
    assert config["kl_loss_coef"] == "0.001"
    assert config["kl_loss_type"] == "low_var_kl"
    assert config["ref_log_prob_micro_batch_size"] == "1"
    assert config["rollout_is"] == "null"
    assert config["enable_thinking"] == "True"
    assert config["data_shuffle"] == "False"


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
        loss_agg_mode="token-mean",
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
