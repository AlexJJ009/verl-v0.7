# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Trainer-path tests for wdl_group_adv_is advantage routing."""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    apply_wdl_group_advantage_positive_fallback,
    apply_wdl_sft_reward_label_advantages,
    compute_advantage,
)


def _make_reward_batch(reward_labels: torch.Tensor, uids: list[str], T: int = 3) -> DataProto:
    token_level_scores = torch.zeros(reward_labels.numel(), T)
    token_level_scores[:, -1] = reward_labels
    return DataProto.from_single_dict(
        {
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_scores.clone(),
            "response_mask": torch.ones(reward_labels.numel(), T),
            "uid": np.array(uids, dtype=object),
        }
    )


def test_group_advantage_loss_preserves_grpo_advantages_and_adds_only_all_correct_fallback():
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    uids = ["all-correct", "all-correct", "all-incorrect", "all-incorrect", "mixed", "mixed"]
    batch = _make_reward_batch(rewards, uids)

    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )
    grpo_advantages = batch.batch["advantages"].clone()

    metrics = {}
    batch = apply_wdl_group_advantage_positive_fallback(
        batch,
        "wdl_group_adv_is",
        metrics=metrics,
        enabled=True,
        coef=1.0,
    )

    effective_advantages = batch.batch["advantages"][:, 0]
    assert torch.allclose(grpo_advantages[:, 0], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0]))
    assert torch.allclose(effective_advantages, torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, -1.0]))
    assert metrics["wdl_group_adv_is/zero_adv_group_fraction"] == pytest.approx(2 / 3)
    assert metrics["wdl_group_adv_is/mixed_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["wdl_group_adv_is/all_correct_fallback_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["wdl_group_adv_is/all_correct_fallback_response_fraction"] == pytest.approx(2 / 6)


def test_group_advantage_loss_is_excluded_from_raw_reward_label_override():
    rewards = torch.tensor([1.0, -1.0])
    batch = _make_reward_batch(rewards, ["p", "p"])
    batch.batch["advantages"] = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])

    metrics = {}
    out = apply_wdl_sft_reward_label_advantages(batch, "wdl_group_adv_is", metrics)

    assert torch.allclose(out.batch["advantages"], torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]))
    assert metrics == {}


def test_wdl_sft_and_wdl_sft_is_keep_raw_reward_label_override():
    rewards = torch.tensor([1.0, -1.0])
    for loss_mode in ("wdl_sft", "wdl_sft_is"):
        batch = _make_reward_batch(rewards, ["p", "p"])
        batch.batch["advantages"] = torch.zeros(2, 3)
        metrics = {}
        out = apply_wdl_sft_reward_label_advantages(batch, loss_mode, metrics)

        assert torch.allclose(out.batch["advantages"], torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]))
        assert metrics["wdl_sft/n_correct"] == 1
        assert metrics["wdl_sft/n_incorrect"] == 1


def test_fallback_uses_true_group_rewards_not_zero_advantages():
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    uids = ["all-correct", "all-correct", "all-incorrect", "all-incorrect"]
    batch = _make_reward_batch(rewards, uids)
    batch.batch["advantages"] = torch.zeros(4, 3)

    batch = apply_wdl_group_advantage_positive_fallback(batch, "wdl_group_adv_is", metrics={})

    assert torch.allclose(batch.batch["advantages"][:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_group_advantage_loss_requires_uid_for_fallback():
    batch = DataProto.from_single_dict(
        {
            "token_level_scores": torch.ones(2, 3),
            "response_mask": torch.ones(2, 3),
            "advantages": torch.zeros(2, 3),
        }
    )

    with pytest.raises(ValueError, match="requires uid"):
        apply_wdl_group_advantage_positive_fallback(batch, "wdl_group_adv_is")
