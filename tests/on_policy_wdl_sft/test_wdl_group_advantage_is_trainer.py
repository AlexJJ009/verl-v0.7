import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    WDL_GROUP_ADV_IS_LOSS_MODE,
    apply_wdl_group_advantage_positive_fallback,
    apply_wdl_sft_reward_label_advantages,
)


def _batch(reward_labels, uids, advantages=None):
    reward_labels = torch.tensor(reward_labels, dtype=torch.float32)
    n = reward_labels.numel()
    token_level_scores = torch.zeros(n, 3)
    token_level_scores[:, -1] = reward_labels
    if advantages is None:
        advantages = torch.zeros(n, 3)
    return DataProto.from_single_dict(
        {
            "token_level_scores": token_level_scores,
            "response_mask": torch.ones(n, 3),
            "advantages": advantages,
            "uid": np.array(uids, dtype=object),
        }
    )


def test_wdl_group_adv_positive_fallback_adds_signal_only_for_all_correct_groups():
    batch = _batch(
        reward_labels=[1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
        uids=["all-correct", "all-correct", "all-incorrect", "all-incorrect", "mixed", "mixed"],
        advantages=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ]
        ),
    )
    metrics = {}

    out = apply_wdl_group_advantage_positive_fallback(
        batch,
        WDL_GROUP_ADV_IS_LOSS_MODE,
        metrics,
        enabled=True,
        coef=1.0,
    )

    adv = out.batch["advantages"]
    assert torch.allclose(adv[0], torch.ones(3))
    assert torch.allclose(adv[1], torch.ones(3))
    assert torch.allclose(adv[2], torch.zeros(3))
    assert torch.allclose(adv[3], torch.zeros(3))
    assert torch.allclose(adv[4], torch.full((3,), 0.5))
    assert torch.allclose(adv[5], torch.full((3,), -0.5))
    assert metrics["wdl_group_adv_is/all_correct_fallback_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["wdl_group_adv_is/all_incorrect_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["wdl_group_adv_is/mixed_group_fraction"] == pytest.approx(1 / 3)


def test_wdl_raw_reward_override_does_not_apply_to_group_adv_loss():
    original_advantages = torch.full((2, 3), 0.25)
    batch = _batch(
        reward_labels=[1.0, -1.0],
        uids=["prompt", "prompt"],
        advantages=original_advantages.clone(),
    )

    out = apply_wdl_sft_reward_label_advantages(batch, WDL_GROUP_ADV_IS_LOSS_MODE, metrics={})

    assert torch.allclose(out.batch["advantages"], original_advantages)
