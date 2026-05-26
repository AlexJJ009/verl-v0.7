import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
    apply_dual_model2_group_advantage_positive_fallback,
    apply_wdl_sft_reward_label_advantages,
    prepare_dual_model2_group_adv_is_rollout_log_probs,
)
from verl.workers.actor.dp_actor import select_old_log_prob_for_policy_loss


def _batch(reward_labels, uids, advantages=None, response_mask=None):
    reward_labels = torch.tensor(reward_labels, dtype=torch.float32)
    n = reward_labels.numel()
    t = 3 if response_mask is None else response_mask.shape[-1]
    token_level_scores = torch.zeros(n, t)
    token_level_scores[:, -1] = reward_labels
    if response_mask is None:
        response_mask = torch.ones(n, t)
    if advantages is None:
        advantages = torch.zeros(n, t)
    return DataProto.from_single_dict(
        {
            "token_level_scores": token_level_scores,
            "response_mask": response_mask,
            "advantages": advantages,
            "uid": np.array(uids, dtype=object),
        }
    )


def test_positive_fallback_adds_signal_only_for_all_correct_groups():
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

    out = apply_dual_model2_group_advantage_positive_fallback(
        batch,
        DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
        metrics,
        coef=1.0,
    )

    adv = out.batch["advantages"]
    assert torch.allclose(adv[0], torch.ones(3))
    assert torch.allclose(adv[1], torch.ones(3))
    assert torch.allclose(adv[2], torch.zeros(3))
    assert torch.allclose(adv[3], torch.zeros(3))
    assert torch.allclose(adv[4], torch.full((3,), 0.5))
    assert torch.allclose(adv[5], torch.full((3,), -0.5))
    assert metrics["dual_model2_group_adv_is/all_correct_fallback_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["dual_model2_group_adv_is/all_incorrect_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["dual_model2_group_adv_is/mixed_group_fraction"] == pytest.approx(1 / 3)


def test_wdl_raw_reward_override_does_not_apply_to_new_loss_mode():
    original_advantages = torch.full((2, 3), 0.25)
    batch = _batch(
        reward_labels=[1.0, -1.0],
        uids=["prompt", "prompt"],
        advantages=original_advantages.clone(),
    )

    out = apply_wdl_sft_reward_label_advantages(batch, DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE, metrics={})

    assert torch.allclose(out.batch["advantages"], original_advantages)


def test_prepare_rollout_log_probs_prefers_rollout_log_probs_and_removes_old_anchor():
    rollout_log_probs = torch.randn(2, 3)
    old_log_probs = torch.randn(2, 3)
    batch = DataProto.from_single_dict(
        {
            "rollout_log_probs": rollout_log_probs,
            "old_log_probs": old_log_probs,
        }
    )

    out = prepare_dual_model2_group_adv_is_rollout_log_probs(
        batch,
        selected_source="sub_model_1",
        loss_mode=DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
    )

    assert torch.allclose(out.batch["log_pi_model2_rollout"], rollout_log_probs)
    assert "old_log_probs" not in out.batch
    assert out.batch["log_pi_model2_rollout"].data_ptr() != rollout_log_probs.data_ptr()


def test_prepare_rollout_log_probs_uses_old_log_probs_fallback_then_removes_it():
    old_log_probs = torch.randn(2, 3)
    batch = DataProto.from_single_dict({"old_log_probs": old_log_probs})

    out = prepare_dual_model2_group_adv_is_rollout_log_probs(
        batch,
        selected_source="sub_model_1",
        loss_mode=DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
    )

    assert torch.allclose(out.batch["log_pi_model2_rollout"], old_log_probs)
    assert "old_log_probs" not in out.batch


def test_prepare_rollout_log_probs_fails_fast_for_wrong_source_or_missing_tensor():
    batch = DataProto.from_single_dict({"rollout_log_probs": torch.zeros(2, 3)})

    with pytest.raises(ValueError, match="model2-only rollout"):
        prepare_dual_model2_group_adv_is_rollout_log_probs(
            batch,
            selected_source="sub_model_0",
            loss_mode=DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
        )

    with pytest.raises(ValueError, match="requires model2 rollout log-probs"):
        prepare_dual_model2_group_adv_is_rollout_log_probs(
            DataProto.from_single_dict({"responses": torch.zeros(2, 3, dtype=torch.long)}),
            selected_source="sub_model_1",
            loss_mode=DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
        )


def test_dp_actor_new_loss_uses_fused_old_log_probs_even_when_on_policy():
    old_log_probs = torch.full((2, 3), -3.0)
    current_log_probs = torch.full((2, 3), -1.0)
    config = type("Config", (), {"policy_loss": {"loss_mode": DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE}})()

    selected = select_old_log_prob_for_policy_loss(
        loss_mode=DUAL_MODEL2_GROUP_ADV_IS_LOSS_MODE,
        config=config,
        model_inputs={"old_log_probs": old_log_probs},
        log_prob=current_log_probs,
        on_policy=True,
    )

    assert selected is old_log_probs
    assert selected is not current_log_probs


def test_dp_actor_default_on_policy_path_still_uses_current_detached_log_probs():
    old_log_probs = torch.full((2, 3), -3.0)
    current_log_probs = torch.full((2, 3), -1.0, requires_grad=True)
    config = type("Config", (), {"policy_loss": {"loss_mode": "vanilla"}})()

    selected = select_old_log_prob_for_policy_loss(
        loss_mode="vanilla",
        config=config,
        model_inputs={"old_log_probs": old_log_probs},
        log_prob=current_log_probs,
        on_policy=True,
    )

    assert torch.allclose(selected, current_log_probs.detach())
    assert selected.requires_grad is False
