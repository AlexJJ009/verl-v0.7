import pytest
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config.actor import ActorConfig


def _config(clip_ratio_low=0.2, clip_ratio_high=0.27, tis_threshold=5.0):
    config = ActorConfig.__new__(ActorConfig)
    config.clip_ratio = 0.2
    config.clip_ratio_low = clip_ratio_low
    config.clip_ratio_high = clip_ratio_high
    config.global_batch_info = {}
    config.policy_loss = {
        "loss_mode": "dual_model2_group_adv_is",
        "gamma_pos_sft": 1.0,
        "tis_threshold": tis_threshold,
    }
    return config


def _loss(
    *,
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    log_pi_model2_rollout,
    config=None,
    rollout_is_weights=None,
    loss_agg_mode="seq-mean-token-sum",
):
    fn = get_policy_loss_fn("dual_model2_group_adv_is")
    return fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config or _config(),
        rollout_is_weights=rollout_is_weights,
        log_pi_model2_rollout=log_pi_model2_rollout,
    )


def test_registered_loss_requires_model2_rollout_log_probs():
    log_prob = torch.zeros(2, 3, requires_grad=True)
    old_log_prob = torch.zeros(2, 3)
    advantages = torch.ones(2, 3)
    response_mask = torch.ones(2, 3)

    with pytest.raises(ValueError, match="requires log_pi_model2_rollout"):
        _loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            log_pi_model2_rollout=None,
        )


def test_loss_rejects_external_rollout_is_weights_and_wrong_aggregation():
    log_prob = torch.zeros(2, 3, requires_grad=True)
    old_log_prob = torch.zeros(2, 3)
    advantages = torch.ones(2, 3)
    response_mask = torch.ones(2, 3)
    model2_rollout = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="forbids external rollout_is_weights"):
        _loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            log_pi_model2_rollout=model2_rollout,
            rollout_is_weights=torch.ones(2, 3),
        )

    with pytest.raises(ValueError, match="seq-mean-token-sum"):
        _loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            log_pi_model2_rollout=model2_rollout,
            loss_agg_mode="token-mean",
        )


def test_only_current_fused_log_prob_receives_gradient():
    old_log_prob = torch.zeros(2, 3, requires_grad=True)
    model2_rollout = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], requires_grad=True)
    response_mask = torch.ones(2, 3)
    log_prob = torch.zeros(2, 3, requires_grad=True)

    loss, metrics = _loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        log_pi_model2_rollout=model2_rollout,
    )
    loss.backward()

    assert log_prob.grad is not None
    assert old_log_prob.grad is None
    assert model2_rollout.grad is None
    assert advantages.grad is None
    assert metrics["dual_model2_group_adv_is/tis_mean"] == pytest.approx(1.0)
    assert metrics["dual_model2_group_adv_is/clipfrac_positive"] == pytest.approx(0.0)
    assert metrics["dual_model2_group_adv_is/clipfrac_negative"] == pytest.approx(0.0)


def test_binary_mask_zeros_only_out_of_bound_train_ratio_tokens():
    old_log_prob = torch.zeros(2, 3)
    log_prob = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # positive token 0 exceeds 1 + clip_ratio_high
            [-0.5, 0.0, 0.0],  # negative token 0 falls below 1 - clip_ratio_low
        ],
        requires_grad=True,
    )
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.ones(2, 3)
    model2_rollout = torch.zeros(2, 3)

    loss, metrics = _loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        log_pi_model2_rollout=model2_rollout,
    )
    loss.backward()

    assert log_prob.grad[0, 0].item() == pytest.approx(0.0)
    assert log_prob.grad[1, 0].item() == pytest.approx(0.0)
    assert log_prob.grad[0, 1].item() < 0.0
    assert log_prob.grad[1, 1].item() > 0.0
    assert metrics["dual_model2_group_adv_is/clipfrac_positive"] == pytest.approx(1.0 / 3.0)
    assert metrics["dual_model2_group_adv_is/clipfrac_negative"] == pytest.approx(1.0 / 3.0)


def test_tis_clip_caps_weight_without_masking_gradient():
    old_log_prob = torch.zeros(1, 2)
    model2_rollout = torch.zeros(1, 2)
    log_prob = torch.full((1, 2), 2.0, requires_grad=True)
    advantages = torch.ones(1, 2)
    response_mask = torch.ones(1, 2)

    loss, metrics = _loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        log_pi_model2_rollout=model2_rollout,
        config=_config(clip_ratio_high=100.0, tis_threshold=5.0),
    )
    loss.backward()

    assert torch.all(log_prob.grad < 0)
    assert torch.allclose(log_prob.grad, torch.full((1, 2), -5.0))
    assert metrics["dual_model2_group_adv_is/tis_max"] == pytest.approx(5.0)
    assert metrics["dual_model2_group_adv_is/tis_clip_fraction"] == pytest.approx(1.0)
