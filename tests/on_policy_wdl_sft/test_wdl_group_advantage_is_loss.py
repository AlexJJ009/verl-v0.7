# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Tests for wdl_group_adv_is policy loss."""

import math

import pytest
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config.actor import ActorConfig


def _make_config(clip_ratio_low=0.2, clip_ratio_high=0.27, beta=0.0) -> ActorConfig:
    config = ActorConfig.__new__(ActorConfig)
    config.clip_ratio = 0.2
    config.clip_ratio_low = clip_ratio_low
    config.clip_ratio_high = clip_ratio_high
    config.global_batch_info = {}
    config.policy_loss = {"loss_mode": "wdl_group_adv_is", "wdl_sft_beta": beta}
    return config


def _loss_fn():
    return get_policy_loss_fn("wdl_group_adv_is")


def test_registry_lookup():
    assert _loss_fn() is not None


def test_neutral_ratio_matches_exact_seq_mean_token_sum():
    log_prob = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], requires_grad=True)
    old_log_prob = log_prob.detach().clone()
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    loss, metrics = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(),
    )

    expected_loss_mat = -advantages * log_prob.detach()
    expected = ((expected_loss_mat * response_mask).sum(dim=-1)).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert metrics["wdl_group_adv_is/ratio_mean"] == pytest.approx(1.0)
    assert metrics["wdl_group_adv_is/ratio_max"] == pytest.approx(1.0)


def test_multiplicative_is_scales_gradient_by_detached_ratio():
    old_log_prob = torch.tensor([[-2.0]])
    log_prob = torch.tensor([[-1.9]], requires_grad=True)
    advantages = torch.tensor([[2.0]])
    response_mask = torch.tensor([[1.0]])

    loss, _ = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(),
    )
    loss.backward()

    rho = math.exp(0.1)
    assert log_prob.grad.item() == pytest.approx(-2.0 * rho, rel=1e-6)


def test_mask_and_is_together_zeroes_clipped_token_and_scales_unclipped_token():
    old_log_prob = torch.tensor([[-2.0, -2.0]])
    log_prob = torch.tensor([[-2.0 + math.log(1.1), -2.0 + math.log(1.5)]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0]])
    response_mask = torch.tensor([[1.0, 1.0]])

    loss, metrics = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(clip_ratio_high=0.27),
    )
    loss.backward()

    assert log_prob.grad[0, 0].item() == pytest.approx(-1.1, rel=1e-6)
    assert log_prob.grad[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["wdl_group_adv_is/clipfrac_positive"] == pytest.approx(0.5)


def test_negative_advantage_lower_mask_zeroes_gradient():
    old_log_prob = torch.tensor([[-2.0]])
    log_prob = torch.tensor([[-2.5]], requires_grad=True)
    advantages = torch.tensor([[-1.0]])
    response_mask = torch.tensor([[1.0]])

    loss, metrics = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(clip_ratio_low=0.2),
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert log_prob.grad.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["wdl_group_adv_is/clipfrac_negative"] == pytest.approx(1.0)


def test_nonneutral_ratio_exact_surrogate_value_with_keep_mask():
    old_log_prob = torch.tensor([[-2.0, -2.0, -2.0], [-1.0, -1.0, -1.0]])
    log_prob = torch.tensor(
        [
            [-2.0 + math.log(1.2), -2.0 + math.log(1.5), -2.0 + math.log(0.9)],
            [-1.0 + math.log(0.7), -1.0 + math.log(0.9), -1.0 + math.log(1.1)],
        ],
        requires_grad=True,
    )
    advantages = torch.tensor([[2.0, 2.0, 0.0], [-3.0, -3.0, -3.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    loss, metrics = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(clip_ratio_low=0.2, clip_ratio_high=0.27),
    )

    ratio = torch.exp(log_prob.detach() - old_log_prob)
    keep = torch.ones_like(ratio)
    keep[(advantages >= 0) & (ratio > 1.27)] = 0
    keep[(advantages < 0) & (ratio < 0.8)] = 0
    expected_loss_mat = -advantages * ratio * keep * log_prob.detach()
    expected = ((expected_loss_mat * response_mask).sum(dim=-1)).mean()

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert metrics["wdl_group_adv_is/clipfrac_positive"] == pytest.approx(0.5)
    assert metrics["wdl_group_adv_is/clipfrac_negative"] == pytest.approx(1 / 3)


def test_all_correct_fallback_positive_loss_and_all_incorrect_zero_loss():
    old_log_prob = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    response_mask = torch.ones(2, 2)

    all_correct_log_prob = old_log_prob.detach().clone().requires_grad_(True)
    all_correct_loss, _ = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=all_correct_log_prob,
        advantages=torch.ones(2, 2),
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(),
    )
    all_correct_loss.backward()

    assert all_correct_loss.item() == pytest.approx(2.0, rel=1e-6)
    assert torch.allclose(all_correct_log_prob.grad, torch.full((2, 2), -0.5))

    all_incorrect_log_prob = old_log_prob.detach().clone().requires_grad_(True)
    all_incorrect_loss, _ = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=all_incorrect_log_prob,
        advantages=torch.zeros(2, 2),
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(),
    )
    all_incorrect_loss.backward()

    assert all_incorrect_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(all_incorrect_log_prob.grad, torch.zeros(2, 2))


def test_no_beta_dependency_and_no_beta_metric():
    old_log_prob = torch.tensor([[-1.0, -1.0]])
    advantages = torch.tensor([[1.0, -1.0]])
    response_mask = torch.tensor([[1.0, 1.0]])

    log_prob_a = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    loss_a, metrics_a = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob_a,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(beta=0.0),
    )

    log_prob_b = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    loss_b, metrics_b = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob_b,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(beta=10.0),
    )

    assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-6)
    assert "actor/wdl_sft_beta" not in metrics_a
    assert "actor/wdl_sft_beta" not in metrics_b


def test_rollout_is_weights_raise_clear_error():
    with pytest.raises(ValueError, match="forbids rollout_is_weights"):
        _loss_fn()(
            old_log_prob=torch.zeros(1, 1),
            log_prob=torch.zeros(1, 1, requires_grad=True),
            advantages=torch.ones(1, 1),
            response_mask=torch.ones(1, 1),
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
            rollout_is_weights=torch.ones(1, 1),
        )


def test_non_seq_mean_token_sum_raises_clear_error():
    with pytest.raises(ValueError, match="seq-mean-token-sum"):
        _loss_fn()(
            old_log_prob=torch.zeros(1, 1),
            log_prob=torch.zeros(1, 1, requires_grad=True),
            advantages=torch.ones(1, 1),
            response_mask=torch.ones(1, 1),
            loss_agg_mode="token-mean",
            config=_make_config(),
        )


def test_long_sequence_contributes_more_than_short_sequence():
    log_prob = torch.full((2, 3), -1.0, requires_grad=True)
    old_log_prob = log_prob.detach().clone()
    advantages = torch.ones(2, 3)
    response_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    loss, _ = _loss_fn()(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_make_config(clip_ratio_high=10.0),
    )

    assert loss.item() == pytest.approx(2.0, rel=1e-6)
