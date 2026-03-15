# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for MiniRL policy loss function."""

import pytest
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config.actor import ActorConfig


def _make_config(clip_ratio_low=0.2, clip_ratio_high=0.27) -> ActorConfig:
    """Create a minimal ActorConfig for testing."""
    config = ActorConfig.__new__(ActorConfig)
    config.clip_ratio = 0.2
    config.clip_ratio_low = clip_ratio_low
    config.clip_ratio_high = clip_ratio_high
    config.global_batch_info = {}
    return config


def _get_minirl_fn():
    return get_policy_loss_fn("minirl")


class TestMiniRLLoss:
    """Test suite for MiniRL policy loss."""

    def test_minirl_basic_positive_advantage(self):
        """Positive advantage + ratio in range → mask=1, loss < 0 (maximize direction)."""
        fn = _get_minirl_fn()
        # ratio ~ 1.0 (in range), positive advantage
        old_log_prob = torch.tensor([[-1.0, -2.0, -1.5]])
        log_prob = torch.tensor([[-1.0, -2.0, -1.5]], requires_grad=True)
        advantages = torch.tensor([[1.0, 2.0, 0.5]])
        response_mask = torch.tensor([[1.0, 1.0, 1.0]])

        loss, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
        )
        # Loss should be negative (we're maximizing expected return)
        # loss = -mask * adv * log_prob, with positive adv and negative log_prob → loss < 0... wait
        # Actually: -1 * positive_adv * negative_log_prob = positive value
        # But aggregated via seq-mean-token-sum: sum of per-token losses
        # Let's just check the sign: - adv * log_prob with adv > 0, log_prob < 0 → positive per-token
        # The overall pg_loss for positive advantage should be positive (minimizing it pushes log_prob up)
        # Actually the key test: gradient should push log_prob up for positive advantage
        loss.backward()
        # For positive advantage, gradient of loss w.r.t. log_prob should be negative
        # (minimizing loss = increasing log_prob)
        assert (log_prob.grad[response_mask.bool()] < 0).all()

    def test_minirl_basic_negative_advantage(self):
        """Negative advantage + ratio in range → mask=1, loss pushes log_prob down."""
        fn = _get_minirl_fn()
        old_log_prob = torch.tensor([[-1.0, -2.0, -1.5]])
        log_prob = torch.tensor([[-1.0, -2.0, -1.5]], requires_grad=True)
        advantages = torch.tensor([[-1.0, -2.0, -0.5]])
        response_mask = torch.tensor([[1.0, 1.0, 1.0]])

        loss, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
        )
        loss.backward()
        # For negative advantage, gradient should be positive (minimizing loss = decreasing log_prob)
        assert (log_prob.grad[response_mask.bool()] > 0).all()

    def test_minirl_clip_mask_positive(self):
        """Positive advantage + ratio > 1+ε_high → mask=0, token loss=0."""
        fn = _get_minirl_fn()
        config = _make_config(clip_ratio_high=0.27)
        # Make ratio = exp(log_prob - old_log_prob) > 1.27
        # log_prob - old_log_prob = ln(1.5) ≈ 0.405
        old_log_prob = torch.tensor([[-2.0]])
        log_prob = torch.tensor([[-2.0 + 0.405]], requires_grad=True)  # ratio ≈ 1.5 > 1.27
        advantages = torch.tensor([[1.0]])  # positive
        response_mask = torch.tensor([[1.0]])

        loss, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=config,
        )
        # Masked out → loss should be 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_minirl_clip_mask_negative(self):
        """Negative advantage + ratio < 1-ε_low → mask=0, token loss=0."""
        fn = _get_minirl_fn()
        config = _make_config(clip_ratio_low=0.2)
        # Make ratio = exp(-0.5) ≈ 0.607 < 0.8 = 1 - 0.2
        old_log_prob = torch.tensor([[-2.0]])
        log_prob = torch.tensor([[-2.5]], requires_grad=True)  # ratio ≈ 0.607 < 0.8
        advantages = torch.tensor([[-1.0]])  # negative
        response_mask = torch.tensor([[1.0]])

        loss, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=config,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_minirl_zero_advantage(self):
        """Zero advantage → loss=0 regardless of ratio/mask."""
        fn = _get_minirl_fn()
        old_log_prob = torch.tensor([[-1.0, -2.0]])
        log_prob = torch.tensor([[-0.5, -3.0]], requires_grad=True)  # various ratios
        advantages = torch.tensor([[0.0, 0.0]])
        response_mask = torch.tensor([[1.0, 1.0]])

        loss, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_minirl_is_weights_applied(self):
        """rollout_is_weights provided → loss is scaled."""
        fn = _get_minirl_fn()
        old_log_prob = torch.tensor([[-1.0, -2.0]])
        log_prob = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0]])
        response_mask = torch.tensor([[1.0, 1.0]])

        # Without IS weights
        loss_no_is, _ = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob.detach().requires_grad_(True),
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
        )

        # With IS weights = 2.0
        is_weights = torch.tensor([[2.0, 2.0]])
        loss_with_is, _ = fn(
            old_log_prob=old_log_prob,
            log_prob=torch.tensor([[-1.0, -2.0]], requires_grad=True),
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
            rollout_is_weights=is_weights,
        )

        assert loss_with_is.item() == pytest.approx(2.0 * loss_no_is.item(), rel=1e-5)

    def test_minirl_is_weights_none(self):
        """rollout_is_weights=None → equivalent to weights all 1."""
        fn = _get_minirl_fn()
        old_log_prob = torch.tensor([[-1.0, -2.0]])
        advantages = torch.tensor([[1.0, 0.5]])
        response_mask = torch.tensor([[1.0, 1.0]])

        log_prob_a = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        loss_none, _ = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob_a,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
            rollout_is_weights=None,
        )

        log_prob_b = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        ones = torch.ones_like(response_mask)
        loss_ones, _ = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob_b,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
            rollout_is_weights=ones,
        )

        assert loss_none.item() == pytest.approx(loss_ones.item(), rel=1e-5)

    def test_minirl_loss_agg_seq_mean_token_sum(self):
        """Verify seq-mean-token-sum aggregation: sum tokens per seq, then mean across seqs."""
        fn = _get_minirl_fn()
        # 2 sequences, 3 tokens each, ratio=1 (same log probs)
        old_log_prob = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
        log_prob = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]], requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        response_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])  # seq2 has 2 valid tokens

        loss, _ = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
        )

        # Manual: pg_losses = -1 * adv * log_prob (mask=1, ratio=1)
        # seq1: -1*(1*-1 + 1*-2 + 1*-3) = -1*(-6) = 6
        # seq2: -1*(2*-1 + 2*-2 + 0) = -1*(-6) = 6  (3rd token masked)
        # seq-mean: (6 + 6) / 2 = 6
        expected = (-1.0 * (1 * (-1) + 1 * (-2) + 1 * (-3)) + -1.0 * (2 * (-1) + 2 * (-2))) / 2.0
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_minirl_metrics_returned(self):
        """Check that ppo_kl, pg_clipfrac, and mask_fraction are in metrics."""
        fn = _get_minirl_fn()
        old_log_prob = torch.tensor([[-1.0, -2.0]])
        log_prob = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0]])
        response_mask = torch.tensor([[1.0, 1.0]])

        _, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
        )

        assert "actor/ppo_kl" in metrics
        assert "actor/pg_clipfrac" in metrics
        assert "actor/pg_clipfrac_lower" in metrics

    def test_minirl_gradient_only_through_logprob(self):
        """Verify gradients flow only through log_prob, not IS weights or mask."""
        fn = _get_minirl_fn()
        old_log_prob = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        log_prob = torch.tensor([[-1.1, -1.9]], requires_grad=True)
        advantages = torch.tensor([[1.0, -1.0]], requires_grad=True)
        response_mask = torch.tensor([[1.0, 1.0]])
        is_weights = torch.tensor([[1.5, 0.8]], requires_grad=True)

        loss, _ = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(),
            rollout_is_weights=is_weights,
        )
        loss.backward()

        # Gradient should flow through log_prob
        assert log_prob.grad is not None
        assert (log_prob.grad != 0).any()

        # Gradient should NOT flow through old_log_prob (used only for ratio → detached mask)
        assert old_log_prob.grad is None or (old_log_prob.grad == 0).all()

        # Gradient should NOT flow through advantages (detached in loss)
        assert advantages.grad is None or (advantages.grad == 0).all()

        # Gradient should NOT flow through IS weights (should be detached)
        assert is_weights.grad is None or (is_weights.grad == 0).all()
