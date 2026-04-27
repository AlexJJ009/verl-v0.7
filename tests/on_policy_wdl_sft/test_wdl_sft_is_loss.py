# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Tests for WDL-SFT-IS (v2) policy loss.

Verifies the IS-corrected version of On-Policy WDL-SFT against:
  - v1 backward compatibility (ratio=1, no clip, no rollout_is, beta=0)
  - Per-token binary clip correctness on positive vs negative samples
  - rollout_is_weights propagation
  - Edge cases: k=0 (all incorrect), N-k=0 (all correct), beta>0 path

Relevant code: verl/trainer/ppo/core_algos.py
  - v1:  compute_wdl_sft_loss, compute_policy_loss_wdl_sft
  - v2:  compute_wdl_sft_is_loss, compute_policy_loss_wdl_sft_is
"""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_wdl_sft_is_loss,
    compute_wdl_sft_loss,
    get_policy_loss_fn,
)
from verl.trainer.ppo.ray_trainer import apply_wdl_sft_reward_label_advantages, compute_advantage
from verl.workers.config.actor import ActorConfig


# ---------- Fixtures / helpers ----------


def _make_config(
    beta: float = 0.0, clip_ratio_low: float = 0.2, clip_ratio_high: float = 0.27
) -> ActorConfig:
    config = ActorConfig.__new__(ActorConfig)
    config.clip_ratio = 0.2
    config.clip_ratio_low = clip_ratio_low
    config.clip_ratio_high = clip_ratio_high
    config.global_batch_info = {}
    # policy_loss is typically a DictConfig; we mimic the interface used by the wrapper.
    config.policy_loss = {"wdl_sft_beta": beta, "loss_mode": "wdl_sft_is"}
    return config


def _broadcast_labels(reward_labels: torch.Tensor, T: int) -> torch.Tensor:
    """Mimic the training-loop convention: advantages = reward_labels broadcast to (N, T)."""
    return reward_labels[:, None].expand(-1, T).contiguous()


def _make_reward_batch(reward_labels: torch.Tensor, T: int = 4) -> DataProto:
    token_level_scores = torch.zeros(reward_labels.numel(), T)
    token_level_scores[:, -1] = reward_labels
    return DataProto.from_single_dict(
        {
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_scores.clone(),
            "response_mask": torch.ones(reward_labels.numel(), T),
            "uid": np.array(["prompt-0"] * reward_labels.numel(), dtype=object),
        }
    )


def _compute_grpo_then_apply_wdl_override(
    reward_labels: torch.Tensor,
    loss_mode: str = "wdl_sft_is",
    T: int = 4,
) -> tuple[DataProto, torch.Tensor, dict]:
    batch = _make_reward_batch(reward_labels, T=T)
    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=reward_labels.numel(),
    )
    grpo_advantages = batch.batch["advantages"].clone()
    metrics = {}
    batch = apply_wdl_sft_reward_label_advantages(batch, loss_mode, metrics)
    return batch, grpo_advantages, metrics


# ---------- Backward compatibility with v1 ----------


class TestBackwardCompatWithV1:
    """When IS signal is neutral (ratio=1, no rollout_is_weights), v2 should match v1 numerically."""

    def setup_method(self):
        torch.manual_seed(42)
        self.N, self.T = 4, 6
        self.log_prob = torch.randn(self.N, self.T, requires_grad=False) * 0.5 - 1.0
        self.response_mask = torch.ones(self.N, self.T)
        self.response_mask[0, -1] = 0.0  # one pad token
        self.response_mask[2, -2:] = 0.0  # two pad tokens

    def test_forward_only_matches_v1(self):
        # Mix of correct and incorrect
        reward_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])

        v1 = compute_wdl_sft_loss(
            log_prob=self.log_prob,
            response_mask=self.response_mask,
            reward_labels=reward_labels,
            beta=0.0,
        )
        v2 = compute_wdl_sft_is_loss(
            old_log_prob=self.log_prob,  # ratio = 1 everywhere
            log_prob=self.log_prob,
            response_mask=self.response_mask,
            reward_labels=reward_labels,
            beta=0.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.27,
            rollout_is_weights=None,
        )

        assert torch.allclose(v1["loss_positive"], v2["loss_positive"], atol=1e-6)
        assert torch.allclose(v1["total_loss"], v2["total_loss"], atol=1e-6)

    def test_bidirectional_matches_v1(self):
        # beta > 0 path, both sets non-empty
        reward_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])

        v1 = compute_wdl_sft_loss(
            log_prob=self.log_prob,
            response_mask=self.response_mask,
            reward_labels=reward_labels,
            beta=0.1,
        )
        v2 = compute_wdl_sft_is_loss(
            old_log_prob=self.log_prob,
            log_prob=self.log_prob,
            response_mask=self.response_mask,
            reward_labels=reward_labels,
            beta=0.1,
            clip_ratio_low=0.2,
            clip_ratio_high=0.27,
        )

        assert torch.allclose(v1["loss_positive"], v2["loss_positive"], atol=1e-6)
        assert torch.allclose(v1["loss_negative"], v2["loss_negative"], atol=1e-6)
        assert torch.allclose(v1["total_loss"], v2["total_loss"], atol=1e-6)


# ---------- Binary clip correctness ----------


class TestBinaryClip:
    """Verify that clipped tokens receive exactly zero gradient and are counted in the diagnostics."""

    def test_upper_clip_on_positive_samples_zeros_gradient(self):
        # Build a scenario where ratio for sample 0 (correct) exceeds 1 + clip_ratio_high.
        N, T = 2, 3
        response_mask = torch.ones(N, T)
        # Sample 0: correct, we'll push log_prob up to produce ratio > 1.27
        # Sample 1: correct with ratio ~ 1 (in range)
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
        log_prob_val = torch.tensor([[-0.5, -1.0, -1.0], [-1.0, -1.0, -1.0]])  # sample 0 pos 0: ratio=e^0.5~1.65
        log_prob = log_prob_val.clone().requires_grad_(True)
        reward_labels = torch.tensor([1.0, 1.0])

        out = compute_wdl_sft_is_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.27,
        )
        out["loss_positive"].backward()

        # The (0, 0) token should have zero gradient because ratio is out of bounds.
        assert log_prob.grad is not None
        assert torch.isclose(log_prob.grad[0, 0], torch.tensor(0.0), atol=1e-7)
        # But (0, 1) and (0, 2) should have nonzero gradient.
        assert log_prob.grad[0, 1].abs() > 0
        # Clipfrac sanity: exactly 1 out of 6 positive tokens clipped.
        assert torch.isclose(out["clipfrac_positive"], torch.tensor(1.0 / 6.0), atol=1e-6)
        assert torch.isclose(out["clipfrac_negative"], torch.tensor(0.0), atol=1e-6)

    def test_lower_clip_on_negative_samples_zeros_gradient(self):
        # Sample 0: incorrect, ratio < 1 - clip_ratio_low (0.8) for token 0.
        N, T = 2, 3
        response_mask = torch.ones(N, T)
        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
        log_prob_val = torch.tensor([[-2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])  # sample 0 pos 0: ratio=e^(-1)~0.37
        log_prob = log_prob_val.clone().requires_grad_(True)
        reward_labels = torch.tensor([-1.0, -1.0])

        out = compute_wdl_sft_is_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=1.0,  # use beta=1 so we can backward on total_loss
            clip_ratio_low=0.2,
            clip_ratio_high=0.27,
        )
        out["total_loss"].backward()

        # Clipped lower-bound token contributes zero gradient.
        assert torch.isclose(log_prob.grad[0, 0], torch.tensor(0.0), atol=1e-7)
        # Unclipped negative tokens contribute nonzero gradient.
        assert log_prob.grad[0, 1].abs() > 0
        assert torch.isclose(out["clipfrac_negative"], torch.tensor(1.0 / 6.0), atol=1e-6)
        assert torch.isclose(out["clipfrac_positive"], torch.tensor(0.0), atol=1e-6)

    def test_negative_samples_NOT_clipped_by_upper_bound(self):
        # A negative sample with ratio > 1 + high should NOT be masked (only the lower bound applies).
        N, T = 1, 2
        response_mask = torch.ones(N, T)
        old_log_prob = torch.tensor([[-1.0, -1.0]])
        log_prob = torch.tensor([[-0.5, -1.0]], requires_grad=True)  # ratio=1.65 on pos 0
        reward_labels = torch.tensor([-1.0])

        out = compute_wdl_sft_is_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=1.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.27,
        )
        out["total_loss"].backward()

        # Negative sample, ratio>1+high but that's NOT a clip event for negatives.
        # Gradient should be nonzero.
        assert log_prob.grad[0, 0].abs() > 0
        assert torch.isclose(out["clipfrac_negative"], torch.tensor(0.0), atol=1e-6)


# ---------- rollout_is_weights ----------


class TestRolloutISWeights:
    def test_weights_multiply_into_loss(self):
        N, T = 2, 3
        log_prob = torch.randn(N, T) * 0.3 - 1.0
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, 1.0])

        base = compute_wdl_sft_is_loss(
            old_log_prob=log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.0,
        )
        weighted = compute_wdl_sft_is_loss(
            old_log_prob=log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.0,
            rollout_is_weights=torch.full((N, T), 0.5),
        )
        # Weighting by 0.5 everywhere should halve the positive loss.
        assert torch.allclose(weighted["loss_positive"], base["loss_positive"] * 0.5, atol=1e-6)


# ---------- Edge cases ----------


class TestEdgeCases:
    def test_k_zero_all_incorrect_forward_only_returns_zero(self):
        # All incorrect + beta=0 → total loss should be exactly 0.
        N, T = 3, 4
        log_prob = torch.randn(N, T, requires_grad=True)
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([-1.0, -1.0, -1.0])

        out = compute_wdl_sft_is_loss(
            old_log_prob=log_prob.detach(),
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.0,
        )
        assert torch.isclose(out["loss_positive"], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(out["total_loss"], torch.tensor(0.0), atol=1e-6)

    def test_k_zero_but_beta_positive_still_produces_reverse_signal(self):
        # V2 DIFFERENCE from V1: when k=0, v1 returned all zeros; v2 computes L- normally.
        N, T = 3, 4
        torch.manual_seed(0)
        log_prob = torch.randn(N, T) * 0.3 - 1.0
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([-1.0, -1.0, -1.0])

        out_v1 = compute_wdl_sft_loss(
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.1,
        )
        out_v2 = compute_wdl_sft_is_loss(
            old_log_prob=log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.1,
        )
        # v1 skips entire prompt → zero
        assert torch.isclose(out_v1["total_loss"], torch.tensor(0.0), atol=1e-6)
        # v2 still gives beta * L- (nonzero, since log_prob has finite values)
        assert out_v2["loss_negative"].abs() > 1e-3
        assert not torch.isclose(out_v2["total_loss"], torch.tensor(0.0), atol=1e-4)

    def test_all_correct_no_reverse_signal(self):
        N, T = 3, 4
        torch.manual_seed(1)
        log_prob = torch.randn(N, T) * 0.3 - 1.0
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, 1.0, 1.0])

        out = compute_wdl_sft_is_loss(
            old_log_prob=log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            reward_labels=reward_labels,
            beta=0.5,
        )
        # N-k=0 → L- = 0
        assert torch.isclose(out["loss_negative"], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(out["total_loss"], out["loss_positive"], atol=1e-6)


# ---------- Trainer reward-label handoff ----------


class TestTrainerRewardLabelOverride:
    """Regression coverage for the trainer-level wdl_sft_is label handoff."""

    @pytest.mark.parametrize("loss_mode", ["wdl_sft", "wdl_sft_is"])
    def test_wdl_modes_overwrite_grpo_advantages_with_raw_reward_labels(self, loss_mode):
        T = 4
        reward_labels = torch.tensor([1.0, 1.0, 1.0, -1.0])

        batch, grpo_advantages, metrics = _compute_grpo_then_apply_wdl_override(
            reward_labels, loss_mode=loss_mode, T=T
        )

        assert not torch.allclose(grpo_advantages[:, 0], reward_labels)
        assert torch.allclose(batch.batch["advantages"], _broadcast_labels(reward_labels, T))
        assert metrics["wdl_sft/n_correct"] == 3
        assert metrics["wdl_sft/n_incorrect"] == 1
        assert metrics["wdl_sft/correct_ratio"] == 0.75

    def test_non_wdl_mode_leaves_grpo_advantages_unchanged(self):
        reward_labels = torch.tensor([1.0, 1.0, 1.0, -1.0])
        batch = _make_reward_batch(reward_labels)
        batch = compute_advantage(
            batch,
            adv_estimator=AdvantageEstimator.GRPO,
            num_repeat=reward_labels.numel(),
        )
        before = batch.batch["advantages"].clone()
        metrics = {}

        batch = apply_wdl_sft_reward_label_advantages(batch, "vanilla", metrics)

        assert torch.allclose(batch.batch["advantages"], before)
        assert metrics == {}

    def test_wdl_sft_is_all_correct_keeps_positive_signal_after_grpo(self):
        T = 4
        reward_labels = torch.ones(4)

        batch, grpo_advantages, metrics = _compute_grpo_then_apply_wdl_override(
            reward_labels, loss_mode="wdl_sft_is", T=T
        )
        assert torch.allclose(grpo_advantages[:, 0], torch.zeros_like(reward_labels))
        assert torch.allclose(batch.batch["advantages"], _broadcast_labels(reward_labels, T))
        assert metrics["wdl_sft/n_correct"] == 4
        assert metrics["wdl_sft/n_incorrect"] == 0

        log_prob = torch.full((reward_labels.numel(), T), -1.0, requires_grad=True)
        loss, loss_metrics = get_policy_loss_fn("wdl_sft_is")(
            old_log_prob=log_prob.detach().clone(),
            log_prob=log_prob,
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["response_mask"],
            config=_make_config(beta=0.0),
        )

        assert loss.requires_grad
        assert loss_metrics["actor/wdl_sft_loss_positive"] > 0
        assert loss_metrics["actor/wdl_sft_loss_total"] > 0

    def test_wdl_sft_is_all_incorrect_keeps_reverse_signal_after_grpo(self):
        T = 4
        reward_labels = -torch.ones(4)

        batch, grpo_advantages, metrics = _compute_grpo_then_apply_wdl_override(
            reward_labels, loss_mode="wdl_sft_is", T=T
        )
        assert torch.allclose(grpo_advantages[:, 0], torch.zeros_like(reward_labels))
        assert torch.allclose(batch.batch["advantages"], _broadcast_labels(reward_labels, T))
        assert metrics["wdl_sft/n_correct"] == 0
        assert metrics["wdl_sft/n_incorrect"] == 4

        log_prob = torch.full((reward_labels.numel(), T), -1.0, requires_grad=True)
        loss, loss_metrics = get_policy_loss_fn("wdl_sft_is")(
            old_log_prob=log_prob.detach().clone(),
            log_prob=log_prob,
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["response_mask"],
            config=_make_config(beta=0.1),
        )

        assert loss.requires_grad
        assert loss_metrics["actor/wdl_sft_loss_negative"] < 0
        assert loss_metrics["actor/wdl_sft_loss_total"] < 0


# ---------- End-to-end wrapper registration ----------


class TestWrapperRegistration:
    """Verify the wdl_sft_is loss is registered and the wrapper runs end-to-end."""

    def test_registry_lookup(self):
        fn = get_policy_loss_fn("wdl_sft_is")
        assert fn is not None

    def test_wrapper_e2e_forward_only(self):
        fn = get_policy_loss_fn("wdl_sft_is")
        torch.manual_seed(7)
        N, T = 4, 5
        log_prob = torch.randn(N, T, requires_grad=True) * 0.5 - 1.0
        old_log_prob = log_prob.detach() + torch.randn(N, T) * 0.1
        reward_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])
        advantages = _broadcast_labels(reward_labels, T)
        response_mask = torch.ones(N, T)

        loss, metrics = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(beta=0.0),
            rollout_is_weights=None,
        )
        assert loss.requires_grad
        assert "actor/wdl_sft_loss_positive" in metrics
        assert "actor/pg_clipfrac" in metrics
        assert "actor/pg_clipfrac_lower" in metrics
        assert "actor/ppo_kl" in metrics
        assert "actor/wdl_sft_beta" in metrics
        assert metrics["actor/wdl_sft_beta"] == 0.0

    def test_wrapper_propagates_beta(self):
        fn = get_policy_loss_fn("wdl_sft_is")
        torch.manual_seed(8)
        N, T = 4, 5
        log_prob = torch.randn(N, T, requires_grad=True) * 0.5 - 1.0
        old_log_prob = log_prob.detach()
        reward_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])
        advantages = _broadcast_labels(reward_labels, T)
        response_mask = torch.ones(N, T)

        _, metrics_b0 = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(beta=0.0),
        )
        _, metrics_b1 = fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_make_config(beta=0.1),
        )
        assert metrics_b0["actor/wdl_sft_beta"] == 0.0
        assert metrics_b1["actor/wdl_sft_beta"] == 0.1
        # With beta=0.1 and nonzero L-, total != L+ anymore.
        assert metrics_b1["actor/wdl_sft_loss_total"] != metrics_b1["actor/wdl_sft_loss_positive"]
