# SPDX-License-Identifier: Apache-2.0

"""Comprehensive unit tests for the On-Policy WDL-SFT loss function.

Tests cover:
- Forward loss (L+) correctness with hand-computed values
- Reverse loss (L-) correctness with hand-computed values
- Combined loss correctness with mixed correct/incorrect responses
- Edge case: all correct (C=N, I=empty)
- Edge case: all incorrect (C=empty, I=N) uses reverse SFT only
- No NaN/Inf for random inputs (fuzz test)
- Gradient flow through the loss
- Various beta values
- Masking behavior (padding tokens excluded)
"""

import torch

from verl.trainer.ppo.core_algos import compute_wdl_sft_loss


class TestForwardLossCorrectness:
    """AC4: Forward loss (L+) correctness with hand-computed values."""

    def test_all_correct_simple(self):
        """All N=4 responses correct, uniform log probs, verify L+ = mean NLL."""
        N, T = 4, 3
        # log_prob[i,t] = -1.0 for all tokens (uniform log probs)
        log_prob = torch.full((N, T), -1.0)
        response_mask = torch.ones(N, T)
        reward_labels = torch.ones(N)  # all correct
        beta = 0.1

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta)

        # seq_log_probs[i] = sum_t(-1.0) = -3.0 for each response
        # L+ = -(1/4) * sum_i(-3.0) = -(1/4)*(-12.0) = 3.0
        # L- = 0 (no incorrect)
        # total = 3.0 + 0.1 * 0.0 = 3.0
        assert abs(result["loss_positive"].item() - 3.0) < 1e-5
        assert abs(result["loss_negative"].item() - 0.0) < 1e-5
        assert abs(result["total_loss"].item() - 3.0) < 1e-5

    def test_all_correct_varying_log_probs(self):
        """All correct with different log probs per response. Hand-computed."""
        N, T = 3, 2
        log_prob = torch.tensor(
            [
                [-0.5, -1.0],  # seq sum = -1.5
                [-2.0, -0.3],  # seq sum = -2.3
                [-1.0, -1.0],  # seq sum = -2.0
            ]
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        # L+ = -(1/3) * (-1.5 + -2.3 + -2.0) = -(1/3)*(-5.8) = 5.8/3 ≈ 1.93333
        expected_lp = 5.8 / 3.0
        assert abs(result["loss_positive"].item() - expected_lp) < 1e-5
        assert abs(result["loss_negative"].item()) < 1e-5
        assert abs(result["total_loss"].item() - expected_lp) < 1e-5

    def test_forward_loss_with_mask(self):
        """Forward loss respects response_mask (excludes padding tokens)."""
        N = 2
        log_prob = torch.tensor(
            [
                [-1.0, -2.0, -3.0, -99.0],  # last token is padding
                [-0.5, -0.5, -99.0, -99.0],  # last two are padding
            ]
        )
        response_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
        reward_labels = torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        # seq_log_probs[0] = -1.0 + -2.0 + -3.0 = -6.0  (padding excluded)
        # seq_log_probs[1] = -0.5 + -0.5 = -1.0          (padding excluded)
        # L+ = -(1/2)*(-6.0 + -1.0) = -(1/2)*(-7.0) = 3.5
        assert abs(result["loss_positive"].item() - 3.5) < 1e-5


class TestReverseLossCorrectness:
    """AC5: Reverse loss (L-) correctness with hand-computed values."""

    def test_all_incorrect_simple(self):
        """All N=4 incorrect → reverse SFT only."""
        N, T = 4, 3
        log_prob = torch.full((N, T), -1.0)
        response_mask = torch.ones(N, T)
        reward_labels = -torch.ones(N)  # all incorrect

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        # k=0, N-k=4
        # L+ = 0
        # L- = (1/4) * sum_i(-3.0) = -3.0
        # total = 0.1 * -3.0 = -0.3
        assert abs(result["total_loss"].item() - (-0.3)) < 1e-5
        assert abs(result["loss_positive"].item()) < 1e-5
        assert abs(result["loss_negative"].item() - (-3.0)) < 1e-5

    def test_reverse_loss_mixed(self):
        """Test L- with known inputs where some are incorrect."""
        N, T = 4, 2
        log_prob = torch.tensor(
            [
                [-1.0, -1.0],  # correct, seq = -2.0
                [-0.5, -0.5],  # incorrect, seq = -1.0
                [-2.0, -2.0],  # incorrect, seq = -4.0
                [-1.5, -1.5],  # correct, seq = -3.0
            ]
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, -1.0, -1.0, 1.0])
        beta = 0.1

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta)

        # k=2, N-k=2
        # L+ = -(1/2)*(-2.0 + -3.0) = -(1/2)*(-5.0) = 2.5
        # L- = (1/2)*(-1.0 + -4.0) = (1/2)*(-5.0) = -2.5
        # total = 2.5 + 0.1*(-2.5) = 2.5 - 0.25 = 2.25
        assert abs(result["loss_positive"].item() - 2.5) < 1e-5
        assert abs(result["loss_negative"].item() - (-2.5)) < 1e-5
        assert abs(result["total_loss"].item() - 2.25) < 1e-5

    def test_reverse_loss_only_incorrect_subset(self):
        """3 correct, 1 incorrect. Verify L- uses 1/(N-k)=1/1 normalization."""
        N, T = 4, 2
        log_prob = torch.tensor(
            [
                [-1.0, -1.0],  # correct
                [-1.0, -1.0],  # correct
                [-1.0, -1.0],  # correct
                [-2.0, -3.0],  # incorrect, seq = -5.0
            ]
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, 1.0, 1.0, -1.0])
        beta = 0.5

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta)

        # k=3, N-k=1
        # L+ = -(1/3)*(-2.0 + -2.0 + -2.0) = -(1/3)*(-6.0) = 2.0
        # L- = (1/1)*(-5.0) = -5.0
        # total = 2.0 + 0.5*(-5.0) = 2.0 - 2.5 = -0.5
        assert abs(result["loss_positive"].item() - 2.0) < 1e-5
        assert abs(result["loss_negative"].item() - (-5.0)) < 1e-5
        assert abs(result["total_loss"].item() - (-0.5)) < 1e-5


class TestCombinedLossCorrectness:
    """AC6: Combined loss correctness with mixed batch."""

    def test_mixed_batch(self):
        """N=8, 5 correct, 3 incorrect. Full hand-computed check."""
        N, T = 8, 2
        # Construct specific log probs
        log_prob = torch.tensor(
            [
                [-0.5, -0.5],  # correct, seq = -1.0
                [-1.0, -1.0],  # correct, seq = -2.0
                [-0.8, -0.2],  # incorrect, seq = -1.0
                [-1.5, -0.5],  # correct, seq = -2.0
                [-0.3, -0.7],  # incorrect, seq = -1.0
                [-2.0, -1.0],  # correct, seq = -3.0
                [-0.1, -0.9],  # incorrect, seq = -1.0
                [-1.0, -2.0],  # correct, seq = -3.0
            ]
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        beta = 0.2

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta)

        # Correct: indices 0,1,3,5,7 → seq sums: -1, -2, -2, -3, -3 → sum = -11
        # k=5, L+ = -(1/5)*(-11) = 11/5 = 2.2
        # Incorrect: indices 2,4,6 → seq sums: -1, -1, -1 → sum = -3
        # N-k=3, L- = (1/3)*(-3) = -1.0
        # total = 2.2 + 0.2*(-1.0) = 2.2 - 0.2 = 2.0
        assert abs(result["loss_positive"].item() - 2.2) < 1e-5
        assert abs(result["loss_negative"].item() - (-1.0)) < 1e-5
        assert abs(result["total_loss"].item() - 2.0) < 1e-5

    def test_beta_zero(self):
        """With beta=0, total loss should equal L+ regardless of L-."""
        N, T = 4, 2
        log_prob = torch.tensor(
            [
                [-1.0, -1.0],  # correct
                [-2.0, -2.0],  # incorrect
                [-1.0, -1.0],  # correct
                [-3.0, -3.0],  # incorrect
            ]
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.0)

        # L+ = -(1/2)*(-2.0 + -2.0) = 2.0
        # total = L+ + 0 * L- = 2.0
        assert abs(result["total_loss"].item() - result["loss_positive"].item()) < 1e-5
        assert abs(result["total_loss"].item() - 2.0) < 1e-5

    def test_beta_one(self):
        """With beta=1.0, both components weighted equally."""
        N, T = 2, 2
        log_prob = torch.tensor(
            [
                [-1.0, -1.0],  # correct, seq = -2.0
                [-2.0, -2.0],  # incorrect, seq = -4.0
            ]
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, -1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=1.0)

        # L+ = -(1/1)*(-2.0) = 2.0
        # L- = (1/1)*(-4.0) = -4.0
        # total = 2.0 + 1.0*(-4.0) = -2.0
        assert abs(result["loss_positive"].item() - 2.0) < 1e-5
        assert abs(result["loss_negative"].item() - (-4.0)) < 1e-5
        assert abs(result["total_loss"].item() - (-2.0)) < 1e-5


class TestEdgeCaseAllCorrect:
    """AC7: All correct (C=N, I=empty) → L- = 0, total = L+ only."""

    def test_all_correct_n8(self):
        """N=8 all correct, verify L- is exactly 0."""
        N, T = 8, 5
        torch.manual_seed(42)
        log_prob = -torch.rand(N, T) * 3 - 0.1  # random negative log probs
        response_mask = torch.ones(N, T)
        reward_labels = torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.5)

        assert result["loss_negative"].item() == 0.0
        assert abs(result["total_loss"].item() - result["loss_positive"].item()) < 1e-6

    def test_all_correct_n1(self):
        """Single correct response."""
        log_prob = torch.tensor([[-0.5, -1.0, -0.3]])
        response_mask = torch.ones(1, 3)
        reward_labels = torch.tensor([1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        # L+ = -(1/1)*(-0.5 + -1.0 + -0.3) = 1.8
        assert abs(result["loss_positive"].item() - 1.8) < 1e-5
        assert result["loss_negative"].item() == 0.0
        assert abs(result["total_loss"].item() - 1.8) < 1e-5


class TestEdgeCaseAllIncorrect:
    """AC8: All incorrect (C=empty, I=N) → reverse SFT only, no NaN/Inf."""

    def test_all_incorrect_n8(self):
        """N=8 all incorrect, loss should be beta-weighted reverse SFT."""
        N, T = 8, 5
        log_prob = torch.full((N, T), -1.0)
        response_mask = torch.ones(N, T)
        reward_labels = -torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        assert abs(result["total_loss"].item() - (-0.5)) < 1e-5
        assert result["loss_positive"].item() == 0.0
        assert abs(result["loss_negative"].item() - (-5.0)) < 1e-5

    def test_all_incorrect_no_nan_inf(self):
        """Ensure no NaN or Inf when all incorrect."""
        N, T = 4, 10
        log_prob = torch.randn(N, T) * 100  # extreme values
        response_mask = torch.ones(N, T)
        reward_labels = -torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.5)

        for key in ["total_loss", "loss_positive", "loss_negative"]:
            assert torch.isfinite(result[key]).all(), f"{key} is not finite"


class TestNoNaNInf:
    """AC9: No NaN/Inf for any valid input (1000 random samples)."""

    def test_random_inputs_no_nan_inf(self):
        """1000 random inputs, all should produce finite outputs."""
        torch.manual_seed(12345)
        for i in range(1000):
            N = torch.randint(1, 16, (1,)).item()
            T = torch.randint(1, 32, (1,)).item()
            log_prob = -torch.rand(N, T) * 10 - 0.01  # negative log probs
            response_mask = (torch.rand(N, T) > 0.2).float()
            # Ensure at least one token per response
            response_mask[:, 0] = 1.0
            # Random rewards: +1 or -1
            reward_labels = torch.where(torch.rand(N) > 0.5, torch.ones(N), -torch.ones(N))
            beta = torch.rand(1).item()

            result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta)

            for key in ["total_loss", "loss_positive", "loss_negative"]:
                assert torch.isfinite(result[key]).all(), (
                    f"Sample {i}: {key}={result[key].item()} is not finite "
                    f"(N={N}, T={T}, k={int((reward_labels > 0).sum().item())})"
                )


class TestGradientFlow:
    """AC10: Backpropagation produces finite gradients."""

    def test_gradient_flow_mixed(self):
        """Mixed correct/incorrect, verify finite gradients on log_prob."""
        N, T = 4, 5
        log_prob = torch.randn(N, T, requires_grad=True)
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)
        result["total_loss"].backward()

        assert log_prob.grad is not None, "No gradient computed"
        assert torch.isfinite(log_prob.grad).all(), "Gradient contains NaN or Inf"

    def test_gradient_flow_all_correct(self):
        """All correct, verify finite gradients."""
        N, T = 4, 5
        log_prob = torch.randn(N, T, requires_grad=True)
        response_mask = torch.ones(N, T)
        reward_labels = torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)
        result["total_loss"].backward()

        assert log_prob.grad is not None
        assert torch.isfinite(log_prob.grad).all()

    def test_gradient_flow_all_incorrect(self):
        """All incorrect uses reverse SFT and produces finite gradients for beta > 0."""
        N, T = 4, 5
        log_prob = torch.randn(N, T, requires_grad=True)
        response_mask = torch.ones(N, T)
        reward_labels = -torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)
        result["total_loss"].backward()

        assert log_prob.grad is not None
        assert torch.isfinite(log_prob.grad).all()
        assert (log_prob.grad[response_mask.bool()] > 0).all()

    def test_gradient_flow_all_incorrect_beta_zero(self):
        """With beta=0, all-incorrect reverse SFT is intentionally disabled."""
        N, T = 4, 5
        log_prob = torch.randn(N, T, requires_grad=True)
        response_mask = torch.ones(N, T)
        reward_labels = -torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.0)
        result["total_loss"].backward()

        assert log_prob.grad is not None
        assert torch.count_nonzero(log_prob.grad).item() == 0

    def test_gradient_direction_correct(self):
        """Verify gradient pushes correct response log probs up (SFT direction)."""
        N, T = 2, 3
        log_prob = torch.tensor(
            [
                [-2.0, -2.0, -2.0],  # correct
                [-2.0, -2.0, -2.0],  # incorrect
            ],
            requires_grad=True,
        )
        response_mask = torch.ones(N, T)
        reward_labels = torch.tensor([1.0, -1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)
        result["total_loss"].backward()

        # For correct responses: L+ = -sum(log_prob), gradient is -1 (want to increase log_prob)
        # For incorrect responses: L- = sum(log_prob), gradient is +beta (want to decrease log_prob)
        grad = log_prob.grad
        # Correct response (index 0): gradient should be negative (decreasing loss = increasing log_prob)
        assert (grad[0] < 0).all(), "Gradient should push correct log probs up"
        # Incorrect response (index 1): gradient should be positive (decreasing beta*L- = decreasing log_prob)
        assert (grad[1] > 0).all(), "Gradient should push incorrect log probs down"

    def test_gradient_finite_random(self):
        """Random inputs gradient flow check (100 samples)."""
        torch.manual_seed(999)
        for _ in range(100):
            N = torch.randint(2, 8, (1,)).item()
            T = torch.randint(2, 16, (1,)).item()
            log_prob = torch.randn(N, T, requires_grad=True)
            response_mask = torch.ones(N, T)
            reward_labels = torch.where(torch.rand(N) > 0.3, torch.ones(N), -torch.ones(N))

            result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)
            result["total_loss"].backward()

            if log_prob.grad is not None:
                assert torch.isfinite(log_prob.grad).all()


class TestFunctionInterface:
    """Test that the function has the expected interface (AC1-3)."""

    def test_returns_dict(self):
        """Return type is a dict with the three expected keys."""
        log_prob = torch.tensor([[-1.0, -1.0]])
        response_mask = torch.ones(1, 2)
        reward_labels = torch.tensor([1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels)
        assert isinstance(result, dict)
        assert "total_loss" in result
        assert "loss_positive" in result
        assert "loss_negative" in result

    def test_default_beta(self):
        """Default beta is 0.1."""
        log_prob = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
        response_mask = torch.ones(2, 2)
        reward_labels = torch.tensor([1.0, -1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels)
        result_explicit = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        assert abs(result["total_loss"].item() - result_explicit["total_loss"].item()) < 1e-7

    def test_all_outputs_are_tensors(self):
        """All returned values are torch.Tensor."""
        log_prob = torch.tensor([[-1.0]])
        response_mask = torch.ones(1, 1)
        reward_labels = torch.tensor([1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels)
        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} is {type(val)}, expected Tensor"

    def test_scalar_outputs(self):
        """All returned tensors are scalar (0-dim)."""
        log_prob = torch.tensor([[-1.0, -1.0]])
        response_mask = torch.ones(1, 2)
        reward_labels = torch.tensor([1.0])

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels)
        for key, val in result.items():
            assert val.dim() == 0, f"{key} has {val.dim()} dims, expected 0"


class TestMaskBehavior:
    """Test that response_mask properly excludes padding tokens."""

    def test_zero_mask_ignored(self):
        """Tokens with mask=0 should not contribute to the loss."""
        log_prob_a = torch.tensor([[-1.0, -2.0, -99.0]])
        mask_a = torch.tensor([[1.0, 1.0, 0.0]])

        log_prob_b = torch.tensor([[-1.0, -2.0, -0.001]])
        mask_b = torch.tensor([[1.0, 1.0, 0.0]])

        reward_labels = torch.tensor([1.0])

        result_a = compute_wdl_sft_loss(log_prob_a, mask_a, reward_labels)
        result_b = compute_wdl_sft_loss(log_prob_b, mask_b, reward_labels)

        assert abs(result_a["total_loss"].item() - result_b["total_loss"].item()) < 1e-6

    def test_variable_length_responses(self):
        """Different response lengths within the same batch."""
        N, T = 3, 5
        log_prob = torch.full((N, T), -1.0)
        response_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],  # 5 tokens
                [1.0, 1.0, 1.0, 0.0, 0.0],  # 3 tokens
                [1.0, 0.0, 0.0, 0.0, 0.0],  # 1 token
            ]
        )
        reward_labels = torch.ones(N)

        result = compute_wdl_sft_loss(log_prob, response_mask, reward_labels, beta=0.1)

        # seq_log_probs: [-5.0, -3.0, -1.0]
        # L+ = -(1/3)*(-5 + -3 + -1) = -(1/3)*(-9) = 3.0
        assert abs(result["loss_positive"].item() - 3.0) < 1e-5
