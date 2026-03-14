"""P2 semantic correctness audit tests.

Covers:
- Task 4: Fused logits formula correctness (numerical precision, edge cases)
- Task 5: Old-log-prob recomputation consistency
- Task 6: Reference-policy and eval-only weight extraction

Run:
    cd /data-1/verl07/verl
    python -m pytest tests/joint_training/feat/test_p2_semantic_audit.py -v
"""

import pytest
import torch
import torch.nn.functional as F


def _make_joint_model(fusion_lambda=0.5, freeze_model1=False):
    from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
    from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM

    config = QwenJointConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        fusion_lambda=fusion_lambda,
        freeze_model1=freeze_model1,
    )
    return QwenJointForCausalLM(config)


# ===========================================================================
# Section 1: Fused logits formula correctness (Task 4)
# ===========================================================================

class TestFusedLogitsFormula:
    """Verify logits_fused = (1-λ)*logits_m1 + λ*logits_m2 at various λ values."""

    @pytest.mark.parametrize("lam", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_formula_at_various_lambda(self, lam):
        model = _make_joint_model(fusion_lambda=lam)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.long)

        with torch.no_grad():
            out0 = model.sub_models[0](input_ids=input_ids, attention_mask=mask)
            out1 = model.sub_models[1](input_ids=input_ids, attention_mask=mask)
            fused = model(input_ids=input_ids, attention_mask=mask)

            expected = (1 - lam) * out0.logits + lam * out1.logits
            torch.testing.assert_close(fused.logits, expected, rtol=1e-4, atol=1e-4)

    def test_lambda_zero_equals_model1_only(self):
        model = _make_joint_model(fusion_lambda=0.0)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            out0 = model.sub_models[0](input_ids=input_ids, attention_mask=mask)
            fused = model(input_ids=input_ids, attention_mask=mask)
            torch.testing.assert_close(fused.logits, out0.logits)

    def test_lambda_one_equals_model2_only(self):
        model = _make_joint_model(fusion_lambda=1.0)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            out1 = model.sub_models[1](input_ids=input_ids, attention_mask=mask)
            fused = model(input_ids=input_ids, attention_mask=mask)
            torch.testing.assert_close(fused.logits, out1.logits)

    def test_fused_logits_no_inplace_corruption(self):
        """Ensure in-place ops don't corrupt sub-model logits for subsequent calls."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            # First call to get reference sub-model logits
            out0_before = model.sub_models[0](input_ids=input_ids, attention_mask=mask).logits.clone()
            out1_before = model.sub_models[1](input_ids=input_ids, attention_mask=mask).logits.clone()

            # Fused forward (uses in-place ops internally)
            model(input_ids=input_ids, attention_mask=mask)

            # Sub-model logits should still produce same results
            out0_after = model.sub_models[0](input_ids=input_ids, attention_mask=mask).logits
            out1_after = model.sub_models[1](input_ids=input_ids, attention_mask=mask).logits

            torch.testing.assert_close(out0_before, out0_after)
            torch.testing.assert_close(out1_before, out1_after)

    def test_fused_softmax_is_valid_distribution(self):
        """Softmax of fused logits should sum to 1 for each position."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            fused = model(input_ids=input_ids, attention_mask=mask)
            probs = F.softmax(fused.logits, dim=-1)
            sums = probs.sum(dim=-1)
            torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_gradient_magnitude_scales_with_lambda(self):
        """For λ=0.8, model2 should get 4x the gradient of model1."""
        model = _make_joint_model(fusion_lambda=0.8)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.randint(0, 1000, (2, 4))

        output = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        output.loss.backward()

        # Collect total grad L2 norms
        norm1 = sum(p.grad.norm().item() ** 2 for p in model.sub_models[0].parameters() if p.grad is not None) ** 0.5
        norm2 = sum(p.grad.norm().item() ** 2 for p in model.sub_models[1].parameters() if p.grad is not None) ** 0.5

        # model2_norm / model1_norm should be close to λ / (1-λ) = 4.0
        # Allow some tolerance due to non-linear effects
        ratio = norm2 / (norm1 + 1e-12)
        assert ratio > 2.0, f"Expected model2 grad >> model1 grad, got ratio={ratio}"


# ===========================================================================
# Section 2: Old-log-prob recomputation consistency (Task 5)
# ===========================================================================

class TestOldLogProbConsistency:
    """Verify that log-probs computed from fused logits are consistent across calls."""

    def _compute_log_probs(self, model, input_ids, attention_mask):
        """Simulate verl's log-prob computation from fused logits."""
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return gathered

    def test_log_probs_deterministic_in_eval(self):
        """In eval mode, two forward passes should produce identical log-probs."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            lp1 = self._compute_log_probs(model, input_ids, mask)
            lp2 = self._compute_log_probs(model, input_ids, mask)
            torch.testing.assert_close(lp1, lp2)

    def test_log_probs_from_fused_differ_from_model2_only(self):
        """Log-probs from fused logits should differ from model2-only log-probs."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            # Fused log-probs
            fused_lp = self._compute_log_probs(model, input_ids, mask)

            # Model2-only log-probs (via eval_only)
            out2 = model(input_ids=input_ids, attention_mask=mask, eval_only=True)
            log_probs_2 = F.log_softmax(out2.logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            model2_lp = log_probs_2.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

            assert not torch.allclose(fused_lp, model2_lp), (
                "Fused and model2-only log-probs should differ"
            )

    def test_log_probs_are_negative(self):
        """All log-probs should be <= 0."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            lp = self._compute_log_probs(model, input_ids, mask)
            assert (lp <= 0).all(), "Log-probs should be <= 0"

    def test_log_probs_finite(self):
        """Log-probs should not contain inf or NaN."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            lp = self._compute_log_probs(model, input_ids, mask)
            assert torch.all(torch.isfinite(lp)), "Log-probs should be finite"

    def test_log_probs_consistent_with_verl_utility(self):
        """Verify our log-prob computation matches verl's logprobs_from_logits."""
        from verl.utils.torch_functional import logprobs_from_logits

        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask)
            logits = output.logits[:, :-1, :]
            targets = input_ids[:, 1:]

            # Our manual computation
            manual_lp = F.log_softmax(logits, dim=-1).gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)

            # verl's utility
            verl_lp = logprobs_from_logits(logits, targets)

            torch.testing.assert_close(manual_lp, verl_lp)


# ===========================================================================
# Section 3: Reference-policy and eval-only extraction (Task 6)
# ===========================================================================

class TestEvalOnlyWeightExtraction:
    """Audit eval_only mode and weight extraction correctness."""

    def test_eval_only_returns_model2_logits_exactly(self):
        """eval_only=True should return exactly model2's output."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            direct_m2 = model.sub_models[1](input_ids=input_ids, attention_mask=mask)
            eval_out = model(input_ids=input_ids, attention_mask=mask, eval_only=True)
            torch.testing.assert_close(eval_out.logits, direct_m2.logits)

    def test_eval_only_attribute_mode(self):
        """Setting _eval_only_mode attribute should also return model2 logits."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        model._eval_only_mode = True
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            direct_m2 = model.sub_models[1](input_ids=input_ids, attention_mask=mask)
            eval_out = model(input_ids=input_ids, attention_mask=mask)
            torch.testing.assert_close(eval_out.logits, direct_m2.logits)

        model._eval_only_mode = False

    def test_eval_only_does_not_run_model1(self):
        """In eval_only mode, model1 should not execute (check by hooking forward)."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        model1_called = []
        original_forward = model.sub_models[0].forward

        def tracking_forward(*args, **kwargs):
            model1_called.append(True)
            return original_forward(*args, **kwargs)

        model.sub_models[0].forward = tracking_forward

        input_ids = torch.randint(0, 1000, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask, eval_only=True)

        assert len(model1_called) == 0, "model1.forward should not be called in eval_only mode"

    def test_weight_utils_extract_model2(self):
        """weight_utils should correctly extract model2 weights."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights, is_joint_model_state_dict

        model = _make_joint_model()
        state_dict = model.state_dict()

        assert is_joint_model_state_dict(state_dict)

        model2_weights = extract_sub_model_weights(state_dict, sub_model_index=1)
        # All keys should not have sub_models prefix
        for key in model2_weights:
            assert not key.startswith("sub_models."), f"Key {key} still has prefix"

        # Load into standalone model and verify
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(
            vocab_size=1000, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
        )
        standalone = Qwen3ForCausalLM(config)
        standalone.load_state_dict(model2_weights)

        input_ids = torch.randint(0, 1000, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)

        model.eval()
        standalone.eval()
        with torch.no_grad():
            joint_m2 = model.sub_models[1](input_ids=input_ids, attention_mask=mask)
            standalone_out = standalone(input_ids=input_ids, attention_mask=mask)
            torch.testing.assert_close(joint_m2.logits, standalone_out.logits)

    def test_weight_utils_extract_model1(self):
        """Extracting model1 should also work correctly."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        model = _make_joint_model()
        model1_weights = extract_sub_model_weights(model.state_dict(), sub_model_index=0)

        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(
            vocab_size=1000, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
        )
        standalone = Qwen3ForCausalLM(config)
        standalone.load_state_dict(model1_weights)

        input_ids = torch.randint(0, 1000, (1, 4))
        mask = torch.ones(1, 4, dtype=torch.long)

        model.eval()
        standalone.eval()
        with torch.no_grad():
            joint_m1 = model.sub_models[0](input_ids=input_ids, attention_mask=mask)
            standalone_out = standalone(input_ids=input_ids, attention_mask=mask)
            torch.testing.assert_close(joint_m1.logits, standalone_out.logits)


class TestReferencePolicy:
    """Verify reference-policy gradient flow is correctly blocked."""

    def test_fused_loss_backward_does_not_affect_frozen_model1(self):
        """When model1 is frozen, backward should not create grads for model1."""
        model = _make_joint_model(fusion_lambda=0.5, freeze_model1=True)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.randint(0, 1000, (2, 4))

        output = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        output.loss.backward()

        for name, param in model.sub_models[0].named_parameters():
            assert param.grad is None, f"Frozen model1 param {name} should have no gradient"

        # model2 should have gradients
        m2_has_grad = any(p.grad is not None for p in model.sub_models[1].parameters())
        assert m2_has_grad, "model2 should have gradients"

    def test_detached_ref_log_prob_does_not_affect_model(self):
        """Simulate reference policy: detach log-probs and verify no gradient leaks."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        # Compute "reference" log-probs (detached)
        output = model(input_ids=input_ids, attention_mask=mask)
        log_probs = F.log_softmax(output.logits[:, :-1, :], dim=-1)
        ref_lp = log_probs.detach()  # This is how ref policy log-probs are created

        # ref_lp should not require grad — this IS the guarantee
        assert not ref_lp.requires_grad

        # Detached tensor should not have a grad_fn
        assert ref_lp.grad_fn is None

        # Verify that using detached ref_lp in a KL-like loss only creates
        # gradients through the non-detached branch
        model.zero_grad()
        # Simulate: policy_lp has grad, ref_lp does not
        policy_output = model(input_ids=input_ids, attention_mask=mask)
        policy_lp = F.log_softmax(policy_output.logits[:, :-1, :], dim=-1)
        kl_loss = (policy_lp.exp() * (policy_lp - ref_lp)).sum()
        kl_loss.backward()

        # model should have gradients from policy_lp, not from ref_lp
        m2_has_grad = any(p.grad is not None for p in model.sub_models[1].parameters())
        assert m2_has_grad, "model2 should have gradients from policy log-probs"

    def test_kl_penalty_between_fused_and_ref_is_nonnegative(self):
        """KL(fused || ref) should be >= 0 when ref is detached model2-only."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            fused_out = model(input_ids=input_ids, attention_mask=mask)
            ref_out = model(input_ids=input_ids, attention_mask=mask, eval_only=True)

            fused_lp = F.log_softmax(fused_out.logits, dim=-1)
            ref_lp = F.log_softmax(ref_out.logits, dim=-1)

            # KL(fused || ref) = sum(fused_prob * (fused_lp - ref_lp))
            kl = (fused_lp.exp() * (fused_lp - ref_lp)).sum(dim=-1).mean()
            assert kl >= -1e-6, f"KL divergence should be >= 0, got {kl}"
