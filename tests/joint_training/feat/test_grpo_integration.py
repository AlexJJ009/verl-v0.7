# SPDX-License-Identifier: Apache-2.0

"""Integration tests simulating the actual GRPO training flow with joint model.

Tests the complete data flow:
1. Joint model forward → fused logits
2. logprobs_from_logits with fused logits
3. GRPO advantage computation
4. Policy loss computation
5. Backward pass through fusion
"""

import torch
import torch.nn.functional as F


def _make_joint_model():
    """Create a small joint model for integration testing."""
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
        fusion_lambda=0.5,
    )
    return QwenJointForCausalLM(config)


class TestGRPOTrainingFlow:
    """Simulate the complete GRPO training step with joint model."""

    def test_compute_old_log_probs_with_fused_logits(self):
        """Step 3: compute old log_probs using fused logits (no grad)."""
        model = _make_joint_model()
        model.eval()

        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

            # logprobs_from_logits: log_softmax → gather
            log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
            response_ids = input_ids[:, 1:]
            old_log_probs = log_probs_all.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

        assert old_log_probs.shape == (batch_size, seq_len - 1)
        assert (old_log_probs <= 0).all()

    def test_compute_new_log_probs_with_gradient(self):
        """Step 5: compute new log_probs with gradients for policy update."""
        model = _make_joint_model()
        model.train()

        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits

        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
        response_ids = input_ids[:, 1:]
        new_log_probs = log_probs_all.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

        # Verify gradient can flow
        loss = -new_log_probs.mean()
        loss.backward()

        # Both sub-models should have gradients
        for idx in [0, 1]:
            has_grad = any(p.grad is not None for p in model.sub_models[idx].parameters())
            assert has_grad, f"sub_models[{idx}] should have gradients"

    def test_full_grpo_step_simulation(self):
        """Simulate complete GRPO step: old_log_prob → advantage → loss → backward."""
        model = _make_joint_model()

        batch_size, seq_len = 8, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        response_mask = torch.ones(batch_size, seq_len - 1)

        # Step 3: compute old_log_probs (no grad)
        model.eval()
        with torch.no_grad():
            old_output = model(input_ids=input_ids, attention_mask=attention_mask)
            old_log_probs = (
                F.log_softmax(old_output.logits[:, :-1, :], dim=-1)
                .gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )

        # Step 4: compute advantages (simulate GRPO group-based)
        # Assume 2 prompts × 4 responses each
        scores = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0])
        group_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        # Normalize within groups
        advantages = torch.zeros_like(scores)
        for group_id in group_indices.unique():
            mask = group_indices == group_id
            group_scores = scores[mask]
            mean = group_scores.mean()
            std = group_scores.std()
            advantages[mask] = (group_scores - mean) / (std + 1e-8)

        # Expand advantages to token level
        token_advantages = advantages.unsqueeze(1).expand(-1, seq_len - 1)

        # Step 5: update policy
        model.train()
        model.zero_grad()

        new_output = model(input_ids=input_ids, attention_mask=attention_mask)
        new_log_probs = (
            F.log_softmax(new_output.logits[:, :-1, :], dim=-1)
            .gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )

        # PPO-clip loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        cliprange = 0.2
        pg_losses1 = -token_advantages * ratio
        pg_losses2 = -token_advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        pg_loss = torch.max(pg_losses1, pg_losses2)

        # Masked mean
        loss = (pg_loss * response_mask).sum() / response_mask.sum()
        loss.backward()

        # Verify loss is a finite scalar
        assert loss.isfinite()
        assert loss.ndim == 0

        # Verify gradients exist
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item()
        assert total_grad_norm > 0, "Model should have non-zero gradient norm"

    def test_eval_only_mode_no_fusion(self):
        """Eval mode should bypass fusion and use model2 only."""
        model = _make_joint_model()
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            fused_out = model(input_ids=input_ids, attention_mask=attention_mask)
            eval_out = model(input_ids=input_ids, attention_mask=attention_mask, eval_only=True)
            model2_out = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)

        # eval_only should match model2 directly
        torch.testing.assert_close(eval_out.logits, model2_out.logits)
        # eval_only should differ from fused
        assert not torch.allclose(eval_out.logits, fused_out.logits)

    def test_frozen_model1_grpo_step(self):
        """GRPO step with frozen model1: only model2 should update."""
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
            fusion_lambda=0.5,
            freeze_model1=True,
        )
        model = QwenJointForCausalLM(config)
        model.train()

        # Save initial model2 weights
        model2_initial = {k: v.clone() for k, v in model.sub_models[1].named_parameters() if v.requires_grad}

        input_ids = torch.randint(0, 1000, (4, 8))
        attention_mask = torch.ones(4, 8, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = (
            F.log_softmax(output.logits[:, :-1, :], dim=-1)
            .gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )

        loss = -log_probs.mean()
        loss.backward()

        # Apply a simple gradient step
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.01 * p.grad

        # model1 should be unchanged (frozen)
        for name, param in model.sub_models[0].named_parameters():
            assert param.grad is None, f"Frozen model1 param {name} has gradient"

        # model2 should have changed
        changed = False
        for name, param in model.sub_models[1].named_parameters():
            if name in model2_initial and not torch.equal(param, model2_initial[name]):
                changed = True
                break
        assert changed, "model2 weights should change after gradient step"


class TestEntropyCombutation:
    """Test entropy computation with fused logits."""

    def test_entropy_from_fused_logits(self):
        """Entropy should be computable from fused logits."""
        model = _make_joint_model()
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

            # Compute entropy
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)

        assert entropy.shape == (2, 8)
        assert (entropy >= 0).all(), "Entropy should be non-negative"
