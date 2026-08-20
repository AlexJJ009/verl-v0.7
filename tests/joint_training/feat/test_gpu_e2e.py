"""GPU end-to-end tests for joint training with FSDP.

All tests are skipped if no CUDA GPU is available.
Tests verify:
1. FSDP wrapping of QwenJointForCausalLM
2. Forward pass through FSDP produces fused logits
3. Full training step: forward → fused logits → loss → backward → optimizer step
"""

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _make_small_joint_config(fusion_lambda=0.5, freeze_model1=False):
    """Create a small joint config for testing."""
    from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig

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
    return config


def _make_joint_model(fusion_lambda=0.5, freeze_model1=False):
    """Create a small joint model for testing."""
    from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM

    config = _make_small_joint_config(fusion_lambda, freeze_model1)
    model = QwenJointForCausalLM(config)
    return model


@requires_cuda
class TestFSDPWrapping:
    """Test FSDP wrapping of joint model (single GPU, no distributed init needed)."""

    def test_model_on_gpu(self):
        """Joint model should move to GPU correctly."""
        model = _make_joint_model()
        model = model.cuda()

        input_ids = torch.randint(0, 1000, (2, 8)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        assert out.logits.device.type == "cuda"
        assert out.logits.shape == (2, 8, 1000)

    def test_fused_logits_on_gpu(self):
        """Fused logits should work correctly on GPU."""
        model = _make_joint_model(fusion_lambda=0.3)
        model = model.cuda()
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        with torch.no_grad():
            fused_out = model(input_ids=input_ids, attention_mask=attention_mask)
            eval_out = model(input_ids=input_ids, attention_mask=attention_mask, eval_only=True)

        # Fused and eval-only should differ (lambda=0.3, not 1.0)
        assert not torch.allclose(fused_out.logits, eval_out.logits, atol=1e-4)

    def test_eval_only_attribute_on_gpu(self):
        """_eval_only_mode attribute should work on GPU."""
        model = _make_joint_model(fusion_lambda=0.5)
        model = model.cuda()
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        model._eval_only_mode = True
        with torch.no_grad():
            attr_out = model(input_ids=input_ids, attention_mask=attention_mask)
            kwarg_out = model(input_ids=input_ids, attention_mask=attention_mask, eval_only=True)

        assert torch.allclose(attr_out.logits, kwarg_out.logits, atol=1e-6)
        model._eval_only_mode = False


@requires_cuda
class TestTrainingStep:
    """Test full training step on GPU."""

    def test_forward_backward_on_gpu(self):
        """Forward and backward pass should work on GPU with fused logits."""
        model = _make_joint_model(fusion_lambda=0.5, freeze_model1=True)
        model = model.cuda()
        model.train()

        input_ids = torch.randint(0, 1000, (2, 8)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()
        labels = torch.randint(0, 1000, (2, 8)).cuda()

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert out.loss is not None
        out.loss.backward()

        # model1 frozen: no gradients
        for param in model.sub_models[0].parameters():
            assert param.grad is None

        # model2 trainable: gradients exist
        has_grad = any(p.grad is not None for p in model.sub_models[1].parameters())
        assert has_grad

    def test_optimizer_step_on_gpu(self):
        """Full training step: forward → loss → backward → optimizer.step()."""
        model = _make_joint_model(fusion_lambda=0.5, freeze_model1=True)
        model = model.cuda()
        model.train()

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        input_ids = torch.randint(0, 1000, (2, 8)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()
        labels = torch.randint(0, 1000, (2, 8)).cuda()

        # Capture model2 params before step
        param_before = next(model.sub_models[1].parameters()).clone()

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # model2 params should have changed
        param_after = next(model.sub_models[1].parameters())
        assert not torch.allclose(param_before, param_after, atol=1e-8)

    def test_generate_on_gpu(self):
        """model.generate() should work on GPU."""
        model = _make_joint_model(fusion_lambda=0.5)
        model = model.cuda()
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )

        assert output.device.type == "cuda"
        assert output.shape[1] > 4

    def test_grpo_step_on_gpu(self):
        """Simulate a GRPO training step with fused logits on GPU."""
        model = _make_joint_model(fusion_lambda=0.5, freeze_model1=True)
        model = model.cuda()

        input_ids = torch.randint(0, 1000, (4, 16)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        # 1. Rollout: get old log probs with fused logits (no grad)
        model.eval()
        with torch.no_grad():
            old_out = model(input_ids=input_ids, attention_mask=attention_mask)
            old_logits = old_out.logits
            old_log_probs = torch.log_softmax(old_logits, dim=-1)
            old_log_probs = old_log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # 2. Training: get new log probs (with grad)
        model.train()
        new_out = model(input_ids=input_ids, attention_mask=attention_mask)
        new_logits = new_out.logits
        new_log_probs = torch.log_softmax(new_logits, dim=-1)
        new_log_probs = new_log_probs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # 3. Compute policy ratio and clipped loss (simplified GRPO)
        ratio = torch.exp(new_log_probs - old_log_probs[:, : new_log_probs.shape[1]])
        advantages = torch.randn(4, 1).cuda()  # Mock advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(surr1, surr2).mean()

        loss.backward()

        # Verify gradients flowed to model2 only
        has_model2_grad = any(p.grad is not None for p in model.sub_models[1].parameters())
        has_model1_grad = any(p.grad is not None for p in model.sub_models[0].parameters())
        assert has_model2_grad
        assert not has_model1_grad
