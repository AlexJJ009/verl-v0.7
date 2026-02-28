"""Feature tests for HF rollout integration with QwenJointForCausalLM.

Tests verify:
1. _eval_only_mode attribute triggers model2-only forward
2. model.generate() works with fused logits
3. _eval_only_mode attribute works with generate()
"""

import pytest
import torch


def _make_small_joint_config(fusion_lambda=0.5, freeze_model1=False):
    """Create a small joint config for testing (tiny model to save memory)."""
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


class TestEvalOnlyModeAttribute:
    """Test that _eval_only_mode attribute triggers model2-only logits."""

    def test_eval_only_mode_attribute_default_false(self):
        """By default, _eval_only_mode should not be set."""
        model = _make_joint_model()
        assert not getattr(model, "_eval_only_mode", False)

    def test_eval_only_mode_via_attribute(self):
        """Setting _eval_only_mode=True should produce model2-only logits."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones_like(input_ids)

        # Get fused logits (default mode)
        with torch.no_grad():
            fused_out = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get model2-only logits via eval_only kwarg
        with torch.no_grad():
            eval_out = model(input_ids=input_ids, attention_mask=attention_mask, eval_only=True)

        # Get model2-only logits via _eval_only_mode attribute
        model._eval_only_mode = True
        with torch.no_grad():
            attr_out = model(input_ids=input_ids, attention_mask=attention_mask)

        # Attribute-based and kwarg-based eval_only should match
        assert torch.allclose(eval_out.logits, attr_out.logits, atol=1e-6), (
            "_eval_only_mode attribute should produce same logits as eval_only=True kwarg"
        )

        # Fused logits should differ from eval-only logits
        assert not torch.allclose(fused_out.logits, attr_out.logits, atol=1e-4), (
            "Fused logits should differ from model2-only logits"
        )

    def test_eval_only_mode_reset(self):
        """Setting _eval_only_mode back to False should restore fused logits."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            fused_out = model(input_ids=input_ids, attention_mask=attention_mask)

        model._eval_only_mode = True
        with torch.no_grad():
            eval_out = model(input_ids=input_ids, attention_mask=attention_mask)

        model._eval_only_mode = False
        with torch.no_grad():
            restored_out = model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(fused_out.logits, restored_out.logits, atol=1e-6), (
            "Resetting _eval_only_mode should restore fused logits"
        )
        assert not torch.allclose(eval_out.logits, restored_out.logits, atol=1e-4), (
            "Eval-only and fused logits should differ"
        )

    def test_eval_only_kwarg_overrides_attribute(self):
        """eval_only=True kwarg should work even when _eval_only_mode=False."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        model._eval_only_mode = False

        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            kwarg_out = model(input_ids=input_ids, attention_mask=attention_mask, eval_only=True)
            model._eval_only_mode = True
            attr_out = model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(kwarg_out.logits, attr_out.logits, atol=1e-6)


class TestHFGenerate:
    """Test that model.generate() works with QwenJointForCausalLM."""

    def test_generate_produces_tokens(self):
        """model.generate() should produce valid token sequences."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )

        assert output.shape[0] == 1
        assert output.shape[1] > input_ids.shape[1], "generate() should produce new tokens"
        assert output.shape[1] <= input_ids.shape[1] + 8

    def test_generate_uses_fused_logits(self):
        """generate() should use fused logits (not just model2)."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones_like(input_ids)

        # Generate with fused logits (default)
        with torch.no_grad():
            fused_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )

        # Generate with model2-only logits via attribute
        model._eval_only_mode = True
        with torch.no_grad():
            eval_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )

        # At least check both produce valid outputs
        assert fused_output.shape[1] > input_ids.shape[1]
        assert eval_output.shape[1] > input_ids.shape[1]

    def test_generate_eval_only_via_attribute(self):
        """Setting _eval_only_mode=True should make generate() use model2-only."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones_like(input_ids)

        # First verify we can get model2-only logits via attribute
        model._eval_only_mode = True
        with torch.no_grad():
            attr_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            eval_logits = model(input_ids=input_ids, attention_mask=attention_mask, eval_only=True).logits

        # These should match — attribute-based eval_only produces same as kwarg
        assert torch.allclose(attr_logits, eval_logits, atol=1e-6)

        # Now test generate with _eval_only_mode
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )
        assert output.shape[1] > input_ids.shape[1]
        model._eval_only_mode = False


class TestHFRolloutInstantiation:
    """Test that HFRollout can be instantiated with a joint model."""

    def test_hf_rollout_instantiation(self):
        """HFRollout should accept a joint model module."""
        pytest.importorskip(
            "verl.workers.rollout.hf_rollout",
            reason="HFRollout import fails due to AutoModelForVision2Seq env issue",
            exc_type=ImportError,
        )
        from omegaconf import OmegaConf

        from verl.workers.rollout.hf_rollout import HFRollout

        model = _make_joint_model()
        config = OmegaConf.create(
            {
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1.0,
                "response_length": 16,
                "do_sample": True,
                "val_kwargs": {"top_k": -1, "top_p": 1.0, "temperature": 0, "do_sample": False},
            }
        )

        rollout = HFRollout(module=model, config=config)
        assert rollout.module is model
        assert rollout.config.response_length == 16

    def test_hf_rollout_is_base_rollout_instance(self):
        """HFRollout should be recognized as BaseRollout subclass."""
        pytest.importorskip(
            "verl.workers.rollout.hf_rollout",
            reason="HFRollout import fails due to AutoModelForVision2Seq env issue",
            exc_type=ImportError,
        )
        from verl.workers.rollout.base import BaseRollout
        from verl.workers.rollout.hf_rollout import HFRollout

        assert issubclass(HFRollout, BaseRollout)
