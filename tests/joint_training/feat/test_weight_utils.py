# SPDX-License-Identifier: Apache-2.0

"""Feature tests for joint training weight extraction utilities.

Tests the pure logic of extracting model2 weights from a joint model's state dict,
which is the core of the dual-mode weight sync in FSDP workers.
"""

from collections import OrderedDict

import torch


def _make_mock_joint_state_dict():
    """Create a mock state dict resembling a QwenJointForCausalLM."""
    state_dict = OrderedDict()
    # model1 weights
    state_dict["sub_models.0.model.embed_tokens.weight"] = torch.randn(1000, 64)
    state_dict["sub_models.0.model.layers.0.self_attn.q_proj.weight"] = torch.randn(64, 64)
    state_dict["sub_models.0.model.layers.0.self_attn.k_proj.weight"] = torch.randn(32, 64)
    state_dict["sub_models.0.model.layers.0.mlp.gate_proj.weight"] = torch.randn(128, 64)
    state_dict["sub_models.0.model.norm.weight"] = torch.randn(64)
    state_dict["sub_models.0.lm_head.weight"] = torch.randn(1000, 64)
    # model2 weights (different values)
    state_dict["sub_models.1.model.embed_tokens.weight"] = torch.randn(1000, 64)
    state_dict["sub_models.1.model.layers.0.self_attn.q_proj.weight"] = torch.randn(64, 64)
    state_dict["sub_models.1.model.layers.0.self_attn.k_proj.weight"] = torch.randn(32, 64)
    state_dict["sub_models.1.model.layers.0.mlp.gate_proj.weight"] = torch.randn(128, 64)
    state_dict["sub_models.1.model.norm.weight"] = torch.randn(64)
    state_dict["sub_models.1.lm_head.weight"] = torch.randn(1000, 64)
    return state_dict


class TestJointWeightUtils:
    """Test weight extraction utilities for joint training."""

    def test_is_joint_model_state_dict(self):
        """Detect if a state dict belongs to a joint model."""
        from verl.models.joint_model.weight_utils import is_joint_model_state_dict

        joint_sd = _make_mock_joint_state_dict()
        assert is_joint_model_state_dict(joint_sd) is True

        normal_sd = OrderedDict(
            {
                "model.embed_tokens.weight": torch.randn(1000, 64),
                "lm_head.weight": torch.randn(1000, 64),
            }
        )
        assert is_joint_model_state_dict(normal_sd) is False

    def test_extract_model2_weights(self):
        """Extract model2 weights with clean key names."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        joint_sd = _make_mock_joint_state_dict()
        model2_sd = extract_sub_model_weights(joint_sd, sub_model_index=1)

        # Keys should not have sub_models prefix
        for key in model2_sd.keys():
            assert not key.startswith("sub_models."), f"Key still has prefix: {key}"

        # Should have the same number of keys as one sub-model
        assert len(model2_sd) == len(joint_sd) // 2

        # Verify specific key names
        assert "model.embed_tokens.weight" in model2_sd
        assert "lm_head.weight" in model2_sd

    def test_extract_model1_weights(self):
        """Extract model1 weights with clean key names."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        joint_sd = _make_mock_joint_state_dict()
        model1_sd = extract_sub_model_weights(joint_sd, sub_model_index=0)

        for key in model1_sd.keys():
            assert not key.startswith("sub_models."), f"Key still has prefix: {key}"
        assert len(model1_sd) == len(joint_sd) // 2

    def test_extracted_weights_match_original(self):
        """Extracted weights should be the same tensors as original."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        joint_sd = _make_mock_joint_state_dict()
        model2_sd = extract_sub_model_weights(joint_sd, sub_model_index=1)

        torch.testing.assert_close(
            model2_sd["model.embed_tokens.weight"],
            joint_sd["sub_models.1.model.embed_tokens.weight"],
        )

    def test_extract_preserves_weight_values_not_references(self):
        """Extracted dict should contain the same tensor data."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        joint_sd = _make_mock_joint_state_dict()
        model2_sd = extract_sub_model_weights(joint_sd, sub_model_index=1)

        original = joint_sd["sub_models.1.lm_head.weight"]
        extracted = model2_sd["lm_head.weight"]
        assert torch.equal(original, extracted)

    def test_is_joint_model_config(self):
        """Detect if a model config is for a joint model."""
        from transformers import Qwen3Config

        from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
        from verl.models.joint_model.weight_utils import is_joint_model_config

        joint_config = QwenJointConfig(vocab_size=1000, hidden_size=64)
        assert is_joint_model_config(joint_config) is True

        normal_config = Qwen3Config(vocab_size=1000, hidden_size=64)
        assert is_joint_model_config(normal_config) is False

    def test_extract_as_generator(self):
        """Extract model2 weights as a generator (for streaming to vLLM)."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights_iter

        joint_sd = _make_mock_joint_state_dict()
        pairs = list(extract_sub_model_weights_iter(joint_sd, sub_model_index=1))

        assert len(pairs) == len(joint_sd) // 2
        for name, param in pairs:
            assert not name.startswith("sub_models.")
            assert isinstance(param, torch.Tensor)


class TestJointModelIntegration:
    """Integration tests: actual model → weight extraction → standalone loading."""

    def test_full_extraction_pipeline(self):
        """Full pipeline: create joint model, extract model2, load standalone."""
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
        from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        # Create joint model
        joint_config = QwenJointConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        joint_model = QwenJointForCausalLM(joint_config)

        # Extract model2 weights
        model2_sd = extract_sub_model_weights(joint_model.state_dict(), sub_model_index=1)

        # Load into standalone model
        standalone_config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        standalone = Qwen3ForCausalLM(standalone_config)
        standalone.load_state_dict(model2_sd)

        # Verify same output
        joint_model.eval()
        standalone.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            joint_out = joint_model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)
            standalone_out = standalone(input_ids=input_ids, attention_mask=attention_mask)

        torch.testing.assert_close(joint_out.logits, standalone_out.logits)
