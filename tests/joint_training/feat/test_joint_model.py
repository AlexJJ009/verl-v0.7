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

"""Feature tests for QwenJointForCausalLM model class.

Tests are written BEFORE implementation (TDD).
They verify:
1. Model configuration and instantiation
2. Forward pass with logit fusion
3. Gradient flow to both sub-models
4. Parameter freezing for model1
5. State dict key structure
6. AutoModel loading with trust_remote_code
"""

import copy
import io

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


def _dynperm_context(input_ids, *, rho, global_step=1, actor_update_index=0):
    batch_size, seqlen = input_ids.shape
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = 0
    active = torch.ones_like(input_ids, dtype=torch.bool)
    active[:, -1] = False
    return {
        "target_ids": targets,
        "sample_ids": torch.arange(batch_size, dtype=torch.int64)[:, None].expand(-1, seqlen),
        "token_positions": torch.arange(seqlen, dtype=torch.int64)[None, :].expand(batch_size, -1),
        "active_mask": active,
        "rho": rho,
        "base_seed": 42,
        "global_step": global_step,
        "actor_update_index": actor_update_index,
        "row_chunk_size": 2,
        "audit_rows": 2,
        "entropy_atol": 2e-6,
        "multiset_atol": 0.0,
    }


class TestQwenJointConfig:
    """Test the QwenJointConfig class."""

    def test_config_creation(self):
        """Config should be created with correct default values."""
        from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig

        config = QwenJointConfig()
        assert config.fusion_lambda == 0.5
        assert config.freeze_model1 is False
        assert config.num_sub_models == 2
        assert config.model_type == "qwen_joint"

    def test_config_custom_fusion_lambda(self):
        """Config should accept custom fusion_lambda."""
        from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig

        config = QwenJointConfig(fusion_lambda=0.3)
        assert config.fusion_lambda == 0.3

    def test_matched_scale_control_config(self):
        from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig

        config = QwenJointConfig(fusion_lambda=0.8, fusion_mode="strong_scaled")
        assert config.fusion_mode == "strong_scaled"

    def test_invalid_fusion_mode_fails_closed(self):
        from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig

        with pytest.raises(ValueError, match="fusion_mode"):
            QwenJointConfig(fusion_mode="unknown")

    def test_config_inherits_qwen3(self):
        """Config should inherit Qwen3 config attributes."""
        config = _make_small_joint_config()
        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2


class TestQwenJointModelInstantiation:
    """Test model instantiation."""

    def test_model_creation(self):
        """Model should be created with two sub-models."""
        model = _make_joint_model()
        assert hasattr(model, "sub_models")
        assert len(model.sub_models) == 2

    def test_sub_models_are_qwen3(self):
        """Each sub-model should be a Qwen3ForCausalLM instance."""
        from transformers import Qwen3ForCausalLM

        model = _make_joint_model()
        for sub_model in model.sub_models:
            assert isinstance(sub_model, Qwen3ForCausalLM)

    def test_state_dict_keys_have_sub_model_prefix(self):
        """State dict keys must have sub_models.0.* and sub_models.1.* prefixes."""
        model = _make_joint_model()
        state_dict = model.state_dict()
        has_sub0 = any(k.startswith("sub_models.0.") for k in state_dict.keys())
        has_sub1 = any(k.startswith("sub_models.1.") for k in state_dict.keys())
        assert has_sub0, "State dict should have sub_models.0.* keys"
        assert has_sub1, "State dict should have sub_models.1.* keys"

    def test_no_shared_parameters(self):
        """Model1 and model2 should NOT share parameters."""
        model = _make_joint_model()
        for p0, p1 in zip(model.sub_models[0].parameters(), model.sub_models[1].parameters(), strict=True):
            assert p0.data_ptr() != p1.data_ptr(), "Sub-models should not share parameters"


class TestQwenJointForwardPass:
    """Test forward pass with logit fusion."""

    @pytest.fixture
    def model_and_inputs(self):
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return model, input_ids, attention_mask

    def test_forward_returns_causal_lm_output(self, model_and_inputs):
        """Forward should return CausalLMOutputWithPast."""
        from transformers.modeling_outputs import CausalLMOutputWithPast

        model, input_ids, attention_mask = model_and_inputs
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(output, CausalLMOutputWithPast)

    def test_forward_logits_shape(self, model_and_inputs):
        """Fused logits should have shape (batch, seq, vocab)."""
        model, input_ids, attention_mask = model_and_inputs
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert output.logits.shape == (2, 8, 1000)

    def test_forward_logit_fusion_is_weighted_average(self, model_and_inputs):
        """Fused logits should be (1-λ)*logits_0 + λ*logits_1."""
        model, input_ids, attention_mask = model_and_inputs
        with torch.no_grad():
            # Get individual sub-model logits
            out0 = model.sub_models[0](input_ids=input_ids, attention_mask=attention_mask)
            out1 = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)

            # Get fused logits
            fused_output = model(input_ids=input_ids, attention_mask=attention_mask)

            expected = 0.5 * out0.logits + 0.5 * out1.logits
            torch.testing.assert_close(fused_output.logits, expected, rtol=1e-4, atol=1e-4)

    def test_forward_with_different_lambda(self):
        """Lambda=0.3 should give (0.7)*logits_0 + (0.3)*logits_1."""
        model = _make_joint_model(fusion_lambda=0.3)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            out0 = model.sub_models[0](input_ids=input_ids, attention_mask=attention_mask)
            out1 = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)
            fused = model(input_ids=input_ids, attention_mask=attention_mask)

            expected = 0.7 * out0.logits + 0.3 * out1.logits
            torch.testing.assert_close(fused.logits, expected, rtol=1e-4, atol=1e-4)

    def test_strong_scaled_control_matches_lambda_times_model2(self):
        from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM

        config = _make_small_joint_config(fusion_lambda=0.8)
        config.fusion_mode = "strong_scaled"
        model = QwenJointForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            model2 = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)
            control = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.testing.assert_close(control.logits, 0.8 * model2.logits, rtol=1e-4, atol=1e-4)

    def test_forward_with_labels_returns_loss(self, model_and_inputs):
        """When labels are provided, forward should return a loss."""
        model, input_ids, attention_mask = model_and_inputs
        labels = torch.randint(0, 1000, input_ids.shape)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert output.loss is not None
        assert output.loss.ndim == 0  # scalar


class TestGradientFlow:
    """Test that gradients flow correctly through logit fusion."""

    def test_gradients_flow_to_both_models(self):
        """Both sub-models should receive gradients."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.train()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.randint(0, 1000, (2, 4))

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()

        # Check model1 has gradients
        for name, param in model.sub_models[0].named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"model1 param {name} has no gradient"
                break

        # Check model2 has gradients
        for name, param in model.sub_models[1].named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"model2 param {name} has no gradient"
                break

    def test_gradient_weights_correspond_to_lambda(self):
        """Gradient magnitudes should scale with (1-λ) and λ respectively."""
        # Use lambda=0.8 (model1 gets 0.2, model2 gets 0.8)
        model = _make_joint_model(fusion_lambda=0.8)
        model.train()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.randint(0, 1000, (2, 4))

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()

        # Collect gradient norms for each sub-model
        grad_norm_0 = sum(p.grad.norm().item() for p in model.sub_models[0].parameters() if p.grad is not None)
        grad_norm_1 = sum(p.grad.norm().item() for p in model.sub_models[1].parameters() if p.grad is not None)

        # model2 (lambda=0.8) should have larger gradient norm than model1 (1-lambda=0.2)
        assert grad_norm_1 > grad_norm_0, (
            f"model2 grad norm ({grad_norm_1}) should be > model1 grad norm ({grad_norm_0})"
        )

    def test_strong_scaled_control_has_zero_model1_gradient(self):
        from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM

        config = _make_small_joint_config(fusion_lambda=0.8)
        config.fusion_mode = "strong_scaled"
        model = QwenJointForCausalLM(config)
        input_ids = torch.randint(0, 1000, (2, 4))
        labels = torch.randint(0, 1000, (2, 4))
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
        output.loss.backward()

        assert all(parameter.grad is None for parameter in model.sub_models[0].parameters())
        assert any(parameter.grad is not None for parameter in model.sub_models[1].parameters())


class TestDynamicPermutationIntegration:
    def test_training_forward_only_permutates_weak_logits(self):
        model = _make_joint_model(fusion_lambda=0.8)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 6))
        attention_mask = torch.ones_like(input_ids)
        context = _dynperm_context(input_ids, rho=1.0)

        with torch.no_grad():
            plain = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_submodel_logits=True,
            )
            permuted = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_submodel_logits=True,
                apply_weak_logit_permutation=True,
                weak_logit_permutation_context=context,
            )

        targets = context["target_ids"]
        assert torch.equal(
            plain.submodel_logits[0].gather(-1, targets.unsqueeze(-1)),
            permuted.submodel_logits[0].gather(-1, targets.unsqueeze(-1)),
        )
        assert torch.equal(
            torch.sort(plain.submodel_logits[0], dim=-1).values,
            torch.sort(permuted.submodel_logits[0], dim=-1).values,
        )
        assert torch.equal(plain.submodel_logits[1], permuted.submodel_logits[1])
        assert not torch.equal(plain.logits[:, :-1], permuted.logits[:, :-1])
        assert permuted.dynamic_permutation_telemetry["dynperm/target_mismatches"] == 0

        with torch.no_grad():
            eval_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eval_only=True,
                apply_weak_logit_permutation=True,
                weak_logit_permutation_context=context,
            )
            direct_model2 = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)
        assert torch.equal(eval_output.logits, direct_model2.logits)

    @pytest.mark.parametrize("freeze_model1", [False, True])
    def test_rho_zero_matches_disabled_forward_backward_and_rng(self, freeze_model1):
        disabled = _make_joint_model(fusion_lambda=0.8, freeze_model1=freeze_model1)
        rho_zero = copy.deepcopy(disabled)
        disabled.train()
        rho_zero.train()
        input_ids = torch.randint(0, 1000, (2, 6))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 1000, (2, 6))
        context = _dynperm_context(input_ids, rho=0.0)

        rng_start = torch.random.get_rng_state().clone()
        plain_output = disabled(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        plain_output.loss.backward()
        rng_after_disabled = torch.random.get_rng_state().clone()

        torch.random.set_rng_state(rng_start)
        zero_output = rho_zero(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            apply_weak_logit_permutation=True,
            weak_logit_permutation_context=context,
        )
        zero_output.loss.backward()
        rng_after_zero = torch.random.get_rng_state().clone()

        assert torch.equal(plain_output.logits, zero_output.logits)
        assert torch.equal(plain_output.loss, zero_output.loss)
        assert torch.equal(rng_after_disabled, rng_after_zero)
        for (plain_name, plain_param), (zero_name, zero_param) in zip(
            disabled.named_parameters(), rho_zero.named_parameters(), strict=True
        ):
            assert plain_name == zero_name
            if plain_param.grad is None:
                assert zero_param.grad is None
            else:
                assert torch.equal(plain_param.grad, zero_param.grad)

    @pytest.mark.parametrize("freeze_model1", [False, True])
    def test_rho_zero_matches_disabled_optimizer_update(self, freeze_model1):
        disabled = _make_joint_model(fusion_lambda=0.8, freeze_model1=freeze_model1)
        rho_zero = copy.deepcopy(disabled)
        disabled_optimizer = torch.optim.AdamW(disabled.parameters(), lr=1e-4)
        zero_optimizer = torch.optim.AdamW(rho_zero.parameters(), lr=1e-4)
        input_ids = torch.randint(0, 1000, (2, 6))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 1000, (2, 6))
        context = _dynperm_context(input_ids, rho=0.0)

        disabled(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
        disabled_optimizer.step()
        rho_zero(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            apply_weak_logit_permutation=True,
            weak_logit_permutation_context=context,
        ).loss.backward()
        zero_optimizer.step()

        for key, value in disabled.state_dict().items():
            assert torch.equal(value, rho_zero.state_dict()[key]), key
        disabled_states = list(disabled_optimizer.state.values())
        zero_states = list(zero_optimizer.state.values())
        assert len(disabled_states) == len(zero_states)
        for disabled_state, zero_state in zip(disabled_states, zero_states, strict=True):
            assert disabled_state.keys() == zero_state.keys()
            for key, value in disabled_state.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, zero_state[key])
                else:
                    assert value == zero_state[key]

    @pytest.mark.parametrize("freeze_model1", [False, True])
    def test_joint_checkpoint_round_trip_replays_next_step_permutation(self, freeze_model1):
        uninterrupted = _make_joint_model(fusion_lambda=0.8, freeze_model1=freeze_model1)
        optimizer = torch.optim.SGD(uninterrupted.parameters(), lr=1e-4)
        input_ids = torch.randint(0, 1000, (2, 6))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 1000, (2, 6))

        step1_context = _dynperm_context(input_ids, rho=1.0, global_step=1)
        uninterrupted(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            apply_weak_logit_permutation=True,
            weak_logit_permutation_context=step1_context,
        ).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        buffer = io.BytesIO()
        torch.save(
            {"model": uninterrupted.state_dict(), "optimizer": optimizer.state_dict(), "global_step": 1},
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        assert any(key.startswith("sub_models.0.") for key in checkpoint["model"])
        assert any(key.startswith("sub_models.1.") for key in checkpoint["model"])

        resumed = _make_joint_model(fusion_lambda=0.8, freeze_model1=freeze_model1)
        resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=1e-4)
        resumed.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        assert checkpoint["global_step"] == 1

        next_context = _dynperm_context(input_ids, rho=1.0, global_step=2)
        uninterrupted_next = uninterrupted(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            apply_weak_logit_permutation=True,
            weak_logit_permutation_context=next_context,
        )
        resumed_next = resumed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            apply_weak_logit_permutation=True,
            weak_logit_permutation_context=next_context,
        )
        assert torch.equal(uninterrupted_next.logits, resumed_next.logits)
        assert uninterrupted_next.dynamic_permutation_telemetry["dynperm/fixed_points"] == 0
        assert resumed_next.dynamic_permutation_telemetry["dynperm/fixed_points"] == 0

        uninterrupted_next.loss.backward()
        resumed_next.loss.backward()
        optimizer.step()
        resumed_optimizer.step()
        for key, value in uninterrupted.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[key]), key


class TestParameterFreezing:
    """Test model1 parameter freezing."""

    def test_freeze_model1(self):
        """When freeze_model1=True, model1 params should not require grad."""
        model = _make_joint_model(freeze_model1=True)

        for name, param in model.sub_models[0].named_parameters():
            assert not param.requires_grad, f"model1 param {name} should be frozen"

        # model2 should still require grad
        for name, param in model.sub_models[1].named_parameters():
            assert param.requires_grad, f"model2 param {name} should require grad"

    def test_frozen_model1_no_gradients(self):
        """Frozen model1 should not accumulate gradients after backward."""
        model = _make_joint_model(freeze_model1=True)
        model.train()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.randint(0, 1000, (2, 4))

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()

        for name, param in model.sub_models[0].named_parameters():
            assert param.grad is None, f"Frozen model1 param {name} should have no gradient"

        # model2 should have gradients
        has_grad = False
        for name, param in model.sub_models[1].named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "model2 should have gradients when model1 is frozen"


class TestLogProbsComputation:
    """Test that log_probs can be computed from fused logits (simulates verl's usage)."""

    def test_logprobs_from_fused_logits(self):
        """Simulate verl's logprobs_from_logits with fused logits."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

        # Simulate logprobs_from_logits: log_softmax and gather
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        assert gathered.shape == (2, 7)
        assert (gathered <= 0).all(), "Log probs should be <= 0"


class TestModel2Extraction:
    """Test extracting model2 state dict for evaluation."""

    def test_extract_model2_state_dict(self):
        """Should be able to extract model2 params with clean key names."""
        model = _make_joint_model()
        full_state_dict = model.state_dict()

        # Extract model2 params and strip prefix
        model2_params = {}
        for key, val in full_state_dict.items():
            if key.startswith("sub_models.1."):
                clean_key = key.replace("sub_models.1.", "", 1)
                model2_params[clean_key] = val

        # Verify keys match a standard Qwen3ForCausalLM
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        standalone_model = Qwen3ForCausalLM(config)
        standalone_keys = set(standalone_model.state_dict().keys())
        extracted_keys = set(model2_params.keys())

        assert standalone_keys == extracted_keys, (
            f"Key mismatch.\nMissing: {standalone_keys - extracted_keys}\nExtra: {extracted_keys - standalone_keys}"
        )

    def test_model2_produces_same_output_as_extracted(self):
        """model2 sub-model should produce same logits as standalone with same weights."""
        model = _make_joint_model()
        model.eval()

        # Extract model2 weights
        model2_params = {}
        for key, val in model.state_dict().items():
            if key.startswith("sub_models.1."):
                clean_key = key.replace("sub_models.1.", "", 1)
                model2_params[clean_key] = val.clone()

        # Create standalone model and load weights
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        standalone = Qwen3ForCausalLM(config)
        standalone.load_state_dict(model2_params)
        standalone.eval()

        # Compare outputs
        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            out_sub = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)
            out_standalone = standalone(input_ids=input_ids, attention_mask=attention_mask)

        torch.testing.assert_close(out_sub.logits, out_standalone.logits)


class TestEvalMode:
    """Test eval_mode that returns only model2 logits."""

    def test_eval_mode_returns_model2_only_logits(self):
        """In eval mode, model should return only model2 logits."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            # Get model2-only logits
            model2_out = model.sub_models[1](input_ids=input_ids, attention_mask=attention_mask)

            # Get eval-mode logits
            eval_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eval_only=True,
            )

        torch.testing.assert_close(eval_out.logits, model2_out.logits)

    def test_eval_mode_differs_from_fused(self):
        """Eval mode logits should differ from fused logits (unless models identical)."""
        model = _make_joint_model(fusion_lambda=0.5)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            fused_out = model(input_ids=input_ids, attention_mask=attention_mask)
            eval_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eval_only=True,
            )

        # They should differ because sub-models have different random init
        assert not torch.allclose(fused_out.logits, eval_out.logits), (
            "Fused and eval-only logits should differ with different sub-models"
        )
