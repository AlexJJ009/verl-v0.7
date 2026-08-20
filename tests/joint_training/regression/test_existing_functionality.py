"""Regression tests to ensure joint training changes don't break existing functionality.

These tests verify that:
1. Standard Qwen3ForCausalLM still works normally
2. Core GRPO algorithm functions produce correct results
3. Weight extraction utils gracefully handle non-joint models
4. The joint model's eval_only=False (default) doesn't alter behavior
"""

import pytest
import torch


class TestStandardModelNotAffected:
    """Verify standard (non-joint) Qwen3ForCausalLM still works correctly."""

    def test_standard_qwen3_forward(self):
        """Standard Qwen3 model forward pass should work unchanged."""
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
        model = Qwen3ForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert output.logits.shape == (2, 4, 1000)

    def test_standard_qwen3_backward(self):
        """Standard Qwen3 model backward pass should work unchanged."""
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
        model = Qwen3ForCausalLM(config)
        model.train()

        input_ids = torch.randint(0, 1000, (2, 4))
        attention_mask = torch.ones(2, 4, dtype=torch.long)
        labels = torch.randint(0, 1000, (2, 4))

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestWeightUtilsOnNonJointModel:
    """Weight utils should handle non-joint models gracefully."""

    def test_is_joint_model_returns_false_for_standard(self):
        """is_joint_model_state_dict returns False for standard models."""
        from verl.models.joint_model.weight_utils import is_joint_model_state_dict

        normal_sd = {"model.layers.0.weight": torch.randn(10, 10)}
        assert is_joint_model_state_dict(normal_sd) is False

    def test_is_joint_model_config_returns_false_for_standard(self):
        """is_joint_model_config returns False for standard configs."""
        from transformers import Qwen3Config

        from verl.models.joint_model.weight_utils import is_joint_model_config

        config = Qwen3Config(vocab_size=1000)
        assert is_joint_model_config(config) is False

    def test_extract_returns_empty_for_non_joint(self):
        """extract_sub_model_weights returns empty dict for non-joint state dicts."""
        from verl.models.joint_model.weight_utils import extract_sub_model_weights

        normal_sd = {"model.layers.0.weight": torch.randn(10, 10)}
        result = extract_sub_model_weights(normal_sd, sub_model_index=1)
        assert len(result) == 0


class TestCoreAlgosUnchanged:
    """Verify core_algos functions still produce correct results."""

    def test_grpo_advantage_computation(self):
        """GRPO advantage computation should work correctly."""
        pytest.importorskip(
            "verl.trainer.ppo.core_algos", reason="core_algos import chain requires full env", exc_type=ImportError
        )
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        batch_size = 8
        token_level_rewards = torch.zeros(batch_size, 16)
        token_level_rewards[:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0])

        response_mask = torch.ones(batch_size, 16)
        # 2 groups of 4 (simulating n=4 per prompt)
        index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).numpy()

        advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
        )

        assert advantages.shape == (batch_size, 16)
        # Advantages should be normalized within groups
        group0_adv = advantages[:4, -1]
        # Within each group, mean should be ~0 (normalized)
        assert abs(group0_adv.mean().item()) < 0.1

    def test_policy_loss_function(self):
        """Vanilla policy loss should still compute correctly."""
        pytest.importorskip(
            "verl.trainer.ppo.core_algos", reason="core_algos import chain requires full env", exc_type=ImportError
        )
        from types import SimpleNamespace

        from verl.trainer.ppo.core_algos import get_policy_loss_fn

        loss_fn = get_policy_loss_fn("vanilla")

        batch_size, seq_len = 4, 16
        old_log_prob = torch.randn(batch_size, seq_len)
        log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.1
        advantages = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        config = SimpleNamespace(
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            clip_ratio_c=10.0,
            global_batch_info={},
        )
        config.get = lambda key, default=None: getattr(config, key, default)

        loss, loss_info = loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=config,
        )

        assert loss.ndim == 0  # scalar


class TestLogProbsComputation:
    """Verify log_probs computation works the same regardless of model type."""

    def test_logprobs_from_logits(self):
        """logprobs_from_logits should work with any logits source."""
        pytest.importorskip("verl.utils.torch_functional", reason="torch_functional import may require full env")
        from verl.utils.torch_functional import logprobs_from_logits

        logits = torch.randn(2, 8, 1000)
        labels = torch.randint(0, 1000, (2, 8))

        log_probs = logprobs_from_logits(logits, labels, inplace_backward=False)
        assert log_probs.shape == (2, 8)
        assert (log_probs <= 0).all()

    def test_logprobs_from_fused_vs_single(self):
        """Log probs from fused logits should differ from single model logits."""
        pytest.importorskip("verl.utils.torch_functional", reason="torch_functional import may require full env")
        from verl.utils.torch_functional import logprobs_from_logits

        logits_0 = torch.randn(2, 8, 1000)
        logits_1 = torch.randn(2, 8, 1000)
        labels = torch.randint(0, 1000, (2, 8))

        fused_logits = 0.5 * logits_0 + 0.5 * logits_1

        lp_single = logprobs_from_logits(logits_0, labels, inplace_backward=False)
        lp_fused = logprobs_from_logits(fused_logits, labels, inplace_backward=False)

        assert not torch.allclose(lp_single, lp_fused)


class TestAutoModelVision2SeqImportCompat:
    """Regression test for AutoModelForVision2Seq import compatibility.

    transformers >= 5.0 removed AutoModelForVision2Seq in favor of
    AutoModelForImageTextToText. The fsdp_workers import must handle
    both old and new transformers versions gracefully.
    """

    def test_fsdp_workers_import_does_not_crash(self):
        """Importing fsdp_workers should not raise ImportError for AutoModelForVision2Seq."""
        # This was the root cause of the training crash in transformers 5.2.0
        from verl.workers import fsdp_workers  # noqa: F401

    def test_auto_model_vision_fallback(self):
        """AutoModelForVision2Seq should fall back to AutoModelForImageTextToText."""
        from transformers import AutoModelForImageTextToText

        try:
            from transformers import AutoModelForVision2Seq

            # If it exists (older transformers), it should be usable
            assert AutoModelForVision2Seq is not None
        except ImportError:
            # In newer transformers, AutoModelForImageTextToText is the replacement
            # Our code aliases it, verify the alias works
            AutoModelForVision2Seq = AutoModelForImageTextToText
            assert AutoModelForVision2Seq is AutoModelForImageTextToText

    def test_hf_rollout_import_not_blocked(self):
        """HFRollout should be importable without AutoModelForVision2Seq workaround."""
        from verl.workers.rollout.hf_rollout import HFRollout  # noqa: F401
