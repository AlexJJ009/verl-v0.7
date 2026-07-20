"""
Tests for the custom_reward_function_latex_verify module.

Validates the 3-tier reward verification pipeline:
  1. LaTeX semantic verification (math_verify + latex2sympy2_extended)
  2. Fallback: verl's math_verify
  3. Fallback: string matching on \\boxed{} extraction

Also tests EOS detection logic and the DAPO reward manager integration.

Run:
    cd /data-1/verl07/verl
    python -m pytest tests/joint_training/reward/test_reward_function_latex_verify.py -v
"""

import sys
import os
import pytest

# ---------------------------------------------------------------------------
# Ensure the reward function module is importable from the recipe directory
# ---------------------------------------------------------------------------
REWARD_FN_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "recipe", "joint_training",
)
REWARD_FN_DIR = os.path.normpath(REWARD_FN_DIR)
if REWARD_FN_DIR not in sys.path:
    sys.path.insert(0, REWARD_FN_DIR)

from custom_reward_function_latex_verify import (
    compute_score_latex_verify,
    verify_with_latex,
    extract_boxed_answer,
    MATH_VERIFY_AVAILABLE,
)


# ===========================================================================
# Section 1: verify_with_latex — LaTeX semantic matching
# ===========================================================================

class TestVerifyWithLatex:
    """Tests for the primary LaTeX semantic verifier."""

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_exact_numeric_match(self):
        pred = "The answer is \\boxed{42}."
        assert verify_with_latex(pred, "42") is True

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_fraction_equivalence(self):
        """Semantic match: 1/2 == 0.5"""
        pred = "Therefore \\boxed{\\frac{1}{2}}."
        assert verify_with_latex(pred, "0.5") is True

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_symbolic_equivalence(self):
        """Semantic match: 2^3 == 8"""
        pred = "We get \\boxed{2^3}."
        assert verify_with_latex(pred, "8") is True

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_wrong_answer(self):
        pred = "The answer is \\boxed{7}."
        assert verify_with_latex(pred, "42") is False

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_sqrt_equivalence(self):
        """sqrt(4) == 2"""
        pred = "\\boxed{\\sqrt{4}}"
        assert verify_with_latex(pred, "2") is True

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_negative_number(self):
        pred = "\\boxed{-3}"
        assert verify_with_latex(pred, "-3") is True

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_large_integer(self):
        pred = "\\boxed{1000000}"
        assert verify_with_latex(pred, "1000000") is True

    @pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify not installed")
    def test_no_boxed_answer_returns_false(self):
        """No boxed content → parser extracts nothing → should return False."""
        pred = "I think the answer is 42, probably."
        result = verify_with_latex(pred, "42")
        # May return False or None depending on extraction; both are acceptable
        assert result is not True


# ===========================================================================
# Section 2: extract_boxed_answer
# ===========================================================================

class TestExtractBoxedAnswer:
    def test_simple_boxed(self):
        assert extract_boxed_answer("\\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_last_boxed_taken(self):
        text = "First \\boxed{1}, then \\boxed{2}."
        assert extract_boxed_answer(text) == "2"

    def test_no_boxed(self):
        assert extract_boxed_answer("There is no boxed answer here.") is None

    def test_empty_boxed(self):
        result = extract_boxed_answer("\\boxed{}")
        assert result == "" or result is None  # implementation-dependent


# ===========================================================================
# Section 3: compute_score_latex_verify — full pipeline
# ===========================================================================

class TestComputeScoreLatexVerify:
    """End-to-end tests for the reward function."""

    def test_correct_answer_with_eos(self):
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="Let me think... The answer is \\boxed{42}.",
            ground_truth="42",
            extra_info={"valid_response_length": 50, "max_resp_len": 1024},
        )
        assert result["score"] == 1.0
        assert result["acc"] is True
        assert result["has_eos"] is True
        assert result["pred"] == "42"

    def test_wrong_answer_with_eos(self):
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="The answer is \\boxed{7}.",
            ground_truth="42",
            extra_info={"valid_response_length": 30, "max_resp_len": 1024},
        )
        assert result["score"] == -1.0
        assert result["acc"] is False
        assert result["has_eos"] is True

    def test_correct_answer_without_eos_gets_penalty(self):
        """Response truncated (no EOS) → reward = -1.0 even if answer is correct."""
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="The answer is \\boxed{42}.",
            ground_truth="42",
            extra_info={"valid_response_length": 1024, "max_resp_len": 1024},
        )
        assert result["score"] == -1.0
        assert result["has_eos"] is False
        assert result["acc"] is False  # penalized for no EOS

    def test_no_boxed_answer(self):
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="I think the answer is 42.",
            ground_truth="42",
            extra_info={"valid_response_length": 30, "max_resp_len": 1024},
        )
        assert result["pred"] == "[NO_BOXED]"
        assert result["score"] == -1.0

    def test_no_extra_info_defaults_to_has_eos(self):
        """When extra_info is None, assume EOS is present."""
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="\\boxed{42}",
            ground_truth="42",
        )
        assert result["has_eos"] is True
        assert result["score"] == 1.0

    def test_fraction_semantic_match(self):
        result = compute_score_latex_verify(
            data_source="math",
            solution_str="Therefore \\boxed{\\frac{3}{6}}.",
            ground_truth="0.5",
            extra_info={"valid_response_length": 40, "max_resp_len": 1024},
        )
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_return_dict_keys(self):
        """Verify all expected keys are present in the result dict."""
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="\\boxed{1}",
            ground_truth="1",
        )
        expected_keys = {
            "score",
            "acc",
            "pred",
            "has_eos",
            "truncated",
            "think_complete",
            "answer_complete",
            "boxed_extraction_success",
            "reward_grader_success",
            "format_contract_success",
            "answer_correct",
            "verification_method",
        }
        assert expected_keys == set(result.keys())

    def test_verification_method_is_populated(self):
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="\\boxed{42}",
            ground_truth="42",
        )
        assert result["verification_method"] in {
            "latex_semantic", "verl_math_verify", "string_match", "no_answer"
        }

    def test_verification_order_uses_semantic_before_fallbacks(self, monkeypatch):
        monkeypatch.setattr(
            "custom_reward_function_latex_verify.verify_with_latex",
            lambda solution_str, ground_truth: True,
        )

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError("fallback should not run when latex semantic verification succeeds")

        monkeypatch.setattr("custom_reward_function_latex_verify.verl_math_verify.compute_score", _unexpected_fallback)

        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="anything",
            ground_truth="42",
        )

        assert result["score"] == 1.0
        assert result["acc"] is True
        assert result["verification_method"] == "latex_semantic"

    def test_verification_order_falls_back_to_string_match_after_semantic_paths_fail(self, monkeypatch):
        monkeypatch.setattr(
            "custom_reward_function_latex_verify.verify_with_latex",
            lambda solution_str, ground_truth: None,
        )

        def _raise_verl_math_verify(*args, **kwargs):
            raise RuntimeError("force string match fallback")

        monkeypatch.setattr(
            "custom_reward_function_latex_verify.verl_math_verify.compute_score",
            _raise_verl_math_verify,
        )

        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="The answer is \\boxed{42}.",
            ground_truth="42",
        )

        assert result["score"] == 1.0
        assert result["acc"] is True
        assert result["verification_method"] == "string_match"

    def test_string_match_fallback(self):
        """When the answer is a plain string that math_verify can't parse,
        the function should fall back to string matching."""
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="\\boxed{hello}",
            ground_truth="hello",
            extra_info={"valid_response_length": 20, "max_resp_len": 1024},
        )
        # Should still match via string fallback
        assert result["score"] == 1.0
        assert result["acc"] is True


# ===========================================================================
# Section 4: Edge cases and robustness
# ===========================================================================

class TestEdgeCases:
    def test_empty_solution(self):
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="",
            ground_truth="42",
        )
        assert result["score"] == -1.0

    def test_empty_ground_truth(self):
        """Edge case: empty ground truth should not crash."""
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="\\boxed{42}",
            ground_truth="",
        )
        assert isinstance(result["score"], float)

    def test_unicode_in_answer(self):
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str="\\boxed{π}",
            ground_truth="π",
        )
        assert isinstance(result["score"], float)

    def test_multiline_response(self):
        solution = """Step 1: We compute 6 * 7 = 42.
Step 2: Therefore the answer is \\boxed{42}.
"""
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str=solution,
            ground_truth="42",
        )
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_thinking_tags_with_boxed(self):
        """Simulate a model response with <think>...</think> tags."""
        solution = "<think>Let me work through this. 6*7=42.</think>\n\nThe answer is \\boxed{42}."
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str=solution,
            ground_truth="42",
        )
        assert result["score"] == 1.0

    def test_multiple_boxed_conflicting_penalized(self):
        """Multiple conflicting boxed expressions — latex_semantic extracts ALL
        boxed content so \\boxed{7} conflicts with ground_truth=42, causing
        the verifier to reject. This is intended: the reward function penalizes
        self-contradictory answers."""
        solution = "First attempt: \\boxed{7}. Wait, let me recalculate... \\boxed{42}."
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str=solution,
            ground_truth="42",
        )
        assert result["score"] == -1.0

    def test_multiple_boxed_same_answer(self):
        """Multiple boxed with the SAME answer — should be correct."""
        solution = "We get \\boxed{42}. Confirmed: \\boxed{42}."
        result = compute_score_latex_verify(
            data_source="gsm8k",
            solution_str=solution,
            ground_truth="42",
        )
        assert result["score"] == 1.0


# ===========================================================================
# Section 5: DAPO reward manager integration smoke test
# ===========================================================================

class TestDAPORewardManagerImport:
    """Verify that the DAPO reward manager can be instantiated with our custom
    reward function, without needing a full training loop."""

    def test_dapo_manager_importable(self):
        from verl.workers.reward_manager.dapo import DAPORewardManager
        assert DAPORewardManager is not None

    def test_dapo_manager_instantiation(self):
        """Create a DAPORewardManager with our custom compute_score function."""
        from verl.workers.reward_manager.dapo import DAPORewardManager
        from unittest.mock import MagicMock
        from omegaconf import OmegaConf

        mock_tokenizer = MagicMock()
        overlong_cfg = OmegaConf.create({
            "enable": True,
            "len": 1024,
            "penalty_factor": 0.5,
            "log": False,
        })

        manager = DAPORewardManager(
            tokenizer=mock_tokenizer,
            num_examine=2,
            compute_score=compute_score_latex_verify,
            max_resp_len=2048,
            overlong_buffer_cfg=overlong_cfg,
        )
        assert manager.compute_score is compute_score_latex_verify
        assert manager.max_resp_len == 2048
        assert manager.overlong_buffer_cfg.enable is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
