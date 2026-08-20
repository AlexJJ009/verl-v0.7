# SPDX-License-Identifier: Apache-2.0

"""P1 tests: reward metadata aggregation metrics in ray_trainer._val_metrics_update().

Tests the following jointTraining/ metrics computed from reward_extra_infos_dict:
- answer_extraction_failure_rate
- verification_method/* distribution
- verifier_pred_gt_disagreement_count
- response_unprintable_or_non_ascii_ratio

Run:
    cd /data-1/verl07/verl
    python -m pytest tests/joint_training/feat/test_p1_reward_aggregation_metrics.py -v
"""

from types import SimpleNamespace

import numpy as np
import pytest


def _make_trainer_for_val_metrics():
    """Create a minimal RayPPOTrainer with just enough state for _val_metrics_update."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._is_joint_training = True
    return trainer


def _make_reward_extra_infos_dict(
    preds=None,
    verification_methods=None,
    answer_corrects=None,
    ground_truths=None,
    rewards=None,
    accs=None,
):
    """Build a reward_extra_infos_dict similar to what _validate collects."""
    n = len(preds) if preds else 4
    return {
        "reward": rewards or [1.0, -1.0, -1.0, -1.0][:n],
        "acc": accs or [1.0, 0.0, 0.0, 0.0][:n],
        "pred": preds or ["42", "[NO_BOXED]", "[NO_BOXED]", "7"],
        "verification_method": verification_methods or ["latex_semantic", "no_answer", "no_answer", "string_match"],
        "answer_correct": answer_corrects or [True, False, False, False],
    }


class TestAnswerExtractionFailureRate:
    def test_metric_present(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        extra = _make_reward_extra_infos_dict(preds=["42", "[NO_BOXED]", "[NO_BOXED]", "7"])

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        assert "jointTraining/answer_extraction_failure_rate" in result

    def test_correct_rate(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        # 2 out of 4 are [NO_BOXED]
        extra = _make_reward_extra_infos_dict(preds=["42", "[NO_BOXED]", "[NO_BOXED]", "7"])

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        rate = result["jointTraining/answer_extraction_failure_rate"]
        assert abs(rate - 0.5) < 1e-6, f"Expected 0.5, got {rate}"

    def test_zero_failures(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 3)
        uids = ["u1", "u2", "u3"]
        extra = _make_reward_extra_infos_dict(
            preds=["42", "7", "100"],
            verification_methods=["latex_semantic"] * 3,
            answer_corrects=[True, False, True],
            rewards=[1.0, -1.0, 1.0],
            accs=[1.0, 0.0, 1.0],
        )

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        assert result["jointTraining/answer_extraction_failure_rate"] == 0.0


class TestVerificationMethodDistribution:
    def test_method_metrics_present(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        extra = _make_reward_extra_infos_dict(
            verification_methods=["latex_semantic", "no_answer", "verl_math_verify", "string_match"]
        )

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        assert "jointTraining/verification_method/latex_semantic" in result
        assert "jointTraining/verification_method/no_answer" in result
        assert "jointTraining/verification_method/verl_math_verify" in result
        assert "jointTraining/verification_method/string_match" in result

    def test_method_proportions(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        # 2 latex_semantic, 1 no_answer, 1 string_match
        extra = _make_reward_extra_infos_dict(
            verification_methods=["latex_semantic", "latex_semantic", "no_answer", "string_match"]
        )

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        assert abs(result["jointTraining/verification_method/latex_semantic"] - 0.5) < 1e-6
        assert abs(result["jointTraining/verification_method/no_answer"] - 0.25) < 1e-6
        assert abs(result["jointTraining/verification_method/string_match"] - 0.25) < 1e-6


class TestVerifierPredGtDisagreement:
    def test_metric_present(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        extra = _make_reward_extra_infos_dict()
        extra["ground_truth"] = ["42", "100", "200", "7"]

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        assert "jointTraining/verifier_pred_gt_disagreement_count" in result

    def test_disagreement_count(self):
        """Count cases where pred matches ground_truth text but answer_correct is False."""
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        extra = {
            "reward": [-1.0, 1.0, -1.0, -1.0],
            "acc": [0.0, 1.0, 0.0, 0.0],
            "pred": ["42", "42", "[NO_BOXED]", "7"],
            "verification_method": ["string_match", "latex_semantic", "no_answer", "string_match"],
            "answer_correct": [False, True, False, False],
            "ground_truth": ["42", "42", "100", "99"],
        }

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        # pred=="42" and ground_truth=="42" but answer_correct==False → 1 disagreement
        assert result["jointTraining/verifier_pred_gt_disagreement_count"] == 1


class TestResponseUnprintableRatio:
    def test_metric_present_when_outputs_available(self):
        trainer = _make_trainer_for_val_metrics()
        data_sources = np.array(["gsm8k"] * 3)
        uids = ["u1", "u2", "u3"]
        extra = _make_reward_extra_infos_dict(
            preds=["42", "7", "100"],
            verification_methods=["latex_semantic"] * 3,
            answer_corrects=[True, False, True],
            rewards=[1.0, -1.0, 1.0],
            accs=[1.0, 0.0, 1.0],
        )
        # Add response_texts to simulate outputs
        extra["response_text"] = [
            "The answer is \\boxed{42}.",
            "The answer is \\boxed{7}.",
            "The answer is \x00\x01\x02 garbage \\boxed{100}.",
        ]

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        assert "jointTraining/response_unprintable_or_non_ascii_ratio" in result
        # 1 out of 3 has unprintable chars
        ratio = result["jointTraining/response_unprintable_or_non_ascii_ratio"]
        assert abs(ratio - 1.0 / 3.0) < 1e-6


class TestNonJointTraining:
    def test_no_joint_metrics_when_not_joint(self):
        """Non-joint training should not include jointTraining/ metrics in validation."""
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer._is_joint_training = False
        data_sources = np.array(["gsm8k"] * 4)
        uids = ["u1", "u2", "u3", "u4"]
        extra = _make_reward_extra_infos_dict()

        result = trainer._val_metrics_update(data_sources, uids, extra, [])
        joint_keys = [k for k in result if k.startswith("jointTraining/")]
        assert len(joint_keys) == 0, f"Non-joint should have no jointTraining/ keys: {joint_keys}"


class TestJointValidationViews:
    def test_metric_view_namespaces_core_aux_and_joint_metrics(self):
        trainer = _make_trainer_for_val_metrics()
        result = trainer._val_metrics_update(
            np.array(["HumanEval+"] * 3),
            ["task-1"] * 3,
            _make_reward_extra_infos_dict(
                preds=["ok", "bad", "bad"],
                rewards=[1.0, -1.0, -1.0],
                accs=[1.0, 0.0, 0.0],
            ),
            [],
            metric_view="model1",
        )

        assert result["val-core/model1/HumanEval+/acc/mean@3"] == pytest.approx(1.0 / 3.0)
        assert result["val-core/model1/HumanEval+/acc/pass@3"] == 1.0
        assert result["val-aux/model1/HumanEval+/acc/std@3"] == pytest.approx(2**0.5 / 3.0)
        assert "jointTraining/model1/answer_extraction_failure_rate" in result

    def test_configured_views_run_model1_then_model2_with_namespaces(self):
        trainer = _make_trainer_for_val_metrics()
        trainer.config = SimpleNamespace(trainer={"joint_validation_views": ["model1", "model2"]})
        calls = []

        def validate(*, weight_view=None, metric_view=None):
            calls.append((weight_view, metric_view))
            return {f"val-core/{metric_view}/HumanEval+/acc/mean@3": 0.5}

        trainer._validate = validate

        metrics = trainer._validate_configured_views()

        assert calls == [("model1", "model1"), ("model2", "model2")]
        assert set(metrics) == {
            "val-core/model1/HumanEval+/acc/mean@3",
            "val-core/model2/HumanEval+/acc/mean@3",
        }
