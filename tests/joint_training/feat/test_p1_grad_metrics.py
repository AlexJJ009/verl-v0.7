# SPDX-License-Identifier: Apache-2.0

"""P1 tests: jointTraining/model_grad_norm_ratio and model_grad_cosine_similarity.

These metrics are computed in dp_actor._compute_joint_grad_norm_metrics().
Tests use a minimal mock model (no distributed, no FSDP) to verify the math.

Run:
    cd /data-1/verl07/verl
    python -m pytest tests/joint_training/feat/test_p1_grad_metrics.py -v
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal mock joint model (replicates sub_models structure without HF deps)
# ---------------------------------------------------------------------------


class _TinyLinearModel(nn.Module):
    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class _MockJointModel(nn.Module):
    """Minimal mock that has a sub_models ModuleList like QwenJointForCausalLM."""

    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.sub_models = nn.ModuleList(
            [
                _TinyLinearModel(in_dim, out_dim),
                _TinyLinearModel(in_dim, out_dim),
            ]
        )


def _populate_gradients(model, grad1_scale=1.0, grad2_scale=1.0):
    """Set deterministic gradients on the mock joint model's sub-models."""
    x = torch.randn(2, 8)
    # Forward through both sub-models and create a weighted loss
    out1 = model.sub_models[0](x)
    out2 = model.sub_models[1](x)
    loss = grad1_scale * out1.sum() + grad2_scale * out2.sum()
    loss.backward()


# ---------------------------------------------------------------------------
# Tests for grad_norm_ratio
# ---------------------------------------------------------------------------


class TestGradNormRatio:
    def test_ratio_present_in_metrics(self):
        """_compute_joint_grad_norm_metrics should return model_grad_norm_ratio."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module = _MockJointModel()
        _populate_gradients(actor.actor_module)

        metrics = actor._compute_joint_grad_norm_metrics()
        assert "jointTraining/model_grad_norm_ratio" in metrics

    def test_ratio_value_is_norm1_over_norm2(self):
        """Ratio should be model1_grad_norm / model2_grad_norm."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module = _MockJointModel()
        _populate_gradients(actor.actor_module, grad1_scale=3.0, grad2_scale=1.0)

        metrics = actor._compute_joint_grad_norm_metrics()
        norm1 = metrics["jointTraining/model1_grad_norm"]
        norm2 = metrics["jointTraining/model2_grad_norm"]
        expected_ratio = norm1 / norm2
        assert abs(metrics["jointTraining/model_grad_norm_ratio"] - expected_ratio) < 1e-5

    def test_ratio_safe_division_when_model2_zero_grad(self):
        """When model2 has zero gradients, ratio should not be inf/nan."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        model = _MockJointModel()
        # Only populate model1 gradients
        x = torch.randn(2, 8)
        out1 = model.sub_models[0](x)
        out1.sum().backward()
        # model2 has no gradients
        actor.actor_module = model

        metrics = actor._compute_joint_grad_norm_metrics()
        ratio = metrics["jointTraining/model_grad_norm_ratio"]
        assert not (ratio != ratio), "ratio should not be NaN"  # NaN check
        assert ratio != float("inf"), "ratio should not be inf"

    def test_grad_norm_shares_sum_to_one(self):
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module = _MockJointModel()
        _populate_gradients(actor.actor_module, grad1_scale=3.0, grad2_scale=1.0)

        metrics = actor._compute_joint_grad_norm_metrics()
        assert metrics["jointTraining/model1_grad_norm_share"] > metrics["jointTraining/model2_grad_norm_share"]
        assert (
            abs(metrics["jointTraining/model1_grad_norm_share"] + metrics["jointTraining/model2_grad_norm_share"] - 1.0)
            < 1e-6
        )


# ---------------------------------------------------------------------------
# Tests for grad_cosine_similarity
# ---------------------------------------------------------------------------


class TestGradCosineSimilarity:
    def test_cosine_sim_present_in_metrics(self):
        """_compute_joint_grad_norm_metrics should return model_grad_cosine_similarity."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module = _MockJointModel()
        _populate_gradients(actor.actor_module)

        metrics = actor._compute_joint_grad_norm_metrics()
        assert "jointTraining/model_grad_cosine_similarity" in metrics

    def test_cosine_sim_range(self):
        """Cosine similarity should be in [-1, 1]."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module = _MockJointModel()
        _populate_gradients(actor.actor_module)

        metrics = actor._compute_joint_grad_norm_metrics()
        cos_sim = metrics["jointTraining/model_grad_cosine_similarity"]
        assert -1.0 <= cos_sim <= 1.0, f"cosine sim {cos_sim} out of range"

    def test_cosine_sim_identical_grads_is_one(self):
        """When both sub-models have identical gradients, cosine sim should be ~1.0."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        model = _MockJointModel()
        # Copy model1 weights to model2 so they're identical
        model.sub_models[1].load_state_dict(model.sub_models[0].state_dict())
        actor.actor_module = model
        _populate_gradients(model, grad1_scale=1.0, grad2_scale=1.0)

        metrics = actor._compute_joint_grad_norm_metrics()
        cos_sim = metrics["jointTraining/model_grad_cosine_similarity"]
        assert cos_sim > 0.99, f"Expected ~1.0 for identical models, got {cos_sim}"

    def test_cosine_sim_zero_when_no_grads(self):
        """When either sub-model has no gradients, cosine sim should be 0."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        model = _MockJointModel()
        # Only model1 gets gradients
        x = torch.randn(2, 8)
        out1 = model.sub_models[0](x)
        out1.sum().backward()
        actor.actor_module = model

        metrics = actor._compute_joint_grad_norm_metrics()
        cos_sim = metrics["jointTraining/model_grad_cosine_similarity"]
        assert cos_sim == 0.0


# ---------------------------------------------------------------------------
# Tests for non-joint model (should return empty dict)
# ---------------------------------------------------------------------------


class TestNonJointModel:
    def test_non_joint_model_returns_empty(self):
        """For a non-joint model, _compute_joint_grad_norm_metrics returns {}."""
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module = nn.Linear(8, 4)  # not a joint model
        metrics = actor._compute_joint_grad_norm_metrics()
        assert metrics == {}
