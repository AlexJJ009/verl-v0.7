"""P1 tests: jointTraining/submodel_logit_disagreement.

The logit disagreement metric measures the mean absolute difference between
the two sub-models' logit distributions (after softmax). It is computed
during forward() and stored as model.last_logit_disagreement.

Run:
    cd /data-1/verl07/verl
    python -m pytest tests/joint_training/feat/test_p1_logit_disagreement.py -v
"""

import torch


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


class TestLogitDisagreement:
    def test_attribute_exists_after_forward(self):
        """Model should expose last_logit_disagreement after forward."""
        model = _make_joint_model()
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)

        assert hasattr(model, "last_logit_disagreement")
        assert model.last_logit_disagreement is not None

    def test_disagreement_is_nonnegative(self):
        """Logit disagreement should be >= 0."""
        model = _make_joint_model()
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)

        assert model.last_logit_disagreement >= 0.0

    def test_identical_submodels_have_zero_disagreement(self):
        """If both sub-models are identical, disagreement should be ~0."""
        model = _make_joint_model()
        # Copy model1 weights to model2
        model.sub_models[1].load_state_dict(model.sub_models[0].state_dict())
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)

        assert model.last_logit_disagreement < 1e-5, (
            f"Expected ~0 for identical models, got {model.last_logit_disagreement}"
        )

    def test_different_submodels_have_nonzero_disagreement(self):
        """Different sub-models should have nonzero disagreement."""
        model = _make_joint_model()
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)

        assert model.last_logit_disagreement > 0.0

    def test_eval_only_mode_no_disagreement(self):
        """eval_only mode only runs model2, so no disagreement should be computed."""
        model = _make_joint_model()
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        # Reset any previous value
        model.last_logit_disagreement = None
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask, eval_only=True)

        assert model.last_logit_disagreement is None

    def test_lambda_zero_no_disagreement_computed(self):
        """When lambda=0, only model1 logits are used. Disagreement is still computed
        since both sub-models still run forward."""
        model = _make_joint_model(fusion_lambda=0.0)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)

        # Both models still run forward, so disagreement should be computed
        assert model.last_logit_disagreement is not None

    def test_disagreement_is_float(self):
        """Disagreement should be a Python float, not a tensor."""
        model = _make_joint_model()
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.long)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=mask)

        assert isinstance(model.last_logit_disagreement, float)
