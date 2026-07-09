import torch

from verl import DataProto
from verl.workers.config.actor import SubmodelKLConfig, SubmodelKLPairConfig
from verl.workers.fsdp_workers import _ref_logprob_tensors_from_actor_outputs


def test_reference_plumbing_preserves_legacy_ref_log_prob_and_submodel_identity():
    from tensordict import TensorDict

    tensors = {
        "ref_log_prob": torch.zeros(2, 3),
        "model1_ref_log_probs": torch.ones(2, 3),
        "model2_ref_log_probs": torch.full((2, 3), 2.0),
    }
    output = DataProto(batch=TensorDict(tensors, batch_size=[2]))

    assert "ref_log_prob" in output.batch
    assert "model1_ref_log_probs" in output.batch
    assert "model2_ref_log_probs" in output.batch
    assert output.batch["ref_log_prob"].shape == output.batch["model1_ref_log_probs"].shape
    assert output.batch["ref_log_prob"].shape == output.batch["model2_ref_log_probs"].shape
    assert torch.all(output.batch["ref_log_prob"] == 0)


def test_ref_logprob_mapping_carries_submodel_outputs():
    outputs = {
        "log_probs": torch.zeros(2, 3),
        "model1_log_probs": torch.ones(2, 3),
        "model2_log_probs": torch.full((2, 3), 2.0),
    }

    tensors = _ref_logprob_tensors_from_actor_outputs(outputs)

    assert set(tensors) == {"ref_log_prob", "model1_ref_log_probs", "model2_ref_log_probs"}
    assert torch.equal(tensors["ref_log_prob"], outputs["log_probs"])
    assert torch.equal(tensors["model1_ref_log_probs"], outputs["model1_log_probs"])
    assert torch.equal(tensors["model2_ref_log_probs"], outputs["model2_log_probs"])


def test_ref_actor_config_can_enable_submodel_logprob_compute():
    submodel_kl = SubmodelKLPairConfig(
        enabled=True,
        model1=SubmodelKLConfig(enabled=True, coef=0.01, kl_type="low_var_kl"),
        model2=SubmodelKLConfig(enabled=False, coef=0.0, kl_type="low_var_kl"),
    )

    assert submodel_kl.is_effective()
    assert submodel_kl.is_model1_effective()
    assert not submodel_kl.is_model2_effective()
