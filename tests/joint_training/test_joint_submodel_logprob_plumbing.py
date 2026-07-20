import os
import unittest

import torch
import torch.nn as nn
from tensordict import TensorDict

from verl import DataProto
from verl.models.joint_model.modeling_joint_qwen3 import QwenJointCausalLMOutputWithPast
from verl.utils.device import get_device_name
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import compute_wdl_sft_loss
from verl.workers.config import (
    FSDPActorConfig,
    OptimizerConfig,
    PolicyLossConfig,
    SubmodelKLConfig,
    SubmodelKLPairConfig,
)


class TinyJointModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        self.sub_models = nn.ModuleList(
            [
                nn.Sequential(nn.Embedding(vocab_size, hidden_size), nn.Linear(hidden_size, vocab_size)),
                nn.Sequential(nn.Embedding(vocab_size, hidden_size), nn.Linear(hidden_size, vocab_size)),
            ]
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, return_submodel_logits=False):
        logits0 = self.sub_models[0](input_ids)
        logits1 = self.sub_models[1](input_ids)
        fused = 0.25 * logits0 + 0.75 * logits1

        class Output:
            pass

        output = Output()
        output.logits = fused
        if return_submodel_logits:
            output.submodel_logits = [logits0, logits1]
        return output


class TestJointSubmodelLogprobPlumbing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29581")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.distributed.init_process_group(backend="gloo", init_method="env://")
        cls.device = torch.device("cuda:0" if get_device_name() == "cuda" else "cpu")

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _actor(self, submodel_kl=None, track_joint_submodel_losses=False, with_optimizer=False):
        model = TinyJointModel().to(self.device)
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=1,
            ppo_epochs=1,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
            rollout_n=1,
            policy_loss=PolicyLossConfig(loss_mode="wdl_sft", wdl_sft_beta=0.1),
            track_joint_submodel_losses=track_joint_submodel_losses,
            submodel_kl=submodel_kl or SubmodelKLPairConfig(),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6) if with_optimizer else None
        return DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=optimizer)

    def _data(self):
        batch_size = 2
        prompt_length = 3
        response_length = 2
        total_length = prompt_length + response_length
        input_ids = torch.randint(0, 32, (batch_size, total_length), device=self.device)
        return DataProto(
            batch=TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(batch_size, total_length, dtype=torch.long, device=self.device),
                    "position_ids": torch.arange(total_length, device=self.device).unsqueeze(0).expand(batch_size, -1),
                    "responses": input_ids[:, -response_length:],
                },
                batch_size=[batch_size],
            ),
            meta_info={"micro_batch_size": 2, "temperature": 1.0, "use_dynamic_bsz": False},
        )

    def test_disabled_submodel_kl_returns_legacy_log_probs_only(self):
        actor = self._actor()
        outputs = actor.compute_log_prob(self._data(), calculate_entropy=False)

        assert set(outputs.keys()) == {"log_probs"}
        assert outputs["log_probs"].shape == (2, 2)

    def test_diagnostic_request_returns_model2_without_enabling_kl(self):
        data = self._data()
        data.meta_info["return_submodel_log_probs"] = [1]
        outputs = self._actor().compute_log_prob(data, calculate_entropy=False)

        assert set(outputs.keys()) == {"log_probs", "model2_log_probs"}
        assert outputs["model2_log_probs"].shape == (2, 2)

    def test_diagnostic_submodel_log_probs_are_detached_without_kl(self):
        actor = self._actor(track_joint_submodel_losses=True)
        micro_batch = {**self._data().batch, "return_submodel_log_probs": True, "submodel_log_prob_grad_indices": []}
        outputs = actor._forward_micro_batch(micro_batch, temperature=1.0, calculate_entropy=False)

        assert not outputs["model1_log_probs"].requires_grad
        assert not outputs["model2_log_probs"].requires_grad

    def test_counterfactual_submodel_losses_match_wdl_helper(self):
        actor = self._actor(track_joint_submodel_losses=True)
        response_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], device=self.device)
        advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=self.device)
        outputs = {
            "model1_log_probs": torch.tensor([[-0.2, -0.3], [-0.7, -0.1]], device=self.device),
            "model2_log_probs": torch.tensor([[-0.1, -0.4], [-0.5, -0.2]], device=self.device),
        }

        metrics = actor._compute_joint_submodel_loss_metrics(outputs, response_mask, advantages)
        expected_model1 = compute_wdl_sft_loss(
            outputs["model1_log_probs"], response_mask, advantages[:, 0], beta=0.1
        )
        expected_model2 = compute_wdl_sft_loss(
            outputs["model2_log_probs"], response_mask, advantages[:, 0], beta=0.1
        )

        assert metrics["jointTraining/model1/wdl_sft_loss_total"] == expected_model1["total_loss"].item()
        assert metrics["jointTraining/model2/wdl_sft_loss_total"] == expected_model2["total_loss"].item()

    def test_training_step_logs_submodel_losses_and_gradient_norms(self):
        actor = self._actor(track_joint_submodel_losses=True, with_optimizer=True)
        data = self._data()
        response_mask = torch.ones_like(data.batch["responses"], dtype=torch.float32)
        data.batch["response_mask"] = response_mask
        data.batch["old_log_probs"] = torch.zeros_like(response_mask)
        data.batch["advantages"] = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=self.device)
        data.meta_info.update({"temperature": 1.0, "pad_token_id": 0})

        before_model1 = next(actor.actor_module.sub_models[0].parameters()).detach().clone()
        before_model2 = next(actor.actor_module.sub_models[1].parameters()).detach().clone()
        metrics = actor.update_policy(data)

        expected_metrics = {
            "jointTraining/model1/wdl_sft_loss_total",
            "jointTraining/model2/wdl_sft_loss_total",
            "jointTraining/model1_grad_norm",
            "jointTraining/model2_grad_norm",
            "jointTraining/model1_grad_norm_share",
            "jointTraining/model2_grad_norm_share",
        }
        assert expected_metrics.issubset(metrics)
        for key in expected_metrics:
            assert torch.isfinite(torch.as_tensor(metrics[key])).all(), key
        assert not torch.equal(before_model1, next(actor.actor_module.sub_models[0].parameters()).detach())
        assert not torch.equal(before_model2, next(actor.actor_module.sub_models[1].parameters()).detach())

    def test_enabled_submodel_kl_returns_fused_and_submodel_log_probs(self):
        submodel_kl = SubmodelKLPairConfig(
            enabled=True,
            model1=SubmodelKLConfig(enabled=True, coef=0.1, kl_type="mse"),
            model2=SubmodelKLConfig(enabled=True, coef=0.2, kl_type="low_var_kl"),
        )
        actor = self._actor(submodel_kl=submodel_kl)
        outputs = actor.compute_log_prob(self._data(), calculate_entropy=False)

        assert outputs["log_probs"].shape == (2, 2)
        assert outputs["model1_log_probs"].shape == (2, 2)
        assert outputs["model2_log_probs"].shape == (2, 2)

    def test_joint_output_schema_keeps_submodel_logits_field(self):
        output = QwenJointCausalLMOutputWithPast(
            logits=torch.zeros(1, 2, 3),
            submodel_logits=(torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0)),
        )

        assert hasattr(output, "submodel_logits")
        assert len(output.submodel_logits) == 2
        assert output.submodel_logits[0].shape == output.logits.shape
