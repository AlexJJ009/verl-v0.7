# Copyright 2026 The verl authors.
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

import copy
import os

import pytest
import torch

from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM
from verl.utils.attention_utils import is_remove_padding_backend_available
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import (
    FSDPActorConfig,
    PolicyLossConfig,
    SubmodelKLConfig,
    SubmodelKLPairConfig,
    WeakLogitPermutationConfig,
)


@pytest.fixture(scope="module", autouse=True)
def _distributed_group():
    initialized_here = False
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29634")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        initialized_here = True
    yield
    if initialized_here:
        torch.distributed.destroy_process_group()


def _model(*, freeze_model1=False):
    config = QwenJointConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        fusion_lambda=0.8,
        freeze_model1=freeze_model1,
    )
    return QwenJointForCausalLM(config)


def _actor(model, *, use_remove_padding):
    config = FSDPActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        use_remove_padding=use_remove_padding,
        ulysses_sequence_parallel_size=1,
        rollout_n=1,
        policy_loss=PolicyLossConfig(loss_mode="wdl_sft"),
        weak_logit_permutation=WeakLogitPermutationConfig(enabled=True, rho=1.0, audit_rows=2),
    )
    return DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=None)


@pytest.mark.parametrize("loss_mode", ["vanilla", "wdl_sft_is", "wdl_group_adv_is", "minirl"])
def test_dynamic_permutation_rejects_ratio_based_policy_losses(loss_mode):
    with pytest.raises(ValueError, match="supports only policy_loss.loss_mode='wdl_sft'"):
        FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=1,
            ppo_epochs=1,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            rollout_n=1,
            policy_loss=PolicyLossConfig(loss_mode=loss_mode),
            weak_logit_permutation=WeakLogitPermutationConfig(enabled=True, rho=1.0),
        )


def test_dynamic_permutation_accepts_teacher_forced_wdl_sft_loss():
    config = FSDPActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        ulysses_sequence_parallel_size=1,
        rollout_n=1,
        policy_loss=PolicyLossConfig(loss_mode="wdl_sft"),
        weak_logit_permutation=WeakLogitPermutationConfig(enabled=True, rho=1.0),
    )
    assert config.policy_loss.loss_mode == "wdl_sft"


def test_dynamic_permutation_rejects_actor_reference_kl():
    with pytest.raises(ValueError, match="does not support actor reference KL"):
        FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=1,
            rollout_n=1,
            policy_loss=PolicyLossConfig(loss_mode="wdl_sft"),
            use_kl_loss=True,
            weak_logit_permutation=WeakLogitPermutationConfig(enabled=True, rho=1.0),
        )


@pytest.mark.parametrize("model_name", ["model1", "model2"])
def test_dynamic_permutation_rejects_effective_submodel_kl(model_name):
    models = {
        "model1": SubmodelKLConfig(),
        "model2": SubmodelKLConfig(),
    }
    models[model_name] = SubmodelKLConfig(enabled=True, coef=0.1)
    with pytest.raises(ValueError, match="does not support effective submodel KL"):
        FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=1,
            rollout_n=1,
            policy_loss=PolicyLossConfig(loss_mode="wdl_sft"),
            submodel_kl=SubmodelKLPairConfig(enabled=True, **models),
            weak_logit_permutation=WeakLogitPermutationConfig(enabled=True, rho=1.0),
        )


def _micro_batch(*, rho, attention_mask=None):
    input_ids = torch.tensor([[0, 10, 11, 3, 4, 5], [20, 21, 22, 6, 7, 8]])
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids),
        "responses": torch.tensor([[3, 4, 5], [6, 7, 8]]),
        "response_mask": torch.tensor([[1, 1, 1], [1, 0, 1]]),
        "dynperm_sample_id": torch.tensor([100, 200]),
        "apply_weak_logit_permutation": True,
        "dynperm_rho": rho,
        "dynperm_base_seed": 42,
        "dynperm_global_step": 9,
        "dynperm_actor_update_index": 1,
        "dynperm_row_chunk_size": 2,
        "dynperm_audit_rows": 2,
        "dynperm_entropy_atol": 2e-6,
        "dynperm_multiset_atol": 0.0,
        "return_submodel_log_probs": False,
        "submodel_log_prob_grad_indices": [],
    }


def test_padded_actor_path_surfaces_validity_telemetry_and_gradients():
    model = _model()
    actor = _actor(model, use_remove_padding=False)
    outputs = actor._forward_micro_batch(_micro_batch(rho=1.0), temperature=1.0)
    assert outputs["log_probs"].shape == (2, 3)
    assert outputs["dynamic_permutation_telemetry"]["dynperm/target_mismatches"] == 0
    outputs["log_probs"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.sub_models[0].parameters())
    assert any(parameter.grad is not None for parameter in model.sub_models[1].parameters())


def test_rho_zero_actor_path_matches_disabled_and_rng_state():
    model = _model()
    actor = _actor(model, use_remove_padding=False)
    disabled = _micro_batch(rho=0.0)
    disabled["apply_weak_logit_permutation"] = False
    enabled = _micro_batch(rho=0.0)
    rng_before = torch.random.get_rng_state().clone()
    plain = actor._forward_micro_batch(disabled, temperature=1.0)["log_probs"]
    rng_after_plain = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng_before)
    zero = actor._forward_micro_batch(enabled, temperature=1.0)["log_probs"]
    rng_after_zero = torch.random.get_rng_state().clone()
    assert torch.equal(plain, zero)
    assert torch.equal(rng_after_plain, rng_after_zero)


def test_remove_padding_matches_padded_actor_path():
    if not is_remove_padding_backend_available():
        pytest.skip("remove-padding parity requires flash_attn or NPU attention backend")
    padded_model = _model()
    rmpad_model = copy.deepcopy(padded_model)
    padded = _actor(padded_model, use_remove_padding=False)
    rmpad = _actor(rmpad_model, use_remove_padding=True)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    micro_batch = _micro_batch(rho=1.0, attention_mask=attention_mask)
    padded_output = padded._forward_micro_batch(copy.deepcopy(micro_batch), temperature=1.0)
    rmpad_output = rmpad._forward_micro_batch(copy.deepcopy(micro_batch), temperature=1.0)
    assert torch.allclose(padded_output["log_probs"], rmpad_output["log_probs"], atol=2e-3, rtol=2e-3)
    assert rmpad_output["dynamic_permutation_telemetry"]["dynperm/invariant_failures"] == 0


def test_frozen_model1_actor_path_keeps_model2_gradient_connected():
    model = _model(freeze_model1=True)
    actor = _actor(model, use_remove_padding=False)
    outputs = actor._forward_micro_batch(_micro_batch(rho=1.0), temperature=1.0)
    outputs["log_probs"].sum().backward()
    assert all(parameter.grad is None for parameter in model.sub_models[0].parameters())
    assert any(parameter.grad is not None for parameter in model.sub_models[1].parameters())
