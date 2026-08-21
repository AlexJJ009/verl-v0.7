# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import math
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_wdl_sft_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import compute_global_grad_l2_norm, logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def build_dynamic_permutation_training_context(
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    sample_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Align next-token identities to full padded causal-logit rows."""

    if (
        input_ids.ndim != 2
        or position_ids.shape != input_ids.shape
        or responses.ndim != 2
        or response_mask.shape != responses.shape
    ):
        raise ValueError("Dynamic Permutation requires 2-D input_ids/responses and an aligned response_mask")
    batch_size, seqlen = input_ids.shape
    response_length = responses.shape[1]
    if sample_ids.shape != (batch_size,):
        raise ValueError(
            f"dynperm_sample_id shape {tuple(sample_ids.shape)} must equal actor microbatch {(batch_size,)}"
        )
    response_start = seqlen - response_length - 1
    if response_start < 0:
        raise ValueError("response sequence is too long to align next-token targets")

    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # The final input position has no causal target and is always inactive.
    target_ids[:, -1] = 0
    active_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    active_mask[:, response_start : response_start + response_length] = response_mask.to(torch.bool)
    token_positions = position_ids.to(device=input_ids.device, dtype=torch.int64)
    sample_ids = sample_ids.to(device=input_ids.device, dtype=torch.int64)[:, None].expand(-1, seqlen)
    return {
        "target_ids": target_ids,
        "sample_ids": sample_ids,
        "token_positions": token_positions,
        "active_mask": active_mask,
    }


def dynamic_permutation_actor_update_index(ppo_epoch: int, mini_batch_index: int, mini_batch_count: int) -> int:
    """Return the optimizer-update identity shared by all accumulation microbatches."""

    if ppo_epoch < 0 or mini_batch_index < 0 or mini_batch_count <= 0 or mini_batch_index >= mini_batch_count:
        raise ValueError("invalid PPO epoch/mini-batch coordinates for Dynamic Permutation")
    return ppo_epoch * mini_batch_count + mini_batch_index


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        dp_group=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.dp_group = dp_group
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)

        dynperm_cfg = self._conf_get(self.config, "weak_logit_permutation")
        self.use_weak_logit_permutation = bool(
            dynperm_cfg is not None and self._conf_get(dynperm_cfg, "enabled", False)
        )
        if self.use_weak_logit_permutation and (self.use_fused_kernels or self.use_prefix_grouper):
            raise NotImplementedError("weak-logit Dynamic Permutation does not support fused kernels or PrefixGrouper")
        if self.use_weak_logit_permutation and self._find_joint_sub_models() is None:
            raise ValueError("weak-logit Dynamic Permutation requires a two-submodel joint actor")

        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    @staticmethod
    def _conf_get(config, key: str, default=None):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    def _enabled_submodel_kl_indices(self) -> list[int]:
        submodel_kl = self._conf_get(self.config, "submodel_kl")
        if submodel_kl is None or not self._conf_get(submodel_kl, "enabled", False):
            return []
        enabled = []
        for idx, name in enumerate(("model1", "model2")):
            model_cfg = self._conf_get(submodel_kl, name)
            if (
                model_cfg is not None
                and self._conf_get(model_cfg, "enabled", False)
                and float(self._conf_get(model_cfg, "coef", 0.0) or 0.0) > 0.0
            ):
                enabled.append(idx)
        return enabled

    @staticmethod
    def _submodel_logprob_key(index: int) -> str:
        return f"model{index + 1}_log_probs"

    @staticmethod
    def _submodel_ref_logprob_key(index: int) -> str:
        return f"model{index + 1}_ref_log_probs"

    @staticmethod
    def _kl_type_code(kl_type: str) -> int:
        return {"kl": 1, "k1": 1, "mse": 2, "k2": 2, "low_var_kl": 3, "k3": 3, "abs": 4, "full": 5}[kl_type]

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        return_submodel_log_probs = bool(micro_batch.get("return_submodel_log_probs", False))
        submodel_log_prob_grad_indices = set(micro_batch.get("submodel_log_prob_grad_indices", []))
        apply_weak_logit_permutation = bool(micro_batch.get("apply_weak_logit_permutation", False))
        if return_submodel_log_probs and (self.use_fused_kernels or self.use_prefix_grouper):
            raise NotImplementedError("submodel KL log-probs do not support fused kernels or PrefixGrouper yet")
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            dynamic_permutation_context = None
            dynamic_permutation_parameters = None
            if apply_weak_logit_permutation:
                dynamic_permutation_context = build_dynamic_permutation_training_context(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    responses=micro_batch["responses"],
                    response_mask=micro_batch["response_mask"],
                    sample_ids=micro_batch["dynperm_sample_id"],
                )
                dynamic_permutation_parameters = {
                    "rho": micro_batch["dynperm_rho"],
                    "base_seed": micro_batch["dynperm_base_seed"],
                    "global_step": micro_batch["dynperm_global_step"],
                    "actor_update_index": micro_batch["dynperm_actor_update_index"],
                    "row_chunk_size": micro_batch["dynperm_row_chunk_size"],
                    "audit_rows": micro_batch["dynperm_audit_rows"],
                    "entropy_atol": micro_batch["dynperm_entropy_atol"],
                    "multiset_atol": micro_batch["dynperm_multiset_atol"],
                }
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if dynamic_permutation_context is not None:
                    for context_key, context_value in list(dynamic_permutation_context.items()):
                        dynamic_permutation_context[context_key] = index_first_axis(
                            rearrange(context_value.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                        ).transpose(0, 1)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    if dynamic_permutation_context is not None:
                        for context_key, context_value in list(dynamic_permutation_context.items()):
                            dynamic_permutation_context[context_key] = torch.zeros(
                                (1, self.ulysses_sequence_parallel_size),
                                device=context_value.device,
                                dtype=context_value.dtype,
                            )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    if dynamic_permutation_context is not None:
                        for context_key, context_value in list(dynamic_permutation_context.items()):
                            context_value, _, _ = ulysses_pad_and_slice_inputs(
                                context_value,
                                position_ids_rmpad=None,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )
                            dynamic_permutation_context[context_key] = context_value

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if dynamic_permutation_context is not None:
                    extra_args["apply_weak_logit_permutation"] = True
                    extra_args["weak_logit_permutation_context"] = {
                        **dynamic_permutation_context,
                        **dynamic_permutation_parameters,
                    }

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_submodel_logits=return_submodel_log_probs,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    submodel_log_probs_rmpad = {}
                    if return_submodel_log_probs:
                        if not hasattr(output, "submodel_logits"):
                            raise RuntimeError("submodel KL requested but model output has no submodel_logits")
                        for sub_idx, sub_logits in enumerate(output.submodel_logits):
                            if sub_idx not in submodel_log_prob_grad_indices:
                                sub_logits = sub_logits.detach()
                            sub_logits_rmpad = sub_logits.squeeze(0)
                            sub_logits_rmpad = sub_logits_rmpad / temperature
                            submodel_log_probs_rmpad[sub_idx] = logprobs_from_logits(
                                logits=sub_logits_rmpad,
                                labels=input_ids_rmpad_rolled,
                                inplace_backward=False,
                            )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    for sub_idx, sub_log_probs in list(submodel_log_probs_rmpad.items()):
                        submodel_log_probs_rmpad[sub_idx] = gather_outputs_and_unpad(
                            sub_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    for sub_idx, sub_log_probs in list(submodel_log_probs_rmpad.items()):
                        submodel_log_probs_rmpad[sub_idx] = sub_log_probs[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_submodel_log_probs = {}
                for sub_idx, sub_log_probs in submodel_log_probs_rmpad.items():
                    full_submodel_log_probs[sub_idx] = pad_input(
                        hidden_states=sub_log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                submodel_log_probs = {
                    sub_idx: full.squeeze(-1)[:, -response_length - 1 : -1]
                    for sub_idx, full in full_submodel_log_probs.items()
                }

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if dynamic_permutation_context is not None:
                    extra_args["apply_weak_logit_permutation"] = True
                    extra_args["weak_logit_permutation_context"] = {
                        **dynamic_permutation_context,
                        **dynamic_permutation_parameters,
                    }

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_submodel_logits=return_submodel_log_probs,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    submodel_log_probs = {}
                    if return_submodel_log_probs:
                        if not hasattr(output, "submodel_logits"):
                            raise RuntimeError("submodel KL requested but model output has no submodel_logits")
                        for sub_idx, sub_logits in enumerate(output.submodel_logits):
                            if sub_idx not in submodel_log_prob_grad_indices:
                                sub_logits = sub_logits.detach()
                            sub_logits = sub_logits / temperature
                            sub_logits = sub_logits[:, -response_length - 1 : -1, :]
                            submodel_log_probs[sub_idx] = logprobs_from_logits(sub_logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy_fn = self.compute_entropy_from_logits
                        if not self.config.entropy_checkpointing:
                            entropy = entropy_fn(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(entropy_fn, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            dynamic_permutation_telemetry = getattr(output, "dynamic_permutation_telemetry", None)
            if dynamic_permutation_telemetry is not None:
                outputs["dynamic_permutation_telemetry"] = dynamic_permutation_telemetry
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if return_submodel_log_probs:
                for sub_idx, sub_log_probs in submodel_log_probs.items():
                    outputs[self._submodel_logprob_key(sub_idx)] = sub_log_probs
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)

        joint_grad_metrics = self._compute_joint_grad_norm_metrics()
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        grad_norm_finite = bool(torch.isfinite(grad_norm).item())
        optimizer_step_applied = 1.0
        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            # GradScaler lowers its scale when it skips optimizer.step() because
            # non-finite gradients were found.
            optimizer_step_applied = float(float(self.scaler.get_scale()) >= scale_before)
        else:
            if not grad_norm_finite:
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
                optimizer_step_applied = 0.0
            else:
                self.actor_optimizer.step()

        joint_grad_metrics.update(
            {
                "actor/grad_clip_event": float(
                    grad_norm_finite and grad_norm.detach().item() > float(self.config.grad_clip)
                ),
                "actor/optimizer_step_applied": optimizer_step_applied,
            }
        )

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm, joint_grad_metrics

    def _find_joint_sub_models(self):
        unwrapped_module = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        for module in unwrapped_module.modules():
            sub_models = getattr(module, "sub_models", None)
            if isinstance(sub_models, nn.ModuleList) and len(sub_models) >= 2:
                return sub_models[0], sub_models[1]
        return None

    def _compute_joint_grad_norm_metrics(self) -> dict[str, float]:
        joint_sub_models = self._find_joint_sub_models()
        if joint_sub_models is None:
            return {}

        process_group = getattr(self.actor_module, "process_group", None)
        model1, model2 = joint_sub_models
        norm1 = compute_global_grad_l2_norm(model1.parameters(), process_group=process_group)
        norm2 = compute_global_grad_l2_norm(model2.parameters(), process_group=process_group)

        # Ratio with safe division
        ratio = norm1 / (norm2 + 1e-8)
        norm_total = norm1 + norm2
        model1_share = norm1 / (norm_total + 1e-8)
        model2_share = norm2 / (norm_total + 1e-8)

        # Cosine similarity between flattened gradients
        cos_sim = self._compute_joint_grad_cosine_similarity(model1, model2, process_group=process_group)

        return {
            "jointTraining/model1_grad_norm": norm1,
            "jointTraining/model2_grad_norm": norm2,
            "jointTraining/model1_grad_norm_share": model1_share,
            "jointTraining/model2_grad_norm_share": model2_share,
            "jointTraining/model_grad_norm_ratio": ratio,
            "jointTraining/model_grad_cosine_similarity": cos_sim,
        }

    def _compute_joint_submodel_loss_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        beta = float(self.config.policy_loss.get("wdl_sft_beta", 0.0))
        reward_labels = advantages[:, 0]
        metrics = {}
        for sub_idx in range(2):
            logprob_key = self._submodel_logprob_key(sub_idx)
            if logprob_key not in outputs:
                raise KeyError(logprob_key)
            with torch.no_grad():
                result = compute_wdl_sft_loss(
                    log_prob=outputs[logprob_key].detach(),
                    response_mask=response_mask,
                    reward_labels=reward_labels,
                    beta=beta,
                )
            prefix = f"jointTraining/model{sub_idx + 1}/wdl_sft"
            metrics[f"{prefix}_loss_positive"] = result["loss_positive"].item()
            metrics[f"{prefix}_loss_negative"] = result["loss_negative"].item()
            metrics[f"{prefix}_loss_total"] = result["total_loss"].item()
        return metrics

    @staticmethod
    def _compute_joint_grad_cosine_similarity(model1: nn.Module, model2: nn.Module, process_group=None) -> float:
        """Compute a sharding-aware cosine without materializing flattened models."""
        accumulator = None
        for parameter1, parameter2 in zip(model1.parameters(), model2.parameters(), strict=True):
            grad1 = parameter1.grad
            grad2 = parameter2.grad
            if grad1 is None or grad2 is None:
                continue
            if grad1.shape != grad2.shape:
                return 0.0
            if accumulator is None:
                accumulator = torch.zeros(3, dtype=torch.float64, device=grad1.device)
            grad1_fp32 = grad1.detach().float()
            grad2_fp32 = grad2.detach().float()
            accumulator[0] += torch.sum(grad1_fp32 * grad2_fp32, dtype=torch.float64)
            accumulator[1] += torch.sum(grad1_fp32.square(), dtype=torch.float64)
            accumulator[2] += torch.sum(grad2_fp32.square(), dtype=torch.float64)
        if accumulator is None:
            return 0.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(accumulator, group=process_group)
        denominator = torch.sqrt(accumulator[1] * accumulator[2])
        if denominator.item() < 1e-12:
            return 0.0
        return torch.clamp(accumulator[0] / denominator, min=-1.0, max=1.0).item()

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        suppress_config_submodel_log_probs = data.meta_info.get("suppress_config_submodel_log_probs", False)
        requested_submodel_log_probs = (
            [] if suppress_config_submodel_log_probs else data.meta_info.get("return_submodel_log_probs", [])
        )
        if requested_submodel_log_probs is True:
            requested_submodel_log_probs = [0, 1]
        elif requested_submodel_log_probs is False:
            requested_submodel_log_probs = []
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                dp_group=self.dp_group,
                same_micro_num_in_dp=self.dp_group is not None,
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        submodel_log_probs_lst: dict[str, list[torch.Tensor]] = {}
        enabled_submodel_kl_indices = [] if suppress_config_submodel_log_probs else self._enabled_submodel_kl_indices()
        returned_submodel_indices = sorted(set(enabled_submodel_kl_indices) | set(requested_submodel_log_probs))
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            model_inputs["return_submodel_log_probs"] = bool(returned_submodel_indices)
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            for sub_idx in returned_submodel_indices:
                key = self._submodel_logprob_key(sub_idx)
                if key in outputs:
                    submodel_log_probs_lst.setdefault(key, []).append(outputs[key])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        submodel_log_probs = {key: torch.concat(values, dim=0) for key, values in submodel_log_probs_lst.items()}
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)
            for key, value in list(submodel_log_probs.items()):
                submodel_log_probs[key] = restore_dynamic_batch(value, batch_idx_list)

        outputs = {"log_probs": log_probs}
        outputs.update(submodel_log_probs)
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        global_step = int(data.meta_info.get("global_step", -1))
        if self.use_weak_logit_permutation and global_step < 0:
            raise ValueError("weak-logit Dynamic Permutation requires restored global_step in actor metadata")

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        enabled_submodel_kl_indices = self._enabled_submodel_kl_indices()
        for sub_idx in enabled_submodel_kl_indices:
            select_keys.append(self._submodel_ref_logprob_key(sub_idx))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if self.use_weak_logit_permutation:
            if "dynperm_sample_id" not in data.batch:
                raise ValueError("weak-logit Dynamic Permutation requires dynperm_sample_id tensor")
            select_keys.append("dynperm_sample_id")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for ppo_epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                actor_update_index = dynamic_permutation_actor_update_index(ppo_epoch, batch_idx, len(mini_batches))
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                        dp_group=self.dp_group,
                        same_micro_num_in_dp=self.dp_group is not None,
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    track_joint_submodel_losses = bool(self.config.get("track_joint_submodel_losses", False))
                    model_inputs["return_submodel_log_probs"] = bool(
                        enabled_submodel_kl_indices or track_joint_submodel_losses
                    )
                    model_inputs["submodel_log_prob_grad_indices"] = enabled_submodel_kl_indices
                    if self.use_weak_logit_permutation:
                        dynperm_cfg = self.config.weak_logit_permutation
                        audit_this_step = bool(dynperm_cfg.audit_invariants) and (
                            global_step % int(dynperm_cfg.audit_frequency) == 0
                        )
                        model_inputs["apply_weak_logit_permutation"] = True
                        model_inputs["dynperm_global_step"] = global_step
                        model_inputs["dynperm_actor_update_index"] = actor_update_index
                        model_inputs["dynperm_rho"] = float(dynperm_cfg.rho)
                        model_inputs["dynperm_base_seed"] = int(dynperm_cfg.seed)
                        model_inputs["dynperm_row_chunk_size"] = int(dynperm_cfg.row_chunk_size)
                        model_inputs["dynperm_audit_rows"] = int(dynperm_cfg.audit_rows) if audit_this_step else 0
                        model_inputs["dynperm_entropy_atol"] = float(dynperm_cfg.entropy_atol)
                        model_inputs["dynperm_multiset_atol"] = float(dynperm_cfg.multiset_atol)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    dynamic_permutation_telemetry = outputs.get("dynamic_permutation_telemetry")
                    if dynamic_permutation_telemetry and self.config.weak_logit_permutation.log_telemetry:
                        for metric_name, metric_value in dynamic_permutation_telemetry.items():
                            if math.isfinite(float(metric_value)):
                                micro_batch_metrics[f"jointTraining/{metric_name}"] = metric_value

                    # Surface logit disagreement from joint model if available
                    unwrapped = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
                    logit_disagreement = getattr(unwrapped, "last_logit_disagreement", None)
                    if logit_disagreement is not None:
                        micro_batch_metrics["jointTraining/submodel_logit_disagreement"] = logit_disagreement
                    logit_diagnostics = getattr(unwrapped, "last_logit_diagnostics", None)
                    if logit_diagnostics:
                        for metric_name, metric_value in logit_diagnostics.items():
                            if math.isfinite(float(metric_value)):
                                micro_batch_metrics[f"jointTraining/{metric_name}"] = metric_value

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if track_joint_submodel_losses:
                        missing_submodel_outputs = [
                            self._submodel_logprob_key(index)
                            for index in range(2)
                            if self._submodel_logprob_key(index) not in outputs
                        ]
                        if missing_submodel_outputs:
                            raise RuntimeError(
                                "track_joint_submodel_losses requires both submodel forward paths; "
                                f"missing={missing_submodel_outputs}"
                            )
                        micro_batch_metrics.update(
                            self._compute_joint_submodel_loss_metrics(
                                outputs=outputs,
                                response_mask=response_mask,
                                advantages=advantages,
                            )
                        )
                        with torch.no_grad():
                            model1_log_prob = outputs[self._submodel_logprob_key(0)]
                            model2_log_prob = outputs[self._submodel_logprob_key(1)]
                            fused_vs_model2 = log_prob - model2_log_prob
                            model1_vs_model2 = model1_log_prob - model2_log_prob
                            valid_mask = response_mask.bool()
                            positive_mask = (advantages[:, :1] > 0) & valid_mask
                            negative_mask = (advantages[:, :1] < 0) & valid_mask

                            def _masked_scalar(values: torch.Tensor, mask: torch.Tensor) -> float:
                                selected = values.masked_select(mask)
                                return selected.mean().item() if selected.numel() else 0.0

                            micro_batch_metrics.update(
                                {
                                    "jointTraining/fused_vs_model2_chosen_token_logprob_delta_mean": _masked_scalar(
                                        fused_vs_model2, valid_mask
                                    ),
                                    "jointTraining/fused_vs_model2_chosen_token_logprob_abs_mean": _masked_scalar(
                                        fused_vs_model2.abs(), valid_mask
                                    ),
                                    "jointTraining/model1_vs_model2_chosen_token_logprob_delta_mean": _masked_scalar(
                                        model1_vs_model2, valid_mask
                                    ),
                                    "jointTraining/fused_vs_model2_positive_token_logprob_delta_mean": _masked_scalar(
                                        fused_vs_model2, positive_mask
                                    ),
                                    "jointTraining/fused_vs_model2_negative_token_logprob_delta_mean": _masked_scalar(
                                        fused_vs_model2, negative_mask
                                    ),
                                }
                            )

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    submodel_kl_total = log_prob.sum() * 0.0
                    for sub_idx in enabled_submodel_kl_indices:
                        model_name = f"model{sub_idx + 1}"
                        model_cfg = self.config.submodel_kl[model_name]
                        logprob_key = self._submodel_logprob_key(sub_idx)
                        ref_key = self._submodel_ref_logprob_key(sub_idx)
                        if logprob_key not in outputs:
                            raise KeyError(logprob_key)
                        if ref_key not in model_inputs:
                            raise KeyError(ref_key)
                        kld = kl_penalty(
                            logprob=outputs[logprob_key],
                            ref_logprob=model_inputs[ref_key],
                            kl_penalty=model_cfg.kl_type,
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        coef = float(model_cfg.coef)
                        policy_loss = policy_loss + kl_loss * coef
                        submodel_kl_total = submodel_kl_total + kl_loss * coef
                        micro_batch_metrics[f"actor/submodel_kl/{model_name}_loss"] = kl_loss.detach().item()
                        micro_batch_metrics[f"actor/submodel_kl/{model_name}_coef"] = coef
                        micro_batch_metrics[f"actor/submodel_kl/{model_name}_type_code"] = self._kl_type_code(
                            model_cfg.kl_type
                        )
                    if enabled_submodel_kl_indices:
                        micro_batch_metrics["actor/submodel_kl/total_loss"] = submodel_kl_total.detach().item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, joint_grad_metrics = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item(), **joint_grad_metrics}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
