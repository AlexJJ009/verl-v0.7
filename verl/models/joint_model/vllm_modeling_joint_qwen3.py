from collections.abc import Iterable, Mapping
from typing import Optional, Union

import torch
from torch import nn

# Patch vLLM layer-index extraction early — this module is imported in every
# vLLM worker process (including those spawned with start_method='spawn')
# when the QwenJointForCausalLM architecture is resolved.
from verl.models.joint_model.vllm_registry import patch_joint_vllm_layer_indexing

patch_joint_vllm_layer_indexing()


class QwenJointForCausalLM(nn.Module):
    """vLLM-native joint Qwen3 model for fused rollout and model2-only eval."""

    supports_lora = True
    supports_pp = True
    embedding_modules = {}
    embedding_padding_modules = []
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    VALID_ROLLOUT_SOURCES = {"fused", "sub_model_0", "sub_model_1"}

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()

        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory

        config = vllm_config.model_config.hf_config
        self.fusion_lambda = getattr(config, "fusion_lambda", 0.5)
        self.hidden_size = config.hidden_size
        self._use_model2_only = False
        self._joint_rollout_source = "fused"

        self.sub_models = nn.ModuleList(
            [
                Qwen3ForCausalLM(vllm_config=vllm_config, prefix=self._child_prefix(prefix, i))
                for i in range(2)
            ]
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            [
                "sub_model_0_hidden_states",
                "sub_model_0_residual",
                "sub_model_1_hidden_states",
                "sub_model_1_residual",
            ],
            config.hidden_size,
        )

    def set_joint_rollout_source(self, source: str) -> None:
        if source not in self.VALID_ROLLOUT_SOURCES:
            raise ValueError(
                f"Invalid joint rollout source {source!r}; expected one of "
                f"{sorted(self.VALID_ROLLOUT_SOURCES)}."
            )
        if self._use_model2_only and source == "sub_model_0":
            raise ValueError("Cannot select sub_model_0 after model2-only weights have been loaded.")
        self._joint_rollout_source = source

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Delegate to the first sub-model's embedding layer."""
        sub = self.sub_models[0]
        if hasattr(sub, "embed_input_ids"):
            return sub.embed_input_ids(input_ids)
        return sub.get_input_embeddings(input_ids)

    @staticmethod
    def _child_prefix(prefix: str, index: int) -> str:
        return f"{prefix}.sub_models.{index}" if prefix else f"sub_models.{index}"

    @staticmethod
    def _split_joint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[bool, list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        model0_weights: list[tuple[str, torch.Tensor]] = []
        model1_weights: list[tuple[str, torch.Tensor]] = []
        model2_only_weights: list[tuple[str, torch.Tensor]] = []
        is_joint = False

        for name, tensor in weights:
            if name.startswith("sub_models.0."):
                is_joint = True
                model0_weights.append((name.removeprefix("sub_models.0."), tensor))
            elif name.startswith("sub_models.1."):
                is_joint = True
                model1_weights.append((name.removeprefix("sub_models.1."), tensor))
            else:
                model2_only_weights.append((name, tensor))

        return is_joint, model0_weights, model1_weights, model2_only_weights

    @staticmethod
    def _extract_tensor_output(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    @staticmethod
    def _is_intermediate_output(output) -> bool:
        return isinstance(output, Mapping) and "hidden_states" in output

    @staticmethod
    def _pack_intermediate_tensors(output0, output1):
        tensor_cls = type(output0)
        merged = {
            f"sub_model_0_{name}": tensor
            for name, tensor in output0.items()
        }
        merged.update(
            {
                f"sub_model_1_{name}": tensor
                for name, tensor in output1.items()
            }
        )
        return tensor_cls(merged)

    @staticmethod
    def _extract_intermediate_tensors(intermediate_tensors, index: int):
        if intermediate_tensors is None:
            return None
        if "hidden_states" in intermediate_tensors:
            return intermediate_tensors

        tensor_cls = type(intermediate_tensors)
        prefix = f"sub_model_{index}_"
        selected = {
            name.removeprefix(prefix): tensor
            for name, tensor in intermediate_tensors.items()
            if name.startswith(prefix)
        }
        return tensor_cls(selected)

    def _pack_hidden_states(self, hidden_states0: torch.Tensor, hidden_states1: torch.Tensor) -> torch.Tensor:
        return torch.cat((hidden_states0, hidden_states1), dim=-1)

    def _unpack_hidden_states(
        self, hidden_states: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(hidden_states, tuple):
            return hidden_states

        if hidden_states.shape[-1] != 2 * self.hidden_size:
            raise ValueError(
                "Joint vLLM hidden states must pack both sub-model outputs along the last dimension. "
                f"Expected trailing dimension {2 * self.hidden_size}, got {hidden_states.shape[-1]}."
            )

        return torch.split(hidden_states, self.hidden_size, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[object] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, object]:
        source = getattr(self, "_joint_rollout_source", "fused")
        if source not in self.VALID_ROLLOUT_SOURCES:
            raise ValueError(
                f"Invalid joint rollout source {source!r}; expected one of "
                f"{sorted(self.VALID_ROLLOUT_SOURCES)}."
            )

        if self._use_model2_only or source in {"sub_model_0", "sub_model_1"}:
            index = 1 if self._use_model2_only or source == "sub_model_1" else 0
            return self.sub_models[index](
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=self._extract_intermediate_tensors(intermediate_tensors, index),
                inputs_embeds=inputs_embeds,
            )

        output0 = self.sub_models[0](
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=self._extract_intermediate_tensors(intermediate_tensors, 0),
            inputs_embeds=inputs_embeds,
        )
        output1 = self.sub_models[1](
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=self._extract_intermediate_tensors(intermediate_tensors, 1),
            inputs_embeds=inputs_embeds,
        )

        output0 = self._extract_tensor_output(output0)
        output1 = self._extract_tensor_output(output1)

        if self._is_intermediate_output(output0):
            return self._pack_intermediate_tensors(output0, output1)

        return self._pack_hidden_states(output0, output1)

    def compute_logits(self, hidden_states, sampling_metadata=None) -> Optional[torch.Tensor]:
        source = getattr(self, "_joint_rollout_source", "fused")
        if self._use_model2_only or source == "sub_model_1":
            return self.sub_models[1].compute_logits(hidden_states)
        if source == "sub_model_0":
            return self.sub_models[0].compute_logits(hidden_states)
        if source != "fused":
            raise ValueError(
                f"Invalid joint rollout source {source!r}; expected one of "
                f"{sorted(self.VALID_ROLLOUT_SOURCES)}."
            )

        hidden_states0, hidden_states1 = self._unpack_hidden_states(hidden_states)
        logits0 = self.sub_models[0].compute_logits(hidden_states0)
        logits1 = self.sub_models[1].compute_logits(hidden_states1)

        if logits0 is None:
            return logits1
        if logits1 is None:
            return logits0

        return (1 - self.fusion_lambda) * logits0 + self.fusion_lambda * logits1

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        is_joint, model0_weights, model1_weights, model2_only_weights = self._split_joint_weights(weights)
        print(
            f"[WDL-SFT VERIFY] vllm_joint.load_weights: is_joint={is_joint}, "
            f"will set _use_model2_only={not is_joint}",
            flush=True,
        )

        if is_joint:
            self._use_model2_only = False
            self._joint_rollout_source = "fused"
            loaded_params = set()
            if model0_weights:
                loaded_params.update(
                    {f"sub_models.0.{name}" for name in self.sub_models[0].load_weights(model0_weights)}
                )
            if model1_weights:
                loaded_params.update(
                    {f"sub_models.1.{name}" for name in self.sub_models[1].load_weights(model1_weights)}
                )
            return loaded_params

        self._use_model2_only = True
        self._joint_rollout_source = "sub_model_1"
        return self.sub_models[1].load_weights(model2_only_weights)
