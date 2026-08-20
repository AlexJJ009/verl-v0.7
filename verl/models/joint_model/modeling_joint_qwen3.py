"""QwenJointForCausalLM: Joint training model with logit fusion.

Two Qwen3ForCausalLM sub-models perform independent forward passes.
In the default ``mixture`` mode their logits are fused as
``(1 - λ) * logits_0 + λ * logits_1``.  ``strong_scaled`` is the
matched-scale no-weak control ``λ * logits_1`` used by the Math WDL causal
experiment; it intentionally leaves model1 out of the active objective.

For evaluation, pass eval_only=True to get only model2's logits.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig


@dataclass
class QwenJointCausalLMOutputWithPast(CausalLMOutputWithPast):
    """CausalLM output that explicitly carries per-submodel logits when requested."""

    submodel_logits: Optional[tuple[torch.FloatTensor, ...]] = None


class QwenJointForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = QwenJointConfig
    base_model_prefix = "sub_models"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: QwenJointConfig):
        super().__init__(config)
        self.sub_models = nn.ModuleList([Qwen3ForCausalLM(config) for _ in range(config.num_sub_models)])
        self.fusion_lambda = config.fusion_lambda
        self.fusion_mode = getattr(config, "fusion_mode", "mixture")
        self.last_logit_disagreement: float | None = None
        self.last_logit_diagnostics: dict[str, float] | None = None

        if config.freeze_model1:
            for param in self.sub_models[0].parameters():
                param.requires_grad = False

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        eval_only=False,
        return_submodel_logits=False,
        **kwargs,
    ):
        # Support eval_only via model attribute (for HF generate() which can't pass custom kwargs)
        eval_only = eval_only or getattr(self, "_eval_only_mode", False)

        if eval_only:
            submodel_index = getattr(self, "_eval_submodel_index", 1)
            return self.sub_models[submodel_index](
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs_list = [None] * len(self.sub_models)
        for submodel_index in range(len(self.sub_models)):
            sub_model = self.sub_models[submodel_index]
            out = sub_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            outputs_list[submodel_index] = out

        lam = self.fusion_lambda
        if self.fusion_mode == "strong_scaled":
            logits = outputs_list[1].logits * lam
        elif lam == 0:
            logits = outputs_list[0].logits
        elif lam == 1:
            logits = outputs_list[1].logits
        else:
            # Avoid materializing multiple full-vocab temporaries at once.
            logits = outputs_list[0].logits * (1 - lam)
            logits.add_(outputs_list[1].logits, alpha=lam)

        # Compute the expensive full-vocabulary diagnostics only when callers
        # explicitly request submodel logits. Ordinary forwards avoid the
        # second pair of softmaxes.
        self.last_logit_disagreement = None
        self.last_logit_diagnostics = None
        if return_submodel_logits:
            with torch.no_grad():
                logits0 = outputs_list[0].logits
                logits1 = outputs_list[1].logits
                orig_shape = logits0.shape  # (bs, seq, vocab)
                flat0 = logits0.reshape(-1, orig_shape[-1])
                flat1 = logits1.reshape(-1, orig_shape[-1])
                # A deterministic token sample keeps these diagnostics useful
                # without adding hundreds of MiB to every training microbatch.
                max_diagnostic_tokens = 8
                if flat0.shape[0] > max_diagnostic_tokens:
                    token_indices = (
                        torch.linspace(
                            0,
                            flat0.shape[0] - 1,
                            steps=max_diagnostic_tokens,
                            device=flat0.device,
                        )
                        .round()
                        .long()
                    )
                    flat0 = flat0.index_select(0, token_indices)
                    flat1 = flat1.index_select(0, token_indices)
                n_tokens = flat0.shape[0]
                _CHUNK = 8
                total_l1 = 0.0
                total_js = 0.0
                top1_equal = 0
                for i in range(0, n_tokens, _CHUNK):
                    p0 = torch.softmax(flat0[i : i + _CHUNK].float(), dim=-1)
                    p1 = torch.softmax(flat1[i : i + _CHUNK].float(), dim=-1)
                    midpoint = 0.5 * (p0 + p1)
                    total_l1 += (p0 - p1).abs().sum().item()
                    total_js += (
                        (
                            0.5
                            * torch.sum(
                                p0 * (torch.log(p0.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))),
                                dim=-1,
                            )
                            + 0.5
                            * torch.sum(
                                p1 * (torch.log(p1.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))),
                                dim=-1,
                            )
                        )
                        .sum()
                        .item()
                    )
                    top1_equal += (p0.argmax(dim=-1) == p1.argmax(dim=-1)).sum().item()
                self.last_logit_disagreement = total_l1 / (n_tokens * orig_shape[-1])
                self.last_logit_diagnostics = {
                    "submodel_total_variation": 0.5 * total_l1 / n_tokens,
                    "submodel_js_divergence": total_js / n_tokens,
                    "submodel_top1_agreement": top1_equal / n_tokens,
                }

        loss = None
        if labels is not None:
            loss = self.sub_models[0].loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        result = QwenJointCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs_list[1].past_key_values,
            hidden_states=outputs_list[1].hidden_states,
            attentions=outputs_list[1].attentions,
            submodel_logits=(
                tuple(out.logits if out is not None else None for out in outputs_list)
                if return_submodel_logits
                else None
            ),
        )
        return result

    def get_input_embeddings(self):
        return self.sub_models[0].get_input_embeddings()

    def set_input_embeddings(self, value):
        for sub_model in self.sub_models:
            sub_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.sub_models[0].get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        for sub_model in self.sub_models:
            sub_model.set_output_embeddings(new_embeddings)
