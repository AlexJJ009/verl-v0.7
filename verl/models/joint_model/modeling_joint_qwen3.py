"""QwenJointForCausalLM: Joint training model with logit fusion.

Two Qwen3ForCausalLM sub-models perform independent forward passes.
Their logits are fused: logits = (1 - λ) * logits_0 + λ * logits_1
Gradients flow to both models weighted by (1-λ) and λ respectively.

For evaluation, pass eval_only=True to get only model2's logits.
"""

import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig


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
        self.sub_models = nn.ModuleList(
            [Qwen3ForCausalLM(config) for _ in range(config.num_sub_models)]
        )
        self.fusion_lambda = config.fusion_lambda
        self.last_logit_disagreement: float | None = None

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
        **kwargs,
    ):
        # Support eval_only via model attribute (for HF generate() which can't pass custom kwargs)
        eval_only = eval_only or getattr(self, '_eval_only_mode', False)

        if eval_only:
            return self.sub_models[1](
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

        outputs_list = []
        for sub_model in self.sub_models:
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
            outputs_list.append(out)

        lam = self.fusion_lambda
        if lam == 0:
            logits = outputs_list[0].logits
        elif lam == 1:
            logits = outputs_list[1].logits
        else:
            # Avoid materializing multiple full-vocab temporaries at once.
            logits = outputs_list[0].logits.mul(1 - lam)
            logits.add_(outputs_list[1].logits, alpha=lam)

        # Compute submodel logit disagreement (mean absolute diff of softmax probs)
        # Chunked along token dim to avoid OOM on large vocab (151k) models.
        # Softmax is per-token along vocab dim, so chunking is mathematically exact.
        with torch.no_grad():
            logits0 = outputs_list[0].logits
            logits1 = outputs_list[1].logits
            orig_shape = logits0.shape  # (bs, seq, vocab)
            flat0 = logits0.reshape(-1, orig_shape[-1])
            flat1 = logits1.reshape(-1, orig_shape[-1])
            n_tokens = flat0.shape[0]
            _CHUNK = 256  # peak ≈ 256 * 151936 * 4 * 3 ≈ 440 MiB
            total_diff = 0.0
            for i in range(0, n_tokens, _CHUNK):
                p0 = torch.softmax(flat0[i:i + _CHUNK].float(), dim=-1)
                p1 = torch.softmax(flat1[i:i + _CHUNK].float(), dim=-1)
                total_diff += (p0 - p1).abs().sum().item()
            self.last_logit_disagreement = total_diff / (n_tokens * orig_shape[-1])

        loss = None
        if labels is not None:
            loss = self.sub_models[0].loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs_list[1].past_key_values,
            hidden_states=outputs_list[1].hidden_states,
            attentions=outputs_list[1].attentions,
        )

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
