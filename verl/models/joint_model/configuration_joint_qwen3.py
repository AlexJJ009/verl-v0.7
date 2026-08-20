# SPDX-License-Identifier: Apache-2.0

"""Configuration class for QwenJointForCausalLM.

Extends Qwen3Config with joint training specific parameters:
- fusion_lambda: weight for logit fusion
- fusion_mode: ``mixture`` or the matched-scale ``strong_scaled`` control
- freeze_model1: whether to freeze the first sub-model
- num_sub_models: number of sub-models (always 2 for now)
"""

from transformers import Qwen3Config


class QwenJointConfig(Qwen3Config):
    model_type = "qwen_joint"

    def __init__(
        self,
        fusion_lambda: float = 0.5,
        fusion_mode: str = "mixture",
        freeze_model1: bool = False,
        num_sub_models: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= fusion_lambda <= 1.0:
            raise ValueError(f"fusion_lambda must be in [0, 1], got {fusion_lambda}")
        if fusion_mode not in {"mixture", "strong_scaled"}:
            raise ValueError(f"fusion_mode must be 'mixture' or 'strong_scaled', got {fusion_mode!r}")
        self.fusion_lambda = fusion_lambda
        self.fusion_mode = fusion_mode
        self.freeze_model1 = freeze_model1
        self.num_sub_models = num_sub_models
