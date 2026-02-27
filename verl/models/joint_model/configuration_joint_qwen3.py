"""Configuration class for QwenJointForCausalLM.

Extends Qwen3Config with joint training specific parameters:
- fusion_lambda: weight for logit fusion
- freeze_model1: whether to freeze the first sub-model
- num_sub_models: number of sub-models (always 2 for now)
"""

from transformers import Qwen3Config


class QwenJointConfig(Qwen3Config):
    model_type = "qwen_joint"

    def __init__(
        self,
        fusion_lambda: float = 0.5,
        freeze_model1: bool = False,
        num_sub_models: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fusion_lambda = fusion_lambda
        self.freeze_model1 = freeze_model1
        self.num_sub_models = num_sub_models
