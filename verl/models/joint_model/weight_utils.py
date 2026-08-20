# SPDX-License-Identifier: Apache-2.0

"""Weight utilities for joint training models.

Provides functions to:
- Detect joint model state dicts and configs
- Extract sub-model weights with clean key names
- Support both dict and generator (streaming) modes
"""

from collections import OrderedDict
from typing import Generator

import torch


def is_joint_model_state_dict(state_dict: dict) -> bool:
    """Check if a state dict belongs to a joint model (has sub_models.X. keys)."""
    return any(k.startswith("sub_models.") for k in state_dict.keys())


def is_joint_model_config(config) -> bool:
    """Check if a model config is for a joint model."""
    return getattr(config, "model_type", None) == "qwen_joint"


def extract_sub_model_weights(
    state_dict: dict[str, torch.Tensor],
    sub_model_index: int = 1,
) -> OrderedDict:
    """Extract a sub-model's weights from a joint model state dict.

    Strips the "sub_models.{index}." prefix from keys.

    Args:
        state_dict: Full joint model state dict
        sub_model_index: Which sub-model to extract (0 or 1)

    Returns:
        OrderedDict with clean key names
    """
    prefix = f"sub_models.{sub_model_index}."
    result = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            clean_key = key[len(prefix) :]
            result[clean_key] = value
    return result


def extract_sub_model_weights_iter(
    state_dict: dict[str, torch.Tensor],
    sub_model_index: int = 1,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Extract sub-model weights as a generator (for streaming to vLLM).

    Args:
        state_dict: Full joint model state dict
        sub_model_index: Which sub-model to extract (0 or 1)

    Yields:
        (clean_key, tensor) pairs
    """
    prefix = f"sub_models.{sub_model_index}."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            yield key[len(prefix) :], value
