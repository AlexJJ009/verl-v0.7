# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


JOINT_VLLM_MODEL_ARCHITECTURES = {
    "QwenJointForCausalLM": "verl.models.joint_model.vllm_modeling_joint_qwen3:QwenJointForCausalLM",
}
JOINT_VLLM_LAYER_INDEX_STRIDE = 1_000_000
JOINT_VLLM_LAYER_NAME_RE = re.compile(r"(?:^|\.)sub_models\.(\d+)\.(?:.*\.)?layers\.(\d+)(?:\.|$)")


def _extract_joint_vllm_layer_index(layer_name: str) -> tuple[int, int] | None:
    match = JOINT_VLLM_LAYER_NAME_RE.search(layer_name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def patch_joint_vllm_layer_indexing() -> bool:
    """Teach vLLM's layer-index helpers to distinguish both joint sub-models."""

    try:
        import vllm.model_executor.models.utils as model_utils
    except Exception:
        return False

    current_extract_layer_index = model_utils.extract_layer_index
    if getattr(current_extract_layer_index, "_verl_joint_patch", False):
        return True

    def patched_extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
        joint_layer_index = _extract_joint_vllm_layer_index(layer_name)
        if joint_layer_index is not None:
            branch_index, layer_index = joint_layer_index
            return branch_index * JOINT_VLLM_LAYER_INDEX_STRIDE + layer_index
        return current_extract_layer_index(layer_name, num_attn_module)

    patched_extract_layer_index._verl_joint_patch = True  # type: ignore[attr-defined]

    model_utils.extract_layer_index = patched_extract_layer_index

    # Patch all modules that import extract_layer_index by name (local bindings).
    # Without this, `from .utils import extract_layer_index` in qwen3.py etc.
    # keeps the original reference even after we patch the source module.
    for module_name in (
        "vllm.utils",
        "vllm.v1.utils",
        "vllm.model_executor.models.qwen3",
        "vllm.v1.worker.utils",
        "vllm.attention.utils.kv_sharing_utils",
    ):
        try:
            module = __import__(module_name, fromlist=["extract_layer_index"])
        except Exception:
            continue
        if hasattr(module, "extract_layer_index"):
            module.extract_layer_index = patched_extract_layer_index

    return True


def register_joint_vllm_model_architectures(model_registry: Any = None) -> bool:
    """Register verl joint-model vLLM architectures if vLLM is available.

    Returns True if the registry was available, even when everything was
    already registered. Returns False if vLLM could not be imported.
    """

    if model_registry is None:
        try:
            from vllm import ModelRegistry as model_registry
        except Exception:
            return False

    patch_joint_vllm_layer_indexing()

    models = getattr(model_registry, "models", {})
    for arch, model_fqcn in JOINT_VLLM_MODEL_ARCHITECTURES.items():
        if arch in models:
            continue
        model_registry.register_model(arch, model_fqcn)

    return True
