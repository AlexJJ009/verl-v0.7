"""Prepare joint model weights from one or two base models.

Mode 1 (single model): Both sub-models share the same initial weights.
Mode 2 (two models):   sub_models.0 gets model1 weights, sub_models.1 gets
                        model2 weights.  Both models must be from the same
                        architecture family (same hidden_size, vocab_size, etc.).

Usage:
    # Single model (original behavior)
    python -m verl.models.joint_model.prepare_joint_weights \
        --base_model_path Qwen/Qwen3-1.7B-Base \
        --output_path .cache/huggingface/QwenJoint-1.7B \
        --fusion_lambda 0.5

    # Two different models
    python -m verl.models.joint_model.prepare_joint_weights \
        --base_model_path  /data-1/.cache/huggingface/Qwen3-1.7B-Base \
        --model2_path      /data-1/checkpoints/Qwen3-1.7B-Base-finetuned-merged \
        --output_path      .cache/huggingface/QwenJoint-1.7B-dual \
        --fusion_lambda 0.55
"""

import argparse
import gc
import json
import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _validate_config_compatibility(config1, config2):
    """Check that two configs are architecturally compatible for joint training."""
    critical_fields = [
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "vocab_size",
    ]
    mismatches = []
    for field in critical_fields:
        v1 = getattr(config1, field, None)
        v2 = getattr(config2, field, None)
        if v1 != v2:
            mismatches.append(f"  {field}: model1={v1}, model2={v2}")
    if mismatches:
        raise ValueError("Model architectures are incompatible for joint training:\n" + "\n".join(mismatches))


def _format_model_source(model_path: str) -> str:
    """Preserve HF repo ids while normalizing local filesystem paths."""
    return os.path.abspath(model_path) if os.path.exists(model_path) else model_path


def prepare_joint_weights(
    base_model_path: str,
    output_path: str,
    fusion_lambda: float = 0.5,
    fusion_mode: str = "mixture",
    freeze_model1: bool = False,
    model2_path: str | None = None,
):
    os.makedirs(output_path, exist_ok=True)

    # --- Load model 1 → immediately prefix into joint dict, then free ---
    print(f"Loading model1 from {base_model_path}...")
    base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model1_state_dict = base_model.state_dict()
    del base_model
    gc.collect()

    print("Creating joint model state dict (sub_models.0)...")
    joint_state_dict = {}
    for key, value in model1_state_dict.items():
        # Clone to break tied-weight memory sharing (e.g. embed_tokens ↔ lm_head)
        # which safetensors cannot serialize.
        joint_state_dict[f"sub_models.0.{key}"] = value.clone()

    if model2_path is None:
        # Clone mode: duplicate model1 into sub_models.1
        print("Using single model (cloned) for both sub-models")
        for key, value in model1_state_dict.items():
            joint_state_dict[f"sub_models.1.{key}"] = value.clone()
    del model1_state_dict
    gc.collect()

    # --- Load model 2 (only if dual-model mode) ---
    if model2_path is not None:
        print(f"Loading model2 from {model2_path}...")
        model2_config = AutoConfig.from_pretrained(model2_path, trust_remote_code=True)
        _validate_config_compatibility(base_config, model2_config)
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model2_state_dict = model2.state_dict()
        del model2
        gc.collect()
        print("Adding sub_models.1 weights...")
        for key, value in model2_state_dict.items():
            joint_state_dict[f"sub_models.1.{key}"] = value.clone()
        del model2_state_dict
        gc.collect()
        print("Using two different models for sub_models.0 and sub_models.1")

    weight_path = os.path.join(output_path, "model.safetensors")
    print(f"Saving weights to {weight_path}...")
    try:
        from safetensors.torch import save_file

        save_file(joint_state_dict, weight_path)
    except ImportError:
        weight_path = os.path.join(output_path, "pytorch_model.bin")
        torch.save(joint_state_dict, weight_path)

    print("Creating config.json...")
    config_dict = base_config.to_dict()
    config_dict["model_type"] = "qwen_joint"
    config_dict["architectures"] = ["QwenJointForCausalLM"]
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_joint_qwen3.QwenJointConfig",
        "AutoModelForCausalLM": "modeling_joint_qwen3.QwenJointForCausalLM",
    }
    if fusion_mode not in {"mixture", "strong_scaled"}:
        raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
    config_dict["fusion_lambda"] = fusion_lambda
    config_dict["fusion_mode"] = fusion_mode
    config_dict["freeze_model1"] = freeze_model1
    config_dict["num_sub_models"] = 2
    config_dict["joint_model_sources"] = {
        "model1": _format_model_source(base_model_path),
        "model2": _format_model_source(model2_path) if model2_path else _format_model_source(base_model_path),
        "mode": "dual" if model2_path else "cloned",
    }

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Copying model source files...")
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ["modeling_joint_qwen3.py", "configuration_joint_qwen3.py"]:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(output_path, fname)
        shutil.copy2(src, dst)

    print("Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    del joint_state_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Joint model saved to {output_path}")
    print(f"  model1: {base_model_path}")
    print(f"  model2: {model2_path or base_model_path} {'(cloned)' if model2_path is None else ''}")
    print(f"  fusion_lambda: {fusion_lambda}")
    print(f"  fusion_mode: {fusion_mode}")
    print(f"  freeze_model1: {freeze_model1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare joint model weights")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--model2_path",
        type=str,
        default=None,
        help="Path to a second model for sub_models.1. "
        "Must be the same architecture as base_model_path. "
        "If omitted, base_model_path is cloned to both sub-models.",
    )
    parser.add_argument("--output_path", type=str, default=".cache/huggingface/QwenJoint-1.7B")
    parser.add_argument("--fusion_lambda", type=float, default=0.5)
    parser.add_argument("--fusion_mode", choices=("mixture", "strong_scaled"), default="mixture")
    parser.add_argument("--freeze_model1", action="store_true")
    args = parser.parse_args()

    prepare_joint_weights(
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        fusion_lambda=args.fusion_lambda,
        fusion_mode=args.fusion_mode,
        freeze_model1=args.freeze_model1,
        model2_path=args.model2_path,
    )
