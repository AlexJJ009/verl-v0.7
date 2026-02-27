"""Prepare joint model weights from a single base model.

Takes a base Qwen3 model and creates a joint model directory with:
- Two copies of the weights under sub_models.0.* and sub_models.1.*
- A config.json with auto_map pointing to our custom classes
- Tokenizer files copied from the base model

Usage:
    python -m verl.models.joint_model.prepare_joint_weights \
        --base_model_path Qwen/Qwen3-1.7B-Base \
        --output_path .cache/huggingface/QwenJoint-1.7B \
        --fusion_lambda 0.5
"""

import argparse
import json
import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def prepare_joint_weights(
    base_model_path: str,
    output_path: str,
    fusion_lambda: float = 0.5,
    freeze_model1: bool = False,
):
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading base model from {base_model_path}...")
    base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_state_dict = base_model.state_dict()

    print("Creating joint model state dict...")
    joint_state_dict = {}
    for key, value in base_state_dict.items():
        joint_state_dict[f"sub_models.0.{key}"] = value.clone()
        joint_state_dict[f"sub_models.1.{key}"] = value.clone()

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
    config_dict["fusion_lambda"] = fusion_lambda
    config_dict["freeze_model1"] = freeze_model1
    config_dict["num_sub_models"] = 2

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

    del base_model, base_state_dict, joint_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Joint model saved to {output_path}")
    print(f"  fusion_lambda: {fusion_lambda}")
    print(f"  freeze_model1: {freeze_model1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare joint model weights")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output_path", type=str, default=".cache/huggingface/QwenJoint-1.7B")
    parser.add_argument("--fusion_lambda", type=float, default=0.5)
    parser.add_argument("--freeze_model1", action="store_true")
    args = parser.parse_args()

    prepare_joint_weights(
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        fusion_lambda=args.fusion_lambda,
        freeze_model1=args.freeze_model1,
    )
