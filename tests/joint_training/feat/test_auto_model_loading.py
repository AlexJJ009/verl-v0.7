"""Tests for AutoModel loading with trust_remote_code.

Verifies the joint model can be loaded through HuggingFace's AutoModel
mechanism, which is how verl loads models in fsdp_workers.py.
"""

import json
import os
import shutil
import tempfile

import pytest
import torch

from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM


class TestAutoModelLoading:
    """Test loading via AutoModelForCausalLM with trust_remote_code."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a temporary directory with model files for AutoModel loading."""
        config = QwenJointConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            fusion_lambda=0.5,
            freeze_model1=False,
        )

        # Create model and save weights
        model = QwenJointForCausalLM(config)
        state_dict = model.state_dict()

        model_dir = tmp_path / "test_joint_model"
        model_dir.mkdir()

        # Save config.json
        config_dict = config.to_dict()
        config_dict["architectures"] = ["QwenJointForCausalLM"]
        config_dict["auto_map"] = {
            "AutoConfig": "configuration_joint_qwen3.QwenJointConfig",
            "AutoModelForCausalLM": "modeling_joint_qwen3.QwenJointForCausalLM",
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_dict, f)

        # Save weights
        torch.save(state_dict, model_dir / "pytorch_model.bin")

        # Copy model source files
        src_dir = os.path.dirname(os.path.abspath(__file__))
        joint_model_dir = os.path.join(os.path.dirname(src_dir), os.pardir, os.pardir, "verl", "models", "joint_model")
        joint_model_dir = os.path.normpath(joint_model_dir)

        for fname in ["modeling_joint_qwen3.py", "configuration_joint_qwen3.py"]:
            src = os.path.join(joint_model_dir, fname)
            shutil.copy2(src, model_dir / fname)

        return str(model_dir)

    def test_auto_model_from_pretrained(self, model_dir):
        """AutoModelForCausalLM.from_pretrained should load joint model."""
        from transformers import AutoModelForCausalLM

        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        assert type(loaded_model).__name__ == "QwenJointForCausalLM"
        assert hasattr(loaded_model, "sub_models")
        assert len(loaded_model.sub_models) == 2

    def test_auto_config_from_pretrained(self, model_dir):
        """AutoConfig.from_pretrained should load joint config."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        assert type(config).__name__ == "QwenJointConfig"
        assert config.fusion_lambda == 0.5
        assert config.model_type == "qwen_joint"

    def test_loaded_model_produces_correct_output(self, model_dir):
        """Loaded model should produce fused logits."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert output.logits.shape == (1, 4, 1000)

    def test_loaded_model_state_dict_roundtrip(self, model_dir):
        """Save → load → forward should produce identical results."""
        from transformers import AutoModelForCausalLM

        # Load model
        model1 = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        model1.eval()

        # Save to new dir and reload
        with tempfile.TemporaryDirectory() as tmp_dir:
            model1.save_pretrained(tmp_dir)

            # Copy source files for trust_remote_code
            src_dir = os.path.dirname(os.path.abspath(__file__))
            joint_model_dir = os.path.normpath(
                os.path.join(os.path.dirname(src_dir), os.pardir, os.pardir, "verl", "models", "joint_model")
            )
            for fname in ["modeling_joint_qwen3.py", "configuration_joint_qwen3.py"]:
                shutil.copy2(os.path.join(joint_model_dir, fname), os.path.join(tmp_dir, fname))

            model2 = AutoModelForCausalLM.from_pretrained(tmp_dir, trust_remote_code=True)
            model2.eval()

        # Compare outputs
        input_ids = torch.randint(0, 1000, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        with torch.no_grad():
            out1 = model1(input_ids=input_ids, attention_mask=attention_mask)
            out2 = model2(input_ids=input_ids, attention_mask=attention_mask)

        torch.testing.assert_close(out1.logits, out2.logits)
