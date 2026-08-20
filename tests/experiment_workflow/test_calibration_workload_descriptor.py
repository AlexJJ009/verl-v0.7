# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "recipe/on_policy_wdl_sft/code_task/calibration_workload_descriptor.py"
    spec = importlib.util.spec_from_file_location("workload_descriptor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_qwen3_1p7b_parameter_counter_matches_pinned_config():
    module = load()
    config = Path(
        "/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json"
    )
    result = module.qwen3_parameter_count(config)
    assert result["version"] == "hf_qwen3_config_parameter_count_v1"
    assert result["total"] == 1_720_567_808
    assert result["components"]["lm_head"] == 0


def test_artifact_hash_accepts_hf_blob_symlink_and_rejects_escape(tmp_path):
    module = load()
    cache = tmp_path / "models--org--model"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots/rev"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "abc").write_bytes(b"payload")
    (snapshot / "config.json").symlink_to("../../blobs/abc")
    assert len(module.artifact_sha256(snapshot)) == 64
    (snapshot / "bad").symlink_to("../../../../outside")
    with pytest.raises((ValueError, FileNotFoundError)):
        module.artifact_sha256(snapshot)


def test_qwen3_counter_rejects_attention_shape_mismatch(tmp_path):
    module = load()
    config = {
        "model_type": "qwen3",
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "tie_word_embeddings": True,
        "attention_bias": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="do not span"):
        module.qwen3_parameter_count(path)


def test_manifest_schema_requires_outcome_schema_v2():
    schema = json.loads((ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/schema.json").read_text())
    assert "calibration_workloads" in schema["required"]
    assert schema["$defs"]["calibrationWorkload"]["properties"]["outcome_schema_version"] == {"const": 2}
    eligibility = schema["$defs"]["calibrationWorkload"]["properties"]["validation_eligibility"]
    assert eligibility["properties"]["max_prompt_length"] == {"const": 1024}
    assert eligibility["properties"]["filter_enabled"] == {"const": True}


def test_real_manifest_has_strict_workload_descriptors():
    manifest = yaml.safe_load((ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml").read_text())
    workloads = manifest["calibration_workloads"]
    assert list(workloads) == ["stage1", "stage2", "stage3"]
    assert [item["role"] for item in workloads["stage2"]["model_sources"]] == ["model1", "model2"]
    assert workloads["stage2"]["rollout_model_parameter_count_sum"] == sum(
        workloads["stage2"]["rollout_model_parameter_counts"]
    )
    assert all(item["outcome_schema_version"] == 2 for item in workloads.values())
    eligibility = [item["validation_eligibility"] for item in workloads.values()]
    assert eligibility[0] == eligibility[1] == eligibility[2]
    assert eligibility[0]["submitted_prompt_count"] == sum(eligibility[0]["per_dataset_eligible_counts"].values())
    assert sum(row["row_count"] for row in workloads["stage1"]["datasets"]) == 1422
