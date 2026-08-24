# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATH = ROOT / "recipe/on_policy_wdl_sft/math_task"
WRAPPER = MATH / "run_math_qwen3_1p7b_wdl_lambda_sweep_c_p60.sh"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_lambda_sweep_c_p60.yaml"


def test_lambda_sweep_manifest_has_only_the_two_primary_c_runs() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["scope"].startswith("C-only primary lambda reconnaissance")
    assert [(run["id"], run["fusion_lambda"]) for run in manifest["runs"]] == [
        ("lambda04-c", 0.4),
        ("lambda05-c", 0.5),
    ]
    assert all(run["fusion_mode"] == "mixture" for run in manifest["runs"])
    assert all(run["freeze_model1"] is False for run in manifest["runs"])
    assert manifest["deferred_controls"] == [
        "D0 matched-scale strong-only at each lambda",
        "fixed-M1 C at each lambda",
    ]


def test_lambda_sweep_contract_matches_c_except_for_lambda_and_identity() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    contract = manifest["training_contract"]
    assert contract == {
        "beta": 0.0,
        "loss_mode": "wdl_sft",
        "kl_enabled": False,
        "rollout_source": "model2",
        "final_step": 60,
        "lr": 1.0e-6,
        "lr_warmup_steps": 0,
        "prompt_batch_size": 64,
        "rollout_n": 8,
        "max_prompt_length": 500,
        "max_response_length": 4096,
        "data_shuffle": False,
        "actor_fsdp_seed": 42,
        "actor_data_loader_seed": 42,
        "rollout_seed": 0,
        "data_seed": 20260719,
        "validation_frequency": 5,
        "save_frequency": 5,
        "validation_n": 3,
        "validation_temperature": 0.2,
        "validation_top_p": 0.95,
        "validation_views": ["model1", "model2"],
        "rollout_gpu_memory_utilization": 0.55,
    }

    wrapper = WRAPPER.read_text(encoding="utf-8")
    for pin in (
        "export TOTAL_TRAINING_STEPS=60",
        "export WDL_SFT_BETA=0.0",
        "export LOSS_MODE=wdl_sft",
        "export LR=1e-6",
        "export DATA_SEED=20260719",
        "export DATA_SHUFFLE=False",
        "export JOINT_TRAINING_ROLLOUT_SOURCE=model2",
        "export TRAIN_PROMPT_BSZ=64",
        "export ROLLOUT_N=8",
        "export MAX_RESPONSE_LENGTH=4096",
        "export VAL_N=3",
        "export ROLLOUT_GPU_MEMORY_UTILIZATION=0.55",
        "actor_rollout_ref.actor.fsdp_config.seed=42",
        "actor_rollout_ref.actor.data_loader_seed=42",
        "+actor_rollout_ref.rollout.seed=0",
    ):
        assert pin in wrapper


def test_lambda_sweep_wrapper_rejects_unregistered_lambda_before_io() -> None:
    env = os.environ | {"FUSION_LAMBDA": "0.8"}
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 64
    assert "permits only FUSION_LAMBDA=0.4 or 0.5" in result.stderr


def test_lambda_sweep_wrapper_has_lambda_tagged_run_cache_and_artifact_identity() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert 'export RUN_PREFIX="MATH-WDL-${lambda_tag^^}-ARM-C-P60-QWEN3-1P7B"' in wrapper
    assert 'export MODEL_PATH="/data-1/.cache/huggingface/math-wdl-${lambda_tag}-arm-c-p60"' in wrapper
    assert 'qwen3_1p7b_wdl_lambda_sweep/${lambda_tag}/standard-c-p60' in wrapper
    assert 'if [ "$#" -ne 0 ]' in wrapper
    assert "accepts no positional/Hydra overrides" in wrapper
