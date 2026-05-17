from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import (
    JOINT_ROLLOUT_SOURCE_TO_ID,
    RayPPOTrainer,
    build_joint_rollout_config,
    select_joint_rollout_output,
)


def _config(custom=None, bypass_mode=False):
    cfg = {
        "actor_rollout_ref": {
            "rollout": {},
        },
        "algorithm": {
            "rollout_correction": {
                "bypass_mode": bypass_mode,
            }
        },
    }
    if custom is not None:
        cfg["actor_rollout_ref"]["rollout"]["custom"] = custom
    return OmegaConf.create(cfg)


def test_joint_rollout_config_absent_preserves_fused_default():
    normalized = build_joint_rollout_config(_config())

    assert normalized == {
        "enabled": False,
        "sources": ["fused"],
        "select": "fused",
        "train_on_selected_only": True,
    }


def test_joint_rollout_config_accepts_dual_model2_selected_mode():
    normalized = build_joint_rollout_config(
        _config(
            {
                "joint_rollout_sources": ["sub_model_0", "sub_model_1"],
                "joint_rollout_select": "sub_model_1",
                "joint_rollout_train_on_selected_only": True,
            }
        )
    )

    assert normalized["enabled"] is True
    assert normalized["sources"] == ["sub_model_0", "sub_model_1"]
    assert normalized["select"] == "sub_model_1"
    assert normalized["train_on_selected_only"] is True


@pytest.mark.parametrize(
    "custom,error_substr",
    [
        ({"joint_rollout_sources": []}, "non-empty"),
        (
            {"joint_rollout_sources": ["model2"], "joint_rollout_select": "model2"},
            "Invalid joint rollout source",
        ),
        (
            {"joint_rollout_sources": ["sub_model_0"], "joint_rollout_select": "sub_model_1"},
            "must be one of",
        ),
        (
            {
                "joint_rollout_sources": ["sub_model_0", "sub_model_1"],
                "joint_rollout_select": "sub_model_1",
                "joint_rollout_train_on_selected_only": False,
            },
            "unsupported",
        ),
    ],
)
def test_joint_rollout_config_fails_fast_for_invalid_combinations(custom, error_substr):
    with pytest.raises(ValueError, match=error_substr):
        build_joint_rollout_config(_config(custom))


def test_joint_rollout_config_rejects_bypass_old_log_prob_mode():
    with pytest.raises(ValueError, match="old_log_probs under the fused training policy"):
        build_joint_rollout_config(
            _config(
                {
                    "joint_rollout_sources": ["sub_model_0", "sub_model_1"],
                    "joint_rollout_select": "sub_model_1",
                    "joint_rollout_train_on_selected_only": True,
                },
                bypass_mode=True,
            )
        )


def test_select_joint_rollout_output_returns_selected_only():
    model1_output = SimpleNamespace(name="model1")
    model2_output = SimpleNamespace(name="model2")

    selected_source, selected_output = select_joint_rollout_output(
        {
            "sub_model_0": model1_output,
            "sub_model_1": model2_output,
        },
        {
            "enabled": True,
            "sources": ["sub_model_0", "sub_model_1"],
            "select": "sub_model_1",
            "train_on_selected_only": True,
        },
    )

    assert selected_source == "sub_model_1"
    assert selected_output is model2_output
    assert selected_output is not model1_output


def test_select_joint_rollout_output_fails_if_selected_source_was_not_generated():
    with pytest.raises(ValueError, match="was not generated"):
        select_joint_rollout_output(
            {"sub_model_0": SimpleNamespace(name="model1")},
            {
                "enabled": True,
                "sources": ["sub_model_0", "sub_model_1"],
                "select": "sub_model_1",
                "train_on_selected_only": True,
            },
        )


class _FakeRepeatedBatch:
    def __init__(self, source):
        self.source = source


class _FakeGenBatch:
    def __init__(self, trainer):
        self.trainer = trainer
        self.repeat_calls = []

    def repeat(self, repeat_times, interleave):
        self.repeat_calls.append((self.trainer.current_source, repeat_times, interleave))
        return _FakeRepeatedBatch(self.trainer.current_source)


class _FakeRolloutManager:
    def __init__(self):
        self.generated_sources = []

    def generate_sequences(self, repeated_batch):
        self.generated_sources.append(repeated_batch.source)
        return SimpleNamespace(source=repeated_batch.source, meta_info={"timing": {}})


class _FakeCheckpointManager:
    def __init__(self):
        self.sleep_calls = 0

    def sleep_replicas(self):
        self.sleep_calls += 1


def _make_trainer_for_generation(joint_rollout_config):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._joint_rollout_config = joint_rollout_config
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(n=8),
        )
    )
    trainer.async_rollout_manager = _FakeRolloutManager()
    trainer.checkpoint_manager = _FakeCheckpointManager()
    trainer.global_steps = 1
    trainer.current_source = "fused"

    def _set_source(source):
        trainer.current_source = source

    trainer._set_joint_rollout_source = _set_source
    return trainer


def test_trainer_dual_generation_calls_each_source_restores_fused_and_selects_model2():
    trainer = _make_trainer_for_generation(
        {
            "enabled": True,
            "sources": ["sub_model_0", "sub_model_1"],
            "select": "sub_model_1",
            "train_on_selected_only": True,
        }
    )
    gen_batch = _FakeGenBatch(trainer)
    metrics = {}

    selected_source, selected_output, outputs_by_source = trainer._generate_training_rollouts(gen_batch, 2, metrics)

    assert trainer.async_rollout_manager.generated_sources == ["sub_model_0", "sub_model_1"]
    assert gen_batch.repeat_calls == [
        ("sub_model_0", 8, True),
        ("sub_model_1", 8, True),
    ]
    assert trainer.current_source == "fused"
    assert selected_source == "sub_model_1"
    assert selected_output is outputs_by_source["sub_model_1"]
    assert selected_output is not outputs_by_source["sub_model_0"]
    assert trainer.checkpoint_manager.sleep_calls == 2
    assert metrics["dual_rollout/selected_source"] == JOINT_ROLLOUT_SOURCE_TO_ID["sub_model_1"]
    assert metrics["dual_rollout/source_count"] == 2
    assert metrics["dual_rollout/prompt_batch_size"] == 2


def test_trainer_no_dual_config_generates_once_without_source_switching():
    trainer = _make_trainer_for_generation(
        {
            "enabled": False,
            "sources": ["fused"],
            "select": "fused",
            "train_on_selected_only": True,
        }
    )
    gen_batch = _FakeGenBatch(trainer)
    metrics = {}

    selected_source, selected_output, outputs_by_source = trainer._generate_training_rollouts(gen_batch, 2, metrics)

    assert trainer.async_rollout_manager.generated_sources == ["fused"]
    assert gen_batch.repeat_calls == [("fused", 8, True)]
    assert trainer.current_source == "fused"
    assert selected_source == "fused"
    assert selected_output is outputs_by_source["fused"]
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert metrics == {}


def test_dual_rollout_recipe_scripts_are_portable_and_opt_in():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_dir = repo_root / "recipe" / "on_policy_wdl_sft" / "dual_submodel_rollout"
    common = recipe_dir / "_common_dual_rollout.sh"
    run_3a = recipe_dir / "run_3a_model2_rollout_beta0.sh"
    run_3b = recipe_dir / "run_3b_model2_rollout_beta01.sh"

    for path in [common, run_3a, run_3b, recipe_dir / "README.md"]:
        assert path.exists()

    common_text = common.read_text()
    run_3a_text = run_3a.read_text()
    run_3b_text = run_3b.read_text()

    assert "joint_rollout_sources=\"[sub_model_0,sub_model_1]\"" in common_text
    assert "joint_rollout_select=sub_model_1" in common_text
    assert "joint_rollout_train_on_selected_only=true" in common_text
    assert 'REPO_ROOT="${REPO_ROOT:-$(cd "${RECIPE_ROOT}/../.." && pwd)}"' in common_text
    assert "DATA_ROOT=${DATA_ROOT:-/data-1/dataset}" in common_text
    assert "TRAIN_FILE=${TRAIN_FILE:-\"${DATA_ROOT}/EnsembleLLM-data-processed/train_rl_format.parquet\"}" in common_text
    assert "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}" in common_text
    assert "override_config.attn_implementation=flash_attention_2" in common_text
    assert "MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-160}" in common_text
    assert "MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-1}" in common_text
    assert "MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-1}" in common_text
    assert "KEEP_BEST_CKPT=${KEEP_BEST_CKPT:-True}" in common_text
    assert "BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-\"val-core/HuggingFaceH4/MATH-500/acc/mean@1\"}" in common_text
    assert "VAL_BEFORE_TRAIN" in common_text
    assert "/mnt/dolphinfs" not in common_text

    assert 'WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}' in run_3a_text
    assert 'WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}' in run_3b_text
    assert 'source "${WRAPPER_SCRIPT_DIR}/_common_dual_rollout.sh" "$@"' in run_3a_text
    assert 'source "${WRAPPER_SCRIPT_DIR}/_common_dual_rollout.sh" "$@"' in run_3b_text
