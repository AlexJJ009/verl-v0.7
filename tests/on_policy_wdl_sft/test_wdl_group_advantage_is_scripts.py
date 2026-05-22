# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Static checks for WDL group-advantage IS launch scripts."""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_DIR = REPO_ROOT / "recipe/on_policy_wdl_sft/group_advantage_is"
PLATFORM_DIR = REPO_ROOT / "platform/hope_group_advantage_is"


def _read(path: Path) -> str:
    return path.read_text()


def test_required_four_layer_files_exist():
    required = [
        FAMILY_DIR / "README.md",
        FAMILY_DIR / "_common_group_adv_is.sh",
        FAMILY_DIR / "run_1a_group_adv_is.sh",
        FAMILY_DIR / "meituan/env.sh",
        FAMILY_DIR / "meituan/jupyter.sh",
        PLATFORM_DIR / "README.md",
        PLATFORM_DIR / "jupyter.sh",
        PLATFORM_DIR / "run.hope",
    ]
    for path in required:
        assert path.exists(), path


def test_shell_scripts_parse_with_bash_n():
    scripts = [
        FAMILY_DIR / "_common_group_adv_is.sh",
        FAMILY_DIR / "run_1a_group_adv_is.sh",
        FAMILY_DIR / "meituan/env.sh",
        FAMILY_DIR / "meituan/jupyter.sh",
        PLATFORM_DIR / "jupyter.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, check=True)


def test_algorithm_defaults_and_no_beta_override():
    common = _read(FAMILY_DIR / "_common_group_adv_is.sh")
    wrapper = _read(FAMILY_DIR / "run_1a_group_adv_is.sh")
    family_text = "\n".join(_read(path) for path in FAMILY_DIR.rglob("*") if path.is_file())

    assert "loss_mode=${LOSS_MODE:-wdl_group_adv_is}" in common
    assert "loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-sum}" in common
    assert "ROLLOUT_IS=${ROLLOUT_IS:-null}" in common
    assert "ROLLOUT_RS=${ROLLOUT_RS:-null}" in common
    assert "use_kl_in_reward=${USE_KL_IN_REWARD:-False}" in common
    assert "use_kl_loss=${USE_KL_LOSS:-False}" in common
    assert "kl_loss_coef=${KL_LOSS_COEF:-0.0}" in common
    assert "NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-false}" in common
    assert "ALL_CORRECT_SFT_FALLBACK=${ALL_CORRECT_SFT_FALLBACK:-true}" in common
    assert "POS_SFT_FALLBACK_COEF=${POS_SFT_FALLBACK_COEF:-1.0}" in common
    assert "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}" in common
    assert "LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-5}" in common
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}" in common
    assert "actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS}" in common
    assert "export LOSS_MODE=${LOSS_MODE:-wdl_group_adv_is}" in wrapper
    assert "WDL_SFT_BETA" not in family_text


def test_overlong_buffer_default_is_safe_for_short_smoke():
    common = _read(FAMILY_DIR / "_common_group_adv_is.sh")

    assert 'if [ -n "${OVERLONG_BUFFER_LEN+x}" ]' in common
    assert 'exceeds MAX_RESPONSE_LENGTH=${max_response_length}' in common
    assert "overlong_buffer_len=${max_response_length}" in common
    assert "+reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}" in common


def test_parent_paths_are_overridable_in_common_script():
    common = _read(FAMILY_DIR / "_common_group_adv_is.sh")
    required_vars = [
        "REPO_ROOT",
        "DATA_ROOT",
        "TRAIN_FILE",
        "TEST_FILES",
        "MODEL_PATH",
        "BASE_CKPT_DIR",
        "LOG_DIR",
        "WANDB_DIR",
        "HF_HOME",
        "RAY_TMPDIR",
        "TMPDIR",
        "VALIDATION_OUTPUT_DIR",
        "CUSTOM_REWARD_FN_PATH",
    ]
    for var in required_vars:
        assert (
            f"{var}=${{" in common
            or f"{var}=\"${{" in common
            or f"export {var}=${{" in common
            or f"export {var}=\"${{" in common
        ), var


def test_no_dolphinfs_paths_in_local_recipe_layers():
    local_text = "\n".join(
        _read(path)
        for path in [
            FAMILY_DIR / "_common_group_adv_is.sh",
            FAMILY_DIR / "run_1a_group_adv_is.sh",
            FAMILY_DIR / "README.md",
        ]
    )
    assert "dolphinfs" not in local_text.lower()
    assert "/mnt/" not in local_text


def test_meituan_smoke_propagation_and_run_script_resolution():
    platform = _read(PLATFORM_DIR / "jupyter.sh")
    adapter = _read(FAMILY_DIR / "meituan/jupyter.sh")

    assert "export SMOKE=${SMOKE:-0}" in platform
    assert "TOTAL_TRAINING_STEPS" in platform
    assert "exec bash \"$REPO/recipe/on_policy_wdl_sft/group_advantage_is/meituan/jupyter.sh\"" in platform
    assert 'if [ "${SMOKE:-0}" = "1" ]' in adapter
    assert 'RUN_SCRIPT="${FAMILY_DIR}/run_${EXPERIMENT//-/_}.sh"' in adapter
    assert 'exec bash "$RUN_SCRIPT"' in adapter


def test_scripts_are_executable():
    for script in [
        FAMILY_DIR / "_common_group_adv_is.sh",
        FAMILY_DIR / "run_1a_group_adv_is.sh",
        FAMILY_DIR / "meituan/env.sh",
        FAMILY_DIR / "meituan/jupyter.sh",
        PLATFORM_DIR / "jupyter.sh",
    ]:
        assert os.access(script, os.X_OK), script
