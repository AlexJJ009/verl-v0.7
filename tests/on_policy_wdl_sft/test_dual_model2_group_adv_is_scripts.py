from pathlib import Path


def test_4a_recipe_uses_existing_dual_rollout_directory_and_required_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_dir = repo_root / "recipe" / "on_policy_wdl_sft" / "dual_submodel_rollout"
    common = recipe_dir / "_common_dual_rollout.sh"
    run_4a = recipe_dir / "run_4a_model2_group_adv_is.sh"
    meituan_env = recipe_dir / "meituan" / "env.sh"
    meituan_jupyter = recipe_dir / "meituan" / "jupyter.sh"
    readme = recipe_dir / "README.md"
    platform_jupyter = repo_root / "platform" / "hope_on_policy_wdl_sft" / "jupyter.sh"

    assert common.exists()
    assert run_4a.exists()
    assert meituan_env.exists()
    assert meituan_jupyter.exists()
    assert readme.exists()

    common_text = common.read_text()
    run_4a_text = run_4a.read_text()
    meituan_env_text = meituan_env.read_text()
    meituan_jupyter_text = meituan_jupyter.read_text()
    readme_text = readme.read_text()
    platform_jupyter_text = platform_jupyter.read_text()

    assert "actor_rollout_ref.rollout.calculate_log_probs=True" in common_text
    assert 'HF_HOME:-}" = "/root/.cache/huggingface"' in common_text
    assert "algorithm.norm_adv_by_std_in_grpo=False" in common_text
    assert "algorithm.rollout_correction.rollout_is=${rollout_is}" in common_text
    assert "actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}" in common_text
    assert "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}" in common_text
    assert "CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}" in common_text
    assert "actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY}" in common_text
    assert "+actor_rollout_ref.actor.policy_loss.gamma_pos_sft=${gamma_pos_sft}" in common_text
    assert "+actor_rollout_ref.actor.policy_loss.tis_threshold=${tis_threshold}" in common_text
    assert "actor_rollout_ref.rollout.val_kwargs.n=3" in common_text
    assert "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP}" in common_text
    assert "trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPTS_TO_KEEP}" in common_text
    assert "+trainer.keep_best_ckpt=${KEEP_BEST_CKPT}" in common_text
    assert '+trainer.best_ckpt_metric_key="${BEST_CKPT_METRIC_KEY}"' in common_text

    assert 'LOSS_MODE=${LOSS_MODE:-dual_model2_group_adv_is}' in run_4a_text
    assert 'JOINT_ROLLOUT_SOURCES=${JOINT_ROLLOUT_SOURCES:-"[sub_model_1]"}' in run_4a_text
    assert "JOINT_ROLLOUT_SELECT=${JOINT_ROLLOUT_SELECT:-sub_model_1}" in run_4a_text
    assert 'TRAIN_FILE=${TRAIN_FILE:-"/data-1/dataset/math/train_rl_format.parquet"}' in run_4a_text
    assert "LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-sum}" in run_4a_text
    assert "GAMMA_POS_SFT=${GAMMA_POS_SFT:-1.0}" in run_4a_text
    assert "TIS_THRESHOLD=${TIS_THRESHOLD:-5.0}" in run_4a_text
    assert "ROLLOUT_IS=${ROLLOUT_IS:-null}" in run_4a_text
    assert "rollout_rs=${ROLLOUT_RS:-null}" in common_text
    assert "USE_KL_LOSS=${USE_KL_LOSS:-False}" in run_4a_text
    assert "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-115}" in run_4a_text
    assert "TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}" in run_4a_text
    assert 'source "${WRAPPER_SCRIPT_DIR}/_common_dual_rollout.sh" "$@"' in run_4a_text
    assert "TRAIN_FILE=${TRAIN_FILE:-$LGX/verl-exp/data/math/train_rl_format.parquet}" in meituan_env_text
    assert "BEST_CKPT_METRIC_KEY=${BEST_CKPT_METRIC_KEY:-val-core/HuggingFaceH4/MATH-500/acc/mean@3}" in meituan_env_text
    assert "run_4a_model2_group_adv_is.sh" in meituan_jupyter_text
    assert "dual_submodel_rollout/meituan/jupyter.sh" in platform_jupyter_text
    assert "4a|4a-dual|4a-model2-group-adv-is" in platform_jupyter_text

    assert "Generate responses only from `sub_model_1`" in readme_text
    assert "/data-1/dataset/math/train_rl_format.parquet" in readme_text
    assert "TOTAL_TRAINING_STEPS=115" in readme_text
    assert "TOTAL_EPOCHS=1" in readme_text
    assert "tmux new-session -s dual_model2_group_adv_is_learning_smoke" in readme_text
    assert "run_4a_model2_group_adv_is.sh" in readme_text
