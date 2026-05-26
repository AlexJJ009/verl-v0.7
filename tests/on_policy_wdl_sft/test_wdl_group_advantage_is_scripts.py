from pathlib import Path


def test_single_model_group_adv_is_math_scripts_and_queue_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    ablation_dir = repo_root / "recipe" / "on_policy_wdl_sft" / "ablation_single_model"
    dual_dir = repo_root / "recipe" / "on_policy_wdl_sft" / "dual_submodel_rollout"
    common = (ablation_dir / "_common_ablation.sh").read_text()
    meituan_env = (ablation_dir / "meituan" / "env.sh").read_text()
    meituan_jupyter = (ablation_dir / "meituan" / "jupyter.sh").read_text()
    run_4b = (ablation_dir / "run_4b_math_base.sh").read_text()
    run_4c = (ablation_dir / "run_4c_math_sft.sh").read_text()
    monitor = (dual_dir / "monitor_4abc_math_queue.sh").read_text()
    generic_monitor = (repo_root / "scripts" / "training_queue_monitor.sh").read_text()
    platform_jupyter = (repo_root / "platform" / "hope_on_policy_wdl_sft" / "jupyter.sh").read_text()
    submit_batch = (repo_root / "platform" / "hope_on_policy_wdl_sft" / "submit_batch.sh").read_text()

    assert "rollout_is=${ROLLOUT_IS:-null}" in common
    assert "actor_rollout_ref.rollout.val_kwargs.n=${val_n}" in common
    assert "CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}" in common
    assert "actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY}" in common
    assert "+actor_rollout_ref.actor.policy_loss.all_correct_sft_fallback=${ALL_CORRECT_SFT_FALLBACK}" in common
    assert "+actor_rollout_ref.actor.policy_loss.pos_sft_fallback_coef=${POS_SFT_FALLBACK_COEF}" in common
    assert "MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$LGX/verl-exp/data/math/train_rl_format.parquet}" in meituan_env
    assert "MATH_TRAIN_FILE not found" in meituan_jupyter
    assert "4b-math-base" in meituan_jupyter
    assert "4c-math-sft" in meituan_jupyter
    assert "4b-math-base|4c-math-sft" in platform_jupyter
    assert "FOUR_ABC_EXPERIMENTS=(4a 4b-math-base 4c-math-sft)" in submit_batch

    for text, prefix in [
        (run_4b, "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1"),
        (run_4c, "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1"),
    ]:
        assert prefix in text
        assert 'LOSS_MODE=${LOSS_MODE:-"wdl_group_adv_is"}' in text
        assert 'TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-115}' in text
        assert 'TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}' in text
        assert 'VAL_N=${VAL_N:-3}' in text
        assert 'BEST_CKPT_METRIC_KEY="val-core/HuggingFaceH4/MATH-500/acc/mean@3"' in text

    assert "run_4a_model2_group_adv_is.sh" in monitor
    assert "run_4b_math_base.sh" in monitor
    assert "run_4c_math_sft.sh" in monitor
    assert "scripts/training_queue_monitor.sh" in monitor
    assert "training_queue_monitor_main" in monitor
    assert "training_queue_monitor_wait_for_resources" in generic_monitor
    assert "training_queue_monitor_wait_for_completion" in generic_monitor
    assert "FINAL_STEP=${FINAL_STEP:-115}" in monitor
    assert "MIN_FREE_GB=${MIN_FREE_GB:-160}" in monitor
