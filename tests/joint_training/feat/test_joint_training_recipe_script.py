from pathlib import Path


def test_joint_training_recipe_uses_vllm_rollout_and_no_sync_mode():
    script_path = Path("/data-1/verl07/verl/recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh")
    script = script_path.read_text()

    assert "ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-vllm}" in script
    assert "ROLLOUT_MODE=${ROLLOUT_MODE:-async}" in script
    assert "ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}" in script
    assert "ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}" in script
    assert "ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-8}" in script
    assert 'if [ "${ROLLOUT_ENGINE}" = "vllm" ]; then' in script
    assert "ROLLOUT_FREE_CACHE_ENGINE_DEFAULT=False" in script
    assert "ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT=False" in script
    assert "ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-${ROLLOUT_FREE_CACHE_ENGINE_DEFAULT}}" in script
    assert "ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-${ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT}}" in script
    assert "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}" in script
    assert "actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE}" in script
    assert "actor_rollout_ref.rollout.mode=sync" not in script
    assert "actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}" in script
    assert "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}" in script
    assert "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}" in script
    assert "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}" in script
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE}" in script
    assert "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}" in script
    assert "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}" in script
    assert "data.filter_overlong_prompts=True" in script
    assert "+actor_rollout_ref.rollout.micro_batch_size=${micro_batch_size}" in script
