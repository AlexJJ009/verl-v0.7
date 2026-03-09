from pathlib import Path


def test_joint_training_recipe_uses_vllm_rollout_and_no_sync_mode():
    script_path = Path("/data-1/verl07/verl/recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh")
    script = script_path.read_text()

    assert "ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-vllm}" in script
    assert 'DEFAULT_CKPT_BASE_DIR="/data-1/checkpoints"' in script
    assert "ROLLOUT_MODE=${ROLLOUT_MODE:-async}" in script
    assert "ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}" in script
    assert "ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}" in script
    assert "LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}" in script
    assert "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" in script
    assert "export TMPDIR=${TMPDIR:-/data-1/tmp}" in script
    assert "export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/data-1/.config/vllm}" in script
    assert "export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}" in script
    assert 'mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$VLLM_CONFIG_ROOT" "$VERL_ZMQ_IPC_DIR"' in script
    assert "MIN_FREE_GB_FOR_CKPT=${MIN_FREE_GB_FOR_CKPT:-30}" in script
    assert 'DEFAULT_CKPT_BASE_DIR_FALLBACK="/data-2/checkpoints/JointTraining/GRPO"' not in script
    assert "MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-2}" in script
    assert "MAX_CRITIC_CKPTS_TO_KEEP=${MAX_CRITIC_CKPTS_TO_KEEP:-2}" in script
    assert 'resolves to the root filesystem (${BASE_CKPT_MOUNT_SOURCE})' in script
    assert 'prefer BASE_CKPT_DIR=/data-1/checkpoints.' in script
    assert 'ERROR: ${BASE_CKPT_DIR} has only $((BASE_CKPT_FREE_KB / 1024 / 1024)) GiB free' in script
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}" in script
    assert "ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-8}" in script
    assert "ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}" in script
    assert "export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}" in script
    assert "export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}" in script
    assert 'if [ "${ROLLOUT_ENGINE}" = "vllm" ]; then' in script
    assert "ROLLOUT_FREE_CACHE_ENGINE_DEFAULT=False" in script
    assert "ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT=False" in script
    assert "ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-${ROLLOUT_FREE_CACHE_ENGINE_DEFAULT}}" in script
    assert "ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-${ROLLOUT_ENABLE_SLEEP_MODE_DEFAULT}}" in script
    assert "USE_REMOVE_PADDING_WAS_SET=${USE_REMOVE_PADDING+x}" in script
    assert "LOG_PROB_MICRO_BATCH_SIZE_WAS_SET=${LOG_PROB_MICRO_BATCH_SIZE+x}" in script
    assert "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}" in script
    assert "GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-4}" in script
    assert "LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-2}" in script
    assert "flash_attn is not installed; disabling USE_REMOVE_PADDING" in script
    assert "LOG_PROB_MICRO_BATCH_SIZE=1" in script
    assert "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True" in script
    assert "actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE}" in script
    assert "actor_rollout_ref.rollout.mode=sync" not in script
    assert "actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}" in script
    assert "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}" in script
    assert "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}" in script
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" in script
    assert "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" in script
    assert "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}" in script
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE}" in script
    assert "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}" in script
    assert "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}" in script
    assert "data.filter_overlong_prompts=True" in script
    assert "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" in script
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}" in script
    assert "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" in script
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}" in script
    assert "+actor_rollout_ref.rollout.micro_batch_size=${GENERATION_MICRO_BATCH_SIZE}" in script
    assert "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP}" in script
    assert "trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPTS_TO_KEEP}" in script
