# Joint Training GRPO Progress

## Status

Phase 2 runtime stabilization is in progress. Joint GRPO training now runs on the vLLM rollout path, but the remaining work is hardening long-running checkpoints and completing a clean end-to-end multi-step run on the target server.

## Landed Since The Initial Bring-Up

1. `384804fe` added the joint vLLM rollout path and the eval-only weight extraction flow for validation.
2. Recipe `6151c24` switched the joint GRPO launcher to `ROLLOUT_ENGINE=vllm` by default, and parent commit `ef0b9671` advanced the submodule pointer to that rollout path.
3. `d0c5d3a` hardened the recipe around rollout memory, checkpoint directories, remove-padding fallback, and host-specific runtime defaults.
4. `428a7e83` stabilized the FSDP actor/update path: shared DP-group wiring, safer old-log-prob entropy behavior, chunked entropy, and lower-peak joint logit fusion.

## Resolved Runtime Failures

1. FSDP actor deadlock on step 2:
   - Root cause: actor/critic dynamic batching used inconsistent DP-group context, so different ranks produced different micro-batch schedules and diverged in NCCL collectives.
   - Fix: pass the worker DP group into dynamic-batch preparation for actor and critic.

2. vLLM startup cache allocation failure:
   - Root cause: colocated rollout with `gpu_memory_utilization=0.6` and vLLM warmup at `max_num_seqs=1024` exhausted KV-cache budget during engine init.
   - Fix: raise rollout budget to `0.75` and cap `max_num_seqs` to `256` in the recipe.

3. Actor-side logits/entropy OOMs during old-log-prob recompute:
   - Root cause: the joint fusion path created extra full-vocab temporaries, and entropy was still being materialized even when `entropy_coeff=0`.
   - Fix: use a lower-peak fused-logits implementation, honor `calculate_entropy`, and chunk dense entropy computation.

4. Remove-padding crash when `flash_attn` was unavailable:
   - Root cause: `use_remove_padding=True` reached the CUDA path that imports `flash_attn`, but the package was not installed in the runtime env.
   - Fix: preflight-disable remove-padding in the recipe and add an explicit runtime guard in `verl/utils/attention_utils.py`.

5. Checkpoint save failure on `/data-2`:
   - Root cause: the failing run in `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K_1772760550.log` still wrote checkpoints under `/data-2`, which resolves to the 60G root filesystem on this host. `torch.save()` then failed mid-write once `/` filled up.
   - Fix: the recipe now defaults to `/data-1/checkpoints`, warns if a manually chosen checkpoint path resolves to the root filesystem, and fails early if the target filesystem does not meet the free-space threshold.

6. Corrupt partial checkpoint shards after a failed save:
   - Root cause: `torch.save()` wrote directly to final shard paths, so a disk-full error could leave broken `.pt` files behind.
   - Fix: FSDP checkpoint shards are now written to `*.tmp`, preflight-checked for disk space, and atomically renamed on success. Partial files are cleaned up on failure.

## Current Recipe Defaults

1. Checkpoints:
   - Default base dir: `/data-1/checkpoints`
   - Minimum free space gate: `MIN_FREE_GB_FOR_CKPT=30`
   - Retention: `MAX_ACTOR_CKPTS_TO_KEEP=2`, `MAX_CRITIC_CKPTS_TO_KEEP=2`

2. Rollout memory:
   - `ROLLOUT_ENGINE=vllm`
   - `ROLLOUT_MODE=async`
   - `ROLLOUT_GPU_MEMORY_UTILIZATION=0.75`
   - `ROLLOUT_MAX_NUM_SEQS=256`
   - `LOG_PROB_MAX_TOKEN_LEN_PER_GPU=1536`
   - `LOG_PROB_MICRO_BATCH_SIZE=2`, auto-fallback to `1` when remove-padding is disabled automatically

3. Safety defaults:
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - `TMPDIR=/data-1/tmp`
   - `USE_REMOVE_PADDING=True` only when `flash_attn` is available

## Test Coverage Added For These Fixes

1. `tests/workers/actor/test_special_dp_actor.py`
2. `tests/workers/critic/test_dynamic_dp_critic.py`
3. `tests/workers/test_fsdp_workers.py`
4. `tests/utils/test_torch_functional.py`
5. `tests/utils/test_attention_utils_on_cpu.py`
6. `tests/joint_training/regression/test_old_log_prob_entropy_gating.py`
7. `tests/joint_training/regression/test_old_log_prob_entropy_skip.py`
8. `tests/joint_training/feat/test_joint_training_recipe_script.py`
9. `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py`

## Remaining Work

1. Re-run the full 8xH800 joint GRPO job from the current code state and confirm it saves checkpoints successfully under `/data-1`.
2. Measure steady-state GPU memory during rollout plus actor old-log-prob recompute to decide whether the current rollout defaults should remain the baseline.
3. Promote the checkpoint preflight/atomic-save logic to any parallel trainer paths that still bypass the shared FSDP checkpoint manager.
