# Stage 1: Implementation Bring-Up and Runtime Stabilization

**Status**: Completed

## Scope

1. Joint model implementation.
2. Eval-only validation path for extracting model2 weights.
3. vLLM rollout integration.
4. Recipe bring-up on the target H800 server.
5. Stabilization of the major runtime blockers:
   - FSDP DP-group mismatch and NCCL deadlock.
   - vLLM cache-budget startup failure.
   - actor old-log-prob and entropy OOMs.
   - remove-padding and `flash_attn` fallback handling.
   - checkpoint path and disk-space failures.
   - `/tmp` and root-filesystem ZMQ / vLLM side effects.
   - missing periodic metric visibility in logs.

## Exit Criteria (All Met)

1. The recipe runs end to end on the target server.
2. Checkpoint save/resume paths are pinned to `/data-1`.
3. Metrics are visible in both W&B history and local logs.
4. Joint-training regression coverage exists for the stabilized paths.

## Major Code/Recipe Milestones

1. `384804fe`: joint vLLM rollout support and eval-only weight extraction.
2. `6151c24` in `recipe/`: switch the joint GRPO recipe to vLLM rollout by default.
3. `d0c5d3a` in `recipe/`: rollout memory and checkpoint-path hardening.
4. `428a7e83`: FSDP actor rollout-path stabilization, entropy gating, and lower-peak fused-logit handling.
5. `73404180`: atomic FSDP checkpoint saves with disk-pressure protection.
6. `5bd62896`: colocated ZMQ socket paths moved off fragile root-mounted defaults.
7. `59a4c534`: recipe and test coverage refresh for stabilized behavior.
8. `5b3aca2` in `recipe/`: persistent local metrics logging.
9. `c4436d2b`: periodic test-step metric printing in the trainer.

## Infrastructure State After Stage 1

1. Checkpoint base dir defaults to `/data-1/checkpoints`.
2. `TMPDIR`, vLLM config, and ZMQ IPC roots are all on `/data-1`.
3. W&B is still offline by default, but metrics are also persisted locally.
4. The launcher auto-falls back when `flash_attn` is unavailable.
5. The trainer now emits merged train-plus-validation summaries at every validation step.

## Observations From The First Successful Run

1. The run is operationally stable, but validation quality is still poor on GSM8K:
   - `val-core/openai/gsm8k/acc/mean@1 = 0.0`
   - `val-aux/openai/gsm8k/reward/mean@1 = -1.0009765625`
2. The next phase is not "make it run", but "understand whether the joint objective is correct, useful, and measurable".
3. The end of the successful run still shows a W&B teardown `BrokenPipeError` in an `atexit` callback (cleanup nuisance, does not invalidate results).
