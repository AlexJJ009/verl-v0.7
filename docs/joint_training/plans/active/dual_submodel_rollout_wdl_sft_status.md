# Dual-Submodel Rollout WDL-SFT Implementation Status

Last updated: 2026-05-18 20:20 CST

## Branch / Commit

- Branch: `feature/on-policy-wdl-sft-dual-rollout`
- Base commit at branch creation: `162bd36d`
- Latest implementation commit: `7675d8b1` (`Implement dual-submodel rollout WDL-SFT`)
- Latest status/docs commit before smoke: `83eac12a` (`Update dual rollout implementation status`)
- Latest smoke/status docs commit: `8dc71fd2` (`Update dual rollout smoke status`)
- Final status cleanup commit: `0161e73b` (`Update dual rollout final status`)
- Final plan/status checklist cleanup commit: `152a506f` (`Align dual rollout completion docs`)
- Launch hardening commit: `6bba3f9f` (`Harden dual rollout launch checks`)
- Recipe submodule base commit: `3895e74`
- Recipe implementation commit: `13c540f` (`Add dual-submodel rollout WDL-SFT recipe`)
- Recipe launch hardening commit: `4682b8c` (`Harden dual rollout launch defaults`)

## Current Milestone

- Implementation, targeted tests, 3A/3B real GPU smoke, final reviewer gates, final documentation update, and temporary smoke-checkpoint cleanup check are complete.
- A real 3A training run exposed a method-level failure: selected model2-only rollout trajectories are trained under fused joint logits, causing a large off-policy distribution gap, very large model2 gradients, output corruption, and MATH-500 validation collapse.
- Detailed failure analysis is recorded in `docs/joint_training/plans/active/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md`.
- Decision: stop the current 3A run and treat this branch's algorithm as a negative result / off-policy ablation, not as the main on-policy WDL-SFT path.

## Completed Milestones

- Created branch `feature/on-policy-wdl-sft-dual-rollout`.
- Created this goal-local status file before major code edits.
- Added config-gated dual rollout validation under `actor_rollout_ref.rollout.custom`.
- Added HF and vLLM joint rollout source switching for `fused`, `sub_model_0`, and `sub_model_1`.
- Added rollout source RPC path through vLLM server/worker extension and HF rollout worker path.
- Refactored trainer generation into `_generate_training_rollouts(...)` and wired dual generation/selected-only training into `fit()`.
- Added required dual rollout metrics and disabled `rollout_is_weights` loss multiplication by default for dual rollout.
- Added recipe folder `recipe/on_policy_wdl_sft/dual_submodel_rollout/` with 3A/3B wrappers and shared launch script.
- Committed recipe submodule changes as `13c540f`.
- Committed parent implementation as `7675d8b1`.
- Hardened 3A/3B launch defaults: `REPO_ROOT` and `DATA_ROOT` are externally overridable, checkpoint free-space fail-fast default is 160 GiB, and checkpoint retention defaults to latest plus best.
- Added targeted unit tests for config validation, source switching, trainer dual/no-dual generation behavior, recipe script checks, and existing WDL-SFT-IS label/beta regressions.
- Ran 3A and 3B real GPU smoke in Docker/tmux with vLLM and FlashInfer.

## Intentionally Changed Files

- `docs/joint_training/plans/active/dual_submodel_rollout_wdl_sft_status.md`
- `docs/joint_training/plans/active/dual_submodel_rollout_wdl_sft.md`
- `verl/models/joint_model/modeling_joint_qwen3.py`
- `verl/models/joint_model/vllm_modeling_joint_qwen3.py`
- `verl/workers/rollout/hf_rollout.py`
- `verl/workers/rollout/vllm_rollout/utils.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/replica.py`
- `verl/workers/fsdp_workers.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/trainer/ppo/ray_trainer.py`
- `tests/on_policy_wdl_sft/test_dual_submodel_rollout.py`
- `tests/joint_training/feat/test_joint_model.py`
- `tests/joint_training/feat/test_vllm_joint_rollout.py`
- `recipe/on_policy_wdl_sft/dual_submodel_rollout/README.md`
- `recipe/on_policy_wdl_sft/dual_submodel_rollout/_common_dual_rollout.sh`
- `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh`
- `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3b_model2_rollout_beta01.sh`

## Pre-Existing Dirty / Untracked Files Not Touched

Observed before implementation edits:

- `.codex/config.toml`
- `docs/joint_training/plans/active/ablation_single_model.md`
- `docs/joint_training/plans/active/wdl_sft_is.md`
- `recipe` submodule/worktree state
- `.claude/skills/experiment-registry`
- `docs/joint_training/plans/active/dual_submodel_rollout_wdl_sft_goal.md`
- `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md`

These must not be staged or committed unless they become intentional task files.

## Tests / Verification

- Static syntax:
  - `bash -n recipe/on_policy_wdl_sft/dual_submodel_rollout/_common_dual_rollout.sh`
  - `bash -n recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh`
  - `bash -n recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3b_model2_rollout_beta01.sh`
  - `python3 -m py_compile verl/trainer/ppo/ray_trainer.py verl/models/joint_model/modeling_joint_qwen3.py verl/models/joint_model/vllm_modeling_joint_qwen3.py verl/workers/rollout/vllm_rollout/utils.py verl/workers/rollout/vllm_rollout/vllm_async_server.py verl/workers/rollout/vllm_rollout/vllm_rollout.py verl/workers/rollout/replica.py verl/workers/fsdp_workers.py verl/experimental/agent_loop/agent_loop.py`
  - Result: pass.
- Docker targeted tests:
  - Command:
    `docker run --rm --gpus all --ipc=host -v /data-1/verl07/verl:/workspace/verl -v /data-1:/data-1 verl-harness bash -lc 'cd /workspace/verl && pytest tests/on_policy_wdl_sft/test_dual_submodel_rollout.py tests/joint_training/feat/test_joint_model.py tests/joint_training/feat/test_vllm_joint_rollout.py tests/on_policy_wdl_sft/test_wdl_sft_is_loss.py -q --tb=short'`
  - Result: `74 passed, 4 skipped, 7 warnings in 48.62s`.

## GPU Smoke

- Shared smoke setup:
  - Docker image: `verl-harness`.
  - GPU path: `docker run --rm --gpus all --ipc=host -v /data-1/verl07/verl:/workspace/verl -v /data-1:/data-1 verl-harness ...`.
  - Default strong model path `/data-1/.cache/Qwen3-4B-Base-SFT-stage-1` was absent on this host, so smoke used same-architecture override `MODEL2_PATH=/data-1/.cache/Qwen3-4B-Base-Code-WDL-M1/checkpoint-39`.
  - First 3A attempt wrote to `/data-1/checkpoints` and failed at checkpoint save because `/data-1` reached 0 GiB free. The generated partial smoke checkpoint was deleted, and both passing smokes used `/dev/shm` for `BASE_CKPT_DIR`, `RAY_TMPDIR`, `TMPDIR`, `VLLM_CONFIG_ROOT`, and `WANDB_DIR`.
  - `VAL_BEFORE_TRAIN=False`, `TEST_FREQ=-1`, `SAVE_FREQ=1`, `TOTAL_TRAINING_STEPS=1`, `TRAIN_PROMPT_BSZ=2`, `TRAIN_PROMPT_MINI_BSZ=1`, `ROLLOUT_AGENT_NUM_WORKERS=1`.
- 3A: PASS.
  - Session: `dual3a_smoke2`.
  - Script: `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh`.
  - Log: `recipe/on_policy_wdl_sft/dual_submodel_rollout/WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0-SMOKE2_1779005075.log`.
  - Metrics: `recipe/on_policy_wdl_sft/dual_submodel_rollout/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0-SMOKE2_1779005075.jsonl`.
  - Evidence: `sub_model_0` and `sub_model_1` both generated with `prompt_batch=2`, selected `sub_model_1`, restored `fused`, reached `Training Progress: 100% 1/1`.
  - Metrics evidence: `dual_rollout/source_count=2`, `dual_rollout/selected_source=1`, `model1_response_count=16`, `model2_response_count=16`, `timing_s/update_actor=3.2038`, `timing_s/save_checkpoint=14.9391`, `timing_s/update_weights=3.7873`, `training/global_step=1`, `actor/wdl_sft_beta=0.0`.
  - Checkpoint evidence: `/dev/shm/.../latest_checkpointed_iteration.txt` contained `1`; `global_step_1/actor` had model/optimizer/extra-state files for ranks 0-7.
- 3B: PASS.
  - Session: `dual3b_smoke2`.
  - Script: `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3b_model2_rollout_beta01.sh`.
  - Log: `recipe/on_policy_wdl_sft/dual_submodel_rollout/WDL-SFT-Qwen3-4B-MATH-3B-DUAL-M2-BETA01-SMOKE2_1779005566.log`.
  - Metrics: `recipe/on_policy_wdl_sft/dual_submodel_rollout/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-3B-DUAL-M2-BETA01-SMOKE2_1779005566.jsonl`.
  - Evidence: wrapper and Hydra command set `WDL_SFT_BETA=0.1` / `wdl_sft_beta=0.1`; config validation passed; `sub_model_0` and `sub_model_1` both generated; selected `sub_model_1`; reached `Training Progress: 100% 1/1`.
  - Metrics evidence: `actor/wdl_sft_beta=0.1`, `actor/pg_loss=-40.1905`, `actor/grad_norm=84.5590`, `jointTraining/model1_grad_norm=52.7146`, `jointTraining/model2_grad_norm=51.9937`, `timing_s/update_actor=3.1565`, `timing_s/save_checkpoint=14.6132`, `timing_s/update_weights=3.8364`, `training/global_step=1`.
  - Checkpoint evidence: `/dev/shm/.../latest_checkpointed_iteration.txt` contained `1`; `global_step_1/actor` had model/optimizer/extra-state files for ranks 0-7.
- Both smokes ended with ignored W&B/torchdata atexit cleanup tracebacks after `Training Progress: 100%`; reviewer accepted these as non-blocking cleanup noise because metrics and checkpoints were already written and there was no `Error executing job` / `RayTaskError` in the relevant failure path.

## Reviewer Gates

- 10.1 Config surface and validation: PASS.
- 10.2 HF joint model source switching: PASS.
- 10.3 vLLM joint model source switching and worker/RPC path: PASS.
- 10.4 Trainer dual-rollout flow and selected-only training: PASS.
- 10.5 Metrics and diagnostics: PASS.
- 10.6 Recipe/scripts and portable launch behavior: PASS.
- 10.7 Unit tests and backward-compatibility tests: initial FAIL because trainer flow/no-config path were only helper-tested; fixed by adding mocked `_generate_training_rollouts(...)` tests. Follow-up reviewer verdict: PASS.
- 10.8 3A GPU smoke: PASS. Reviewer caveats: same-architecture model2 override due missing default path; checkpoint/temp dirs under `/dev/shm` because `/data-1` was full; ignored W&B/torchdata atexit exceptions after successful completion.
- 10.9 3B GPU smoke: PASS. Reviewer caveats: same environment deviations as 3A; reviewer could not live-inspect `/dev/shm` after the fact, but accepted captured checkpoint evidence plus log/metric save timing.
- 10.10 Final documentation/status update: PASS. Reviewer caveat was that final docs were still uncommitted at review time; main-agent follow-up committed the final status update and aligned the source-plan checklist with the recorded smoke evidence.

## Open Blockers / Decisions

- Do not stage pre-existing dirty files: `.codex/config.toml`, `docs/joint_training/plans/active/ablation_single_model.md`, `docs/joint_training/plans/active/wdl_sft_is.md`, `.claude/skills/experiment-registry`, `docs/joint_training/plans/active/dual_submodel_rollout_wdl_sft_goal.md`, or `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md`.
- Temporary smoke checkpoints under `/dev/shm/dual_rollout_smoke_checkpoints` are already absent as of the final live check.
- `/data-1` currently has only about 46 GiB free, so the hardened 3A/3B launch scripts will fail fast for a real training run unless `BASE_CKPT_DIR` points to a mount with at least the configured `MIN_FREE_GB_FOR_CKPT` or space is freed.
- 3A real run `WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0_1779027403` should not be continued as a candidate checkpoint-producing run. Evidence:
  - validation is model2-only, so collapse is not a fused-eval artifact;
  - MATH-500 `acc/mean@3` moved from about `0.5968` at step 0 to `0.3448` at step 25 and `0.0390` at step 50;
  - validation output `weird_frac` moved from `0.151` at step 0 to `0.997` at step 50;
  - step 65 showed `jointTraining/model2_grad_norm=13288.8`, `actor/grad_norm=13339.8`, `model2_response_len_mean=3986.5`, and `response_length/clip_ratio=0.9180`.
- Mainline recommendation after this negative result: return to the previous fused-rollout/fused-training method and tune label quality within that on-policy distribution.

## Next Concrete Action

- Stop the current 3A training process, preserve logs/metrics/validation dumps, and switch back to the previous branch for follow-up work.
