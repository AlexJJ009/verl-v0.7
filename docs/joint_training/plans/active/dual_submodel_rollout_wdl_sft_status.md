# Dual-Submodel Rollout WDL-SFT Implementation Status

Last updated: 2026-05-17 15:50 CST

## Branch / Commit

- Branch: `feature/on-policy-wdl-sft-dual-rollout`
- Base commit at branch creation: `162bd36d`
- Latest implementation commit: `7675d8b1` (`Implement dual-submodel rollout WDL-SFT`)
- Recipe submodule base commit: `3895e74`
- Recipe implementation commit: `13c540f` (`Add dual-submodel rollout WDL-SFT recipe`)

## Current Milestone

- Pre-smoke implementation and unit/reviewer gates complete.
- Next: create scoped commits, then run real GPU smoke for 3A and 3B inside Docker/tmux.

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
- Added targeted unit tests for config validation, source switching, trainer dual/no-dual generation behavior, recipe script checks, and existing WDL-SFT-IS label/beta regressions.

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

- 3A: not run.
- 3B: not run.

## Reviewer Gates

- 10.1 Config surface and validation: PASS.
- 10.2 HF joint model source switching: PASS.
- 10.3 vLLM joint model source switching and worker/RPC path: PASS.
- 10.4 Trainer dual-rollout flow and selected-only training: PASS.
- 10.5 Metrics and diagnostics: PASS.
- 10.6 Recipe/scripts and portable launch behavior: PASS.
- 10.7 Unit tests and backward-compatibility tests: initial FAIL because trainer flow/no-config path were only helper-tested; fixed by adding mocked `_generate_training_rollouts(...)` tests. Follow-up reviewer verdict: PASS.
- 10.8 3A GPU smoke: pending.
- 10.9 3B GPU smoke: pending.
- 10.10 Final documentation/status update: pending.

## Open Blockers / Decisions

- Need create scoped commits in the recipe submodule and parent repo without staging pre-existing dirty files.
- Need run 3A/3B real GPU smoke in Docker/tmux.

## Next Concrete Action

- Commit pre-smoke implementation, then launch 3A GPU smoke.
