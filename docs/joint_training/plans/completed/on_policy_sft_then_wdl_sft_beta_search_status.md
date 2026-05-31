# Stage 1 On-Policy SFT Beta Search Status

Last updated: 2026-05-28

## Current State

- Branch: `feature/on-policy-wdl-sft`
- Current main-repo HEAD: `b5e575a8`
- Current recipe submodule HEAD: `5190572`
- Current milestone: IMPLEMENTED AND SMOKE-PASSED for the current Stage 1 execution contract.
- Current goal: Stage 1 single-model On-Policy SFT beta grid, `0.0..1.0` at `0.1` intervals.
- Explicitly deferred: Stage 2 joint On-Policy WDL-SFT. Do not execute or expand Stage 2 in the current goal.
- Full training authorization: not granted. Do not launch the full Stage 1 beta grid without a separate user approval.
- Smoke authorization: granted for a short usability/runtime validation; completed successfully at 10 steps.
- Plan review: subagent `Pascal` returned WARN, all WARN findings addressed.

## Current Stage 1 Scope Files

- `docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search.md`
- `docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search_status.md`
- `docs/joint_training/guides/training_script_index.md`
- `recipe/on_policy_wdl_sft/staged_v1/run_s1_base_sft.sh`
- `recipe/on_policy_wdl_sft/staged_v1/run_s1_beta_0.sh`
- `recipe/on_policy_wdl_sft/staged_v1/run_s1_beta_01.sh` .. `run_s1_beta_10.sh`
- `recipe/on_policy_wdl_sft/staged_v1/run_stage1_beta_search_queue.sh`
- `recipe/on_policy_wdl_sft/staged_v1/meituan/env.sh`
- `recipe/on_policy_wdl_sft/staged_v1/meituan/jupyter.sh`
- `recipe/on_policy_wdl_sft/staged_v1/README.md`
- `recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh`
- `platform/hope_staged_v1/run.hope`
- `platform/hope_staged_v1/submit_beta_search.sh`
- `platform/hope_staged_v1/README.md`

## Legacy / Deferred Files Touched Earlier

These files may have been touched before the scope correction, but they are not current acceptance criteria:

- `recipe/on_policy_wdl_sft/staged_v1/_run_stage2_common.sh`
- any `run_s2_beta_*.sh` wrapper
- `recipe/on_policy_wdl_sft/staged_v1/run_stage2_beta_search_queue.sh`

## Script State

- Stage 1 base wrapper exists and is configured for single-model Base training:
  - `LOSS_MODE=wdl_sft`
  - `JOINT_TRAINING=False`
  - `WDL_SFT_BETA` is env-overridable and defaults to `0.0`
  - KL is disabled through reward and actor loss knobs.
  - rollout correction is disabled with `ROLLOUT_IS=null`, `ROLLOUT_RS=null`, and `ROLLOUT_CALCULATE_LOG_PROBS=False`.
- Stage 1 beta wrappers exist for `0.0..1.0` at `0.1` intervals.
- Stage 1 local queue exists and runs beta wrappers sequentially after checking checkpoint free space, W&B free space, GPU use, conflicting sessions/containers, and incomplete checkpoint collisions.
- Meituan four-layer launch path exists and supports `s1-beta-0`, `s1-beta-01` .. `s1-beta-10`.
- W&B sync helper exists: `recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh`.

## Defaults

- `TOTAL_TRAINING_STEPS=150` for pilot/full authorized runs.
- `TOTAL_TRAINING_STEPS=10` for Stage 1 usability smoke.
- Stage 1 beta grid: `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0`.
- `VAL_N=3`.
- `VAL_BEFORE_TRAIN=False`.
- `TEST_FREQ=5`.
- `SAVE_FREQ=5`.
- `DATA_SEED=20260528`.
- `WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1`.
- `WANDB_MODE=offline`.

## Checks Run

- `bash -n recipe/on_policy_wdl_sft/staged_v1/*.sh recipe/on_policy_wdl_sft/staged_v1/meituan/*.sh platform/hope_staged_v1/*.sh recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh`
  - Result: passed after the latest edits.
- `platform/hope_staged_v1/submit_beta_search.sh --dry-run s1-beta-0 s1-beta-05 s1-beta-10`
  - Result: passed; rendered Stage 1 beta experiments with `TOTAL_TRAINING_STEPS=150`, `VAL_N=3`, `VAL_BEFORE_TRAIN=False`, W&B offline mode, and Base model init.
- Required current-scope file existence check:
  - Result: passed.
- Live disk check after user cleanup:
  - `/data-1` had about `206G` free before the successful smoke, above the queue guardrail of `160G`.
- Live disk check after successful smoke:
  - `/data-1` had about `160G` free after retaining the 10-step smoke checkpoint, exactly at the default queue guardrail. Full grid launch should re-check space and likely needs additional cleanup or archival planning before authorization.
- Final static checks after status update:
  - `bash -n recipe/on_policy_wdl_sft/staged_v1/*.sh recipe/on_policy_wdl_sft/staged_v1/meituan/*.sh platform/hope_staged_v1/*.sh recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh`: passed.
  - `platform/hope_staged_v1/submit_beta_search.sh --dry-run s1-beta-0 s1-beta-05 s1-beta-10`: passed.
  - `git diff --check`: passed.
  - `git -C recipe diff --check`: passed.

## Smoke State

Historical smoke command shape:

```bash
tmux new-session -s staged_v1_s1_smoke4
RUN_PREFIX=ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1 \
TOTAL_TRAINING_STEPS=10 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=5 \
VAL_N=3 \
WANDB_MODE=offline \
WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1 \
WDL_SFT_BETA=0.0 \
bash recipe/on_policy_wdl_sft/staged_v1/run_s1_base_sft.sh
```

Smoke attempt 1:

- tmux session: `staged_v1_s1_smoke`
- run id: `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1_1779952267`
- result: configuration, dataset loading, model loading, vLLM startup, and W&B offline logging succeeded, but `VAL_BEFORE_TRAIN=True` caused full pre-training validation before the 10-step smoke. This was too heavy for the intended usability smoke.
- action taken: tmux session was killed.

Smoke attempt 2:

- tmux session: `staged_v1_s1_smoke2`
- result: failed immediately because `/data-1/verl07/run_train.sh` does not exist on this host.

Smoke attempt 3:

- tmux session: `staged_v1_s1_smoke3`
- result: failed during Hydra validation because `_common_ablation.sh` set `actor_rollout_ref.model.joint_training=...` without the required `+` prefix for a new key.
- action taken: `_common_ablation.sh` now uses `+actor_rollout_ref.model.joint_training=${JOINT_TRAINING:-False}`.

Smoke attempt 4:

- tmux session: `staged_v1_s1_smoke4`
- run id: `ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1_1779958052`
- result: PASSED. The run reached `training/global_step=10`, saved `global_step_10`, and exited the tmux session.
- checkpoint dir: `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1_1779958052`
- latest checkpoint marker: `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1_1779958052/latest_checkpointed_iteration.txt` contained `10`.
- retained checkpoint: `global_step_10`; `global_step_5` was removed by latest/best retention.
- checkpoint size after completion: about `47G`.
- log file: `/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1_1779958052.log`
- metrics file: `/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicySFT-Then-WDLSFT-StagedV1/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1_1779958052.jsonl`
- W&B offline dir: `/data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1/wandb/offline-run-20260528_085053-8sae6tp8`
- runtime cleanup state after completion:
  - tmux session missing, expected after completion.
  - no Docker training container remained in `docker ps`.
  - all 8 GPUs were idle with about `1 MiB` memory used each.
- key config evidence in the run log:
  - `loss_mode=wdl_sft`
  - `+actor_rollout_ref.actor.policy_loss.wdl_sft_beta=0.0`
  - `+actor_rollout_ref.model.joint_training=False`
  - `algorithm.rollout_correction.rollout_is=null`
  - `algorithm.rollout_correction.rollout_rs=null`
  - `actor_rollout_ref.rollout.calculate_log_probs=False`
  - `trainer.val_before_train=False`
  - `trainer.test_freq=-1`
  - `trainer.save_freq=5`
  - `trainer.total_training_steps=10`
  - `data.seed=20260528`
- key final metrics at step 10:
  - `actor/wdl_sft_beta=0.0`
  - `actor/kl_loss=0.0`
  - `actor/ppo_kl=0.0`
  - `wdl_sft/n_correct=67`
  - `wdl_sft/n_incorrect=445`
  - `wdl_sft/correct_ratio=0.130859375`
  - `actor/grad_norm=672.3891296386719`
  - `response/aborted_ratio=0.0`
  - `timing_s/save_checkpoint=12.989398869220167`
- warning after completion:
  - The log ended with W&B/DataLoader cleanup-time exceptions after `Training Progress: 100%|...| 10/10` and after `global_step_10` was saved. This did not prevent metrics/checkpoint completion, but it should be watched in longer runs.

## W&B Sync State

- The completed smoke has one W&B offline run:
  - `/data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1/wandb/offline-run-20260528_085053-8sae6tp8`
- No W&B offline run has been synced yet.
- Sync command shape:

```bash
WANDB_SYNC_DIR=/data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-SMOKE4-BETA0-V1/wandb/offline-run-20260528_085053-8sae6tp8 \
bash recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh
```

## Reviewer Verdicts

- Method reviewer: PASS for current scope after WARN findings were addressed.
- Script/Meituan reviewer: PASS for current scope after syntax checks and Meituan dry-run rendering.
- Ops/W&B reviewer: PASS for current scope; queue guardrails, offline W&B path, and sync helper exist.
- Runtime reviewer: PASS for short Stage 1 usability smoke; full beta-grid runtime remains untested because it has not been authorized.

## Open Decisions / Blockers

- Full Stage 1 beta-grid training requires explicit user authorization.
- Current local disk is only at the `160G` default guardrail after the smoke checkpoint; re-check and free/archive space before starting the full grid.
- Stage 2 requires a separate future plan/revision after Stage 1 beta search completes.
- Longer runs should watch for W&B/DataLoader cleanup warnings, although the short smoke saved metrics and checkpoint successfully before those cleanup-time warnings.

## Next Concrete Action

1. If the user authorizes full execution, launch `recipe/on_policy_wdl_sft/staged_v1/run_stage1_beta_search_queue.sh` in tmux.
2. Monitor progress through tmux, Docker, GPU utilization, per-run logs, metrics JSONL, checkpoint markers, and W&B offline directories.
3. Sync W&B offline runs only when explicitly desired or when the run handoff requires cloud visibility.
