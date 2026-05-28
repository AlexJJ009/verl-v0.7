# Stage 1 On-Policy SFT Beta Search Status

Last updated: 2026-05-28

## Current State

- Branch: `feature/on-policy-wdl-sft`
- Current milestone: READY FOR EXECUTION. Plan hardening is complete; next step is Stage 1 script/monitor implementation.
- Current goal: Stage 1 single-model On-Policy SFT beta grid, `0.0..1.0` at `0.1` intervals.
- Explicitly deferred: Stage 2 joint On-Policy WDL-SFT. Do not execute or expand Stage 2 in the current goal.
- Full training authorization: not granted.
- Smoke authorization: not currently granted.
- Plan review: subagent `Pascal` returned WARN, all WARN findings addressed.

## Current Stage 1 Scope Files

- `docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search.md`
- `docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search_status.md`
- `docs/joint_training/guides/training_script_index.md`
- `recipe/on_policy_wdl_sft/staged_v1/run_s1_base_sft.sh`
- `recipe/on_policy_wdl_sft/staged_v1/meituan/env.sh`
- `recipe/on_policy_wdl_sft/staged_v1/README.md`
- `recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh`
- `platform/hope_staged_v1/run.hope`
- `platform/hope_staged_v1/README.md`

## Legacy / Deferred Files Touched Earlier

These files may have been touched before the scope correction, but they are not current acceptance criteria:

- `recipe/on_policy_wdl_sft/staged_v1/_run_stage2_common.sh`
- any `run_s2_beta_*.sh` wrapper
- `recipe/on_policy_wdl_sft/staged_v1/run_stage2_beta_search_queue.sh`

## Script State

- Stage 1 wrapper exists: `recipe/on_policy_wdl_sft/staged_v1/run_s1_base_sft.sh`.
- Stage 1 beta wrappers for `0.0..1.0` are required by the revised plan but have not yet been implemented in this plan-only step.
- Stage 1 beta queue/monitor is required by the revised plan but has not yet been implemented in this plan-only step.
- Legacy/future Stage 2 files may exist, but they are no longer acceptance criteria for the current goal.
- Meituan four-layer launch path exists under `platform/hope_staged_v1/` and `recipe/on_policy_wdl_sft/staged_v1/meituan/`.
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

- `bash -n recipe/on_policy_wdl_sft/staged_v1/*.sh recipe/on_policy_wdl_sft/staged_v1/meituan/*.sh platform/hope_staged_v1/*.sh`
  - Result: passed after the latest edits.
- `platform/hope_staged_v1/submit_beta_search.sh --dry-run s1-base-sft`
  - Result: passed; rendered `VAL_BEFORE_TRAIN=False`, `TOTAL_TRAINING_STEPS=150`, `VAL_N=3`, W&B project, and W&B offline mode.

## Smoke State

Historical requested smoke shape, only if the user authorizes execution again:

```bash
tmux new-session -s staged_v1_s1_smoke
TOTAL_TRAINING_STEPS=10 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=5 \
VAL_N=3 \
WANDB_MODE=offline \
WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1 \
bash recipe/on_policy_wdl_sft/staged_v1/run_s1_base_sft.sh
```

Smoke attempt 1:

- tmux session: `staged_v1_s1_smoke`
- run id: `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1_1779952267`
- checkpoint dir: `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1_1779952267`
- log file: `/data-1/verl07/verl/recipe/on_policy_wdl_sft/staged_v1/staged_v1_s1_smoke.log`
- W&B offline dir: `/data-1/wandb_runs/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BASE-V1/wandb/offline-run-20260528_071432-7utidal6`
- result: configuration, dataset loading, model loading, vLLM startup, and W&B offline logging succeeded, but `VAL_BEFORE_TRAIN=True` caused full pre-training validation before the 10-step smoke. This was too heavy for the intended usability smoke.
- action taken: tmux session was killed.

Current execution boundary:

- The user clarified the current work is plan-authoring only.
- Do not start smoke attempt 2 unless the user explicitly authorizes execution.
- If execution is later authorized, first verify that no stale Docker container or tmux session from the previous smoke is still running.
- Smoke attempt 2 should use `VAL_BEFORE_TRAIN=False`, `TEST_FREQ=-1`, and an explicit `WDL_SFT_BETA` value.

## W&B Sync State

- No staged v1 training run has completed yet.
- No W&B offline run has been synced yet.
- Sync command shape:

```bash
WANDB_SYNC_DIR=/path/to/wandb/offline-run-* \
bash recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh
```

## Reviewer Verdicts

- Method reviewer: WARN from subagent `Pascal`, then addressed.
  - Finding 1: status file listed a Stage 2 file under intentional changes. Fixed by splitting current Stage 1 scope files from legacy/deferred Stage 2 files.
  - Finding 2: plan banned KL penalty but did not explicitly require KL config keys to be disabled. Fixed by adding explicit KL-off settings to Stage 1 wrapper requirements.
- Script/Meituan reviewer: pending after Stage 1 beta wrappers/queue are implemented.
- Ops/W&B reviewer: pending after Stage 1 monitor and W&B sync path are finalized.
- Runtime reviewer: not applicable until the user authorizes a smoke or training run.

## Open Decisions / Blockers

- Stage 1 beta wrappers and queue/monitor still need implementation after user authorization.
- Full Stage 1 beta-grid training requires explicit user authorization.
- Stage 2 requires a separate future plan/revision after Stage 1 beta search completes.

## Next Concrete Action

1. Commit and push the ready-for-execution plan/status update.
2. Implement Stage 1 beta wrappers for `0.0..1.0` and the Stage 1 queue/monitor.
3. Run shell syntax and AFO render checks after script edits.
4. Wait for explicit user authorization before any smoke or full training.
