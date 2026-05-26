# Training Script Index

This file is the collaboration index for runnable training scripts in this
branch. Keep it factual and short: script purpose, creation time, last use, and
current status. Update it whenever a training script is created or used.

The shared policy is
`docs/joint_training/constraints/experiment_tracking/training_script_index_policy.md`.

Do not put full launch commands, monitor commands, or operational playbooks in
this index. Those belong in the relevant on-demand guide or workflow, so agents
and collaborators can load run mechanics only when they need them.

Times are local server time (`Asia/Shanghai`, CST).

## Dual-Submodel Rollout WDL-SFT

| Script | Purpose | Created | Last used | Status / notes |
|---|---|---:|---:|---|
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/_common_dual_rollout.sh` | Shared local launcher for this branch's dual-submodel rollout family. | before 2026-05-25 | 2026-05-25 | Active. Supports 3A/3B dual-source rollout and 4A model2-only rollout via environment overrides. Used by 4A smoke/stability gates. |
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh` | 3A failed-path reproduction: dual-source rollout, model2-selected training, `wdl_sft_is`, `WDL_SFT_BETA=0.0`. | before 2026-05-25 | not used in this session | Active for reproduction/comparison only. |
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3b_model2_rollout_beta01.sh` | 3B failed-path reproduction: dual-source rollout, model2-selected training, `wdl_sft_is`, `WDL_SFT_BETA=0.1`. | before 2026-05-25 | not used in this session | Active for reproduction/comparison only. |
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_4a_model2_group_adv_is.sh` | Revised algorithm: model2-only rollout, fused training, `dual_model2_group_adv_is`, group advantage, detached TIS, binary staleness mask. | 2026-05-25 | 2026-05-25 | Active. Defaults to `/data-1/dataset/math/train_rl_format.parquet`, `TOTAL_TRAINING_STEPS=115`, `TOTAL_EPOCHS=1`, validation `n=3`, latest+best checkpoint retention. The earlier 300-step launch `WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS-LEARNING_1779716557` was stopped after the schedule mismatch was found; 256-token smoke/stability is plumbing-only evidence. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_4b_math_base.sh` | Single-model Base ablation for the current group-advantage IS algorithm: `wdl_group_adv_is` on MATH train. | 2026-05-25 | queued 2026-05-25 | Active. Defaults to Base init, `/data-1/dataset/math/train_rl_format.parquet`, `TOTAL_TRAINING_STEPS=115`, `TOTAL_EPOCHS=1`, validation `n=3`, latest+best checkpoint retention. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_4c_math_sft.sh` | Single-model SFT ablation for the current group-advantage IS algorithm: `wdl_group_adv_is` on MATH train. | 2026-05-25 | queued 2026-05-25 | Active. Defaults to SFT init, `/data-1/dataset/math/train_rl_format.parquet`, `TOTAL_TRAINING_STEPS=115`, `TOTAL_EPOCHS=1`, validation `n=3`, latest+best checkpoint retention. |
| `scripts/training_queue_monitor.sh` | Project-level generic sequential queue monitor used by experiment-specific wrappers. | 2026-05-26 | pending relaunch | Active. Owns shared tmux launch, disk/GPU gates, completion detection, and fail-fast queue behavior. |
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/monitor_4abc_math_queue.sh` | Thin queue wrapper for 4A dual-model training, then 4B Base single-model training, then 4C SFT single-model training. | 2026-05-25 | queued 2026-05-25 | Active. Defines 4ABC queue arrays and delegates shared behavior to `scripts/training_queue_monitor.sh`; waits for `FINAL_STEP=115` before continuing. |
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/meituan/jupyter.sh` | Meituan AFO adapter for 4A dual-submodel rollout family. | 2026-05-26 | not submitted | Active. Routes `EXPERIMENT=4a` / `4a-dual` to `run_4a_model2_group_adv_is.sh` after sourcing dolphinfs path overrides. |
| `platform/hope_on_policy_wdl_sft/` | Unified Meituan entry for 1X, 2X, and 4ABC WDL-SFT experiments. | before 2026-05-26 | not submitted for 4ABC | Active. `jupyter.sh` routes 4A to `dual_submodel_rollout/meituan/jupyter.sh` and 4B/4C to `ablation_single_model/meituan/jupyter.sh`; `submit_batch.sh --4abc` renders separate AFO submissions. |
