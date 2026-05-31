# Active Plan Index

Last organized: 2026-05-29.

This directory should contain only plans that still need decisions, reruns, or
follow-up experiments. Completed, superseded, or negative-result plans should
live in `../completed/`.

## Active

| Plan | Status | Why it remains active |
| --- | --- | --- |
| `stage2_model2_rollout_fused_loss_fast_validation.md` | Active Stage 2 fast-validation plan | New Stage 2 scope: Model2-only rollout from merged Stage 1 best checkpoints, fused Model1+Model2 WDL-SFT loss, both submodels trainable, non-overlap Stage 2 data shard after the Stage 1 consumed prefix, and two default 75-step validation runs from Stage 1 beta `0.0` and `0.1`. |
| `on_policy_sft_then_wdl_sft_beta_search.md` | Active Stage 1 implementation and experiment plan | Current scope is Stage 1 only: single-model on-policy SFT from Base with `loss_mode=wdl_sft`, no joint model, and `wdl_sft_beta=0.0..1.0` at `0.1` intervals. Stage 2 joint WDL-SFT is deferred until Stage 1 produces a selected checkpoint and the user authorizes a new phase. |
| `wdl_group_advantage_is_goal.md` | Active implementation contract | New method revision: group advantages + all-correct positive-SFT fallback + explicit mixed-policy old/current IS for joint rollout/joint training, with `norm_adv_by_std_in_grpo=false`, no `rollout_is_weights`, no KL penalty, no length normalization, no `beta`, and a required complete Meituan four-layer launch family. |
| `wdl_sft_is.md` | Active, but historical sections are stale | The post-fix rerun matrix is still open. Pre-fix 1A/1B/1C results remain useful history, but should not be treated as spec-correct current runs. |
| `ablation_single_model.md` | Active / partially completed | The SFT-init ablation has useful results, but the post-fix rerun matrix is incomplete and several rows remain not started. |

## Archived During This Cleanup

| Archived plan | Reason |
| --- | --- |
| `../completed/on_policy_wdl_sft_v1.md` | Superseded by `wdl_sft_is.md` on 2026-04-19. |
| `../completed/dual_submodel_rollout_wdl_sft.md` | Implementation completed and smoke-passed, but real 3A training produced a method-level negative result. |
| `../completed/dual_submodel_rollout_wdl_sft_goal.md` | Implementation contract preserved for traceability; no longer an active goal. |
| `../completed/dual_submodel_rollout_wdl_sft_status.md` | Final implementation/smoke/failure status preserved with the archived plan. |
| `../completed/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md` | Detailed gradient, loss, and distribution analysis for the 3A negative result. |
