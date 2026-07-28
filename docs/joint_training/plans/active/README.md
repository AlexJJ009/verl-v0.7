# Active Plan Index

Last organized: 2026-06-02.

This directory should contain only plans that still need decisions, reruns, or
follow-up experiments. Completed, superseded, or negative-result plans should
live in `../completed/`.

## Active

| Plan | Status | Why it remains active |
| --- | --- | --- |
| `boxed_matched_stage1_stage2_chain.md` | Active execution plan | Current boxed-prompt rerun: beta `0.0` Stage1 -> fixed Model2 merge -> beta `0.0` Stage2, then beta `0.1` Stage1 -> fixed Model2 merge -> beta `0.1` Stage2. |
| `plateau_handoff_stage1_stage2.md` | Active experiment plan, implementation ready | New registered experiment: take Model2 from early Stage1 plateau checkpoints (primary: step 60) and run short Stage2 (primary: 40 steps) to test whether early handoff preserves the local Stage2 gain while avoiding late collapse. Primary runnable wrappers and queue exist; results are not complete. |
| `kodcode_instruct2507_ctx8k_stage2_step60.md` | Active experiment design, script changes pending approval | New KodCode Instruct2507 CTX8K Stage2 plan: matched-beta Stage1 step60 -> Stage2 40 steps, effective step100, with same-budget Stage1 step100 and official offline eval as the comparison口径. |
| `kodcode_instruct2507_ctx8k_stage2_p40_submodel_kl_ablation.md` | Active experiment design, script changes pending Main Agent decision | Formal Stage2 per-model KL ablation: inherit the existing P40 beta `0.1`, `fusion_lambda=0.8`, step40 -> Stage2 60-step no-KL baseline, then add model1-only, model2-only, and both-on KL runs to test stability. |
| `kodcode_qwen3_1p7b_instruct_ctx8k_stage1_stage2.md` | Active experiment plan, scripted not launched | New Qwen3-1.7B code pipeline: final format cold-start SFT weights -> KodCode Stage1 beta `0.0/0.1` -> conservative cold-start Stage2 P40 lambda `0.8` no-KL vs model2-only KL, with raw step70/75/80 retained only as format-learning diagnostic evidence. |
| `qwen3_1p7b_code_stage123_plateau_breakthrough.md` | Active design, dry-run ready, formal training blocked | Primary FRAC25 step40 -> 20-step Stage2 -> 40-step Stage3 chain, with P60 approval-gated; all phases share one canonical L40S resource profile and full LiveCodeBench. |
| `qwen3_1p7b_stage123_model2_kl_split_stage3.md` | Active experiment plan, implementation ready | Two matched Stage2 arms (no KL vs model2-only KL), dual model1/model2 `n=3` validation, then independent Stage3 continuations from each extracted submodel. |
| `qwen3_1p7b_math_stage123.md` | Active experiment plan, V1/V2 invalidated, V3 not launched | Fixed-seed disjoint MATH shards and Math-7 validation; cold-start V1/V2 trained only answer tokens because Qwen Thinking per-turn rendering was masked incorrectly. V3 uses whole-message tokenization before any Stage123 handoff. |
| `rebuttal_rlvr_post_training_comparison.md` | G0 conditional-checkpoint assumption accepted; G1a/G1b/G2-local passed; model paths and Meituan G3/G4 evidence pending | Six-cell conditional comparison of colleague-supplied AM-1.4M ordinary-SFT and public offline-WDL-SFT checkpoints using identical 7,500-row MATH RLVR. Standard-GRPO v2, strict scorer, fixed seeds/order, direct worker bash entry, parallel Hope handoff, best/latest retention, and automatic post-success release are frozen. |
| `../goals/experiment-execution-core-consolidation/plan.md` | Goal Plan v0.2.0 review pending | First serial Goal: deliver a smaller CPU-verified execution core by removing Stage123 policy from shared layers and deleting redundant checker, receipt, queue, test, and review-document accretion. GPU qualification and experiment execution are deferred Goals. |
| `code_task_extension_on_policy_wdl_sft.md` | Active research plan, not implementation ready | Experiment B plan for extending Stage1 -> Stage2 On-Policy WDL-SFT to code tasks. It gates implementation on executable reward, sandbox/dependency, data-conversion, and offline-eval validation. |
| `code_task_scripts_implementation_plan.md` | Active implementation plan, not launched | Concrete script-development plan for code-task data conversion, reward smoke, Stage1/Stage2 wrappers, queue/monitor, offline eval, Meituan-ready env overrides, and shared main/reviewer acceptance checks. |
| `joint_submodel_kl_regularization.md` | Active implementation plan, reviewed READY / user-confirmed | Adds independent KL regularization for model1/model2 inside joint WDL-SFT, with per-submodel references, off-equivalence tests, and a blocking 5-step all-KL-mode local smoke matrix. |
| `wdl_group_advantage_is_goal.md` | Active implementation contract | New method revision: group advantages + all-correct positive-SFT fallback + explicit mixed-policy old/current IS for joint rollout/joint training, with `norm_adv_by_std_in_grpo=false`, no `rollout_is_weights`, no KL penalty, no length normalization, no `beta`, and a required complete Meituan four-layer launch family. |
| `wdl_sft_is.md` | Active, but historical sections are stale | The post-fix rerun matrix is still open. Pre-fix 1A/1B/1C results remain useful history, but should not be treated as spec-correct current runs. |
| `ablation_single_model.md` | Active / partially completed | The SFT-init ablation has useful results, but the post-fix rerun matrix is incomplete and several rows remain not started. |

## Experiment Design Documents

| Document | Purpose |
| --- | --- |
| `../../reports/deepcoder_preview_code_task_transfer_design.md` | DeepCoder-Preview code-task transfer design and data-domain rationale. This is not a `/goal` execution plan. |
| `../../reports/deepcoder_kodcode_failure_analysis.md` | Postmortem for the DeepCoder data switch: compares KodCode and DeepCoder Stage1 results, identifies reward sparsity plus stdin/stdout-vs-function interface mismatch, and records the decision basis for returning the main code-task experiments to KodCode. |

## Archived During This Cleanup

| Archived plan | Reason |
| --- | --- |
| `../completed/on_policy_wdl_sft_v1.md` | Superseded by `wdl_sft_is.md` on 2026-04-19. |
| `../completed/dual_submodel_rollout_wdl_sft.md` | Implementation completed and smoke-passed, but real 3A training produced a method-level negative result. |
| `../completed/dual_submodel_rollout_wdl_sft_goal.md` | Implementation contract preserved for traceability; no longer an active goal. |
| `../completed/dual_submodel_rollout_wdl_sft_status.md` | Final implementation/smoke/failure status preserved with the archived plan. |
| `../completed/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md` | Detailed gradient, loss, and distribution analysis for the 3A negative result. |
| `../completed/on_policy_sft_then_wdl_sft_beta_search.md` | Stage 1 implementation/smoke goal is complete and superseded by the boxed matched execution plan. |
| `../completed/on_policy_sft_then_wdl_sft_beta_search_status.md` | Status for the completed Stage 1 implementation/smoke goal. |
| `../completed/stage2_model2_rollout_fused_loss_fast_validation.md` | Original Stage 2 fast-validation goal reached local PASS; superseded by boxed-prompt rerun after prompt-format diagnosis. |
| `../completed/stage2_model2_rollout_fused_loss_fast_validation_status.md` | Status for the completed original Stage 2 fast-validation goal. |
| `../completed/deepcoder_stage1_training_execution_plan.md` | DeepCoder Stage1 execution contract is preserved, but the data switch is now a negative transfer result; main code-task experiments return to KodCode. |
