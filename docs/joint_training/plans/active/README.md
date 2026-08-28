# Active Plan Index

Last organized: 2026-07-26.

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
| `qwen3_1p7b_code_stage123_plateau_breakthrough.md` | Paused after two Stage1 sources; no Stage2 started | The corrected Code matrix produced beta `0.0/0.1` Stage1 sources, then the queue stopped at Stage1-control admission. Code controls/Stage2/Stage3 are deprioritized while Math runs the core WDL causal test. |
| `qwen3_1p7b_code_acd0_p60_beta0.md` | A/C/D0 P60 completed; follow-up peak study planned | Transfers the successful Math 60-step A/C/D0 causal design to Code with beta `0`, lambda `0.8`, no Stage3, strict native-EOS admission, and the official Code-3 environment. A completed P60 at Code-3 mean@3 42.87%; D0 completed P60 but peaked at P20 and regressed to 35.29%; C completed P60 at 49.01%, giving C-D0 +13.72pp and C-A +6.14pp under the single-seed online endpoint. |
| `qwen3_1p7b_stage123_model2_kl_split_stage3.md` | Active experiment plan, legacy inputs invalidated | Two matched Stage2 arms and split Stage3 remain the design, but the historical answer-only Model1/Stage1 inputs cannot launch; regenerate the matrix after CoT-v3 retraining. |
| `qwen3_1p7b_math_stage123.md` | V3 and WDL-first causal P60 complete; common n=256 attached; multi-seed open | Math is primary: continuous C uses $0.2z_1+0.8z_2$; D0 is the matched-scale no-weak $0.8z_2$ control. Common offline confirms C's pass@1 gain but not all-k dominance; fixed-M1, GRPO and DynPerm extensions now refine the mechanism claim. |
| `qwen3_1p7b_standard_rlvr_grpo_matrix.md` | Strict-scorer Math retrain complete; common n=256 offline eval complete; multi-seed remains open | Math canonical GRPO baseline matrix now has aligned Cold P200, Stage1 effective P200, and C-P60→GRPO effective P200 trajectories plus common `n=256` pass@k. C remains stronger than both pure-GRPO arms; C+GRPO adds only `+0.405 pp` pass@1 and no high-k gain. Two pure-GRPO release-gate terminal events still need reconciliation before registry publication. |
| `qwen3_1p7b_offline_passk_diversity_eval.md` | Math A/C/D0/fixed-M1/strict-GRPO n=256 complete; DynPerm offline and Code matrix remain open | Common Math thinking-enabled evaluation uses Qwen3-native T=0.6, eight n=32 shards, exact `2,798 prompts / 716,288 responses / n=256` coverage, paired uncertainty and response-diversity receipts. It confirms a pass@1 gain for C while exposing high-k coverage trade-offs. |
| `qwen3_1p7b_on_policy_sft_baseline_extension.md` | Fixed-M1 Stage1 P60 training and common n=256 offline eval complete | Frozen weak guidance reproduces nearly all of C's pass@1 gain but trails C at pass@128/256; trainable Model1 is not necessary for the main pass@1 effect, while joint adaptation may affect tail coverage and stability. |
| `qwen3_1p7b_wdl_mechanism_program.md` | DynPerm rho=0/0.25/0.5 P60 complete; rho=1 Standard-C complete/fixed-M1 failed; lambda 0.7/0.9 online factorial complete; offline/multi-seed pending | DynPerm establishes dependence on real assignment or the cross-model geometry that the intervention also changes. The lambda factorial further rejects a single adaptive-feedback story: lambda 0.7 gives C > fixed > D0, while lambda 0.9 gives fixed > D0 ≈ C. Common offline, second seed, Align/true-Random/Anti and scalar-geometry controls remain required before stronger causal attribution. |
| `feishu_cli_experiment_knowledge_sync.md` | Active Research Hub architecture; implementation not started | Replaces the earlier Obsidian-like draft with a private Git-backed submodule: curated Feishu publication, remote-first human edits, recoverable three-way sync, attributable Git history, pre-push/root gates, local CI, periodic pull, and Feature Story acceptance. |
| `../goals/feishu-research-hub-sync/plan.md` | Goal Plan v1 independently reviewed READY; implementation not started | Frozen staged implementation and independent verification contract for the private Feishu Research Hub repository/submodule. |
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
| `../../reports/qwen3_1p7b_math_stage123_matrix_results_20260723.md` | Result attachment for the completed 16-run Qwen3-1.7B Math Stage123 matrix: all validation steps, effective-step curves, fixed-order data analysis, training-dynamics analysis, and conclusions linked back to the design plan. |
| `../../reports/qwen3_1p7b_wdl_mechanism_discovery_story_20260816.md` | Chinese long-form mechanism research narrative: reconstructs the evidence-to-hypothesis path from A/B/C/D0 and dual-model gains through reward-projected hard self-training, exact product-of-experts/Chernoff analysis, fixed-M1, Dynamic Permutation, geometry/semantic controls, no-online-WM1 controllers, reverse tests, and a Math-first execution contract. |
| `../../reports/on_policy_wdl_feishu_ssot.md` | Single-source-of-truth map and synchronization gate for the Git Markdown and complete Feishu publication replicas, including the rule that CSV evidence must be uploaded as a real attachment. |

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
