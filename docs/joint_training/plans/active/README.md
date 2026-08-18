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
| `qwen3_1p7b_math_stage123.md` | V3 matrix complete; WDL-first causal P60 implementation in admission | Math is primary: continuous C uses $0.2z_1+0.8z_2$; D is direct strong-only $z_2$ and D0 is the required matched-scale no-weak $0.8z_2$ control. The plan separates additive weak-logit contribution, practical value, pipeline allocation, and stability, and defines manipulation, loss, dual-view validation, health, logit, and confirmatory metrics. |
| `qwen3_1p7b_standard_rlvr_grpo_matrix.md` | Active retrain contract; 2026-08-18 audit withdrew old Math GRPO and Code Job 38 from formal fairness claims | Math+Code canonical GRPO baseline matrix: strict task reward contracts, exact config allowlist, continuous 100-step Cold Start + GRPO versus 60-step Stage1 + GRPO, prompt batch 64, group N=8, actor KL 0.001, common n=256 pass@k, multi-seed and cost accounting. |
| `qwen3_1p7b_offline_passk_diversity_eval.md` | Active evaluation design; 8×TP1 L40S entry and sharding/merge code prepared, scheduler canary pending | Common Math+Code thinking-enabled offline evaluation for CS0/S1-P0/A/C/D0/GRPO: Qwen3-native T=0.6 decoder with an explicit non-universal-standard boundary, pass@1...256, response diversity, 8×n32 shards, exact-coverage merge, official code scorers, paired uncertainty, and cost receipts. |
| `qwen3_1p7b_on_policy_sft_baseline_extension.md` | Active design/report; fixed-M1 code and dry-runs complete, no new training launched | Math-first baseline and mechanism extension: OPSFT/GFT literature audit, CS0/Stage1 fixed-WM1 P60 controls, Dynamic Perturbation evidence boundary and no-online-WM1 surrogate ladder, plus a later continual-learning design. |
| `qwen3_1p7b_wdl_mechanism_program.md` | Active theory-derived mechanism design; DynPerm implementation not started | Unifies reward-projected hard self-distillation with an exact product-of-experts/Chernoff decomposition, then defines Math-first fixed-M1, weak-entropy-preserving permutation, affinity-direction controls, a no-online-WM1 target-margin controller, and reverse rollout/loss experiments. |
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
