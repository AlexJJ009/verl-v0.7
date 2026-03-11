# Joint Training GRPO Progress

## Current Status

Stage 1 is complete.

The first stage covered implementation bring-up plus runtime stabilization of the new joint-training GRPO path. The milestone is now closed because the current launcher completed one end-to-end training session successfully:

1. `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh` completed `100/100` steps in `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262.log`.
2. Checkpoints were saved successfully at global steps `20`, `40`, `60`, `80`, and `100` under `/data-1/checkpoints/Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262/`.
3. Validation was executed every `5` steps and the log now prints merged validation and training metrics at each validation point.
4. A persistent local metrics file was created at `recipe/joint_training/metrics/JointTraining/Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262.jsonl`.

This does not mean the algorithm is semantically mature. It means the system now runs through the intended training lifecycle without the earlier infrastructure failures.

## Stage Boundary

### Stage 1: Finished

Scope:

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

Exit criteria met:

1. The recipe runs end to end on the target server.
2. Checkpoint save/resume paths are pinned to `/data-1`.
3. Metrics are visible in both W&B history and local logs.
4. Joint-training regression coverage exists for the stabilized paths.

### Stage 2: Starting Now

Scope:

1. Debug algorithm-internal and logic-level issues beneath the system bring-up layer.
2. Add joint-training-specific metrics so the fused-policy behavior is observable, not only the generic PPO/GRPO metrics.
3. Identify failure modes and fallback behavior that are intrinsic to joint training rather than generic runtime issues.
4. Improve robustness and compatibility so the joint algorithm behaves more like a mature first-class training path inside `verl`.

## What Landed In Stage 1

### Major Code/Recipe Milestones

1. `384804fe`: joint vLLM rollout support and eval-only weight extraction.
2. `6151c24` in `recipe/`: switch the joint GRPO recipe to vLLM rollout by default.
3. `d0c5d3a` in `recipe/`: rollout memory and checkpoint-path hardening.
4. `428a7e83`: FSDP actor rollout-path stabilization, entropy gating, and lower-peak fused-logit handling.
5. `73404180`: atomic FSDP checkpoint saves with disk-pressure protection.
6. `5bd62896`: colocated ZMQ socket paths moved off fragile root-mounted defaults.
7. `59a4c534`: recipe and test coverage refresh for stabilized behavior.
8. `5b3aca2` in `recipe/`: persistent local metrics logging.
9. `c4436d2b`: periodic test-step metric printing in the trainer.

### Infrastructure State Now

1. Checkpoint base dir defaults to `/data-1/checkpoints`.
2. `TMPDIR`, vLLM config, and ZMQ IPC roots are all on `/data-1`.
3. W&B is still offline by default, but metrics are also persisted locally.
4. The launcher auto-falls back when `flash_attn` is unavailable.
5. The trainer now emits merged train-plus-validation summaries at every validation step.

## Current Observations From The First Successful Run

1. The run is operationally stable, but validation quality is still poor on GSM8K:
   - `val-core/openai/gsm8k/acc/mean@1 = 0.0`
   - `val-aux/openai/gsm8k/reward/mean@1 = -1.0009765625`
2. This means the next phase is not “make it run”, but “understand whether the joint objective is correct, useful, and measurable”.
3. The end of the successful run still shows a W&B teardown `BrokenPipeError` in an `atexit` callback. It does not invalidate training results, but it is a cleanup nuisance worth tracking separately.

## Latest Stage 2 Instrumentation Pass

The current Stage 2 observability pass added the missing evidence needed to diagnose why joint GRPO currently receives no useful reward signal.

### What Changed

1. Validation generations are now surfaced in three places:
   - sampled examples to stdout/log via `trainer.log_val_generations`
   - full jsonl dumps under `trainer.validation_data_dir`
   - row-per-sample tracking tables with prompt, response, ground truth, score, and reward extra-info fields
2. Joint-training actor updates now emit:
   - `jointTraining/model1_grad_norm`
   - `jointTraining/model2_grad_norm`
3. Validation-step metric summaries now include `jointTraining/` metrics, so the joint-specific values appear in the same summary block as the usual train/validation metrics.
4. The vLLM async server now closes all reserved startup port sockets before launch. This fixed the real E2E regression where the first instrumentation rerun failed with `EADDRINUSE` on `data_parallel_master_port`.

### Evidence From The Latest Live Run

Run IDs:

1. `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obs-1773202012_1773202014.log`
   - failed during vLLM startup with `EADDRINUSE`
   - root cause: reserved startup sockets were not all released before vLLM bound its distributed ports
2. `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253.log`
   - completed the full `100/100` training schedule successfully after the port-lifecycle fix
   - dumped full validation generations to `recipe/joint_training/validation/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253/0.jsonl`
   - continued dumping follow-up validation generations through `recipe/joint_training/validation/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253/100.jsonl`
   - emitted the first training metrics to `recipe/joint_training/metrics/JointTraining/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253.jsonl`
   - saved checkpoints successfully at global steps `20`, `40`, `60`, `80`, and `100`
   - completed in `1:17:37`
   - all sampled validation rows so far used `verification_method = "verl_math_verify"` rather than the intended first-stage LaTeX semantic verifier
   - the final step still showed `val-core/openai/gsm8k/acc/mean@1 = 0.0`, `val-aux/openai/gsm8k/reward/mean@1 = -1.0`, `jointTraining/model1_grad_norm = 0.0`, and `jointTraining/model2_grad_norm = 0.0`
   - training still ends with the previously known W&B offline teardown nuisance: `BrokenPipeError` in an `atexit` callback after the run is already complete

The earliest post-validation training metrics from the fixed run show the expected instrumentation and the expected failure pattern:

1. `actor/grad_norm = 0.0`
2. `jointTraining/model1_grad_norm = 0.0`
3. `jointTraining/model2_grad_norm = 0.0`
4. `critic/score/min = -1.0`
5. `critic/rewards/max = -1.0`
6. `critic/advantages/mean = 0.0`

This confirms that the new metrics are wired correctly and that the current gradient collapse is downstream of the reward collapse, not a logging omission.

### What The Validation Samples Show

The new validation outputs make the failure mode more specific than the scalar metrics alone:

1. Some responses are coherent and even numerically correct, but still receive `score = -1.0` because the reward extractor cannot recover the expected final-answer format. A representative example produced the correct final number `7`, yet `pred = [NO_BOXED]`.
2. Some responses are fluent but semantically wrong. One sample returned `\\boxed{45000}` for a problem whose ground truth is `70000`.
3. Some responses are clearly off-domain or partially garbled, including mixed-language fragments and unrelated API/interface text.
4. By step `10`, the live validation dump also exposed a verifier mismatch: some rows already had extracted answers such as `pred = "21"` with `gts = "21"` and `pred = "18"` with `gts = "18"`, but still logged `answer_correct = false`, `score = -1.0`, and `verification_method = "verl_math_verify"`.
5. The verifier mismatch is not a one-off sample:
   - `0.jsonl`: `147 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `5.jsonl`: `151 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `10.jsonl`: `133 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `80.jsonl`: `167 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `100.jsonl`: `149 / 1319` rows had `pred == gts` but `answer_correct = false`
6. The format failure is also large-scale rather than anecdotal:
   - `0.jsonl`: `1075 / 1319` rows had `pred = [NO_BOXED]`
   - `5.jsonl`: `1060 / 1319` rows had `pred = [NO_BOXED]`
   - `10.jsonl`: `1079 / 1319` rows had `pred = [NO_BOXED]`
   - `95.jsonl`: `1091 / 1319` rows had `pred = [NO_BOXED]`
   - `100.jsonl`: `1071 / 1319` rows had `pred = [NO_BOXED]`
7. The late-run samples show that the failure modes persist all the way to the end of training:
   - step `80`: a row with `pred = "21"` and `gts = "21"` still received `answer_correct = false`
   - step `90`: a row with `pred = "7"` and `gts = "7"` still received `answer_correct = false`
   - step `100`: sampled outputs still include pure off-domain or garbled text such as isolated Hebrew or code-like junk
8. By step `50` and step `75`, the real run also emitted W&B warnings about serializing strings of `100754` and `104912` bytes while logging validation generations. They did not stop the run, but they show that sending all validation rows to tracking has a real payload cost.

Therefore the current `-1` reward saturation is not one bug class. At minimum it combines:

1. answer-extraction / formatting failures
2. ordinary reasoning errors
3. pathological response drift or corruption
4. at least one reward-verification mismatch even after answer extraction succeeds
5. failure modes that remain stable through a full `100`-step run rather than disappearing after early warmup

The instrumentation gap is now closed enough to distinguish those cases on the next run.

## Stage 2 Priorities

1. Add joint-training-specific metrics:
   - fused-vs-submodel logit drift
   - per-submodel contribution statistics
   - fusion-lambda-sensitive KL / entropy / disagreement metrics
   - eval-only model2 versus fused-policy gap
2. Audit semantic correctness of:
   - rollout logits
   - old-log-prob recomputation
   - reference-policy comparisons
   - validation/eval-only weight extraction
3. Hard-test edge cases:
   - resume from checkpoints
   - partial checkpoint retention
   - fallback paths without `flash_attn`
   - W&B online/offline switching
   - vLLM and HF rollout parity where applicable
4. Reduce the number of recipe-only guards by moving stable protections into reusable framework code where that is justified.

## Regression Coverage Status

Recent regression and feature coverage now includes:

1. `tests/joint_training/`
2. `tests/workers/actor/test_special_dp_actor.py`
3. `tests/joint_training/regression/test_validation_generation_logging.py`
4. `tests/joint_training/feat/test_vllm_joint_rollout.py`
5. `tests/workers/critic/test_dynamic_dp_critic.py`
6. `tests/workers/test_fsdp_workers.py`
7. `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py`
8. `tests/utils/test_attention_utils_on_cpu.py`
9. `tests/utils/test_torch_functional.py`

The latest verification pass for `tests/joint_training` plus `tests/workers/actor/test_special_dp_actor.py` completed successfully with `157 passed`.

## Pending

The following metrics are not yet implemented, but they are now high-priority candidates for the next Stage 2 pass:

1. `jointTraining/answer_extraction_failure_rate`
   - example basis: `pred = [NO_BOXED]`, empty extraction, malformed final answer markers
2. `jointTraining/verification_method/*` distribution
   - needed to tell whether all rewards come from the same verifier path or from mixed fallback behavior
3. `jointTraining/fused_vs_eval_model2_acc_gap`
   - compare fused-policy validation against the eval-only model2 path directly
4. `jointTraining/model_grad_norm_ratio`
   - `model1_grad_norm / model2_grad_norm` to expose chronic update imbalance
5. `jointTraining/model_grad_cosine_similarity`
   - needed to see whether the two submodels are learning in aligned or conflicting directions
6. `jointTraining/submodel_logit_disagreement`
   - mean/max absolute fused-input disagreement before fusion
7. `jointTraining/response_unprintable_or_non_ascii_ratio`
   - useful because the latest validation samples show both garbled fragments and off-domain multilingual drift
8. `jointTraining/verifier_pred_gt_disagreement_count`
   - count rows where extracted `pred` already matches `gts` textually but `answer_correct` is still false
9. `jointTraining/validation_tracking_payload_bytes`
   - monitor the size cost of sending all validation rows to tracking backends

## Companion Documents

1. `docs/joint_training/verl_joint_training_investigation_report.md`
2. `docs/joint_training/GRPO_Joint_Training_Target_v1.md`
3. `docs/joint_training/server_migration_pitfalls.md`
4. `docs/joint_training/stabilization_experience_notes.md`
5. `docs/joint_training/runReport/2026-03-11_joint_validation_logging_and_grad_norm_report.md`
