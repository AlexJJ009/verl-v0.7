# KodCode Qwen3-1.7B Format Cold-Start -> Stage1 -> Stage2

- Status: ACTIVE EXPERIMENT PLAN, SCRIPTED NOT LAUNCHED
- Updated: 2026-07-06
- Branch: `feature/on-policy-wdl-sft`
- Recipe family: `recipe/on_policy_wdl_sft/code_task/`
- Primary init: `/data-1/model_weights/format_cold_start/qwen3-1p7b-kodcode-format-sft`

## Objective

Move the Qwen3-1.7B KodCode experiment from raw chat/instruct init to the final
code format cold-start SFT weights, then test whether early Stage1 handoff into
Stage2 can improve code-task performance without spending Stage1 steps learning
the output format.

The current primary pipeline is:

```text
Qwen3-1.7B raw/chat -> code format cold-start SFT -> KodCode Stage1 -> Stage2
```

The final cold-start HF directory is:

```text
/data-1/model_weights/format_cold_start/qwen3-1p7b-kodcode-format-sft
```

This directory is the default `INIT_MODEL_PATH` for the cold-start Stage1
wrappers. The raw Qwen3-1.7B Stage1 scripts remain useful, but only as baseline
and diagnostic controls.

## Baseline Evidence From Raw Qwen3-1.7B

The prior raw Qwen3-1.7B Stage1 curve indicated that around step 75/80 the model
started to learn the required code answer format. That conclusion remains useful
as baseline evidence:

- raw Stage1 step 70/75/80 is mainly a format-learning diagnostic, not the new
  best Stage2 source;
- raw Stage1 step 100 or step 150 can still be used as same-family baselines;
- raw step40 should not be the primary Stage2 handoff, because it precedes the
  observed format transition and mixes format acquisition with policy learning.

The new question is whether code format cold-start moves useful Stage1/Stage2
handoff earlier: P20/P40/P60 are now candidate handoff points, with P40 as the
first conservative scripted queue.

## Experiment Questions

| Question | Decision signal |
| --- | --- |
| Does format cold-start remove the raw step75/80 format-learning delay? | Cold-start Stage1 should show valid fenced Python answer rates and online pass@1 before raw step75/80. |
| Which Stage1 beta should feed Stage2? | Compare beta `0.0` and beta `0.1` Stage1 curves under the same cold-start init. |
| Does Stage2 still help after the format problem is removed? | Compare Stage2 effective step100 against cold-start Stage1 same-budget checkpoints. |
| Does model2-only KL still help the deployable submodel? | At matched beta/lambda, model2-only KL should preserve or improve Stage2 final pass@1 vs no-KL. |
| Is lambda `0.8` enough for the first cold-start matrix? | Use lambda `0.8` as the default conservative matrix; expand to `0.6/0.7/0.8/0.9` only after the first four rows are interpretable. |

## Primary Matrix

### Stage1

| Run | Init | beta | Steps | Prefix |
| --- | --- | ---: | ---: | --- |
| S1-CS-B0 | code format cold-start SFT | `0.0` | 150 | `ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA0-V1` |
| S1-CS-B01 | code format cold-start SFT | `0.1` | 150 | `ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA01-V1` |

Stage1 settings:

- KodCode-Light-RL-10K train data
- HumanEval+, MBPP+, and LiveCodeBench v5 subset128 online validation
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=4096`
- `ROLLOUT_MAX_MODEL_LEN=8192`
- `TRAIN_PROMPT_BSZ=64`
- `ROLLOUT_N=8`
- `TOTAL_TRAINING_STEPS=150`
- `SAVE_FREQ=5`
- `TEST_FREQ=5`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.60`
- `GENERATION_MICRO_BATCH_SIZE=32`
- `LOG_PROB_MICRO_BATCH_SIZE=8`
- protected handoff checkpoints: `40,60,80,100,120,150`

### Stage2

Do not use raw Stage1 step40 as the main handoff. The first scripted Stage2
matrix uses cold-start Stage1 P40 as the conservative early-handoff candidate:

```text
handoff_step = 40
beta in {0.0, 0.1}, matched Stage1 -> Stage2
fusion_lambda = 0.8
KL mode in {no_kl, model2_only_kl}
```

That gives 4 rows by default. The queue supports extension to a lambda sweep by
setting:

```bash
COLDSTART_STAGE2_FUSION_LAMBDAS="0.6 0.7 0.8 0.9"
START_INDEX=...
END_INDEX=...
```

P20 and P60 remain the next handoff candidates after the P40 matrix is read. If
P40 is too late relative to the cold-start Stage1 curve, add P20. If P40 improves
but drifts, add P60 or shorten Stage2.

Common Stage2 settings:

- `STAGE2_HANDOFF_STEP=40`
- `TOTAL_TRAINING_STEPS=60`
- `JOINT_TRAINING_ROLLOUT_SOURCE=model2`
- `LOSS_MODE=wdl_sft`
- `LR=5e-7`
- `TRAIN_PROMPT_BSZ=64`
- `ROLLOUT_N=8`
- `TRAIN_PROMPT_MINI_BSZ=512`
- `DATA_SEED=20260604`
- `DATA_SHUFFLE=False`
- `VAL_N=1`
- `VAL_TEMPERATURE=0.2`
- `VAL_TOP_P=0.95`
- `TEST_FREQ=5`
- `SAVE_FREQ=5`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.45`
- `GENERATION_MICRO_BATCH_SIZE=32`
- `LOG_PROB_MICRO_BATCH_SIZE=8`
- `MAX_ACTOR_CKPTS_TO_KEEP=1`
- `KEEP_BEST_CKPT=True`

Model2-only KL rows:

```text
SUBMODEL_KL_ENABLED=true
SUBMODEL_KL_MODEL1_ENABLED=false
SUBMODEL_KL_MODEL1_COEF=0.0
SUBMODEL_KL_MODEL2_ENABLED=true
SUBMODEL_KL_MODEL2_COEF=0.01
SUBMODEL_KL_MODEL2_TYPE=low_var_kl
SUBMODEL_KL_MODEL2_REF_PATH=<merged cold-start Stage1 step40 Model2>
```

No-KL rows keep all per-submodel KL switches off.

## Runnable Scripts

Cold-start Stage1:

```text
recipe/on_policy_wdl_sft/code_task/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_beta_0.sh
recipe/on_policy_wdl_sft/code_task/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_beta_01.sh
recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage1_queue.sh
recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage1_notify.sh
```

Cold-start Stage2:

```text
recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_qwen3_1p7b_instruct_ctx8k_p40_common.sh
recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage2_p40_m2kl_vs_nokl_queue.sh
recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage2_p40_m2kl_vs_nokl_notify.sh
```

Existing raw 1.7B controls:

```text
recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage1_queue.sh
recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep_queue.sh
```

The cold-start queues reuse:

```text
recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh
recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh
scripts/training_queue_monitor.sh
```

Meituan single-wrapper routes are available for the cold-start Stage1 wrappers
and the common cold-start Stage2 wrapper:

```text
s1-code-kodcode-qwen3-1p7b-coldstart-ctx8k-beta-0
s1-code-kodcode-qwen3-1p7b-coldstart-ctx8k-beta-01
s2-code-kodcode-qwen3-1p7b-coldstart-ctx8k-p40-common
```

## Launch Gates

| Gate | Requirement |
| --- | --- |
| G0 | Cold-start HF directory exists and has config, tokenizer, chat template, `model.safetensors`, and `format_cold_start_source.json`. |
| G1 | Stage1 queue dry-run validates both wrappers. |
| G2 | Stage1 real queue has explicit `ALLOW_KODCODE_QWEN3_1P7B_COLDSTART_CTX8K_STAGE1_TRAINING=1`. |
| G3 | Stage2 queue dry-run validates selected rows and shows distinct `RUN_PREFIX`, `MODEL2_CACHE_TAG`, `FUSION_LAMBDA`, KL flags, and provenance paths. |
| G4 | Stage2 real queue has explicit `ALLOW_KODCODE_QWEN3_1P7B_COLDSTART_CTX8K_STAGE2_P40_M2KL_VS_NOKL_TRAINING=1`. |
| G5 | `/data-1` free space is at least 300G for Stage1 and 500G for Stage2. Do not delete checkpoints as part of launch; archive/delete requires separate path-level confirmation. |
| G6 | If the 1.7B throughput overrides OOM or destabilize vLLM/FSDP, first retry with `GENERATION_MICRO_BATCH_SIZE=16 LOG_PROB_MICRO_BATCH_SIZE=4`; keep `TRAIN_PROMPT_BSZ`, `ROLLOUT_N`, sequence length, validation decode, and data seed unchanged. |

## Fair Comparison Rules

- Count format cold-start SFT cost when comparing against raw Qwen3-1.7B.
- Compare cold-start Stage2 effective step100 against cold-start Stage1 same-budget checkpoints, not only against raw Stage1.
- Treat raw step70/75/80 as a format-learning diagnostic baseline. It is not a
  free compute-equivalent baseline for the cold-start pipeline.
- Report raw, cold-start Stage1, and cold-start Stage2 as separate families in
  any registry/import summary.
- Keep decode settings explicit for every online/offline comparison.
- Treat micro-batch and vLLM memory-utilization settings as throughput-only
  knobs. They may differ between Stage1 and Stage2 because Stage2 carries a
  larger joint-memory profile, but they must be recorded and must not be mixed
  with changes to `TRAIN_PROMPT_BSZ`, `ROLLOUT_N`, sequence length, data order,
  or decoding policy.

## Validation And Checkpoint Retention Tradeoff

The default keeps dense online validation every 5 steps and checkpoint save
every 5 steps. This is intentionally expensive: it gives enough resolution to
detect early cold-start effects, Stage2 peak-then-drift behavior, and handoff
candidate quality without rerunning the same matrix.

Use this policy for the first cold-start Stage1 pair and the first P40 Stage2
matrix:

- Keep `TEST_FREQ=5`, `SAVE_FREQ=5`, `VAL_N=1`, `VAL_TEMPERATURE=0.2`, and
  `VAL_TOP_P=0.95`.
- Stage1 protects handoff checkpoints `40,60,80,100,120,150`; these are the
  source of future P40/P60/P80/P100 Stage2 decisions. Steps 10/20/30 are kept
  in metrics only: they are useful for diagnosing cold-start learning dynamics,
  but not worth checkpoint retention unless a separate early-handoff ablation is
  explicitly requested.
- Stage2 keeps latest plus best, with save candidates every 5 steps; this avoids
  losing a short-run Stage2 peak between steps 40 and 60.
- After a checkpoint is selected for publication, merge/upload/verify the HF
  weight, then delete only the path-level confirmed local checkpoint.

If disk becomes the bottleneck, prefer reducing the matrix width before reducing
diagnostic resolution: run fewer lambda/KL rows first, then archive selected
checkpoints. Only lower `TEST_FREQ` or `SAVE_FREQ` for a separately named
fast-diagnostic family, because sparse validation can hide the exact peak or
drift point and force a retrain.

## Expected Results And Interpretation

| Observation | Interpretation |
| --- | --- |
| Cold-start Stage1 has valid format and pass@1 before raw step75/80 | Cold-start did its intended job; early P20/P40 handoff becomes meaningful. |
| Cold-start Stage1 only catches raw after step75/80 | SFT cost may not buy useful training acceleration; raw pipeline remains competitive after compute accounting. |
| Stage2 no-KL improves early but drifts by step60 | Same target-mismatch/drift pattern as 4B; model2-only KL remains justified. |
| Stage2 model2-only KL preserves no-KL peak and improves final | Evidence that model2 anchoring helps the deployable submodel. |
| All Stage2 rows underperform cold-start Stage1 same-budget baseline | Stage2 handoff is not justified for this scale/model/data setting. |
| beta `0.1` damages format or pass@1 under cold-start | Reverse term remains unsafe at 1.7B; prioritize beta `0.0`. |

## Reporting

Report online metrics separately from official offline eval.

Online metrics:

- HumanEval+ `pass@1`
- MBPP+ `pass@1`
- LiveCodeBench v5 subset128 `pass@1`
- `wdl_sft/correct_ratio`
- `response_length/clip_ratio`
- `response/aborted_ratio`
- `actor/grad_norm`
- `actor/submodel_kl/model2_loss`
- `actor/submodel_kl/model2_coef`
- `actor/submodel_kl/total_loss`

Offline eval candidates:

- cold-start Stage1 beta `0.0` and beta `0.1` best/final
- cold-start Stage2 P40 no-KL and model2-only KL rows for each beta
- raw Stage1 step70/75/80 only as diagnostic comparison
- any row whose online LCB subset gain changes the decision

Use the existing official code offline eval口径:

- HumanEval+
- MBPP+
- BigCodeBench
- LiveCodeBench `release_v5`
- `N_SAMPLES=3`
- `TEMPERATURE=1.0`
- `TOP_P=0.95`
- `MAX_TOKENS=4096`
- `SEED=42`
- `ENABLE_THINKING=true`
