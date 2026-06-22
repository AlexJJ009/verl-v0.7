# DeepCoder vs KodCode Code-Task Failure Analysis

- Status: EXPERIMENT RESULT REPORT
- Created: 2026-06-22
- Branch: `feature/on-policy-wdl-sft`
- Scope: Explain the evidence behind the degraded code-task Stage1 On-Policy
  WDL-SFT result after replacing KodCode with DeepCoder-Preview.
- Decision: return the primary code-task experiments to KodCode unless a
  follow-up DeepCoder ablation explicitly fixes data difficulty and interface
  mismatch.

## Executive Summary

The DeepCoder transfer should be treated as a negative result for the current
setup, not as proof that DeepCoder itself is unusable. The evidence does not
isolate a single causal factor, but it strongly supports the combination of:

1. DeepCoder produces far fewer correct on-policy rollouts than KodCode, leaving
   too few positive examples for the SFT part of the loss.
2. DeepCoder train tasks are mostly contest-style stdin/stdout programs, while
   the online validation suite, HumanEval+ and MBPP+, is function-completion
   EvalPlus. This creates a train/eval interface mismatch.
3. DeepCoder has much heavier hidden-test payloads and longer prompts. The
   reward is binary all-tests-pass, so harder tasks make the effective reward
   much sparser.
4. Increasing reverse-SFT weight from beta `0.1` to `0.5` increased the pressure
   on incorrect outputs but did not improve validation. It also increased late
   output length and truncation.

Therefore, the clean conclusion is:

> DeepCoder-Preview is not a good drop-in Stage1 replacement for KodCode under
> the current On-Policy WDL-SFT code-task setup. It changes both task difficulty
> and task interface, while the online validation remains function-style. The
> resulting correct rollout density is too low for the current SFT-style
> objective to learn reliably.

## Compared Runs

All runs use Qwen3-4B-Base, code-task online validation on HumanEval+ and MBPP+,
`VAL_N=1`, `VAL_TEMPERATURE=0.2`, `VAL_TOP_P=0.95`, and
`MAX_RESPONSE_LENGTH=4096`, unless noted otherwise.
The comparison is matched on the main training/validation knobs we control here
such as model family, LR, beta family, rollout count, validation decode, and
response length. It is not a strict apples-to-apples dataset-size or epoch
coverage comparison: DeepCoder changes data source, interface, prompt/test
payload, and effective train-set exposure.

| Run | Source metrics |
| --- | --- |
| KodCode beta `0.0` | `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION-R2_1780811946.jsonl` |
| KodCode beta `0.1` | `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION-R2_1780833499.jsonl` |
| DeepCoder beta `0.0` | `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780976139.jsonl` |
| DeepCoder beta `0.1` | `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781282660.jsonl` |
| DeepCoder beta `0.5` | `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA05-V1-FULL_1782059353.jsonl` |

The local experiment registry currently records the KodCode runs as trusted.
The DeepCoder runs above are parsed directly from metrics/log artifacts and have
not yet been imported into the registry.

Metric field definitions used below:

- `HumanEval+ best` / `MBPP+ best`: the maximum value of
  `val-core/<benchmark>/acc/pass@1` across validation checkpoints in that run.
- `HumanEval+ final` / `MBPP+ final`: the same fields at the last available
  validation checkpoint. For completed 150-step runs this is step 150; for
  DeepCoder beta `0.0`, the run stopped at train step 134 and the last
  validation checkpoint is step 130.
- `Train correct ratio`: `wdl_sft/correct_ratio`, a rollout-level batch ratio
  of correct sampled responses among the current training batch.
- `Response clip`: `response_length/clip_ratio`, the fraction of rollout
  responses clipped at `MAX_RESPONSE_LENGTH=4096`.

## Result Comparison

| Dataset / beta | Last train / val step | HumanEval+ best | HumanEval+ final | MBPP+ best | MBPP+ final | Train correct ratio final | Response clip final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KodCode beta `0.0` | 150 / 150 | 76.22% | 75.61% | 69.84% | 67.72% | 46.29% | 6.84% |
| KodCode beta `0.1` | 150 / 150 | 78.66% | 78.66% | 69.58% | 67.99% | 45.51% | 2.93% |
| DeepCoder beta `0.0` | 134 / 130 | 67.68% | 65.24% | 69.58% | 67.20% | 0.20% | 1.95% |
| DeepCoder beta `0.1` | 150 / 150 | 68.29% | 68.29% | 69.84% | 68.78% | 10.55% | 27.54% |
| DeepCoder beta `0.5` | 150 / 150 | 67.07% | 64.63% | 70.11% | 67.72% | 11.72% | 29.88% |

Reading:

- KodCode strongly improves HumanEval+ and reaches about `79%` pass@1.
- DeepCoder stays around `67-68%` HumanEval+ best and falls to `64.6%` final for
  beta `0.5`.
- DeepCoder beta `0.5` has the highest final correct ratio among DeepCoder runs,
  but that does not transfer to better online validation.
- DeepCoder beta `0.1` and `0.5` both show high final response clipping, around
  `28-30%`.

## Evidence for Reward Sparsity

The strongest direct evidence is on-policy train rollout correctness.

| Run | Correct ratio first | Correct ratio best | Correct ratio final |
| --- | ---: | ---: | ---: |
| KodCode beta `0.0` | 6.45% | 63.67% | 46.29% |
| KodCode beta `0.1` | 8.40% | 66.99% | 45.51% |
| DeepCoder beta `0.0` | 3.12% | 5.27% | 0.20% |
| DeepCoder beta `0.1` | 0.59% | 18.16% | 10.55% |
| DeepCoder beta `0.5` | 0.59% | 18.55% | 11.72% |

This is the clearest observed failure mechanism for an SFT-style objective.
KodCode creates a large pool of correct rollouts after the first few dozen
steps, so positive SFT has something meaningful to imitate. DeepCoder does not
under the current prompt, reward, and validation setup. Raising beta mostly
increases the influence of heterogeneous negative outputs instead of creating
more correct programs.

The beta `0.5` final losses show this clearly:

| Metric | DeepCoder beta `0.5` step 150 |
| --- | ---: |
| `actor/wdl_sft_loss_positive` | 707.30 |
| `actor/wdl_sft_loss_negative` | -1797.67 |
| `actor/wdl_sft_loss_total` | -191.54 |

The negative term dominates at beta `0.5`. This confirms that beta `0.5` changes
the loss scale, but in this run it did not fix the availability of high-quality
positive samples.

## Dataset Difference

Current dataset artifacts:

| Dataset | Rows | Parquet size | Prompt token stats | Prompt over 1024 | Difficulty proxy easy / medium / hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| KodCode-Light-RL-10K | 10,000 | 18.65 MB | mean 289; median 221; p90 549; p99 805 | 9 | 8,335 / 1,612 / 53 |
| DeepCoder-Preview prompt1024 | 19,241 | 4.40 GB | kept median 604; kept p90 854; kept p99 999 | already filtered | from unfiltered audit: 1,620 / 16,423 / 6,931 |

Additional DeepCoder source distribution after prompt1024 filtering:

| Source | Rows |
| --- | ---: |
| PrimeIntellect Synthetic-1 | 13,180 |
| TACO verified | 5,773 |
| LiveCodeBench v5 | 288 |

The manifest drift matters: older design docs mention larger intermediate row
counts, but the current train file used by recent runs is:

```text
/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet
kept_rows=19,241 from input_rows=20,707
```

The DeepCoder Stage1 wrappers set this path explicitly through `CODE_TRAIN_FILE`
and `TRAIN_FILE`:

```text
recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_0_retention.sh
recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_01_retention.sh
recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_05_full.sh
```

The most important structural difference is hidden-test load:

| Dataset | Test count mean | Median | Max |
| --- | ---: | ---: | ---: |
| KodCode | 5.71 | 6 | 18 |
| DeepCoder prompt1024 | 102.36 | 102 | 1,226 |

DeepCoder has about `18x` more tests per task on average. With binary
all-tests-pass reward, this likely contributes to partially correct solutions
being scored negative much more often.

The payload sizes show the same thing:

| Dataset | Compressed reward/test payload |
| --- | ---: |
| KodCode `ground_truth` column | 7.35 MB |
| DeepCoder `reward_model` column | 4,372.83 MB |

DeepCoder is not just more rows. Each row carries far more execution evidence,
which makes reward execution stricter and more expensive.

## Interface and Validation Mismatch

DeepCoder is mostly a stdin/stdout complete-program dataset. The prepared prompt
requires writing an executable Python program that reads from standard input and
writes to standard output.

KodCode is function-style. The reward contains an `entry_point` and pytest
hidden tests.

HumanEval+ and MBPP+ online validation are also function-completion EvalPlus
benchmarks. They score functions with `base_input` and `plus_input`, not
stdin/stdout contest programs.

This creates two mismatches:

1. **Training-interface mismatch:** DeepCoder teaches complete stdin/stdout
   programs, while the online validation tasks expect function definitions.
2. **Reward-density mismatch:** DeepCoder tasks often require passing around
   100 stdin/stdout tests, while KodCode/HumanEval+/MBPP+ are more forgiving
   function-style tasks in the context of this model.

This explains why DeepCoder can fail even if the reward executor itself is
working correctly.

## Reward and Extraction Health

The stable DeepCoder runs do not show a global reward-harness failure.

At DeepCoder beta `0.5` final validation:

| Signal | Value |
| --- | ---: |
| HumanEval+ extraction fail | 0.00% |
| MBPP+ extraction fail | 0.26% |
| HumanEval+ timeout | 1.22% |
| MBPP+ timeout | 0.26% |
| Compile/runtime/dependency errors | approximately 0 |

The raw final validation dump has 542 rows:

```text
recipe/on_policy_wdl_sft/code_task/validation/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA05-V1-FULL_1782059353/150.jsonl
pass=362, wrong_answer=176, timeout=3, extraction_fail=1
```

Therefore, the stable DeepCoder failures should not be attributed primarily to
broken extraction, sandboxing, or executor dependency failures.

There was one earlier DeepCoder beta `0.1` run with near-total format collapse:

```text
ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781184148
HumanEval+ pass@1 final = 0.61%
MBPP+ pass@1 final = 1.06%
extraction_fail around 98%
response_length clip around 97%
```

That run is useful as a failure-mode example, but it is not representative of
the later stable DeepCoder beta `0.1` or beta `0.5` runs.

## Output-Length and Repetition Failure Mode

DeepCoder training also triggers long and repetitive outputs:

| Run | Final response mean | Final clip ratio |
| --- | ---: | ---: |
| KodCode beta `0.0` | 682.6 | 6.84% |
| KodCode beta `0.1` | 484.9 | 2.93% |
| DeepCoder beta `0.1` | 2,074.1 | 27.54% |
| DeepCoder beta `0.5` | 2,160.1 | 29.88% |

For beta `0.5`, response length was healthy near the best checkpoint but
degraded by the final checkpoint:

| Step | HumanEval+ | MBPP+ | response mean | clip ratio |
| ---: | ---: | ---: | ---: | ---: |
| 55 | 67.07% | 67.20% | 677.0 | 4.88% |
| 150 | 64.63% | 67.72% | 2,160.1 | 29.88% |

This suggests that high beta does not repair sparsity. It can instead amplify
late-stage instability by pushing on many incorrect outputs.

## Alternative Explanations Considered

### 1. The dataset is simply too difficult

Supported, but incomplete.

Evidence:

- DeepCoder has many more medium/hard tasks.
- DeepCoder prompts are longer.
- DeepCoder has about `18x` more tests per task.
- DeepCoder on-policy correct ratio is much lower.

However, this explanation alone misses the interface mismatch with online
validation.

### 2. The train/eval task type changed

Strongly supported.

DeepCoder trains stdin/stdout complete programs. HumanEval+/MBPP+ validate
function completion. KodCode is much closer to HumanEval+/MBPP+ because it is
also function-style.

This is likely a major reason KodCode improves HumanEval+ strongly while
DeepCoder does not.

### 3. The reward or extraction implementation is broken

Not supported for the stable DeepCoder runs.

Stable beta `0.1` and beta `0.5` have low extraction fail, low timeout, and
near-zero compile/runtime/dependency errors on validation.

### 4. beta was too small

Not supported.

Raising beta from `0.1` to `0.5` increased correct ratio only slightly
(`10.55%` to `11.72%` final), while HumanEval+ final fell from `68.29%` to
`64.63%` and clip ratio rose from `27.54%` to `29.88%`.

### 5. DeepCoder needs more steps

Possible but not sufficient.

150 steps covers almost a full KodCode epoch but only about half of the current
DeepCoder prompt1024 training set. More steps may help data coverage, but the
observed late-stage output length and clipping suggest that simply extending the
same setup could worsen repetition and truncation unless the interface/difficulty
issue is addressed.

## Recommended Decision

Return the main code-task experiments to KodCode.

For the current On-Policy WDL-SFT SFT-stage objective, KodCode is a better
matched training dataset because:

- it gives enough correct on-policy rollouts for positive SFT;
- it is function-style, closer to HumanEval+/MBPP+;
- it does not induce the same late-stage response-length explosion;
- it already produced stronger HumanEval+ validation results.

DeepCoder should not be treated as a drop-in replacement. It should only be
revisited under one of these controlled ablations:

1. **Easy DeepCoder subset:** keep easy/medium tasks, cap test count at 20 or 30,
   and exclude high-payload sources such as LCBv5.
2. **Interface-matched validation:** add DeepCoder dev/official-test validation
   alongside HumanEval+/MBPP+ to separate in-domain learning from function-eval
   transfer.
3. **Function-style conversion:** convert a subset of DeepCoder tasks into
   function-wrapper tasks and compare against stdin/stdout prompts.
4. **Bucketed diagnostics:** report pass rate by `num_tests`, prompt length,
   source, and difficulty proxy.
5. **Offline code benchmark suite:** evaluate checkpoints on HumanEval+, MBPP+,
   BigCodeBench, LiveCodeBench, and DeepCoder official-test instead of selecting
   solely from HumanEval+.

Minimum acceptance for a future DeepCoder ablation should be defined before
launch. A practical gate is: train correct ratio should reach the same order of
magnitude as KodCode rather than staying below `20%`; response clip should stay
near the KodCode range rather than `25-30%`; and any claimed DeepCoder benefit
must be shown both in-domain on DeepCoder dev/official-test and out-of-domain on
HumanEval+/MBPP+.

## Practical Reporting Sentence

For paper or lab reporting:

> We attempted to replace KodCode with DeepCoder-Preview for the code-task
> Stage1 data. Under matched core training and validation settings, the transfer
> degraded HumanEval+ performance while also changing dataset size, train-set
> exposure, hidden-test payload, and task interface. Diagnostics show that
> DeepCoder produces much lower on-policy correct-rollout density and changes
> the task interface from function-style problems to stdin/stdout contest
> programs, while our online validation remains
> HumanEval+/MBPP+ function completion. Increasing the reverse-SFT coefficient
> to beta `0.5` did not solve the sparsity issue and increased late-stage output
> truncation. We therefore return the main code-task experiments to KodCode and
> leave DeepCoder for future interface- and difficulty-controlled ablations.
