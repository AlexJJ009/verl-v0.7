# Joint GRPO Validation Logging And Joint Gradient Metrics Report

## 1. Background: Why This Task Was Necessary

The latest joint-training log showed a clear symptom cluster:

1. `critic/score/min` stayed at `-1`
2. `critic/rewards/max` stayed at `-1`
3. `val-core/openai/gsm8k/acc/mean@1` stayed at `0`
4. the PPO / GRPO losses and gradient signal collapsed toward `0`

Those scalar metrics were enough to prove that training was not learning, but not enough to explain why.

Before this patch, the log did not print validation prompt / response samples. That meant we could not distinguish among three very different root causes:

1. the model output was garbled
2. the model output was readable but mathematically wrong
3. the model output was close or even correct, but the reward extractor could not parse the expected final-answer format

This task therefore had two concrete goals:

1. expose validation content in a debug-friendly way
2. expose per-submodel gradient norms so both sides of joint training can be observed directly

## 2. Code Changes

### 2.1 Validation logging and tracking

The validation logging path was extended in these places:

1. `verl/trainer/ppo/ray_trainer.py`
   - build structured validation sample rows from prompt, response, ground truth, score, `uid`, `data_source`, and reward extra info
   - print a bounded number of validation samples directly to stdout / run log
   - allow a separate limit for tracking backends so logs can stay small while tracking can still receive the full validation set
   - include `jointTraining/` metrics in the periodic validation-step metric summary
2. `verl/utils/tracking.py`
   - change validation generation logging to a row-per-sample table instead of a single wide row
   - preserve useful reward-debug fields such as `pred`, `answer_correct`, `has_eos`, and `verification_method`
3. `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh`
   - add `VAL_GENERATIONS_TO_LOG`, default `3`
   - add `VAL_GENERATIONS_TO_TRACKING`, default `-1`
   - add `VALIDATION_DATA_DIR`, default `recipe/joint_training/validation/${WANDB_RUN_NAME}`
   - pass the new knobs into the trainer config

### 2.2 Joint gradient metrics

The joint gradient metrics were added in these places:

1. `verl/utils/torch_functional.py`
   - add `compute_global_grad_l2_norm()` so grad norms can be computed with distributed all-reduce semantics
2. `verl/workers/actor/dp_actor.py`
   - locate the two joint submodels from the actor module
   - compute `jointTraining/model1_grad_norm`
   - compute `jointTraining/model2_grad_norm`
   - emit both metrics at the same optimizer-step boundary as `actor/grad_norm`, before clipping

### 2.3 E2E-only runtime fix uncovered during the rerun

The first real rerun after the logging changes failed before training could proceed:

1. file: `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
2. symptom: vLLM startup `EADDRINUSE` inside distributed `TCPStore`
3. root cause: `_master_sock` was closed before launch, but `_dp_rpc_sock` and `_dp_master_sock` were left reserved
4. fix: close all reserved startup sockets together before the real server bind

This bug was not the original goal of the task, but it blocked the real E2E verification and therefore had to be fixed.

## 3. What The New Logging Now Shows

The new validation output is intentionally split across three sinks:

1. stdout / log:
   - a small sample, default `3`
   - readable enough for humans during live debugging
2. tracking:
   - configurable count, currently defaulted to all validation samples with `-1`
   - suitable for W&B table inspection
3. disk:
   - full jsonl dumps in `trainer.validation_data_dir`

Each validation row now carries enough information to reason about reward failure:

1. prompt
2. response
3. ground truth
4. reward / score
5. extracted prediction
6. extraction / verification metadata

This closes the earlier observability gap where a scalar `-1` reward said “something failed” but not “what failed”.

## 4. Tests Executed

All Python / pytest commands were run inside the Docker container (`verl-train:cu126`):

```bash
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29517
python -m pytest tests/joint_training tests/workers/actor/test_special_dp_actor.py
```

Result:

1. `157 passed`
2. the new or extended coverage includes:
   - `tests/joint_training/regression/test_validation_generation_logging.py`
   - `tests/workers/actor/test_special_dp_actor.py`
   - `tests/joint_training/feat/test_joint_training_recipe_script.py`
   - `tests/joint_training/feat/test_vllm_joint_rollout.py`
   - `tests/joint_training/regression/test_test_step_metric_logging.py`

What those tests verify:

1. validation samples are built with reward extra-info preserved
2. tracking tables are row-per-sample
3. stdout and tracking limits for validation generation logging behave as intended
4. joint submodel grad norms are emitted during actor update
5. the recipe script exports and passes the new logging knobs
6. the vLLM async server closes every reserved port socket before launch

## 5. Real E2E Training Runs

### 5.1 First rerun: instrumentation exposed a startup regression

Command:

```bash
RUN_PREFIX=Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obs-$(date +%s) \
bash recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh
```

Observed result:

1. log: `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obs-1773202012_1773202014.log`
2. failure: `EADDRINUSE`
3. location: vLLM startup path on `data_parallel_master_port`

This run proved that the new trainer / recipe config was wired, but the job could not reach validation because of the reserved-port lifecycle bug.

### 5.2 Second rerun: validation logging and joint metrics are live

Command:

```bash
RUN_PREFIX=Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-$(date +%s) \
bash recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh
```

Observed artifacts:

1. log:
   - `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253.log`
2. validation dumps:
   - `recipe/joint_training/validation/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253/0.jsonl`
   - `recipe/joint_training/validation/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253/5.jsonl`
   - `recipe/joint_training/validation/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253/10.jsonl`
   - ... through `recipe/joint_training/validation/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253/100.jsonl`
3. local metrics:
   - `recipe/joint_training/metrics/JointTraining/Joint-GRPO-Qwen3-1.7B-GSM8K-stage2obsfix-1773202252_1773202253.jsonl`
4. training progress:
   - the run completed `100 / 100` training steps in `1:17:37`
   - checkpoint saves completed successfully at global steps `20`, `40`, `60`, `80`, and `100`
   - the final validation summary still reported `val-core/openai/gsm8k/acc/mean@1 = 0.0`
   - the final training summary still reported `actor/grad_norm = 0.0`, `jointTraining/model1_grad_norm = 0.0`, and `jointTraining/model2_grad_norm = 0.0`
5. verifier path:
   - all rows in every validation dump from `0.jsonl` through `100.jsonl` used `verification_method = "verl_math_verify"`
   - the intended first-stage LaTeX semantic verifier did not appear in the live validation data

Observed validation behavior:

1. format failure is real:
   - some responses are coherent, but `pred = [NO_BOXED]`
   - example: a gum-counting problem produced a readable answer but still received `score = -1.0`
2. semantic failure is also real:
   - one sampled answer produced `\\boxed{45000}` even though the ground truth is `70000`
3. off-domain or garbled generation is also real:
   - one sampled answer drifted into unrelated `ApiModelProperty` / JAX-RS text mixed with Korean
4. verifier mismatch is also real:
   - by step `10`, the validation dump contained rows such as `pred = "21"` with `gts = "21"` and `pred = "18"` with `gts = "18"`, yet `answer_correct = false`, `score = -1.0`, and `verification_method = "verl_math_verify"`
5. the verifier mismatch is statistically significant, not anecdotal:
   - `0.jsonl`: `147 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `5.jsonl`: `151 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `10.jsonl`: `133 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `80.jsonl`: `167 / 1319` rows had `pred == gts` but `answer_correct = false`
   - `100.jsonl`: `149 / 1319` rows had `pred == gts` but `answer_correct = false`
6. the formatting issue is also large-scale:
   - `0.jsonl`: `1075 / 1319` rows had `pred = [NO_BOXED]`
   - `5.jsonl`: `1060 / 1319` rows had `pred = [NO_BOXED]`
   - `10.jsonl`: `1079 / 1319` rows had `pred = [NO_BOXED]`
   - `95.jsonl`: `1091 / 1319` rows had `pred = [NO_BOXED]`
   - `100.jsonl`: `1071 / 1319` rows had `pred = [NO_BOXED]`
7. the late-run samples prove that these failures persist to the end of training:
   - step `80` still contains rows with `pred = "21"` and `gts = "21"` but `answer_correct = false`
   - step `90` still contains rows with `pred = "7"` and `gts = "7"` but `answer_correct = false`
   - step `100` still contains clearly off-domain or garbled outputs, including isolated Hebrew text and code-like junk
8. by step `50` and again by step `75`, W&B warned about serializing strings of `100754` and `104912` bytes while logging validation data
9. the run still ended with the pre-existing W&B teardown `BrokenPipeError` in the `atexit` callback
10. therefore the reward collapse is not explained by one bug class alone

Observed training metrics:

1. `actor/grad_norm = 0.0`
2. `jointTraining/model1_grad_norm = 0.0`
3. `jointTraining/model2_grad_norm = 0.0`
4. `critic/score/min = -1.0`
5. `critic/rewards/max = -1.0`
6. `critic/advantages/mean = 0.0`

Teaching point:

This was not a one-off success limited to the initial validation. The second validation cycle at step `10` still produced:

1. sampled validation outputs in the main log
2. full jsonl generation dumps
3. merged validation-plus-training summaries containing `jointTraining/model1_grad_norm` and `jointTraining/model2_grad_norm`
4. `val-core/openai/gsm8k/acc/mean@1 = 0.0` in the merged validation summary

Interpretation:

1. the new metrics are correctly wired into the real training job
2. the current gradient collapse is a downstream consequence of reward collapse
3. the logging gap is no longer the main blocker
4. the live run now points to a likely reward-verification bug in addition to format and generation issues

## 6. Why The New Metrics Matter

The two new joint metrics answer a question that generic PPO logging cannot answer:

1. is model1 receiving gradient?
2. is model2 receiving gradient?
3. are both being updated at comparable scale?

In the current run both values are `0.0`, but that is still useful:

1. it proves the metrics are present in the real job
2. it proves the failure is upstream of parameter update magnitude
3. once reward starts moving, these metrics will immediately show whether one submodel dominates the learning signal

## 7. Recommended Next Metrics

The following metrics were not implemented in this task, but they should be the next additions:

1. answer-extraction failure rate such as `[NO_BOXED]`
2. parse-failure buckets by `verification_method`
3. response garbling indicators such as non-printable or abnormal character ratio
4. `jointTraining/model_grad_norm_ratio`
5. `jointTraining/model_grad_cosine_similarity`
6. fused-policy versus eval-only model2 validation gap
7. submodel logits disagreement statistics before fusion
8. a verifier-disagreement counter for rows where extracted `pred` already matches `gts` textually but `answer_correct` is still false
9. a validation-tracking payload metric so “log all rows to tracking” can be monitored instead of guessed

## 8. Status At Report Time

At the time this report was finalized:

1. the code changes were in place
2. the test suite passed
3. the second real run completed `100 / 100` steps successfully
4. checkpoint saves succeeded at every scheduled boundary: `20`, `40`, `60`, `80`, and `100`
5. the observability changes remained stable for the entire run
6. the algorithmic failure also remained stable for the entire run:
   - `val-core/openai/gsm8k/acc/mean@1 = 0.0`
   - `val-aux/openai/gsm8k/reward/mean@1 = -1.0`
   - `actor/grad_norm = 0.0`
   - `jointTraining/model1_grad_norm = 0.0`
   - `jointTraining/model2_grad_norm = 0.0`

The main result of this task is therefore not “the model is fixed”. The main result is that the next debugging round can now distinguish:

1. formatting and answer-extraction failure
2. semantic reasoning failure
3. off-domain or garbled generation
4. reward-verifier mismatch after answer extraction already succeeds
