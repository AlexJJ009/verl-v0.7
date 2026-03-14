# Joint GRPO Run 1773032262: Metric Collapse Analysis

## Scope

This note analyzes the completed run:

- Log: `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262.log`
- Local metrics: `recipe/joint_training/metrics/JointTraining/Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262.jsonl`

The purpose is not to change the implementation yet. The purpose is to identify the actual root causes behind:

1. response length collapsing to about `2` tokens,
2. `actor/grad_norm` peaking abnormally high before collapse,
3. `actor/grad_norm`, `actor/pg_loss`, and `critic/advantages/*` becoming `0`,
4. why this happened in the current joint-training design and recipe.

## Executive Conclusion

The dominant cause of the collapse in this run is **not** the fused-logit principle by itself.

The dominant cause is a **reward-design/configuration error**:

- `overlong_buffer_len == max_response_length == 1024` in the recipe
- the DAPO reward manager therefore computes `expected_len = max_resp_len - overlong_buffer_len = 0`
- this makes **every non-empty response receive a length penalty**
- because the run produced **no positive reward samples at all**, the optimizer learned the only remaining easy strategy:
  **emit EOS as early as possible**

That is why:

- response length fell from about `385` to `2`,
- validation reward looked "better" while validation accuracy stayed `0.0`,
- once all `n=4` rollouts per prompt converged to the same 2-token wrong answer, GRPO groupwise standardization produced zero advantages,
- then `actor/grad_norm`, `actor/pg_loss`, and the PPO update all went to zero.

The joint-training design still has real Stage 2 issues, but they are **secondary** in explaining this particular run:

1. the training policy is the fused policy, while validation uses `model2` only,
2. the two sub-models are initialized from the same base weights and are not symmetry-broken,
3. fused-logit training uses a normalized geometric mean over token probabilities, so disagreement on EOS or answer-format tokens can suppress those tokens.

Those are real design risks. They are just not the first-order cause of the observed collapse in this run.

## What The Metrics Show

### Phase 1: Normal-length but all-negative regime

Early steps have long responses and non-zero gradients:

- `step 1`: `response_length/mean = 308.09`, `actor/grad_norm = 2.11`
- `step 5`: `response_length/mean = 355.09`, `actor/grad_norm = 1.97`
- `step 10`: `response_length/mean = 205.15`, `actor/grad_norm = 4.97`

But an important warning sign is already present:

- `critic/score/max` is never positive
- in fact, over the whole run, `max(critic/score/max) = -1.0009765625`

So even before collapse, the sampled rollouts are not producing any positively rewarded answers.

### Phase 2: Collapse frontier

The collapse starts between steps `15` and `27`:

| Step | Resp len mean | Reward mean | Grad norm | PG loss |
|---|---:|---:|---:|---:|
| 15 | 32.64 | -1.01594 | 14.01 | 2.0993 |
| 16 | 13.04 | -1.00637 | 17.64 | 1.4512 |
| 17 | 12.96 | -1.00633 | 23.53 | 1.1952 |
| 18 | 4.88 | -1.00238 | 29.02 | 0.9725 |
| 19 | 2.69 | -1.00131 | 16.76 | 0.4067 |
| 20 | 2.09 | -1.00102 | 19.29 | 0.1188 |
| 25 | 2.02 | -1.00098 | 8.76 | 0.0346 |
| 26 | 2.01 | -1.00098 | 2.27 | 0.0131 |
| 27 | 2.00 | -1.00098 | 0.00 | 0.0000 |

At `step 18`, the raw logged gradient norm peaks at about `29.02`. By `step 27`, the system is fully collapsed.

### Phase 3: Fully degenerate regime

From `step 27` onward, the run is almost flat:

- `response_length/mean = 2.0`
- `response_length/min = 2.0`
- `response_length/max = 2.0`
- `critic/score/mean = -1.0009765625`
- `critic/advantages/mean = 0.0`
- `critic/advantages/min = 0.0`
- `critic/advantages/max = 0.0`
- `actor/grad_norm = 0.0`
- `actor/pg_loss = 0.0`

The run still "moves" in wall-clock time, but learning is already dead.

## Why Validation Looked Slightly Better While Accuracy Stayed 0

Validation metrics stayed:

- `val-core/openai/gsm8k/acc/mean@1 = 0.0`
- `val-aux/openai/gsm8k/answer_correct/mean@1 = 0.0`
- `val-aux/openai/gsm8k/has_eos/mean@1 = 1.0`

Yet validation reward moved from about `-1.1608` to `-1.0010`.

This was not genuine improvement in reasoning quality.

It can be explained almost entirely by the length penalty becoming smaller as responses became shorter.

For example:

- validation reward at step `0`: `-1.160777...`
- validation reward at step `20`: `-1.001056...`
- validation reward at step `30+`: `-1.0009765625`

Given the current reward formula, the terminal value `-1.0009765625` corresponds exactly to a wrong answer with response length `2`.

So the validation "improvement" was mostly:

`wrong but shorter`

not:

`more correct`

## Primary Root Cause: Overlong Penalty Is Active For Every Response

### Relevant code path

In the recipe:

- `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh`
  - `overlong_buffer_len=$((1024 * 1))`
  - `max_response_length=1024`

In the reward manager:

- `verl/workers/reward_manager/dapo.py`
  - `expected_len = self.max_resp_len - overlong_buffer_len`
  - `exceed_len = valid_response_length - expected_len`
  - `overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)`

With the current recipe values:

- `max_resp_len = 1024`
- `overlong_buffer_len = 1024`
- therefore `expected_len = 0`

So for every non-empty response of length `L`:

`overlong_reward = min(-L / 1024 * 0.5, 0) = -L / 2048`

That means every token is penalized, not just overlong responses.

### Exact consequence

The custom reward function first returns:

- correct with EOS: `+1.0`
- wrong with EOS: `-1.0`
- no EOS: `-1.0`

Then DAPO adds the length penalty.

So the actual reward becomes:

- correct: `1 - L/2048`
- wrong: `-1 - L/2048`

Examples:

| Response length `L` | Wrong reward | Correct reward |
|---|---:|---:|
| 2 | `-1.0009765625` | `0.9990234375` |
| 5 | `-1.00244140625` | `0.99755859375` |
| 32 | `-1.015625` | `0.984375` |
| 205 | `-1.10009765625` | `0.89990234375` |
| 355 | `-1.17333984375` | `0.82666015625` |
| 1024 | `-1.5` | `0.5` |

### Why this causes collapse

If positive rewards existed often enough, correctness would still dominate length because the correct-vs-wrong gap remains `2.0`.

But that is not what happened in this run.

What happened is:

1. there were effectively no positive samples,
2. the reward landscape therefore consisted almost entirely of negative values,
3. among wrong answers, shorter responses are always better,
4. the easiest way to maximize reward is to terminate immediately.

This precisely matches the observed trajectory.

## Secondary Root Cause: GRPO Collapses To Zero Once Group Rewards Become Identical

### Relevant code path

The recipe uses:

- `n_resp_per_prompt = 4`

The trainer repeats each prompt `n=4` times and groups by `uid`.

GRPO advantage is computed in:

- `verl/trainer/ppo/core_algos.py`
  - `compute_grpo_outcome_advantage()`

It computes, per prompt-group:

`adv_i = (r_i - mean(group)) / (std(group) + epsilon)`

and then broadcasts that scalar across response tokens using `response_mask`.

### Why this produces zero gradients

Once all four responses for the same prompt collapse to the same 2-token wrong answer:

- all four rewards are the same
- group standard deviation becomes `0`
- each `(r_i - mean(group))` is also `0`
- therefore all advantages become `0`

Once advantages are zero:

- PPO policy loss is zero
- actor gradients are zero
- gradient norm is zero

This is exactly what the metrics show from `step 27` onward.

## Why The Gradient Norm Spike Is Real But Not What It First Looks Like

The reported `actor/grad_norm` is the value returned by `clip_grad_norm_()` in:

- `verl/workers/actor/dp_actor.py`

That is the **pre-clipping total norm**, not the post-clipping update magnitude.

So a peak like `29.0` does **not** mean the optimizer applied a norm-29 update. The actual update was clipped by:

- `actor_rollout_ref.actor.grad_clip = 1.0`

Still, the spike is meaningful. It says the raw policy gradient became highly coherent near the collapse boundary.

The likely mechanism is:

1. reward increasingly favors short outputs,
2. the policy starts placing mass on early EOS,
3. many sequences become very similar,
4. under `loss_agg_mode="token-mean"`, gradients concentrate on very few early response tokens,
5. the same early-stop direction is reinforced across many samples at once.

So the spike is a genuine signal of collapse pressure, even though the actual applied update is clipped.

## Joint Training Design Issues That Matter For Stage 2

These are not the main cause of this run's collapse, but they are real algorithmic concerns.

### 1. Fused logits form a normalized geometric mean

The joint model fuses logits as:

`z_fused = (1 - lambda) * z1 + lambda * z2`

This implies:

`p_fused(y)` is proportional to `p1(y)^(1-lambda) * p2(y)^lambda`

So a token only receives large fused probability when both sub-models agree on it.

This means the user's concern is mathematically valid in general:

- if model1 wants EOS strongly,
- but model2 wants to continue,
- the fused EOS probability can be substantially reduced.

Simple 2-way example with `lambda = 0.5`:

- if `p1(EOS)=0.8` and `p2(EOS)=0.05`, then `p_fused(EOS) ~= 0.3145`

So EOS and answer-format disagreement can indeed distort rollout behavior.

### 2. But this exact run was not primarily an EOS-disagreement run

The joint weights are prepared by duplicating the same base model into:

- `sub_models.0.*`
- `sub_models.1.*`

and the run used:

- `fusion_lambda = 0.5`
- `freeze_model1 = false`

So the current system starts from a perfectly symmetric duplicated model and does not explicitly break symmetry.

That means the intended "model1 wants EOS, model2 does not" behavior is not the first thing to blame in this run. In the current implementation, the two branches are much closer to:

- duplicated copies of the same base policy

than:

- deliberately complementary policies with meaningfully different termination behavior

This is a core Stage 2 design issue: the present implementation does not yet create a strong reason for the two branches to specialize.

### 3. Training policy and validation policy are not the same

During training rollout, the fused policy is used.

During validation, the trainer switches to `eval_only=True`, and the rollout uses `model2` weights only.

So the system trains one policy but validates another.

This creates a structural observability problem:

- training metrics describe the fused policy,
- validation metrics describe model2 alone.

If those two drift apart, current metrics will not tell us clearly whether:

1. the fused training policy improved,
2. model2 improved,
3. one improved while the other got worse.

## Additional Concern: Reward Contract Mismatch With GSM8K Prompt Format

The GSM8K dataset prompt explicitly asks the model to:

- "output the final answer after `####`"

But the custom reward function is LaTeX-oriented first:

- LaTeX semantic parse
- then `verl_math_verify`
- then boxed-answer string fallback

This does not prove the reward function is wrong.

However, it does mean the reward contract is no longer the original GSM8K `####` contract.

That is a Stage 2 audit item, because if the model follows the prompt format but the reward parser prefers another style, the system may undercount genuine improvements.

This report does not claim that this mismatch caused the current collapse. The data only supports calling it a plausible secondary concern that needs instrumentation.

## Most Important Interpretation

The run did **not** fail because "joint training is inherently impossible".

The run failed because:

1. the recipe injected a global token-length penalty into every response,
2. the run produced no positive samples,
3. GRPO then optimized the only easy improvement direction: shorter wrong answers,
4. once all samples in each prompt-group were equally short and equally wrong, GRPO had no variance left and learning stopped.

The joint-training design may still have intrinsic issues, but they were masked by this more immediate reward-collapse mechanism.

## Stage 2 Recommendations

No implementation changes are proposed in this note. These are the next things that should be tested or instrumented.

### Reward and collapse instrumentation

Add explicit metrics for:

1. base correctness reward before any overlong penalty,
2. overlong penalty magnitude,
3. final reward after penalty,
4. fraction of positive rewards,
5. fraction of prompt-groups with reward std exactly `0`,
6. response length percentiles,
7. EOS-at-position-1 / 2 / 4 / 8 rates.

### Joint-specific metrics

Add metrics for:

1. fused-policy entropy at early response positions,
2. `model1` vs `model2` EOS probability difference,
3. `model1` vs `model2` KL or JSD on response tokens,
4. fused-policy vs `model2` reward on the same validation batch,
5. fused-policy vs `model2` response length on the same validation batch.

### Reward contract audit

Measure:

1. `verification_method` distribution,
2. parse-failure rate,
3. `####`-formatted answers that still receive zero reward,
4. correctness rate under the original GSM8K scorer versus the current custom scorer.

### Algorithm design audit

Clarify whether the intended joint-training principle is:

1. two truly different policies,
2. a frozen teacher plus trainable student,
3. a fused rollout policy but model2-only deployment policy,
4. or something else.

Right now the code path mixes:

- fused-policy rollout and PPO,
- model2-only validation,
- symmetric duplicated initialization,

which is not yet a stable algorithmic story.

## Bottom Line

For run `1773032262`, the shortest correct summary is:

- The observed collapse is real.
- The first-order cause is the reward configuration, not GPU/runtime instability.
- The abnormal gradient spike is a pre-clipping collapse signal, not evidence that `free()` or memory release is broken.
- The later zero gradients are the expected GRPO result after all four rollouts per prompt become identically short and identically wrong.
- The joint-training design still needs Stage 2 work, but the current run must not be used as evidence against the joint principle before the reward-collapse mechanism is removed from the experiment.
