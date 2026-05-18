# Dual-Submodel Rollout WDL-SFT 3A Failure Analysis

- Date: 2026-05-18
- Branch: `feature/on-policy-wdl-sft-dual-rollout`
- Run: `WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0_1779027403`
- Training script: `recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh`
- Log: `recipe/on_policy_wdl_sft/dual_submodel_rollout/WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0_1779027403.log`
- Metrics: `recipe/on_policy_wdl_sft/dual_submodel_rollout/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0_1779027403.jsonl`
- Validation dumps: `recipe/on_policy_wdl_sft/dual_submodel_rollout/validation/WDL-SFT-Qwen3-4B-MATH-3A-DUAL-M2-BETA0_1779027403/{0,25,50}.jsonl`

## Executive Summary

The 3A real training run exposed a method-level problem in the dual-submodel
rollout design. The implementation generates label trajectories from
`sub_model_1` / model2, then trains the fused joint policy on those model2-only
trajectories. This creates a large off-policy gap:

```text
label trajectory:      y ~ pi_model2
training likelihood:  log pi_fused(y | x)
```

Because generation is autoregressive, the entire prefix seen during training is
from model2. Model1 did not generate this trajectory, but it participates in the
fused logits used for the SFT loss. If model1 assigns low probability to tokens
that model2 generated, the fused log-probability becomes low and the SFT loss
gradient becomes large. This is the opposite of the original on-policy
motivation, which was to reduce training/rollout distribution mismatch.

The run should be treated as a negative result for the current 3A algorithmic
form, not as a successful continuation of on-policy WDL-SFT.

## Experiment Setup

### Data Flow in This Branch

The dual rollout implementation follows this flow:

1. Generate `rollout.n=8` responses from `sub_model_0`.
2. Generate `rollout.n=8` responses from `sub_model_1`.
3. Score both source batches for diagnostics.
4. Select only `sub_model_1` for training.
5. Recompute `old_log_probs` under the fused joint actor policy.
6. Update both submodels through fused joint logits.

The selected training data are therefore model2-only rollouts. The training
forward/backward path is still fused.

Relevant implementation points:

- `ray_trainer.py` resets rollout source to `fused` after generation and selects
  only the configured source for training.
- `ray_trainer.py` recomputes `old_log_probs` with the actor joint policy.
- `dp_actor.py` computes current `log_prob` from the actor joint model during
  update.
- `core_algos.py` computes `wdl_sft_is` loss from those fused log-probs.

### Model2 Used in This Run

This run did not use the historical canonical model2 from the earlier 1A/1B/1C
runs. The temporary model2 was copied from:

```text
/root/buaa/czh/EnsembleLLM/weights/llmboost_ablation/Qwen3-4B-Base-Math-m1step_fixed/stage1_m1/checkpoint-218
```

to:

```text
/data-1/.cache/Qwen3-4B-Base-SFT-stage-1
```

The source documentation records this checkpoint as `fixed_ck218`, with a
reference MATH-500 score around `67.0` under its original greedy/thinking eval
setup. In the 3A run, step 0 online validation reached only about `0.5968`
MATH-500 `acc/mean@3`. This suggests the temporary model2 is weaker and less
format-stable than the prior model2, but the subsequent collapse is too large
to attribute to model2 quality alone.

### Validation Path

The validation path was checked and is model2-only, not fused:

- `_validate() entered, _is_joint_training=True`
- `extracting model2-only weights (sub_model_index=1)`
- vLLM loads `is_joint=False` and sets `_use_model2_only=True`

Therefore, the low validation scores are not explained by accidentally
validating the fused model.

Validation settings:

- `actor_rollout_ref.rollout.val_kwargs.n=3`
- `do_sample=True`
- `temperature=1.0`
- `top_p=0.95`
- validation datasets: MATH-500 and AIME-2025

Sampling validation can reduce absolute accuracy, but it cannot explain the
observed step 0 to step 50 collapse or the severe output corruption.

## Observed Training and Validation Behavior

### Online Validation Metrics

| Step | MATH-500 `acc/mean@3` | MATH-500 `acc/best@3` | Notes |
| --- | ---: | ---: | --- |
| 0 | 0.5968 | 0.6780 | Initial model2-only validation; lower than previous canonical model2, but usable. |
| 25 | 0.3448 | 0.4171 | Large early degradation. |
| 50 | 0.0390 | 0.0617 | Near-collapse. |

The first concerning value was not the initial step. The true failure is the
rapid downward trajectory after training starts.

### Output Quality Diagnostics

Validation dumps show visible corruption:

| Step | Accuracy in dump | Avg output chars | `weird_frac` | `no_boxed_frac` | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.5754 | 7102.9 | 0.151 | 0.300 | Some initial format/noise issues. |
| 25 | 0.3276 | 10629.7 | 0.891 | 0.600 | Severe mixed-language / malformed text. |
| 50 | 0.0370 | 11867.2 | 0.997 | 0.781 | Almost all outputs corrupted or missing boxed answer. |

The corruption includes mixed English/Chinese fragments, full-width
characters, Cyrillic/Greek-looking characters, repeated self-correction loops,
and long outputs that fail to converge to an answer. This is a model-output
quality failure, not just a verifier strictness issue.

## Gradient Analysis

### Key Gradient and Length Metrics

| Step | model2 correct ratio | model2 mean response len | response clip ratio | actor grad norm | model1 grad norm | model2 grad norm | positive loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.3008 | 3515.5 | 0.6797 | 4627.7 | 463.8 | 4590.2 | 164.18 |
| 50 | 0.0781 | 3962.7 | 0.9121 | 9445.9 | 470.3 | 9386.9 | 426.34 |
| 63 | 0.0469 | 3938.9 | 0.9102 | 10451.2 | 284.2 | 10422.2 | 222.90 |
| 64 | 0.0605 | 3951.5 | 0.9004 | 9258.9 | 293.7 | 9222.9 | 304.40 |
| 65 | 0.0488 | 3986.5 | 0.9180 | 13339.8 | 288.5 | 13288.8 | 293.95 |

Three points matter:

1. **Gradients are dominated by model2.** By step 50 and later, model2 grad norm
   is thousands to over ten thousand, while model1 grad norm remains hundreds.
   The training update is effectively forcing model2 to chase a high-loss fused
   objective on model2-generated trajectories.
2. **Responses are near the maximum length.** From step 50 onward, mean response
   length is close to the 4096-token limit and clip ratio is around 0.90. The
   model is learning long, unstable reasoning traces.
3. **Correct samples are sparse but high-impact.** At step 50 only 40 of 512
   selected model2 rollouts are correct. The loss averages by number of correct
   sequences, not by number of correct tokens, so a small number of long correct
   rollouts can dominate the update.

### Interpretation

The gradients are large because the loss is applied to model2-generated
sequences under a fused likelihood. For many tokens:

```text
log pi_model2(y_t | prefix_model2) may be reasonable
log pi_fused(y_t | prefix_model2) may be much lower
```

The SFT objective maximizes `log pi_fused` on these tokens. When the fused
policy assigns low probability to a long model2 trajectory, the sequence-level
negative log-likelihood becomes large. Since gradients flow through both
submodels and model2 already supports the sequence better than model1, model2
receives the dominant update and rapidly drifts.

This behavior matches the observed training metrics and the validation output
corruption.

## Loss Function Analysis

### Current `wdl_sft_is` Loss

The current implementation computes:

```text
L+ = - sum_{i in correct} sum_t keep_{i,t} log pi_train(y_i,t | prefix_i) / |C|
L- =   sum_{j in incorrect} sum_t keep_{j,t} log pi_train(y_j,t | prefix_j) / |I|
L  = L+ + beta * L-
```

For 3A, `beta=0.0`, so only `L+` affects training:

```text
L = L+
```

Important implications:

- The loss is **sequence token-sum**, not token-mean.
- Normalization is by number of correct responses `|C|`, not by the number of
  correct tokens.
- Long correct answers have much larger influence than short correct answers.
- With `beta=0`, incorrect rollouts do not push down bad trajectories.
- The ratio clip in `wdl_sft_is` compares current fused actor log-probs to
  recomputed old fused actor log-probs. It does not correct the mismatch between
  model2 rollout policy and fused training policy.

### Rollout IS Is Not Applied to This Gap

The run uses `loss_mode=wdl_sft_is`, but the dual rollout implementation does
not apply `rollout_is_weights` as loss weights by default. The shared launch
script sets:

```text
algorithm.rollout_correction.rollout_is=null
```

Even when rollout correction is computed from `rollout_log_probs`, the dual
rollout trainer path removes `rollout_is_weights` before actor update and logs:

```text
dual_rollout/rollout_is_loss_weight_applied = 0
```

This was originally intended to avoid applying an ill-defined correction during
the first dual-rollout implementation. In this experiment, it means there is no
effective correction for:

```text
pi_model2 rollout policy -> pi_fused training policy
```

### Why the Loss Amplifies the Failure

The loss is not wrong as a local SFT objective, but it is being applied to the
wrong distribution for this algorithmic intent. It asks the fused policy to
imitate high-quality model2 samples even when fused policy is far from model2 on
the same prefix. With long model2 trajectories, sequence-level loss scales up
quickly:

```text
-sum_t log pi_fused(y_t | prefix_model2)
```

The observed `actor/wdl_sft_loss_positive` values, response lengths, and
gradient norms are consistent with this amplification.

## Distribution Difference Analysis

### Earlier Fused-Rollout Branch

The earlier branch used fused logits during rollout:

```text
prefix_t ~ pi_fused
y_t ~ softmax((1 - lambda) z1(prefix_t) + lambda z2(prefix_t))
training likelihood = log pi_fused(y_t | prefix_t)
```

This is on-policy with respect to the fused training policy. The prefix is
generated by the same policy family that is later trained. The vLLM joint model
and the FSDP joint model compute fused logits for the same trajectory type.

### Current Dual-Submodel 3A Branch

The current 3A branch uses:

```text
prefix_t ~ pi_model2
y_t ~ pi_model2(. | prefix_t)
training likelihood = log pi_fused(y_t | prefix_t)
```

The fusion formula for a fixed prefix is still the same, but the prefix is not
from the fused policy. In autoregressive generation, this distinction is
fundamental. Hidden states and logits at time `t` depend on all previous tokens:

```text
prefix_t = (x, y_1, ..., y_{t-1})
```

If `prefix_t` was generated by model2, model1 may be far off-distribution on
that prefix. The fused logits can therefore be much worse than model2 logits,
even though model2 produced a correct final answer.

### Sequence-Level Ratio Accumulation

The mismatch is multiplicative over tokens:

```text
pi_fused(y | x) / pi_model2(y | x)
= product_t pi_fused(y_t | x, y_<t) / pi_model2(y_t | x, y_<t)
```

For long math answers, this product can have extremely high variance. Even
moderate per-token disagreement compounds across thousands of tokens. This is
why the current 3A design can produce very large gradients even when each
individual token-level discrepancy appears small.

### Why Adding an Extra Fused Rollout Does Not Fully Fix It

A proposed alternative was:

1. Run model2 rollout for high-quality labels.
2. Run fused rollout as in the previous branch.
3. Train with the previous fused forward/backward path while using model2
   rollout results as labels.

This still leaves the core mismatch if the actual trained tokens come from
model2 rollout. The fused rollout is useful for diagnostics or for selecting
which prompts to trust, but it does not make:

```text
y_model2 ~ pi_model2
log pi_fused(y_model2)
```

on-policy. To recover on-policy stability, the trained trajectory itself must
come from the fused policy, or the method needs a separate off-policy
distillation design with explicit correction and much tighter optimization
controls.

## Method-Level Conclusion

The current 3A method should be classified as:

```text
off-policy model2-to-fused distillation
```

It should not be treated as an on-policy WDL-SFT variant. The implementation
does what the plan specified, but the plan's algorithmic assumption appears
incorrect: using higher-quality model2-only rollouts as SFT labels does improve
label quality, but it also breaks the distribution alignment that made the
previous fused-rollout method stable.

The failure mode is not merely a hyperparameter issue. Smaller LR, tighter
gradient clipping, length normalization, and rollout IS may reduce damage, but
the core data-flow mismatch remains.

## Recommended Decision

### For the Current 3A Run

Stop the run and record it as a negative result. Continuing is unlikely to
produce useful checkpoints because validation has already collapsed and output
quality is visibly corrupted.

### For the Main Research Line

Return to the previous fused-rollout/fused-training branch as the mainline:

```text
y ~ pi_fused
train log pi_fused(y)
validate model2-only
```

Then tune within that stable distribution:

- increase rollout `n` if label coverage is insufficient;
- adjust temperature/top-p for label quality;
- use length-normalized or partially length-normalized SFT loss;
- keep `beta=0` unless model1-specific failure is explicitly being tested;
- tighten `grad_clip`;
- add deterministic validation alongside sampled `n=3` validation;
- keep model2-only validation as the primary reported target.

### If This Off-Policy Direction Is Revisited

Treat it as a separate algorithm, not as the same on-policy method. Minimum
requirements before another full run:

- dump per-source rollout text for model1, model2, and fused;
- compute and log `pi_fused / pi_model2` diagnostics on selected samples;
- use token-mean or capped sequence loss;
- use much smaller LR;
- use tight grad clipping;
- consider freezing model1 or training model2-only first;
- add a short 1/3/10-step smoke gate that checks output corruption before
  running 300 steps.

## Reporting Statement

The faithful summary for advisor reporting is:

> The dual-submodel rollout branch successfully implemented model1/model2
> separate rollout and selected model2 data for training, but the first real 3A
> run showed that the algorithm is not on-policy with respect to the fused
> training objective. Model2-generated prefixes are trained under fused logits,
> causing a large autoregressive distribution mismatch. Because the WDL-SFT loss
> is sequence token-sum over correct samples and the selected correct samples are
> long, the mismatch produces very large model2 gradients, rapid response-length
> saturation, severe output corruption, and MATH-500 validation collapse by step
> 50. This is best treated as a negative result for the current method design and
> motivates returning to fused-rollout/fused-training as the mainline while
> recording model2-only rollout distillation as a separate, high-risk off-policy
> ablation.
