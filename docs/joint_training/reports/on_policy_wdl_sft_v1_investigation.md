# On-Policy WDL-SFT v1 Historical Investigation

- Date: 2026-05-28
- Scope: check whether the simpler On-Policy Weak-Driven SFT algorithm already exists in this project, whether it has run experiments, and what those experiments imply for returning to On-Policy SFT / On-Policy WDL-SFT.
- Main repo inspected: `/root/buaa/local_data1/verl07/verl`
- Branch at inspection time: `feature/on-policy-wdl-sft`
- Registry inspected: `/data-1/experiment_registry/experiment_registry.sqlite`

## Executive Summary

The simpler algorithm exists. It is the v1 `wdl_sft` path on `feature/on-policy-wdl-sft`, implemented before the later `wdl_sft_is` and `wdl_group_adv_is` variants.

It matches the target shape:

- rollout policy: fused weak/strong joint policy, using the existing joint model family;
- training likelihood: fused log-prob, so gradients flow through both submodels;
- loss: SFT-style sequence cross entropy on correct rollouts, plus optional reverse SFT on incorrect rollouts;
- no token-level rollout importance sampling;
- no GRPO-style group advantage used by the loss.

The important caveat is terminology: the launch scripts set `adv_estimator=grpo` to fit the verl training pipeline, but the trainer overwrites the advantages with raw reward labels for `wdl_sft`. So v1 is not using GRPO group advantages as the actual learning signal.

Historically, this family is recorded as EXP-12 through EXP-15 in `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md` and in the experiment registry. The best clean v1 forward-only run is EXP-13 / M5.5. Reverse SFT at low learning rate did not destroy model2, but it caused a serious model1 format-compliance collapse. That is the key historical lesson for designing the next simpler method.

Against the closest same-budget GRPO-style baseline available offline (`ABL-MINIRL-01 / 2Z-SFT`, single-model MiniRL from the same SFT-stage-1 init), v1 WDL-SFT did not win on MATH-500. The best v1 model2 result is `79.6%` mean@3, while 2Z-SFT reaches `80.7%`; the clean forward-only M5.5 model2 is `78.6%`, about `2.1 pp` lower than 2Z-SFT.

## Where The Algorithm Exists

### Code Path

The v1 loss is registered as `wdl_sft` in `verl/trainer/ppo/core_algos.py`.

Relevant facts from code:

- `compute_policy_loss_wdl_sft(...)` is registered with `@register_policy_loss("wdl_sft")`.
- Its docstring explicitly says it does not use `old_log_prob`, `rollout_is_weights`, or `loss_agg_mode`.
- It reads reward labels from `advantages[:, 0]`, where the trainer has already written per-response `+1/-1` correctness labels.
- `compute_wdl_sft_loss(...)` implements:
  - positive SFT loss on correct rollouts;
  - optional reverse SFT on incorrect rollouts;
  - skip when all responses are incorrect;
  - no importance ratio or group advantage.

Source anchors:

- `verl/trainer/ppo/core_algos.py:1861` registers `wdl_sft`.
- `verl/trainer/ppo/core_algos.py:1877-1883` documents raw reward labels and no IS usage.
- `verl/trainer/ppo/core_algos.py:1926-1938` documents positive SFT, reverse SFT, and boundary cases.
- `verl/trainer/ppo/core_algos.py:1979-1993` implements `L+ + beta * L-`.

The trainer-side label bridge is in `ray_trainer.py`:

- `WDL_SFT_REWARD_LABEL_LOSS_MODES = {"wdl_sft", "wdl_sft_is"}`.
- `apply_wdl_sft_reward_label_advantages(...)` writes raw `token_level_scores.sum(dim=-1)` into the `advantages` tensor.

Source anchors:

- `verl/trainer/ppo/ray_trainer.py:90`
- `verl/trainer/ppo/ray_trainer.py:138-158`

### Rollout / Joint Model Semantics

The method note describes the common WDL-SFT family as using weak and strong logits to form a fused policy, rolling out under that fused policy, and training both submodels through fused log-prob.

Source anchors:

- `docs/joint_training/courses/method_on_policy_wdl_sft.tex:18`
- `docs/joint_training/courses/method_on_policy_wdl_sft.tex:31-44`

This is the closest match to the algorithm described in the request: on-policy rollout under the same fused policy whose log-prob is used by the loss.

## Runnable Scripts

The concrete v1 scripts are under `recipe/on_policy_wdl_sft/`.

| Script | Meaning | Loss | Reverse SFT | IS |
|---|---|---:|---:|---:|
| `run_on_policy_wdl_sft_qwen3_4b_math.sh` | original M5 long bidirectional run | `wdl_sft` | `beta=0.1` default | off |
| `run_on_policy_wdl_sft_qwen3_4b_math_m5_5.sh` | M5.5 forward-only baseline | `wdl_sft` | `beta=0.0` | off |
| `run_on_policy_wdl_sft_qwen3_4b_math_m5_6.sh` | M5.6 reverse-SFT rerun at lower lr | `wdl_sft` | `beta=0.1` | off |
| `run_on_policy_wdl_sft_qwen3_4b_math_lr3.sh` | LR3 forward-only high-lr run | `wdl_sft` | `beta=0.0` | off |

Script evidence:

- M5.5 declares forward-only WDL-SFT, joint model, `loss_mode="wdl_sft"`, `WDL_SFT_BETA=0.0`, and `rollout_is="null"`: `recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_m5_5.sh:3-10`, `:180-207`.
- M5.6 declares bidirectional WDL-SFT, `loss_mode="wdl_sft"`, `WDL_SFT_BETA=0.1`, and `rollout_is="null"`: `recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_m5_6.sh:3-10`, `:180-207`.

Git evidence:

- The WDL-SFT loss implementation commit `26f94294` is contained by `feature/on-policy-wdl-sft`, `origin/feature/on-policy-wdl-sft`, and the later `feature/on-policy-wdl-sft-dual-rollout`.
- The training-loop integration commit `f12eb0b3` is also contained by those branches.

So the implementation is not missing. It is already in the current branch lineage.

## Registry And Experiment Records

The registry contains v1 `wdl_sft` entries in the shared `verl` project form and newer branch-scoped entries for later methods.

Relevant registry projects:

| project id | name | meaning |
|---:|---|---|
| 2 | `verl` | selected on-policy-wdl-sft branch import; holds old v1/v2 historical records |
| 3 | `verl:feature/on-policy-wdl-sft` | branch-scoped current-branch form; holds later label-fix / group-advantage records |
| 4 | `verl:feature/on-policy-wdl-sft-dual-rollout` | dual-rollout sibling branch records |

V1 experiment records found:

| Registry experiment | Display name | Trust | Meaning |
|---|---|---|---|
| `verl.sft.math.qwen3_4b.v1_wdl_sft.exp_13` | EXP-13 v1_wdl_sft | trusted | M5.5 forward-only baseline |
| `verl.sft.math.qwen3_4b.v1_wdl_sft.exp_14` | EXP-14 v1_wdl_sft | usable_with_caution | M5.6 reverse-SFT run; model1 collapse caveat |
| `verl.sft.math.qwen3_4b.v1_wdl_sft.exp_15` | EXP-15 v1_wdl_sft | usable_with_caution | LR3 high-lr forward-only; stopped at best step 125 |
| `verl.online.math.v1_wdl_sft.exp_13` | EXP-13 v1_wdl_sft online validation | trusted | online curve for M5.5 |
| `verl.online.math.v1_wdl_sft.exp_14` | EXP-14 v1_wdl_sft online validation | usable_with_caution | online curve for M5.6 |
| `verl.online.math.v1_wdl_sft.exp_15` | EXP-15 v1_wdl_sft online validation | usable_with_caution | online curve for LR3 |

Registry caveat: the old v1 rows have eval artifacts and online-validation artifacts, but no populated `training_runs` rows. The experiment rows, eval rows, model rows, artifacts, source markdown, and source eval JSON paths are present.

Source records show the registry imported:

- `/data-1/verl07/verl/recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md`
- `/data-1/verl07/verl/recipe/on_policy_wdl_sft/INFERENCE_RESULTS.md`
- v1 eval metric JSONs under `/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/`, `/data-1/model_weights/WDL-SFT-4B-MATH-M5-6/`, and `/data-1/model_weights/WDL-SFT-4B-MATH-LR3/`.

## Historical Results

### Offline MATH-500 Results From Registry

| Run | Config | Model | Step | MATH-500 mean@3 | pass@3 | extraction_fail | Delta vs 2Z-SFT final mean@3 | Trust |
|---|---|---|---:|---:|---:|---:|---:|---|
| EXP-13 / M5.5 | v1 `wdl_sft`, beta=0, lr=5e-7 | model2 | 300 | 78.6 | 87.4 | 16.7 | -2.1 pp | trusted |
| EXP-13 / M5.5 | v1 `wdl_sft`, beta=0, lr=5e-7 | model1 | 300 | 70.5 | 86.6 | 6.8 | n/a | trusted |
| EXP-14 / M5.6 | v1 `wdl_sft`, beta=0.1, lr=5e-7 | model2 | 300 | 79.1 | 86.8 | 16.9 | -1.6 pp | usable_with_caution |
| EXP-14 / M5.6 | v1 `wdl_sft`, beta=0.1, lr=5e-7 | model1 | 300 | 48.9 | 78.2 | 26.9 | n/a | usable_with_caution |
| EXP-15 / LR3 | v1 `wdl_sft`, beta=0, lr=1e-6 | model2 | 125 | 79.6 | 87.6 | 14.6 | -1.1 pp | usable_with_caution |
| EXP-15 / LR3 | v1 `wdl_sft`, beta=0, lr=1e-6 | model1 | 125 | 63.7 | 84.4 | 12.9 | n/a | usable_with_caution |
| ABL-MINIRL-01 / 2Z-SFT | single-model MiniRL / GRPO-style baseline, same SFT init, lr=5e-7 | single | 275 | 79.6 | 88.4 | 3.4 | -1.1 pp | trusted |
| ABL-MINIRL-01 / 2Z-SFT | single-model MiniRL / GRPO-style baseline, same SFT init, lr=5e-7 | single | 300 | 80.7 | 89.2 | 2.9 | baseline | trusted |

These values are also summarized in `recipe/on_policy_wdl_sft/INFERENCE_RESULTS.md:720-778`.

The 2Z-SFT baseline is not a joint-model run, so it is not identical in architecture. It is still the closest offline same-budget GRPO-style comparison in the registry: same model2 init (`Qwen3-4B-Base-SFT-stage-1`), same EnsembleLLM train file, same 300-step horizon, same `lr=5e-7`, same `rollout.n=8`, and the same offline n=3 evaluation pipeline. Its launcher explicitly says the boundary against the joint 1A/1B/1C family is that joint training is disabled and everything else is held fixed.

Source anchors for this comparison:

- `recipe/on_policy_wdl_sft/ablation_single_model/run_2z_sft.sh:3-11`
- `recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh:16-27`
- `recipe/on_policy_wdl_sft/INFERENCE_RESULTS.md:555-625`

There is also a stricter current-branch "standard GRPO" family (`2G MATH-data`) in the registry:

| Run | Config | Eval type | Best MATH-500 metric | Caveat |
|---|---|---|---:|---|
| `2G-MATH-BASE` | single-model vanilla GRPO, Base init, Hendrycks MATH train, 115 steps | online mean@1 | 74.8 at step 115 | different train data and no offline n=3 row in registry |
| `2G-MATH-SFT` | single-model vanilla GRPO, SFT-stage-1 init, Hendrycks MATH train, 115 steps | online mean@1 | 79.0 at step 100 | different train data and no offline n=3 row in registry |

I did not put 2G into the main offline table because it is a different data budget and metric: Hendrycks MATH train, 115 steps, online mean@1. It is useful as a modern standard-GRPO sanity check, but not a clean offline apples-to-apples comparison with EXP-13/14/15.

Source anchors for 2G:

- `docs/joint_training/guides/training_script_index.md:27-30`
- registry rows `experiment_id=30` and `experiment_id=31`, with best MATH-500 online mean@1 imported from training metrics.

### Result Reading

For the original v1 `wdl_sft` implementation:

- Forward-only WDL-SFT works, but it does not beat the closest GRPO-style baseline on MATH-500. M5.5 model2 is `78.6%` mean@3, while 2Z-SFT final is `80.7%`.
- Raising lr to `1e-6` improves v1 model2 to `79.6%` at step 125, but this is still below 2Z-SFT final by `1.1 pp` and carries drift risk.
- Adding reverse SFT at `lr=5e-7` gives model2 only `+0.5 pp` mean@3 over forward-only M5.5 (`79.1` vs `78.6`) and slightly lower pass@3 (`86.8` vs `87.4`).
- The same reverse-SFT run damages model1 badly: MATH-500 mean@3 falls from `70.5` to `48.9` and extraction failure rises from `6.8%` to `26.9%`.

Answer to "should we add Reverse SFT?": not as the default. The measured model2 gain is small and not robust across metrics, while the model1 damage is large. If the next method explicitly discards model1 and only deploys model2, reverse SFT can be treated as an ablation, but the historical result does not justify making it the main path.

### Online / Experiment Index Notes

The experiment index says:

- EXP-12 / M5 used bidirectional `wdl_sft` with `beta=0.1` and lr `1e-6`; it diverged after an early peak and motivated the forward-only lower-lr M5.5 run.
- EXP-13 / M5.5 is the forward-only baseline.
- EXP-14 / M5.6 trained through the low-lr reverse-SFT setup; online it did not simply explode, but offline model1 evaluation exposed the failure.
- EXP-15 / LR3 is forward-only at lr `1e-6`, with best step 125.

Source anchors:

- `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md:16-32`
- `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md:36-40`
- `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md:70-77`
- `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md:81-91`

## What Happened With Positive And Negative Samples

The v1 design uses reward labels directly:

- correct rollout: standard SFT, maximize probability;
- incorrect rollout: reverse SFT if `beta > 0`, minimize probability;
- all-correct group: positive SFT only;
- all-incorrect group: skipped.

This gave one stable default and one strong warning:

1. Forward-only (`beta=0`) is the cleanest historical baseline.
2. Reverse SFT (`beta=0.1`) can leave model2 roughly healthy, but it damages model1 heavily.

The model1 damage is not a small metric fluctuation. On MATH-500, EXP-14 model1 drops from EXP-13 model1 `70.5` to `48.9` mean@3, while extraction failure rises from `6.8%` to `26.9%`. In later v2 `wdl_sft_is`, the attempt to rescue reverse SFT with lower-bound clipping did not fix this; EXP-17 model1 was even worse. The project notes now treat `beta > 0` as unsafe unless model1 is explicitly discarded.

Source anchors:

- `recipe/on_policy_wdl_sft/INFERENCE_RESULTS.md:758-780`
- `recipe/on_policy_wdl_sft/INFERENCE_RESULTS.md:847-854`
- `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md:132-146`

## Branches That Are Related But Not The Same

### `feature/on-policy-wdl-sft`

This is the branch that contains the original v1 `wdl_sft` algorithm and the later v2/v3 work. The target simple implementation already exists here.

### `feature/on-policy-wdl-sft-dual-rollout`

This branch implemented a different idea: generate per-submodel rollouts, select model2 rollouts, and train the fused joint policy on those trajectories.

That is not the same as the simpler v1 on-policy fused-rollout algorithm. The 3A dual-rollout run was archived as a negative result because it became:

```text
y ~ pi_model2
train with log pi_fused(y | x)
```

That is model2-to-fused off-policy distillation, not on-policy WDL-SFT.

Source anchors:

- `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft.md:14-22`
- `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft.md:32-39`
- `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md:14-32`
- `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md:38-58`
- `docs/joint_training/courses/method_on_policy_wdl_sft.tex:583-588`
- `docs/joint_training/courses/method_on_policy_wdl_sft.tex:754-760`

## Interpretation For The Next Method Iteration

If the goal is to return to a simple On-Policy SFT / On-Policy WDL-SFT direction, the clean restart point should be v1 `wdl_sft`, not `wdl_sft_is`, not `wdl_group_adv_is`, and not dual-submodel rollout 3A.

Recommended default baseline:

- rollout from fused policy;
- train fused log-prob through both submodels;
- `loss_mode=wdl_sft`;
- `rollout_is=null`;
- `beta=0`;
- start from the M5.5 schedule (`lr=5e-7`, 300-step short run) unless the new experiment changes only one axis.

The unresolved design question is how to use negative samples without repeating the reverse-SFT model1 collapse. The historical evidence argues against naive negative reverse SFT as the default. Safer candidate directions:

1. Positive-only selected SFT: use only correct rollouts; skip incorrect rollouts.
2. Positive-only with all-incorrect fallback: when a group has no correct rollout, skip the prompt or keep a low-weight diagnostic objective, but do not reverse-SFT by default.
3. Filtered / thresholded negatives: if negatives are used, constrain them to local format-safe components rather than applying sequence-level reverse likelihood to the full answer.
4. Separate On-Policy SFT single-model baseline: run the same correct-only selected SFT without joint fusion to separate the value of the fused rollout/joint architecture from the selected-SFT loss.

The key historical result to preserve is this:

> The original simple fused-rollout `wdl_sft` exists and ran. It is useful as a baseline, but naive reverse SFT on incorrect rollouts is not a safe default because it damages the weak/model1 anchor even when model2 remains competitive.
