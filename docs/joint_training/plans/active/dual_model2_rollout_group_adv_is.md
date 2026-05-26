# Dual Model2 Rollout Group-Advantage IS — Active Plan

- Status: IMPLEMENTED - plumbing smoke/stability complete; learning-signal smoke required before full training
- Created: 2026-05-25
- Branch: `feature/on-policy-wdl-sft-dual-rollout`
- Loss mode: `dual_model2_group_adv_is`
- Historical negative result: `dual_submodel_rollout_wdl_sft_3a_failure_analysis.md`
- Related implementation reference: parent-branch `wdl_group_adv_is`
- Primary external reference: `docs/joint_training/references/external/stabilizing_rl_with_llms_source/colm2024_conference.tex`

Implementation evidence:

- 3-step smoke: `DUAL-M2-GROUP-ADV-IS-SMOKE4_1779704487`
- 10-step stability gate: `DUAL-M2-GROUP-ADV-IS-10STEP_1779704885`
- Registry project: `verl:feature/on-policy-wdl-sft-dual-rollout`
- Registry training runs:
  `verl.branch.feature_on_policy_wdl_sft_dual_rollout.dual_model2_group_adv_is.smoke4_1779704487`
  and
  `verl.branch.feature_on_policy_wdl_sft_dual_rollout.dual_model2_group_adv_is.10step_1779704885`
- Runtime caveat: both smoke/stability samples were all-incorrect groups, so
  positive-fallback behavior is covered by unit tests rather than runtime data.
- Smoke policy update: the 256-token smoke is now classified as plumbing-only.
  Before launching full 4A training, run a production-context learning-signal
  smoke following `docs/joint_training/constraints/smoke_learning_signal_policy.md`.

## 1. Objective

Revise the failed 3A dual-rollout method into a MiniRL-style, group-advantage
loss for the branch's intended data path:

```text
rollout policy:   model2 only
training policy:  fused two-submodel joint model
eval target:      model2 only
```

The new algorithm keeps the advisor's direction of adding importance sampling,
but does not treat the 3A failure as a simple hyperparameter issue. The failure
source was the off-policy mismatch:

```text
y ~ pi_model2
train log pi_fused(y)
```

The new loss must explicitly account for this mismatch while keeping training
variance controlled enough for a real smoke run.

## 2. Method Boundary

### In Scope

- Generate rollout trajectories only from `sub_model_1` / model2.
- Train with fused joint logits, updating both submodels.
- Use group-mean advantages.
- Preserve positive SFT signal for all-correct groups.
- Use MiniRL-style detached token-level IS weight in the loss.
- Use MiniRL-style binary mask for fused old-current policy staleness.
- Use token-level Truncated Importance Sampling (TIS).
- Keep model2-only validation as the primary target.
- Record this branch's experiments under the existing experiment-registry
  project `verl:feature/on-policy-wdl-sft-dual-rollout`.

### Out Of Scope For First Implementation

- Do not re-enable model1 rollout for training.
- Do not train on both model1 and model2 trajectories.
- Do not use PPO clipped surrogate.
- Do not use external `rollout_is_weights` in addition to the loss-internal IS.
- Do not use actor KL loss.
- Do not use reverse-SFT `beta`.
- Do not add all-incorrect negative fallback.
- Do not normalize group advantage by standard deviation.
- Do not change checkpoint format.

## 3. Loss Design

For prompt group `g`, response row `i`, token `t`, reward `R_i in {+1, -1}`,
and response mask `M_{i,t}`, define group advantage:

```text
A_i = R_i - mean_{j in g(i)} R_j
```

Do not divide by group standard deviation:

```text
norm_adv_by_std_in_grpo = false
```

Add all-correct positive-SFT fallback:

```text
F_i = 1 if every response in group g(i) is correct, else 0
G_i = A_i + gamma_pos * F_i
```

Default:

```text
gamma_pos = 1.0
```

All-incorrect groups keep `G_i = 0` and do not update. This is intentional; do
not add a negative fallback in the first implementation.

### 3.1 Token-Level IS Weight

MiniRL's token-level IS weight is adapted to the branch's rollout/training
policies:

```text
w_is(i,t) = pi_current_fused(y_i,t | x_i, y_i,<t)
          / pi_model2_rollout(y_i,t | x_i, y_i,<t)
```

Equivalently:

```text
w_is = (pi_current_fused / pi_old_fused)
     * (pi_old_fused / pi_model2_rollout)
```

This means the fused model's own old-current ratio is already included in the
continuous IS weight. The full `w_is` enters the loss only as a detached scalar
weight.

Apply token-level TIS:

```text
w_tis(i,t) = min(w_is(i,t), c_tis)
```

Default:

```text
c_tis = 5.0
```

If the first smoke run shows abnormal gradient norm or non-finite metrics, do
not silently change this method's default. Record the blocker and open a
separate ablation decision for smaller `c_tis` only after the default design has
been diagnosed.

### 3.2 Binary Mask

Use a separate MiniRL-style binary mask only for fused old-current staleness:

```text
r_train(i,t) = pi_current_fused(y_i,t | x_i, y_i,<t)
             / pi_old_fused(y_i,t | x_i, y_i,<t)
```

```text
m(i,t) = 0 if G_i > 0 and r_train(i,t) > 1 + eps_high
m(i,t) = 0 if G_i < 0 and r_train(i,t) < 1 - eps_low
m(i,t) = 1 otherwise
```

Defaults:

```text
eps_high = 0.27
eps_low = 0.20
```

The binary mask is not a second behavior-IS mask. It only gates tokens whose
current fused policy has drifted too far from old fused during actor updates.

### 3.3 Final Objective

Use the MiniRL-style score-function form:

```text
L = - mean_i sum_t [
      M_i,t
    * sg(G_i)
    * sg(w_tis(i,t))
    * sg(m(i,t))
    * log pi_current_fused(y_i,t | x_i, y_i,<t)
]
```

The loss aggregation must be:

```text
loss_agg_mode = seq-mean-token-sum
```

Do not add a length-normalization fallback to this plan. If a future experiment
needs `token-mean` or capped token-sum, that should be a separate method
revision because it changes the MiniRL first-order approximation boundary.

## 4. Required Tensor Provenance

The implementation must make these tensor sources explicit:

| Tensor | Source | Gradient use |
|---|---|---|
| `log_pi_model2_rollout` | model2 rollout policy for the sampled tokens | detached denominator only |
| `old_log_prob` / `log_pi_old_fused` | fused actor before update for selected model2 trajectories | detached denominator / mask reference |
| `log_prob` / `log_pi_current_fused` | current fused actor during actor update | only gradient path |
| `advantages` | group-mean reward advantage plus all-correct fallback | detached scalar coefficient |
| `w_tis` | token-level TIS-capped `current_fused / model2_rollout` | detached scalar coefficient |
| `m` | MiniRL binary staleness mask from `current_fused / old_fused` | detached token gate |

Implementation must fail fast if `log_pi_model2_rollout` is unavailable, because
without it the method silently regresses to the failed 3A loss.

## 5. Config Defaults

First implementation defaults:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      joint_rollout_sources: ["sub_model_1"]
      joint_rollout_select: "sub_model_1"
      joint_rollout_train_on_selected_only: true
  actor:
    loss_agg_mode: seq-mean-token-sum
    use_kl_loss: false
    policy_loss:
      loss_mode: dual_model2_group_adv_is
      gamma_pos_sft: 1.0
      tis_threshold: 5.0
      wdl_sft_beta: 0.0  # ignored by this loss; keep beta disabled

algorithm:
  norm_adv_by_std_in_grpo: false
  rollout_correction:
    rollout_is: null
```

Actor ratio defaults:

```text
clip_ratio_low = 0.20
clip_ratio_high = 0.27
```

## 6. Metrics

The loss/trainer path must log at least:

```text
dual_model2_group_adv_is/tis_mean
dual_model2_group_adv_is/tis_max
dual_model2_group_adv_is/tis_clip_fraction
dual_model2_group_adv_is/train_ratio_mean
dual_model2_group_adv_is/train_ratio_max
dual_model2_group_adv_is/clipfrac_positive
dual_model2_group_adv_is/clipfrac_negative
dual_model2_group_adv_is/mixed_group_fraction
dual_model2_group_adv_is/all_correct_fallback_group_fraction
dual_model2_group_adv_is/all_incorrect_group_fraction
dual_rollout/model2_correct_ratio
dual_rollout/model2_response_len_mean
dual_rollout/selected_source
actor/grad_norm
```

The metrics must distinguish:

- TIS clipping of the continuous IS weight;
- binary mask clipping of old-current staleness;
- response correctness distribution by group.

## 7. Execution Work Packages

Implementation must proceed in dependency order. Do not start a full training
run until every package below is complete and reviewer-gated.

| Package | Owner | Primary files | Required evidence |
|---|---|---|---|
| WP1 Loss function and registry | Main Agent | `verl/trainer/ppo/core_algos.py` | Registered `dual_model2_group_adv_is`; unit tests pass; reviewer confirms formula and gradient path. |
| WP2 Config schema | Main Agent | `verl/workers/config/actor.py` or the branch's active actor config module | New loss knobs are accepted without relying only on Hydra `+` overrides; invalid configs fail clearly. |
| WP3 Rollout log-prob capture | Main Agent | `verl/trainer/ppo/ray_trainer.py`, rollout worker/model paths as needed | Actual model2 rollout token log-probs are preserved as `log_pi_model2_rollout` or an equivalent explicit tensor. |
| WP4 Trainer tensor plumbing | Main Agent | `verl/trainer/ppo/ray_trainer.py`, actor update data path | Selected model2 batch contains model2 rollout log-probs, old fused log-probs, current fused log-probs, response mask, and group advantages. |
| WP5 Advantage routing | Main Agent | `verl/trainer/ppo/ray_trainer.py` | New loss is excluded from WDL raw-label override; all-correct fallback is added after group advantage and before actor update. |
| WP6 Tests | Main Agent, optional test-only Worker Sub-Agent | `tests/on_policy_wdl_sft/`, targeted trainer tests | Unit and trainer tests cover Section 7 acceptance cases. |
| WP7 Recipe scripts | Main Agent | `recipe/on_policy_wdl_sft/dual_submodel_rollout/` | Runnable common/script updates in the existing branch recipe directory, real-run launcher, README, validation `n=3`, checkpoint retention, dataset/model defaults. |
| WP8 Script index and docs | Main Agent | `docs/joint_training/guides/training_script_index.md`, this plan/status notes if added | Branch-local script index updated after script creation/use; no launch manual embedded in the index. |
| WP9 GPU smoke and stability gate | Main Agent | tmux session, recipe logs, metrics JSONL, checkpoint dir | 1-3 step smoke evidence and 10-step gate evidence recorded with paths and metrics snippets. |
| WP10 Experiment registry | Main Agent | `/data-1/experiment_registry/experiment_registry.sqlite` via the project registry workflow | Any real run/smoke intended as experiment evidence is recorded under `verl:feature/on-policy-wdl-sft-dual-rollout` with method metadata. |

Dependency rules:

1. WP1 and WP2 must land before trainer integration can be considered runnable.
2. WP3 must land before WP4; trainer tests that do not exercise real
   `log_pi_model2_rollout` provenance are insufficient.
3. WP4 and WP5 must land before recipe scripts are used for real smoke.
4. WP6 must pass before WP9 starts.
5. WP7 and WP8 must be complete before any real run beyond a tiny smoke.
6. Reviewer Sub-Agent gates must PASS before the Main Agent marks the
   implementation complete.

## 8. Acceptance Criteria

### Code Completion Standard

Implementation is not complete when the loss function alone is written. It is
complete only when the end-to-end training path is runnable and guarded:

- A registered policy loss named `dual_model2_group_adv_is` exists.
- The loss implements Section 3 exactly:
  - detached group coefficient `G_i`;
  - detached TIS-capped token-level IS weight `current_fused / model2_rollout`;
  - one detached MiniRL binary mask based on `current_fused / old_fused`;
  - gradient path only through `log_pi_current_fused`;
  - aggregation fixed to `seq-mean-token-sum`.
- The trainer preserves the rollout-time model2 token log-probs as
  `log_pi_model2_rollout` or an equivalently named tensor through actor update.
- The trainer recomputes `old_log_prob` under the fused actor policy, not under
  model2.
- The actor update can access all three token log-prob sources in the same
  selected batch:
  - rollout model2 log-probs;
  - old fused log-probs;
  - current fused log-probs.
- The new loss is excluded from `apply_wdl_sft_reward_label_advantages(...)`.
  It must consume GRPO/group advantages plus the all-correct fallback, not raw
  `+1/-1` reward labels.
- External `rollout_is_weights` must be disabled and rejected for this loss.
- The config layer accepts the new loss knobs without ad-hoc Hydra `+` fields
  being the only way to run the recipe.
- Missing provenance tensors must fail fast with a clear error. Silent fallback
  to the old 3A behavior is a blocking failure.
- The implementation must add focused tests before any real training launch.

### Unit Tests

Add or update targeted tests under:

```text
tests/on_policy_wdl_sft/
```

Recommended test files:

```text
tests/on_policy_wdl_sft/test_dual_model2_group_adv_is_loss.py
tests/on_policy_wdl_sft/test_dual_model2_group_adv_is_trainer.py
tests/on_policy_wdl_sft/test_dual_model2_group_adv_is_scripts.py
```

Required test command:

```bash
pytest tests/on_policy_wdl_sft/test_dual_model2_group_adv_is_loss.py \
       tests/on_policy_wdl_sft/test_dual_model2_group_adv_is_trainer.py \
       tests/on_policy_wdl_sft/test_dual_model2_group_adv_is_scripts.py \
       -q --tb=short
```

- Loss computes nonzero positive-fallback signal for all-correct groups.
- Loss produces zero update for all-incorrect groups.
- Mixed groups push correct rows up and incorrect rows down.
- TIS caps `w_is` but does not drop samples.
- Binary mask gates only tokens violating `r_train` bounds.
- `rollout_is_weights` non-`None` raises a clear error.
- Missing `log_pi_model2_rollout` raises a clear error.
- `loss_agg_mode != seq-mean-token-sum` raises a clear error.

### Trainer Tests

- `dual_model2_group_adv_is` is excluded from WDL raw-label advantage override.
- GRPO/group advantages are preserved before positive fallback is added.
- `norm_adv_by_std_in_grpo=false` is used in the recipe.
- Selected rollout source is model2 only.
- `old_log_prob` is fused, not model2.
- `log_pi_model2_rollout` is preserved through actor update.

### Training Script Deliverables

Business-code implementation must include runnable training scripts. Do not
leave script creation as a separate follow-up after the loss is implemented.

Use the existing branch recipe directory. Do not create a new sibling recipe
folder for this method:

```text
recipe/on_policy_wdl_sft/dual_submodel_rollout/
```

Add or modify scripts in that existing directory, at minimum:

```text
_common_dual_submodel_rollout.sh or the existing shared launcher used by this directory
run_4a_model2_group_adv_is.sh
README.md
```

If a monitor, queue script, or smoke-specific launcher becomes part of the real
workflow, it must live in the same existing recipe directory or a clearly linked
branch-local path. Whenever a runnable training, shared launcher, monitor, or
queue script is created, materially changed, or used, update:

```text
docs/joint_training/guides/training_script_index.md
```

The branch-local index should remain factual and directory-like. Full launch
commands and monitor playbooks belong in the recipe README or relevant guide,
not in the index.

Required training-data and model defaults:

```text
train_files=/data-1/dataset/math/train_rl_format.parquet
model1=/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base
model2=/data-1/.cache/Qwen3-4B-Base-SFT-stage-1
```

Required online validation defaults:

```text
actor_rollout_ref.rollout.val_kwargs.n=3
```

Do not use validation `n=1` for this experiment family. The online validation
must report model2-only validation as the primary score.

Required checkpoint behavior:

- Keep latest checkpoints for normal resume/debugging.
- Track and preserve best checkpoints according to the configured validation
  metric.
- Best checkpoint retention must not be broken by latest-checkpoint pruning.
- The recipe must set or document the best-checkpoint metric key explicitly.

The first real 4A MATH training script should target exactly one filtered MATH
epoch: `TOTAL_TRAINING_STEPS=115` and `TOTAL_EPOCHS=1`. This mirrors the
branch's single-model MATH scripts. The shared launcher may still carry older
300-step defaults for 3A/3B reproduction, so the 4A thin wrapper must pin its
own schedule explicitly. Smoke scripts or smoke env overrides must be clearly
labeled and must not replace the real-run launcher.

Smoke overrides must not remove the learning signal. In particular, for this
math RL method, do not lower `MAX_RESPONSE_LENGTH` from the real-run default
`4096` or lower `N_RESP_PER_PROMPT` from the real-run default `8` when the run
is intended to validate algorithmic training behavior. A short-context run such
as `MAX_RESPONSE_LENGTH=256` is a plumbing-only smoke because truncated
responses receive reward `-1`, all-incorrect groups have zero group advantage,
and the policy gradient can be exactly zero.

### Smoke Gates

Before any full run:

1. Run a 1-3 step GPU smoke in tmux using production-context generation
   settings. Only reduce `TOTAL_TRAINING_STEPS`, run/checkpoint/log paths, and
   validation/save cadence.
2. Check that no non-finite loss, ratio, or grad metric appears.
3. Check `actor/grad_norm` does not show immediate 3A-style explosion.
4. Check `tis_clip_fraction`, `clipfrac_positive`, and `clipfrac_negative`.
5. Check that the run produced a learning signal, not only a successful code
   path.
6. If unstable, stop and record the blocker; do not silently switch to a
   lower `tis_threshold` inside the same default method.

Only after smoke passes should a longer 10-step stability gate be launched.
Follow the detailed policy in
`docs/joint_training/constraints/smoke_learning_signal_policy.md`.

Smoke pass/fail criteria:

- Required metrics fields exist in the metrics JSONL.
- `tis_mean`, `tis_max`, `train_ratio_mean`, `train_ratio_max`, loss values,
  and `actor/grad_norm` are finite.
- `tis_clip_fraction`, `clipfrac_positive`, and `clipfrac_negative` are finite
  and in `[0, 1]`.
- `dual_rollout/selected_source` identifies model2 / `sub_model_1`.
- `dual_rollout/model2_correct_ratio` and `dual_rollout/model2_response_len_mean`
  are present.
- `response_length/clip_ratio` must not indicate deterministic truncation across
  the whole smoke. If every response is clipped because the smoke lowered
  `MAX_RESPONSE_LENGTH`, the run is plumbing-only and does not satisfy this
  gate.
- Learning-signal evidence must be present: at least one logged step should have
  `mixed_group_fraction > 0` or `all_correct_fallback_group_fraction > 0`, and
  at least one logged step should have nonzero advantage/gradient evidence such
  as `critic/advantages/max > 0` and finite `actor/grad_norm > 0`.
- `actor/grad_norm` must not show immediate 3A-style explosion. Concrete first
  smoke threshold: no non-finite value and no step with `actor/grad_norm > 5000`
  in the first 3 completed update steps. If this threshold is violated, stop and
  record blocker evidence rather than tuning in place.
- Smoke evidence must include tmux session name, log path, metrics JSONL path,
  exact run name, and the command or script path used.

10-step stability gate pass/fail criteria:

- Completes 10 actor update steps without non-finite metrics or process crash.
- Required metrics remain present and finite for every logged train step.
- `actor/grad_norm` must not exceed the 3A collapse anchor range. Concrete
  threshold: no step with `actor/grad_norm > 10000` before step 10.
- Online validation settings are visible in config/logs as `val_kwargs.n=3`.
- Latest checkpoint state and best-checkpoint metadata are either produced or,
  if the gate is too short for save frequency, the configured save/best-checkpoint
  settings are printed and verified from logs/config.
- If all 10 steps are all-incorrect groups with `actor/grad_norm=0`, the gate
  can count only as plumbing/stability evidence. It must not be treated as
  evidence that the algorithm can train.
- Gate evidence must include log path, metrics JSONL path, validation output
  directory if validation ran, checkpoint directory, and a short metrics snippet.

### Implementation Acceptance Checklist

The Main Agent must provide this checklist before asking for final acceptance:

```text
Code:
- changed files:
- loss mode registration evidence:
- tensor provenance evidence:

Tests:
- pytest command:
- pytest result:

Reviewer Sub-Agent:
- formula/provenance reviewer verdict:
- trainer/data-path reviewer verdict:
- script/runtime reviewer verdict:

Smoke:
- tmux session:
- run name:
- log path:
- metrics JSONL path:
- checkpoint path:
- validation output path:
- key metrics snippet:

Scripts:
- recipe files:
- training_script_index.md diff summary:

Experiment registry:
- project:
- experiment/run id or explicit reason not recorded:
```

## 9. Reviewer Sub-Agent Protocol

Reviewer Sub-Agent review is mandatory. The Reviewer Sub-Agent is independent
from the Main Agent and provides feedback only. The Main Agent owns all code
changes, integration, fixes, final test execution, and final evidence.
The Main Agent must not self-accept the implementation; final acceptance is
based on Reviewer Sub-Agent verdicts plus the recorded evidence checklist.

### Reviewer Rules

- Reviewer Sub-Agents do not edit files unless the Main Agent explicitly creates
  a separate worker task restricted to test-only paths.
- Reviewer Sub-Agents must inspect actual diffs and relevant source files, not
  only the plan text.
- Reviewer Sub-Agents must cite concrete `file:line` evidence or exact test/log
  output for every PASS, WARN, or FAIL.
- Main Agent must address every FAIL before proceeding.
- WARN items may proceed only if the Main Agent records the risk, rationale, and
  a concrete follow-up or reason it is acceptable for this implementation.
- If reviewers disagree, Main Agent must preserve both views and resolve the
  conflict explicitly before running real training.

### Required Reviewer Sub-Agents

| Reviewer | Scope | Edit permission | Required output |
|---|---|---|---|
| Formula/provenance reviewer | Section 3 loss formula, gradient path, detach semantics, TIS, binary mask, fail-fast behavior | No edits | PASS/WARN/FAIL table with `file:line` evidence. |
| Trainer/data-path reviewer | model2 rollout source, `log_pi_model2_rollout` preservation, fused `old_log_prob`, selected batch tensors, advantage routing | No edits | PASS/WARN/FAIL table with `file:line` evidence. |
| Script/runtime reviewer | recipe scripts, dataset/model paths, validation `n=3`, latest/best checkpoint settings, script index update, smoke command shape | No edits | PASS/WARN/FAIL table with script/config/log evidence. |

Optional Worker Sub-Agent:

- May be used only for test authoring.
- Ownership is restricted to `tests/on_policy_wdl_sft/`.
- Main Agent must review all worker changes and remains responsible for final
  correctness.

### Reviewer Checklist

Before implementation is considered ready for a real run, reviewer output must
confirm:

- The formula in code matches Section 3 exactly.
- There is only one binary mask: the MiniRL old-current staleness mask.
- TIS is a continuous capped scalar weight, not a mask.
- External `rollout_is_weights` is disabled.
- The only gradient path in the loss is `log_pi_current_fused`.
- The method records `log_pi_model2_rollout` from the actual rollout policy.
- The code cannot silently fall back to old 3A behavior.
- Training scripts use `/data-1/dataset/math/train_rl_format.parquet`.
- Online validation is configured as `val_kwargs.n=3`, not `n=1`.
- Latest and best checkpoint retention are both configured or explicitly
  verified from the script/log.
- `docs/joint_training/guides/training_script_index.md` is updated after any
  runnable script is created or used.
- Smoke and 10-step gate evidence meet Section 8 pass/fail criteria.

Reviewer output format:

```text
Reviewer: <formula/provenance | trainer/data-path | script/runtime>
Verdict: PASS | WARN | FAIL

Findings:
- PASS/WARN/FAIL: <short finding> [file:line or exact test/log evidence]

Blocking items:
- <required Main Agent action, or "None">

Residual risks:
- <risk, or "None">
```

## 10. Database Ownership

All experiments for this method belong in the existing SQLite experiment
registry project:

```text
verl:feature/on-policy-wdl-sft-dual-rollout
```

Do not create a new physical database. This method is a new experiment family
within the current branch/research direction. Ablations of TIS threshold,
fallback coefficient, or smoke/stability variants should remain under this
branch project with method metadata identifying `dual_model2_group_adv_is`.
