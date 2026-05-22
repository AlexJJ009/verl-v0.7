# Goal: WDL Group-Advantage IS Loss With Positive-SFT Fallback

- Status: ACTIVE IMPLEMENTATION CONTRACT - ready for execution
- Target branch: `feature/on-policy-wdl-sft`
- Target loss mode: `wdl_group_adv_is`
- Target recipe family: `recipe/on_policy_wdl_sft/group_advantage_is/`
- Goal status file: `docs/joint_training/plans/active/wdl_group_advantage_is_status.md`
- Main references:
  - Current WDL-SFT v2 plan: `docs/joint_training/plans/active/wdl_sft_is.md`
  - Current method note: `docs/joint_training/courses/method_on_policy_wdl_sft.tex`
  - GFT reference: `docs/joint_training/references/external/gft_arxiv_2604_14258.md`
  - GFT source appendix: `docs/joint_training/references/external/gft_arxiv_2604_14258_source/sec/gft_appendix.tex`
  - Meituan launch playbook: `docs/joint_training/guides/meituan_platform.md`
- Last updated: 2026-05-20

## 1. Objective

Implement a new WDL loss variant that rewrites the current WDL-SFT objective as
an RL-style token-level policy-gradient surrogate, then adds group advantage,
an all-correct positive-SFT fallback, and explicit current-policy/old-policy
importance sampling.

The target algorithm keeps the current branch's intended data path:

```text
mixed joint rollout -> reward -> mixed-policy old_log_probs -> mixed-policy actor update
```

Both behavior and target policies for the new IS ratio are the fused/mixed joint
policy at different training times:

```text
rho_{i,t} = pi_theta^mix(y_{i,t} | x_i, y_{i,<t}) /
            pi_old^mix(y_{i,t} | x_i, y_{i,<t})
```

This is intentionally different from the rejected dual-rollout direction where
model2-generated trajectories were reweighted against fused/joint likelihood.
No model2-vs-joint-model importance sampling is part of this goal.

The implementation goal is method correctness, launchability, script
portability, and smoke validation. Full 300-step performance proof is outside
the first implementation goal.

This file is the implementation contract for the next coding session. The
executor should treat the requirements, tests, scripts, reviewer gates, and done
definition below as blocking acceptance criteria.

## 2. Background Findings

### 2.1 Current WDL-SFT-IS Behavior

Current `wdl_sft_is` already computes an old/current ratio:

```text
rho_{i,t} = exp(log_prob_{i,t} - old_log_prob_{i,t})
```

but that ratio is only used as a detached binary keep/drop selector. It is not
multiplied into the token loss as an importance-sampling weight.

Current `wdl_sft_is` also overrides `advantages` with raw reward labels before
actor update. Therefore the current loss does not use GRPO-style group
advantages even though the trainer computes them earlier.

### 2.2 GFT Reference Relevance

The GFT paper is useful because it rewrites SFT into an RL-style form and gives
a token-level loss based on group advantages. For this project, use it as a
method reference, not as code truth. The local implementation must define the
surrogate loss and its autograd path explicitly.

### 2.3 Feasibility In The Current Branch

The current branch uses joint rollout and joint actor training. When rollout is
the fused joint policy and training also evaluates fused joint logits, the
old/current ratio is a standard proximal-policy ratio:

```text
pi_theta^mix / pi_old^mix
```

This is method-level coherent as a proximal old/current correction. It corrects
the training-time drift between the frozen old policy for the sampled batch and
the current actor policy being optimized. It is not a complete correction for
possible vLLM rollout vs FSDP actor numerical mismatch; `rollout_is_weights`
remain intentionally disabled for this first method revision.

## 3. Non-Negotiable Boundaries

### In Scope

- Add a new loss mode, default name `wdl_group_adv_is`.
- Keep rollout from the fused/mixed joint policy.
- Keep actor training on fused/mixed joint logits.
- Keep both submodels trainable.
- Use group advantages instead of raw reward-label C/I partition.
- Preserve positive SFT learning signal for all-correct groups through the
  fallback term in Section 4.2.
- Remove `beta` from the new algorithm; negative samples enter through negative
  group advantages.
- Add explicit multiplicative IS using `pi_theta^mix / pi_old^mix`.
- Keep the existing old/current binary trust-region mask unless a reviewer
  finds a concrete mathematical or implementation blocker.
- Preserve `seq-mean-token-sum` aggregation by default and fail fast otherwise.
- Add a new portable recipe/script family compatible with the Meituan layered
  launch playbook.
- Create the complete Meituan four-layer launch path in the first
  implementation pass; do not leave platform files as a follow-up.
- Run real GPU smoke with vLLM and FlashInfer before marking implementation
  complete.

### Out Of Scope For First Implementation

- Do not use `rollout_is_weights`.
- Do not use KL penalty in reward.
- Do not use actor KL loss.
- Do not introduce length normalization or switch to token-mean aggregation.
- Do not reintroduce `beta` in the new loss.
- Do not add model2-vs-joint-model importance sampling.
- Do not implement dual-submodel rollout.
- Do not require a full 300-step experiment as implementation acceptance.
- Do not claim equivalence to standard GRPO unless the remaining differences in
  Section 8 are removed in a separate goal.

## 4. Target Algorithm

For prompt group `g`, rollout responses are `y_i`, token index is `t`, and
scalar rewards are `R_i in {+1, -1}`.

### 4.1 Group Advantage

Default first implementation:

```text
A_i = R_i - mean_{j in group(i)} R_j
```

Hard default: `algorithm.norm_adv_by_std_in_grpo=False`. Do not add a
std-normalized experiment path in this goal. Keeping mean-centered advantages
avoids changing advantage scale, positive-fallback scale, and IS semantics in
the same first experiment. A closer-to-GRPO std-normalized run requires a
separate goal.

Boundary condition:

- all-correct group: all `A_i = 0`, but the positive-SFT fallback in Section
  4.2 must still produce an update;
- all-incorrect group: all `A_i = 0`, no update from that group;
- mixed group: correct samples have positive `A_i`, incorrect samples have
  negative `A_i`.

This is not a pure group-advantage algorithm because all-correct groups retain
the positive SFT signal. Negative samples still enter only through group
advantages in mixed groups; there is no `beta` reverse-SFT term.

### 4.2 All-Correct Positive-SFT Fallback

Pure group advantage would produce zero gradient for all-correct groups. This
goal must preserve the current WDL-SFT family's useful positive SFT signal on
those groups.

For each response row:

```text
all_correct_g = 1 if all rewards in group g are +1 else 0
F_i = all_correct_{group(i)}
gamma_pos_sft = 1.0
G_i = A_i + gamma_pos_sft * F_i
```

`G_i` is the effective policy coefficient used by the loss. It has these
required behaviors:

- all-correct group: `A_i=0`, `F_i=1`, `G_i=+1`, positive SFT update;
- all-incorrect group: `A_i=0`, `F_i=0`, `G_i=0`, no update;
- mixed group: `F_i=0`, `G_i=A_i`, group-advantage positive/negative update.

The fallback must be computed where group rewards are still available, normally
in the trainer pipeline before actor update. A loss function that only sees
zero-valued `advantages` cannot distinguish all-correct from all-incorrect
groups, so implementing this only inside `core_algos.py` is invalid unless the
loss receives explicit fallback/group metadata.

Implementation choices allowed:

1. Store `G_i` in the `advantages` tensor passed to actor loss and log the raw
   GRPO advantage metrics before augmentation.
2. Pass an explicit detached fallback tensor to the loss and combine it with
   `advantages` there.

Either choice must be tested. The status file must record which choice was
implemented.

### 4.3 Mixed-Policy IS Ratio

For every real response token:

```text
Delta_{i,t} = log pi_theta^mix(y_{i,t}) - log pi_old^mix(y_{i,t})
rho_{i,t} = exp(clamp(Delta_{i,t}, -20, 20))
```

`old_log_probs` must be computed under the fused/mixed training policy for the
sampled batch before actor update. `log_prob` is computed under the current
fused/mixed actor during actor update.

The IS ratio must be detached before multiplication in the surrogate loss:

```text
rho_sg = stop_gradient(rho)
```

Rationale: this implements a policy-gradient surrogate with a fixed sample
weight. Backpropagating through the ratio would optimize a different objective
and is not the intended policy-gradient estimator.

### 4.4 Trust-Region Binary Mask

Keep the current MiniRL-style detached binary mask, generalized from reward
label sign to effective coefficient sign:

```text
keep_{i,t} = 1  if G_i >= 0 and rho_{i,t} <= 1 + clip_ratio_high
keep_{i,t} = 1  if G_i <  0 and rho_{i,t} >= 1 - clip_ratio_low
keep_{i,t} = 0  otherwise
```

The mask is detached and only gates token participation. It is not the IS
weight itself.

This means the first implementation uses:

```text
effective_weight_{i,t} = stop_gradient(G_i) *
                         stop_gradient(rho_{i,t}) *
                         stop_gradient(keep_{i,t})
```

### 4.5 Loss With Preserved `seq-mean-token-sum`

The loss matrix is:

```text
loss_{i,t} = - effective_weight_{i,t} *
               log pi_theta^mix(y_{i,t} | x_i, y_{i,<t})
```

Aggregate with the existing `seq-mean-token-sum` semantics:

```text
L = mean_i sum_t response_mask_{i,t} loss_{i,t}
```

Do not normalize by token count. Long responses keep larger gradient energy,
matching the current WDL-SFT family.

Hard guard: `wdl_group_adv_is` must raise a clear configuration error if
`loss_agg_mode != "seq-mean-token-sum"`. The function signature default is not
sufficient because actor config can pass `token-mean`.

Implementation guard: the scalar loss must be produced through the shared
`agg_loss(...)` helper with the actor `config.global_batch_info`, not through a
local `.mean()` or a hand-written sequence average. This keeps the loss scale
consistent under FSDP data parallelism and dynamic batching.

### 4.6 Expected Gradient

For unclipped real tokens, the intended actor gradient is:

```text
grad L = - G_i * rho_{i,t} *
         grad log pi_theta^mix(y_{i,t} | x_i, y_{i,<t})
```

For clipped or padded tokens, the gradient is zero.

This is gradient-accurate for the declared surrogate. Because the implementation
uses token-level truncated IS and a binary mask, it is not an unbiased
sequence-level off-policy estimator. That tradeoff is accepted for stability.

### 4.7 Data-Flow And Grouping Invariants

These invariants are part of the implementation contract. They are intended to
prevent a superficially correct loss from silently receiving the wrong grouping
or policy tensors.

1. The rollout group is the set of `n` responses sampled for the same original
   prompt. Group boundaries must come from the trainer's prompt/uid structure,
   not from incidental tensor order after an uncontrolled shuffle.
2. Rewards, advantages, `old_log_probs`, current `log_prob`, and
   `response_mask` must stay row-aligned from rollout collection through actor
   update.
3. `advantages` passed into `wdl_group_adv_is` must be the GRPO-computed group
   advantages plus the explicit all-correct positive-SFT fallback, or the raw
   GRPO advantages plus a separate fallback tensor. They must not be replaced
   by raw `+1/-1` reward labels.
4. `old_log_probs` must be recomputed under the fused/mixed actor policy for
   the sampled batch before the actor update. Rollout-time model2-only,
   submodel-only, or dual-rollout likelihoods are invalid for this goal.
5. If any implementation flattens, reorders, chunks, or pads the batch before
   computing group metrics, it must either preserve explicit group ids or prove
   by test that the group reconstruction is exact.
6. True group-level metrics such as `zero_adv_group_fraction` and
   `mixed_group_fraction` must be computed where group identity is available
   in the trainer pipeline, or the implementation must explicitly pass group
   metadata into the metric layer. Tensor-only approximations inside the loss
   function must be named as response-level metrics, not group metrics.
7. The all-correct fallback must be computed from true group rewards. It cannot
   be inferred from `advantages == 0` because all-correct and all-incorrect
   groups both have zero group advantage.

## 5. Required Config Surface

New loss mode:

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: wdl_group_adv_is
```

Required defaults for first scripts:

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false
  use_kl_in_reward: false
  rollout_correction:
    rollout_is: null
    rollout_rs: null

actor_rollout_ref:
  actor:
    use_kl_loss: false
    kl_loss_coef: 0.0
    loss_agg_mode: seq-mean-token-sum
    ppo_epochs: 1
  rollout:
    calculate_log_probs: false
```

Acceptance:

- `algorithm.norm_adv_by_std_in_grpo=false` is mandatory for the first
  implementation scripts and smoke. Do not make the first run std-normalized.
- `wdl_group_adv_is` must consume GRPO-computed group advantages plus the
  all-correct positive-SFT fallback. It must not consume raw reward labels as a
  C/I partition.
- `apply_wdl_sft_reward_label_advantages(...)` must not override advantages for
  `wdl_group_adv_is`.
- `wdl_sft` and `wdl_sft_is` must keep their current raw-label behavior.
- New recipe/script must set `ROLLOUT_IS=null` or equivalent.
- New recipe/script must keep both KL controls disabled.
- New recipe/script must keep `LOSS_AGG_MODE=seq-mean-token-sum`.
- `wdl_sft_beta` must not be read by the new loss.
- `wdl_group_adv_is` must fail fast if `rollout_is_weights` is non-None.
- `wdl_group_adv_is` must fail fast if `loss_agg_mode` is not
  `seq-mean-token-sum`.
- `wdl_group_adv_is` must not emit `actor/wdl_sft_beta` or any beta-specific
  metric.

## 6. Required Recipe And Launch Scripts

The first implementation must create the complete four-layer Meituan launch
path. This is not optional and must not be deferred to a later session.

Layer 3 and Layer 4 recipe family files:

```text
recipe/on_policy_wdl_sft/group_advantage_is/
├── README.md
├── _common_group_adv_is.sh
├── run_1a_group_adv_is.sh
└── meituan/
    ├── env.sh
    └── jupyter.sh
```

Layer 1 and Layer 2 platform shim files:

```text
platform/hope_group_advantage_is/
├── README.md
├── jupyter.sh
└── run.hope
```

Layer mapping:

- Layer 1: `platform/hope_group_advantage_is/run.hope` and README template for
  the AFO `hope_dir`.
- Layer 2: `platform/hope_group_advantage_is/jupyter.sh`, a family-level shim
  that requires `EXPERIMENT`, locates the repo, propagates `SMOKE=1`, and execs
  Layer 3.
- Layer 3: `recipe/on_policy_wdl_sft/group_advantage_is/meituan/env.sh` and
  `meituan/jupyter.sh`, owning dolphinfs path overrides and family-level
  routing.
- Layer 4: `recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh`,
  a thin per-experiment wrapper sourced into `_common_group_adv_is.sh`.

Script rules:

- `run_1a_group_adv_is.sh` must be a thin wrapper only: export
  experiment-specific knobs, then source `_common_group_adv_is.sh`.
- `_common_group_adv_is.sh` owns shared environment setup, checkpoint/resume
  handling, Hydra launch, and shared defaults.
- `meituan/env.sh` owns dolphinfs path overrides and high-churn temp dirs.
- `meituan/jupyter.sh` resolves `EXPERIMENT` to `run_${EXPERIMENT//-/_}.sh`,
  validates prerequisites, then `exec bash "$RUN_SCRIPT"`.
- `platform/hope_group_advantage_is/jupyter.sh` must be family-level only:
  require `EXPERIMENT`, locate the repo, respect `SMOKE=1`, and exec the
  recipe-family Meituan adapter.
- `platform/hope_group_advantage_is/run.hope` must be a reusable template with
  `EXPERIMENT=1a-group-adv-is` or the final chosen variant name.

Portability rules:

- Every path in `run_*.sh` and `_common_group_adv_is.sh` must use
  `${VAR:-local-default}`.
- No dolphinfs path may appear outside `meituan/env.sh` or platform shim docs.
- External callers must be able to override parent paths for repo, data,
  models, checkpoints, logs, caches, Ray temp, generic temp, validation output,
  and wandb/offline logging.
- Required overridable variables include at least:
  - `REPO_ROOT`
  - `DATA_ROOT`
  - `TRAIN_FILE`
  - `TEST_FILES`
  - `MODEL_PATH`
  - `BASE_CKPT_DIR`
  - `LOG_DIR`
  - `WANDB_DIR`
  - `HF_HOME`
  - `RAY_TMPDIR`
  - `TMPDIR`
  - `VALIDATION_OUTPUT_DIR`
  - `CUSTOM_REWARD_FN_PATH`
- `VLLM_ATTENTION_BACKEND=FLASHINFER` must be forced or defaulted in the common
  script and preserved by Meituan adapters.
- FSDP training must keep `attn_implementation=flash_attention_2`.
- On Meituan, `WANDB_MODE=offline` and high-churn temp dirs must default to
  container-local `/tmp`, not dolphinfs.
- The common script must expose `PPO_EPOCHS` as an environment override and map
  it to `actor_rollout_ref.actor.ppo_epochs`; this is required for the
  ratio-path smoke.
- The script family must follow `docs/joint_training/guides/meituan_platform.md`
  from day one. Running locally must not require any Meituan layer; running on
  Meituan must not require editing the per-experiment wrapper.

Algorithm defaults in the new scripts:

```bash
LOSS_MODE=wdl_group_adv_is
LOSS_AGG_MODE=seq-mean-token-sum
PPO_EPOCHS=1
ROLLOUT_IS=null
ROLLOUT_RS=null
USE_KL_IN_REWARD=False
USE_KL_LOSS=False
KL_COEF=0.0
KL_LOSS_COEF=0.0
NORM_ADV_BY_STD_IN_GRPO=false
ALL_CORRECT_SFT_FALLBACK=true
POS_SFT_FALLBACK_COEF=1.0
```

`WDL_SFT_BETA` must not be required, exported, printed, or passed to Hydra for
this recipe family.

## 7. Implementation Tasks

1. Create or update the goal-local status file:
   `docs/joint_training/plans/active/wdl_group_advantage_is_status.md`.
2. Add the new loss function in `verl/trainer/ppo/core_algos.py`.
3. Register it as `wdl_group_adv_is`.
4. Implement formula:
   - read `advantages` as group advantages, plus the all-correct positive-SFT
     fallback if the trainer stores `G_i` in `advantages`;
   - otherwise read an explicit fallback tensor and compute `G_i` before
     token-level loss;
   - compute `rho = exp(log_prob - old_log_prob)`;
   - detach `G_i`, `rho`, and binary keep mask;
   - multiply `rho` into token losses;
   - aggregate by calling `agg_loss(loss_mat=..., loss_mask=response_mask,
     loss_agg_mode="seq-mean-token-sum", **config.global_batch_info)`;
   - do not hand-roll local `.mean()` or local-only sequence averaging;
   - raise if `rollout_is_weights is not None`;
   - raise if `loss_agg_mode != "seq-mean-token-sum"`.
5. Ensure `apply_wdl_sft_reward_label_advantages(...)` excludes
   `wdl_group_adv_is`.
6. Add config or script defaults that disable:
   - `rollout_is_weights`;
   - KL in reward;
   - actor KL loss;
   - length/token normalization.
7. Implement all files listed in Section 6 in one pass, including the complete
   Meituan four-layer launch path.
8. Add metrics for no-gradient group visibility:
   - trainer-level `wdl_group_adv_is/zero_adv_group_fraction`
   - trainer-level `wdl_group_adv_is/mixed_group_fraction`
   - trainer-level `wdl_group_adv_is/all_correct_fallback_group_fraction`
   - trainer-level `wdl_group_adv_is/all_correct_fallback_response_fraction`
   - loss-level `wdl_group_adv_is/zero_adv_response_fraction`
   - `wdl_group_adv_is/ratio_mean`
   - `wdl_group_adv_is/ratio_max`
   - `wdl_group_adv_is/ratio_mean_pos_adv`
   - `wdl_group_adv_is/ratio_max_pos_adv`
   - `wdl_group_adv_is/ratio_mean_neg_adv`
   - `wdl_group_adv_is/ratio_max_neg_adv`
   - `wdl_group_adv_is/clipfrac_positive`
   - `wdl_group_adv_is/clipfrac_negative`
   - existing `actor/grad_norm` must be recorded in smoke evidence.
9. Add unit tests for the new loss.
10. Add trainer-path tests showing `wdl_group_adv_is` preserves GRPO advantage
    semantics and adds only the all-correct positive-SFT fallback.
11. Add script tests or shell syntax checks for the new recipe and platform
    shim.
12. Run CPU/unit tests.
13. Run intermediate real GPU probes when a milestone becomes runnable, as
    defined in Section 10.1. These probes are allowed before the full goal is
    complete and should be used to catch runtime issues early.
14. Run the final real GPU smoke with vLLM + FlashInfer for 1 to 3 steps.
15. Update method docs after implementation, including the RL-form derivation
    and the difference from standard GRPO.
16. Update this goal and the status file with implementation notes, tests,
    smoke evidence, and deviations.
17. Run the project agent-context sync or record why it is blocked.
18. Record the exact sync command, result, and any bridge-file diff in the
    status file.

## 8. Relationship To Standard GRPO

After this goal, the loss will share the following with GRPO:

- group-based outcome advantages;
- old/current policy ratio;
- token-level policy-gradient surrogate;
- mixed positive and negative updates from advantage sign.

It will still differ from standard GRPO in these ways:

- default advantage is mean-centered but not std-normalized;
- all-correct groups retain a positive SFT fallback term, so the method is not
  pure group advantage;
- aggregation remains `seq-mean-token-sum`, not token-mean or Dr.GRPO-style
  token normalization;
- clipping is a detached binary keep/drop mask, not the PPO/GRPO min-clipped
  surrogate;
- no KL penalty is used;
- actor policy is the fused joint policy, so gradients flow through both
  submodels via fused logits.

Therefore the first implementation should be described as:

```text
WDL group-advantage policy-gradient with all-correct positive-SFT fallback,
token-level IS, and binary mask
```

It should not be described as standard GRPO.

## 9. Tests

Required unit/script tests:

1. **Neutral ratio**: when `old_log_prob == log_prob`, `rho=1` and the loss
   reduces to `-G * log_prob` with `seq-mean-token-sum`, where
   `G = A + all_correct_fallback`.
2. **Multiplicative IS**: with a hand-constructed non-1 ratio, gradient
   magnitude changes by the detached `rho` factor.
3. **Mask + IS together**: clipped tokens have zero gradient; unclipped tokens
   have gradient scaled by `rho`.
4. **Positive advantage upper mask**: `A>0` token with `rho > 1+high` is zeroed.
5. **Negative advantage lower mask**: `A<0` token with `rho < 1-low` is zeroed.
6. **No beta dependency**: changing `wdl_sft_beta` does not change
   `wdl_group_adv_is`.
7. **No rollout_is dependency**: passing `rollout_is_weights` to the new loss
   raises a clear error.
8. **Preserve WDL raw-label path**: `wdl_sft` and `wdl_sft_is` still get raw
   labels from `apply_wdl_sft_reward_label_advantages(...)`.
9. **New loss preserves GRPO semantics plus fallback**: after
   `compute_advantage(...)`, `loss_mode=wdl_group_adv_is` must not overwrite
   advantages with raw labels; it may only add the explicit all-correct
   positive-SFT fallback.
10. **All-same group behavior**: all-correct groups produce positive SFT loss
    through the fallback, while all-incorrect groups produce zero policy loss.
11. **Aggregation**: a long sequence with the same advantage contributes more
    than a short sequence, proving `seq-mean-token-sum` rather than token-mean.
12. **Exact surrogate value**: hand-computed loss equals
    `-(G.detach() * rho.detach() * keep.detach() * log_prob)` after masking and
    `seq-mean-token-sum` aggregation.
13. **Detached ratio**: gradients do not flow through `rho`; a test must fail if
    the implementation uses non-detached `ratio * log_prob`.
14. **Hard aggregation guard**: `loss_agg_mode="token-mean"` raises a clear
    error.
15. **Script defaults**: shell/static tests prove `ROLLOUT_IS=null`, both KL
    controls are false/zero, `LOSS_AGG_MODE=seq-mean-token-sum`, and no
    `WDL_SFT_BETA` Hydra override exists in the new recipe.
16. **Meituan path portability**: shell/static tests prove `run_*.sh` and
    `_common_group_adv_is.sh` use overridable env vars for all parent paths and
    contain no dolphinfs hard-code.
17. **Meituan smoke propagation**: shell/static or dry-run tests prove
    `SMOKE=1` propagates from `platform/hope_group_advantage_is/jupyter.sh` to
    `recipe/on_policy_wdl_sft/group_advantage_is/meituan/jupyter.sh` and into
    the final recipe launch knobs.
18. **Group identity preservation**: trainer tests prove
    `zero_adv_group_fraction` and `mixed_group_fraction` are computed from true
    prompt groups. If group metadata is not passed into the loss, tests must
    prove any loss-level zero-adv metric is named as response-level only.
19. **Aggregation helper usage**: unit tests or code review evidence prove the
    new loss uses `agg_loss(..., **config.global_batch_info)` and does not
    bypass distributed loss scaling.
20. **Hard no std-normalization default**: script/static tests prove
    `NORM_ADV_BY_STD_IN_GRPO=false` and
    `algorithm.norm_adv_by_std_in_grpo=false` in the new launch family.
21. **Fallback metric/test coverage**: trainer tests prove the fallback is
    computed from true group rewards, not inferred from zero advantages, and
    metrics expose all-correct fallback frequency.

Suggested test locations:

```text
tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py
tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py
tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py
```

## 10. GPU Smoke Acceptance

### 10.1 Intermediate GPU Probes During Implementation

The executor may and should use the GPU before final acceptance to run short
real training probes when a milestone becomes runnable. These probes are
especially expected after:

- core loss registration and dispatch are wired;
- trainer-side GRPO advantage plus all-correct fallback routing is wired;
- the recipe/common script can launch locally;
- Meituan/platform shims are added or changed enough to affect launch
  environment.

Intermediate probes are allowed to run for only 1 to 3 training steps and do
not replace unit tests, reviewer gates, or the final smoke. They are an
explicit acceptance-support mechanism for finding runtime-only bugs early.

Intermediate probe requirements:

- Use real GPU execution, not CPU-only mocks.
- Use vLLM rollout with `VLLM_ATTENTION_BACKEND=FLASHINFER` whenever the
  runnable milestone includes rollout.
- Use tmux for any probe expected to take more than a few minutes.
- Keep probe scale small, for example `TOTAL_TRAINING_STEPS=1..3`,
  `TRAIN_PROMPT_BSZ=2`, `TRAIN_PROMPT_MINI_BSZ=1`, validation disabled unless
  the milestone being tested needs validation.
- Preserve method defaults: `ROLLOUT_IS=null`,
  `LOSS_AGG_MODE=seq-mean-token-sum`,
  `NORM_ADV_BY_STD_IN_GRPO=false`, KL disabled, no beta.
- Record command, tmux session if used, log path, result, and any observed
  failure in the status file.

Intermediate probe acceptance evidence:

- the process reaches the intended milestone, such as actor loss call, one
  actor update, script launch validation, or one complete training step;
- logs show `loss_mode=wdl_group_adv_is` when actor loss is exercised;
- no unexpected `rollout_is_weights`, KL, beta, or non-`seq-mean-token-sum`
  behavior appears;
- relevant new metrics appear once their code path is reached.

If an intermediate probe finds a bug, OOM, hang, non-finite metric, missing
metric, or method-contract violation, treat it as a real implementation failure:
debug, identify the cause, fix it, rerun the relevant unit/static tests, and
rerun the probe or a stricter one before moving to the dependent milestone.

### 10.2 Final GPU Smoke

Smoke command shape:

```bash
tmux new-session -s wdl_group_adv_is_smoke
# inside tmux:
TOTAL_TRAINING_STEPS=1 \
TRAIN_PROMPT_BSZ=2 \
TRAIN_PROMPT_MINI_BSZ=1 \
PPO_EPOCHS=1 \
ROLLOUT_AGENT_NUM_WORKERS=1 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=1 \
ROLLOUT_IS=null \
LOSS_MODE=wdl_group_adv_is \
LOSS_AGG_MODE=seq-mean-token-sum \
bash recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh
```

Acceptance:

- run is launched inside tmux for any long-running smoke or full training;
- real GPU run, not CPU-only;
- vLLM rollout is used;
- `VLLM_ATTENTION_BACKEND=FLASHINFER` is active;
- actor training uses `attn_implementation=flash_attention_2`;
- `use_kl_in_reward=False`;
- `actor.use_kl_loss=False`;
- `algorithm.rollout_correction.rollout_is=null`;
- `actor.loss_agg_mode=seq-mean-token-sum`;
- `actor.ppo_epochs=1` for the minimal infrastructure smoke, unless running the
  follow-up ratio-path smoke below;
- `algorithm.norm_adv_by_std_in_grpo=False`;
- all-correct positive-SFT fallback is enabled with coefficient 1.0;
- `loss_mode=wdl_group_adv_is`;
- at least one actor update completes;
- logs/metrics show nonzero ratio diagnostics when multiple minibatches or
  epochs make `pi_theta != pi_old`;
- ratio diagnostics are split by advantage sign, including negative-advantage
  high-ratio visibility;
- `actor/grad_norm` is recorded and checked for obvious spikes;
- no `rollout_is_weights` are passed into the new loss;
- no `wdl_sft_beta` metric is emitted for the new loss;
- zero-advantage group metrics are logged.
- all-correct fallback metrics are logged.

If the minimal smoke keeps `rho=1` because the first actor update is fully
on-policy, run a required follow-up ratio-path smoke on the same rollout batch
with `actor_rollout_ref.actor.ppo_epochs=2` or another explicit same-batch
multi-minibatch/multi-epoch setup. Merely running more global training steps is
not sufficient evidence, because each step may recompute fresh `old_log_probs`.
The follow-up acceptance condition is:

- `wdl_group_adv_is/ratio_max != 1`, or
  `abs(wdl_group_adv_is/ratio_mean - 1) > 1e-6`;
- sign-split ratio metrics are present;
- no large unexplained `actor/grad_norm` spike appears.

## 11. Agent Work Split

The implementation should be split into owner/reviewer pairs. Reviewers are
quality gates and do not own final integration.

| Subtask | Owner agent | Reviewer gate |
| --- | --- | --- |
| Method/formula doc | Method owner | Math reviewer |
| Core loss implementation | Loss owner | Loss reviewer |
| Trainer advantage routing | Trainer owner | Pipeline reviewer |
| Recipe/config and Meituan launch | Script owner | Config/Meituan reviewer |
| Unit tests | Test owner | Test reviewer |
| GPU smoke | Smoke owner | Runtime reviewer |
| Final docs/status | Docs owner | Final reviewer |

Reviewer input protocol:

- Every reviewer subagent must receive:
  - this goal file;
  - the goal-local status file;
  - the relevant diff or commit hash;
  - the specific subtask acceptance criteria;
  - available test/smoke outputs for that subtask.
- The reviewer feedback must be given back to the main agent.
- The main agent must address every `FAIL` before moving on, unless the user
  explicitly accepts the failed state.
- For every `WARN`, the main agent must either fix it or record the accepted
  limitation plus follow-up in the status file.

Reviewer verdicts must be one of:

- `PASS`: acceptance criteria are met.
- `WARN`: usable with recorded limitation.
- `FAIL`: blocking issue; main agent must fix before continuing unless user
  explicitly accepts the failure.

Reviewer feedback and main-agent response must be summarized in the status file.

Hard ordering rule: an owner subtask is not complete, and the main agent must
not move into a dependent milestone or milestone commit, until the corresponding
reviewer gate has `PASS` or a recorded `WARN` with accepted limitation and
follow-up. A `FAIL` blocks dependent work unless the user explicitly accepts
the failed state in writing.

## 12. Reviewer Gates

### 12.1 Method/Formulation Reviewer

PASS requires:

- loss formula uses mixed-policy `pi_theta / pi_old`;
- no model2-vs-joint IS appears;
- beta is absent from the new algorithm;
- group advantages explain negative samples;
- all-correct groups retain positive SFT through the fallback term;
- all-incorrect groups still produce zero loss;
- `norm_adv_by_std_in_grpo=false` is preserved for the first implementation;
- `seq-mean-token-sum` is explicitly preserved;
- the doc states this is not pure group advantage and not standard GRPO.

FAIL if the formula silently reintroduces `rollout_is_weights`, KL penalty, or
token-length normalization.

### 12.2 Core Loss Reviewer

PASS requires code evidence that:

- `wdl_group_adv_is` is registered;
- the effective coefficient `G_i` is group advantage plus only the all-correct
  positive-SFT fallback, not raw reward labels;
- ratio is multiplied into the token loss;
- ratio is detached;
- binary mask is detached;
- aggregation is `seq-mean-token-sum`;
- aggregation calls the shared `agg_loss(...)` helper with
  `config.global_batch_info`;
- non-`seq-mean-token-sum` loss aggregation raises;
- `rollout_is_weights` cannot affect the loss;
- `wdl_sft_beta` cannot affect the loss;
- no `actor/wdl_sft_beta` metric is emitted.

FAIL if ratio is only used as a mask or if gradients flow through the ratio.

### 12.3 Trainer Pipeline Reviewer

PASS requires:

- `compute_advantage(...)` still runs before actor update;
- `wdl_group_adv_is` is excluded from the raw-label override;
- `wdl_sft` and `wdl_sft_is` remain unchanged;
- the all-correct positive-SFT fallback is computed from true group rewards
  before group identity is lost;
- `old_log_probs` are computed under the fused/mixed actor policy;
- no dual-rollout source or model2-only data path is introduced;
- zero-advantage group metrics are computed from prompt groups, not from the
  flattened batch in a way that loses group identity.
- tensor-only zero-advantage metrics inside the loss, if any, are named as
  response-level metrics rather than group-level metrics.

FAIL if new loss receives raw labels instead of group advantages plus the
explicit all-correct fallback.

### 12.4 Config/Script And Meituan Reviewer

PASS requires:

- scripts set `ROLLOUT_IS=null`;
- both KL controls are false/zero;
- `LOSS_AGG_MODE=seq-mean-token-sum`;
- `NORM_ADV_BY_STD_IN_GRPO=false`;
- all-correct fallback defaults are enabled with coefficient 1.0;
- no `WDL_SFT_BETA` is required for the new run;
- all files in Section 6 exist, covering all four Meituan layers;
- `run_1a_group_adv_is.sh` is a thin wrapper;
- `_common_group_adv_is.sh` owns shared Hydra launch logic;
- every parent path is overridable by environment variable;
- `meituan/env.sh` contains dolphinfs-specific path overrides and high-churn
  temp dirs under `/tmp`;
- `meituan/jupyter.sh` resolves `EXPERIMENT` to the run script and `exec`s it;
- `platform/hope_group_advantage_is/jupyter.sh` is family-level and respects
  `SMOKE=1`;
- tests prove `SMOKE=1` reaches the final recipe launch path;
- `PPO_EPOCHS` is supported and maps to `actor_rollout_ref.actor.ppo_epochs`;
- `bash -n` passes for all shell scripts.

FAIL if launch defaults produce `rollout_is_weights` for the new loss.

### 12.5 Test Reviewer

PASS requires:

- all tests in Section 9 are implemented or explicitly mapped to existing
  coverage;
- reported commands and outputs are exact;
- failure cases prove the ratio is multiplicative, not mask-only;
- tests prove the exact surrogate value and no gradient through ratio;
- tests prove all-correct fallback gives positive SFT and all-incorrect groups
  remain zero-loss;
- tests prove script portability and Meituan compatibility.

FAIL if there is no gradient-level test for IS scaling.

### 12.6 Runtime Reviewer

PASS requires:

- intermediate GPU probe evidence for runnable milestones, or a recorded reason
  why no intermediate probe was useful before final smoke;
- real GPU smoke evidence;
- vLLM + FlashInfer evidence;
- at least one actor update;
- logs/metrics proving new loss mode and disabled unwanted tricks;
- sign-split ratio diagnostics and `actor/grad_norm` evidence;
- tmux session name, command, log path, and run output are recorded in the
  status file.

WARN is acceptable if the first smoke has `rho=1` but all infrastructure works;
the status file must then require a follow-up same-batch multi-epoch or
multi-minibatch smoke as defined in Section 10.

## 13. Branch, Commit, And Status Discipline

Work on `feature/on-policy-wdl-sft` unless the user explicitly asks for a new
branch. Do not touch unrelated dirty files.

Maintain the goal-local status file:

```text
docs/joint_training/plans/active/wdl_group_advantage_is_status.md
```

The status file must include:

- current branch and latest relevant commit;
- current task/milestone;
- completed milestones;
- files changed intentionally;
- tests run and exact results;
- intermediate GPU probe commands, results, and fixes;
- GPU smoke command and result;
- reviewer verdicts;
- project agent-context sync command, result, and bridge-file diff summary;
- open blockers or user decisions needed;
- next concrete action.

Read/write rules:

- Create the status file before major code edits.
- At the start of any resumed session or after context compaction, read this
  status file before continuing implementation.
- Before any expected context compaction, long pause, or handoff, update the
  status file with the live state.
- After every meaningful commit, update the status file with commit hash,
  summary, tests, reviewer verdicts, and next action.
- Before final completion, update the status file to point at final commits and
  verification results.

Commit expectations:

- Commit promptly after coherent, verifiable milestones rather than leaving the
  whole implementation uncommitted until the end.
- Once a milestone has passing required tests/reviewer gate, commit it before
  moving to the next dependent milestone, unless there is an explicit blocker
  recorded in the status file.
- Do not accumulate more than one coherent milestone of uncommitted intentional
  changes. If work must remain uncommitted because tests are failing or a user
  decision is needed, record the reason and exact working-tree state in the
  status file.
- Before committing, inspect the working tree and include only intended files.
- Do not commit unrelated dirty files that existed before this goal.
- If user-owned dirty files block a clean commit boundary, record the situation
  in the status file and ask before touching or staging them.

Suggested commit boundaries:

1. Method/status docs.
2. Core loss + trainer routing.
3. Tests.
4. Recipe/scripts and Meituan adapters.
5. Smoke/debug fixes.
6. Final docs/status.

## 14. Done Definition

This goal is complete only when all of the following are true:

- `wdl_group_adv_is` is implemented and registered.
- The new loss consumes GRPO advantages plus the all-correct positive-SFT
  fallback, not raw reward labels as a C/I partition.
- All-correct groups retain positive SFT signal through a tested fallback term.
- All-incorrect groups remain zero-loss under the new method.
- Multiplicative detached old/current IS is proven by exact-value and gradient
  tests.
- `rollout_is_weights` non-None fails fast for the new loss.
- Non-`seq-mean-token-sum` aggregation fails fast for the new loss.
- `wdl_sft_beta` does not affect the new loss and no beta metric is emitted.
- Both KL controls are disabled in the recipe and proven in script/smoke
  evidence.
- `algorithm.norm_adv_by_std_in_grpo=false` is preserved in scripts, tests, and
  smoke evidence.
- The full script family in Section 6 exists, including all four Meituan
  layers, and passes shell/static checks.
- Meituan layered launch compatibility is implemented completely. Missing
  Meituan platform files are a blocking failure for this goal.
- Required unit/trainer/script tests pass with exact commands recorded.
- Any intermediate GPU probe failures discovered during implementation are
  fixed, with rerun evidence recorded.
- A real GPU vLLM + FlashInfer smoke runs inside tmux and completes at least
  one actor update.
- Reviewer gates in Section 12 have `PASS`, or `WARN` with accepted limitation
  and follow-up in status.
- All `FAIL` findings are fixed or explicitly accepted by the user.
- Method docs, this goal, active plan index, and CLAUDE/AGENTS bridge are
  updated; if bridge sync is blocked, the exact blocker is recorded.
- The goal-local status file is current and sufficient for a new agent to resume
  without reading chat history.

## 15. Post-Completion Training Handoff

After every item in Section 14 is complete, the executor must start one real
training run for this method and supervise it for 30 completed training steps.
This is part of the goal contract, not an optional follow-up.

Launch and supervision requirements:

- Start the training run inside tmux.
- Record the exact tmux session name, launch command, log path, checkpoint
  path, and run id in the status file.
- Supervise until the log/metrics prove `training/global_step >= 30`.
- During the 30-step supervision window, monitor for at least:
  - Python exceptions, Ray worker failures, vLLM failures, reward-function
    failures, and hanging actor updates;
  - CUDA OOM or CPU/RAM pressure;
  - non-finite loss, non-finite ratio metrics, or obvious `actor/grad_norm`
    explosions;
  - unexpected `rollout_is_weights`, KL, beta, or non-`seq-mean-token-sum`
    behavior;
  - missing all-correct fallback metrics or missing sign-split ratio metrics.
- If 30 steps complete cleanly, update the status file with the observed
  metrics, leave the training running, detach from tmux, and end the session.
- If a bug, OOM, hang, non-finite metric, or method-contract violation appears
  before 30 supervised steps, do not end the session. Debug the failure, find
  the concrete cause, fix it, rerun the required tests/smoke as appropriate,
  relaunch training, and continue supervision until 30 clean steps complete.
- If the same blocker repeats and cannot be resolved without user input,
  record the exact evidence in the status file and ask the user only after the
  concrete failure mode has been identified.

The training may continue after the session ends. The session ends only after
the implementation contract is complete and the 30-step supervision requirement
has passed.

## 16. Open Decisions Before Implementation

These defaults are chosen for the first implementation and should only be
changed with an explicit note in the status file:

1. Advantage std normalization defaults to `false`.
2. Binary mask is kept and IS is added multiplicatively.
3. `rollout.calculate_log_probs` should default to `false` for the new script
   unless an existing runtime dependency requires it.
4. The new loss raises or fails fast if `rollout_is_weights` is passed in.
