# Smoke Learning-Signal Policy

This policy applies to RL training smoke tests in this branch, especially
math-reasoning experiments whose reward depends on complete boxed answers.

## Principle

A smoke test for an algorithmic training method must preserve the conditions
needed to produce a learning signal. A run that only proves the code path can
execute is a plumbing smoke, not an algorithm acceptance smoke.

For math RL in this worktree, too-small response budgets can deterministically
erase the learning signal:

```text
MAX_RESPONSE_LENGTH too small -> response truncates before EOS
truncated response -> reward = -1
all responses in a prompt group reward -1
group advantage = reward - group_mean = 0
policy loss = 0 and actor grad_norm = 0
```

Therefore, a smoke that sets `MAX_RESPONSE_LENGTH=256` for this method is not
valid evidence that the algorithm can train. It is only valid as a low-cost
plumbing check for rollout, tensor provenance, logging, and checkpoint writing.

## Valid 4A Pre-Training Smoke

For `dual_model2_group_adv_is`, the pre-training smoke should use the same
learning-relevant defaults as the real experiment launcher:

```text
LOSS_MODE=dual_model2_group_adv_is
JOINT_ROLLOUT_SOURCES=[sub_model_1]
JOINT_ROLLOUT_SELECT=sub_model_1
TRAIN_FILE=/data-1/dataset/math/train_rl_format.parquet
LOSS_AGG_MODE=seq-mean-token-sum
GAMMA_POS_SFT=1.0
TIS_THRESHOLD=5.0
ROLLOUT_IS=null
MAX_RESPONSE_LENGTH=4096
N_RESP_PER_PROMPT=8
actor_rollout_ref.rollout.val_kwargs.n=3
```

Safe smoke overrides are limited to operational scope:

- `TOTAL_TRAINING_STEPS`
- run name / log directory / checkpoint directory
- `SAVE_FREQ`, `TEST_FREQ`, `VAL_BEFORE_TRAIN`
- storage guard values such as `MIN_FREE_GB_FOR_CKPT`

Do not lower `MAX_RESPONSE_LENGTH`, `N_RESP_PER_PROMPT`, or change the reward
function to make the smoke cheaper unless the run is explicitly labeled
`plumbing-only`.

## Acceptance Checks

A learning-signal smoke must check both stability and non-degeneracy.

Stability checks:

- required ratio, TIS, loss, and gradient metrics are present and finite;
- `tis_clip_fraction`, `clipfrac_positive`, and `clipfrac_negative` are in
  `[0, 1]`;
- `dual_rollout/selected_source` is model2 / `sub_model_1`;
- `actor/grad_norm` does not show immediate 3A-style explosion.

Learning-signal checks:

- not every logged step has `all_incorrect_group_fraction=1.0`;
- at least one logged step has either `mixed_group_fraction > 0` or
  `all_correct_fallback_group_fraction > 0`;
- at least one logged step has nonzero group advantage evidence such as
  `critic/advantages/max > 0`;
- at least one logged step has finite `actor/grad_norm > 0`.

If a production-context smoke still produces all-incorrect groups, do not mark
the algorithm as failed. Mark the run as `no_learning_signal_evidence`, record
the source metrics, and run a larger gate or a known-easy training slice before
launching a full experiment.

## Recording Rule

When recording smoke results in docs or the experiment registry, distinguish:

- `plumbing-only smoke`: proves code path execution only;
- `learning-signal smoke`: proves the code path can produce nonzero policy
  gradient under production-like generation settings.

