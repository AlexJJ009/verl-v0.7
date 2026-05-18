# Goal: Dual-Submodel Rollout WDL-SFT First Implementation

- Status: ARCHIVED 2026-05-18 - implementation completed on `feature/on-policy-wdl-sft-dual-rollout`; 3A then failed as a method-level negative result
- Source plan: `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft.md`
- Target branch: `feature/on-policy-wdl-sft-dual-rollout`
- Target recipe: `recipe/on_policy_wdl_sft/dual_submodel_rollout/`
- Goal status file: `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft_status.md`
- Failure analysis: `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md`
- Last updated: 2026-05-18

> Archive note: this file is the original implementation contract. It is kept
> for traceability. The completed status and negative-result analysis are now
> the authoritative read points.

## 1. Objective

Implement a usable first version of Dual-Submodel Rollout WDL-SFT.

The goal is to decouple data generation from fused-logit training:

1. Generate rollout trajectories from `sub_model_0` and `sub_model_1` separately.
2. Select `sub_model_1` / model2 rollout data for training by default.
3. Keep the actor training forward/backward path on fused joint logits.
4. Keep both submodels trainable.
5. Verify the implementation with real GPU vLLM + FlashInfer smoke runs.

This goal is about infrastructure correctness and launchability. It does not
require proving performance against 1A, 2A, MiniRL, or other algorithms.

The rollout method is prompt-aligned: for each training step, start from one
prompt batch, then generate one response group per configured rollout source
from that same prompt batch. Do not draw separate prompt batches per source.

## 2. Non-Negotiable Boundaries

### In Scope

- Add a config-gated dual-rollout mode for joint models.
- Keep current fused rollout behavior as the default unless the new recipe opts in.
- Support rollout sources:
  - `fused`
  - `sub_model_0`
  - `sub_model_1`
- Generate both model1 and model2 rollout batches in the dual recipe.
- Select only model2 rollout data for default training.
- Preserve fused-logit actor training.
- Preserve `freeze_model1=False` behavior.
- Preserve `wdl_sft_is` as the default loss for 3A/3B.
- Preserve `WDL_SFT_BETA` as an external override.
- Run real GPU smoke tests through vLLM with `VLLM_ATTENTION_BACKEND=FLASHINFER`.
- Keep FSDP training on `attn_implementation=flash_attention_2`.

### Out of Scope For First Implementation

- Do not train on combined model1 + model2 rollout data.
- Do not select best responses across model1/model2 by reward.
- Do not add a new reward function.
- Do not change checkpoint format.
- Do not require full 300-step training as implementation acceptance.
- Do not require performance comparison against earlier runs as implementation acceptance.
- Do not make Meituan submission itself a blocker, but keep scripts compatible
  with the Meituan layered launch style.

## 3. Algorithm Defaults

First implementation defaults:

| Field | Default |
|---|---|
| rollout sources | `["sub_model_0", "sub_model_1"]` |
| selected training source | `sub_model_1` |
| train on selected only | `true` |
| training forward | fused joint logits |
| loss mode | `wdl_sft_is` |
| beta | 3A: `0.0`, 3B: `0.1` |
| lr | `5e-7` |
| rollout.n | `8` per source |
| rollout engine | vLLM |
| vLLM attention backend | FlashInfer |

`rollout.n=8` means 8 responses per prompt per rollout source. In dual mode,
model1 generates 8 responses and model2 generates 8 responses. The selected
model2 training batch still has 8 responses per prompt, matching the 1A data
budget for actor update.

## 4. Data-Flow Invariants

The first implementation must preserve this data flow:

```text
prompt_batch
  -> repeat the same prompts for each configured rollout source
  -> set rollout source = sub_model_0; generate source-specific responses
  -> set rollout source = sub_model_1; generate source-specific responses
  -> score each source batch for source-specific metrics
  -> select sub_model_1 batch for training
  -> compute old_log_probs for the selected batch under fused training policy
  -> apply WDL reward-label override to the selected batch
  -> actor update with fused joint logits
```

Hard boundaries:

- Non-selected source batches are diagnostics only.
- `token_level_scores`, `advantages`, `old_log_probs`, `rollout_log_probs`,
  `response_mask`, and actor-update tensors must come from the selected source
  only.
- Model1 rewards or metadata must not be mixed into the selected model2
  training batch.
- Prompt ordering must remain aligned enough to compare source-level metrics
  for the same original prompt batch.
- After rollout generation, the rollout source must be restored to a safe
  default (`fused`) unless the next operation explicitly sets a source.

## 5. IS / Mask Semantics

Keep the `wdl_sft_is` ratio binary mask between actor `log_prob` and
`old_log_prob`. This remains a training-time stability mechanism for updates
over the selected batch.

For dual rollout, `old_log_prob` must be computed under the fused training
policy for the selected batch, not under the model2 rollout policy. The ratio
mask is a local training-time trust-region proxy for fused-policy updates. It
is not an importance-sampling correction from model2 rollout policy to fused
training policy.

Default behavior for `rollout_is_weights`:

- Do not multiply `rollout_is_weights` into the loss by default in dual rollout.
- Keep rollout/FSDP log-prob correction diagnostics if they are cheap and do not
  complicate the first implementation.
- Do not add a performance ablation for this switch in the first implementation.

Hard acceptance for the first implementation:

- The default dual-rollout training path must not pass or consume
  `rollout_is_weights` as a multiplicative factor inside
  `compute_policy_loss_wdl_sft_is`.
- It is acceptable either to disable generation of `rollout_is_weights` for the
  dual recipe or to generate it for diagnostics, as long as the loss does not
  consume it by default.
- If diagnostics are retained, their names must make clear that they are
  diagnostics, not loss weights.

Reason: in fused-rollout v2, `rollout_is_weights` mostly corrected vLLM-vs-FSDP
numerical mismatch for the same intended fused policy. In dual rollout, selected
data comes from model2 while training uses fused logits. Multiplying
`rollout_is_weights` into the loss would become a real cross-policy reweighting
from model2 data toward fused policy, which conflicts with the first
implementation's design: model2 is deliberately the data policy; fused logits
are deliberately the training-time gradient amplifier.

## 6. Config Surface

Preferred config location:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      joint_rollout_sources: ["sub_model_0", "sub_model_1"]
      joint_rollout_select: "sub_model_1"
      joint_rollout_train_on_selected_only: true
```

Acceptance:

- The base config must remain backward compatible and continue to use fused
  rollout unless a recipe explicitly opts into dual rollout.
- The new recipe must opt in explicitly.
- Prefer `actor_rollout_ref.rollout.custom.*` to avoid unnecessary global
  config-schema churn.
- If implementation shows that `custom.*` is awkward or inconsistent with
  local config patterns, formal rollout fields are allowed, but the reason must
  be documented in the implementation notes and backward compatibility must be
  preserved.

Config validation requirements:

- Valid sources are exactly `fused`, `sub_model_0`, and `sub_model_1`.
- `joint_rollout_sources` must be non-empty.
- `joint_rollout_select` must be one of `joint_rollout_sources`.
- For the first implementation, `joint_rollout_train_on_selected_only=false`
  is unsupported and must fail fast with a clear error.
- If no dual-rollout config is provided, behavior must remain the existing
  single fused-rollout path.
- Invalid source names or invalid combinations must fail before launching a
  long-running training job.

## 7. Script And Launch Requirements

Create:

```text
recipe/on_policy_wdl_sft/dual_submodel_rollout/
├── README.md
├── _common_dual_rollout.sh
├── run_3a_model2_rollout_beta0.sh
└── run_3b_model2_rollout_beta01.sh
```

Script rules:

- `run_*.sh` files are thin wrappers only: export experiment-specific knobs,
  then source `_common_dual_rollout.sh`.
- `_common_dual_rollout.sh` owns environment setup, checkpoint/resume handling,
  Hydra launch, and shared defaults.
- Every path must use default-local-overridable-everything style:
  `${VAR:-/data-1/...}`.
- External callers must be able to override parent paths for repo, data, model,
  checkpoints, logs, caches, temporary dirs, and validation outputs.
- Keep project-relative paths usable for recipe-local files, especially reward
  function and validation/log output folders.
- Do not hard-code dolphinfs paths in `run_*.sh` or `_common_dual_rollout.sh`.
- Keep compatibility with the Meituan four-layer launch design. Meituan
  `env.sh` / `jupyter.sh` and platform shim can be added later without changing
  per-run script semantics.
- The launch must force or default to `VLLM_ATTENTION_BACKEND=FLASHINFER`.
- The model override must keep FSDP training on `flash_attention_2`.

## 8. Implementation Tasks

1. Inspect current dirty worktree and checkpoint/branch only with user approval.
2. Create implementation branch `feature/on-policy-wdl-sft-dual-rollout`.
3. Create or update the goal-local status file.
4. Add config-gated rollout source selection.
5. Add HF joint model source switching.
6. Add vLLM joint model source switching.
7. Add worker/RPC path to switch rollout source across vLLM workers.
8. Update trainer rollout flow:
   - generate once per configured rollout source,
   - reward each source batch,
   - record per-source metrics,
   - select `sub_model_1` batch for training,
   - continue old-log-prob, rollout correction diagnostics, advantage labels,
     and actor update on selected batch only.
9. Preserve `apply_wdl_sft_reward_label_advantages(...)` behavior for both
   `wdl_sft` and `wdl_sft_is`.
10. Add dual rollout metrics.
11. Add recipe folder and scripts.
12. Add/update targeted tests.
13. Run CPU/unit tests.
14. Commit coherent implementation checkpoints on the new branch.
15. Run real GPU smoke for 3A.
16. Run real GPU smoke for 3B as the last validation task.
17. Update the goal-local status file before any compact/resume boundary and
    after each meaningful commit.
18. Update the source plan with implementation notes and changed decisions.

## 9. Branch, Commit, And Status Discipline

All implementation work should happen on
`feature/on-policy-wdl-sft-dual-rollout` unless the user explicitly chooses a
different branch.

Commit expectations:

- Commit promptly after coherent milestones instead of leaving the whole
  implementation uncommitted until the end.
- Suggested commit boundaries:
  - config + source-switching model changes,
  - trainer dual-rollout flow,
  - recipe/scripts,
  - tests,
  - smoke/debug fixes,
  - documentation/status updates.
- Before committing, inspect the working tree and include only intended files.
- Do not commit unrelated dirty files that existed before this goal.
- If user-owned dirty files block a clean commit boundary, record the situation
  in the status file and ask before touching or staging them.

Maintain the goal-local status file:

```text
docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft_status.md
```

This status file is scoped to this one development goal, not to the entire
project. It is the durable handoff point for context compaction or a new agent
session.

The status file must include:

- current branch and latest relevant commit;
- current task/milestone;
- completed milestones;
- files changed intentionally;
- tests run and exact results;
- GPU smoke status for 3A/3B;
- open blockers or user decisions needed;
- next concrete action.

Read/write rules:

- After creating the implementation branch, create the status file before major
  code edits.
- At the start of any resumed session or after context compaction, read this
  status file before continuing implementation.
- Before any expected context compaction, long pause, or handoff, update the
  status file with the live state.
- After every meaningful commit, update the status file with the commit hash,
  summary, tests, and next action.
- Before final completion, update the status file to point at the final commits
  and verification results.

## 10. Reviewer Gates

Every implementation subtask must have an independent reviewer subagent gate
before the main agent treats that subtask as complete.

Reviewer requirements:

- The reviewer subagent must receive:
  - the current goal file,
  - the goal-local status file,
  - the relevant diff or commit hash,
  - the specific subtask acceptance criteria,
  - test or smoke outputs available for that subtask.
- The reviewer must respond with a clear verdict:
  - `PASS`: subtask meets its acceptance criteria.
  - `WARN`: subtask is usable but has known limitations or follow-up items.
  - `FAIL`: subtask does not meet acceptance criteria or has blocking issues.
- The reviewer feedback must be given back to the main agent.
- The main agent must address every `FAIL` before moving on, unless the user
  explicitly accepts the failed state.
- For `WARN`, the main agent must either fix the issue or record the accepted
  limitation and follow-up in the status file.
- Reviewer feedback, verdict, and main-agent response must be summarized in the
  goal-local status file.

Minimum reviewer-gated subtasks:

1. Config surface and validation.
2. HF joint model source switching.
3. vLLM joint model source switching and worker/RPC path.
4. Trainer dual-rollout flow and selected-only training.
5. Metrics and diagnostics.
6. Recipe/scripts and portable launch behavior.
7. Unit tests and backward-compatibility tests.
8. 3A GPU smoke.
9. 3B GPU smoke.
10. Final documentation/status update.

The reviewer is a quality gate, not the implementation owner. The main agent
remains responsible for integrating feedback, fixing issues, committing changes,
and updating the status file.

Per-subtask reviewer acceptance criteria:

### 10.1 Config Surface And Validation

Reviewer must verify:

- Dual rollout is opt-in only; no config means existing fused rollout behavior.
- Valid sources are exactly `fused`, `sub_model_0`, `sub_model_1`.
- Empty `joint_rollout_sources` fails fast.
- Unknown source fails fast.
- `joint_rollout_select` not present in `joint_rollout_sources` fails fast.
- `joint_rollout_train_on_selected_only=false` fails fast for this first
  implementation.
- New recipe explicitly sets the dual rollout config.

PASS requires code and tests or direct evidence covering all bullets.
FAIL if any invalid config can silently launch a long-running job, or if the
default fused path is changed.

### 10.2 HF Joint Model Source Switching

Reviewer must verify:

- `fused` mode still returns weighted fused logits with the existing lambda
  semantics.
- `sub_model_0` mode uses model1 logits and does not depend on model2 logits.
- `sub_model_1` mode uses model2 logits and does not depend on model1 logits.
- Switching source does not alter training defaults unless explicitly set.
- Source state is restored or safely scoped so later operations do not inherit a
  stale source accidentally.

PASS requires targeted tests or an equivalent deterministic check. FAIL if
source switching can affect ordinary fused training without opt-in.

### 10.3 vLLM Joint Model Source Switching And Worker/RPC Path

Reviewer must verify:

- vLLM joint model implements the same source semantics as the HF model.
- Source switching reaches every vLLM worker used for generation.
- Source switching happens before generation for each source.
- The implementation works with vLLM + FlashInfer assumptions and does not
  require a fallback attention backend.
- The previous model2-only validation/eval behavior remains intact.

PASS requires tests, mocks, or smoke evidence sufficient to prove worker source
selection is not local-only on the driver. FAIL if only the driver-side object
is switched while worker model instances may remain fused.

### 10.4 Trainer Dual-Rollout Flow And Selected-Only Training

Reviewer must verify:

- One prompt batch is reused across all configured rollout sources.
- Each source generates its own response group.
- Non-selected source batches are diagnostics only.
- `token_level_scores`, `advantages`, `old_log_probs`, `rollout_log_probs`,
  `response_mask`, and actor-update tensors are from the selected source only.
- `old_log_probs` for selected data are computed under fused training policy.
- Actor update uses fused joint logits and keeps both submodels trainable.
- `apply_wdl_sft_reward_label_advantages(...)` still applies to `wdl_sft` and
  `wdl_sft_is`.
- Existing no-dual-config trainer path remains fused single-rollout.

PASS requires selector tests plus code inspection of tensor provenance. FAIL if
model1 reward/log-prob/metadata can enter selected model2 training tensors.

### 10.5 Metrics And Diagnostics

Reviewer must verify:

- Required metrics exist:
  - `dual_rollout/model1_correct_ratio`
  - `dual_rollout/model2_correct_ratio`
  - `dual_rollout/model1_response_len_mean`
  - `dual_rollout/model2_response_len_mean`
  - `dual_rollout/selected_source`
- There is source-switch or response-count evidence for each generated source.
- Metrics are source-specific and do not mix model1/model2 values.
- Missing optional diagnostics do not block PASS if required metrics are present.
- If `rollout_is_weights` diagnostics exist, their names do not imply they are
  active loss weights.

PASS requires logs, tests, or smoke output showing required metrics or their
file-logger equivalents. WARN is acceptable for rough metric naming only if
the selected source and per-source generation are still provable. FAIL if the
smoke cannot prove both sources actually generated.

### 10.6 Recipe/Scripts And Portable Launch Behavior

Reviewer must verify:

- Required files exist in `recipe/on_policy_wdl_sft/dual_submodel_rollout/`.
- `run_3a_model2_rollout_beta0.sh` sets beta `0.0`.
- `run_3b_model2_rollout_beta01.sh` sets beta `0.1`.
- `run_*.sh` files are thin wrappers that export knobs and source
  `_common_dual_rollout.sh`.
- `_common_dual_rollout.sh` owns shared launch logic.
- Paths use `${VAR:-local-default}` style and do not hard-code dolphinfs.
- External parent paths for data/model/checkpoints/logs/cache/temp can be
  overridden.
- Launch defaults to or forces `VLLM_ATTENTION_BACKEND=FLASHINFER`.
- FSDP model override keeps `attn_implementation=flash_attention_2`.
- Smoke command can disable pre-train validation via env override.

PASS requires shell inspection and, where practical, `bash -n` or equivalent
syntax checks. FAIL if local `/data-1` paths are unconditional or dolphinfs
paths appear outside Meituan adapters.

### 10.7 Unit Tests And Backward-Compatibility Tests

Reviewer must verify:

- Tests cover config validation failures.
- Tests cover HF source switching.
- Tests cover vLLM source switching or a justified mock equivalent.
- Tests cover selected-only trainer behavior.
- Tests cover no-config backward compatibility.
- Tests cover WDL reward-label override for both `wdl_sft` and `wdl_sft_is`.
- Tests cover beta nonzero reverse-SFT signal at unit level.
- The reported test command and result are exact and reproducible.

PASS requires the relevant tests to pass. WARN may be used only when a vLLM
full-construction test is replaced by a documented mock because real vLLM is
too heavy for CPU tests. FAIL if selected-only behavior or backward
compatibility lacks coverage.

### 10.8 3A GPU Smoke

Reviewer must verify:

- Smoke ran on real GPU, not CPU-only or dry-run.
- vLLM was used.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` was active.
- Pre-train validation was disabled unless explicitly being tested.
- Both `sub_model_0` and `sub_model_1` generated responses from the same prompt
  batch.
- `selected_source=sub_model_1` is visible in logs or metrics.
- At least one actor update completed.
- No shape mismatch occurred in `old_log_probs`, `rollout_log_probs`,
  `advantages`, or `response_mask`.
- Checkpoint/update-weight path did not crash.

PASS requires smoke command, run location/session, and relevant log excerpts or
metric paths. FAIL if the run succeeds but cannot prove both source paths were
used.

### 10.9 3B GPU Smoke

Reviewer must verify:

- 3B smoke ran after 3A smoke and unit tests.
- 3B used `run_3b_model2_rollout_beta01.sh` or an equivalent command proving
  `WDL_SFT_BETA=0.1`.
- Real GPU, vLLM, and FlashInfer were active.
- Training started and completed the configured smoke steps.
- Detailed metric quality is not a blocking criterion for 3B smoke.

PASS requires launch/completion evidence. FAIL if beta is not proven to be
`0.1`, or if the run bypasses the real dual-rollout launch path.

### 10.10 Final Documentation/Status Update

Reviewer must verify:

- Source plan records implementation notes and deviations from the goal.
- Goal-local status file records latest commit, tests, 3A/3B smoke status,
  reviewer verdicts, and next action.
- All `FAIL` reviewer findings are fixed or explicitly accepted by the user.
- All `WARN` findings are either fixed or recorded as accepted limitations.
- Final changed files are intentional and no unrelated dirty files are included
  in commits.

PASS requires the status file and plan to be current. FAIL if the final state
cannot be resumed from the status file after context compaction.

## 11. Required Metrics

At minimum, log:

- `dual_rollout/model1_correct_ratio`
- `dual_rollout/model2_correct_ratio`
- `dual_rollout/model1_response_len_mean`
- `dual_rollout/model2_response_len_mean`
- `dual_rollout/selected_source`
- one source-switch or batch-count evidence log/metric per rollout source

Optional if available without heavy plumbing:

- per-source extraction failure rate
- per-source truncation rate
- rollout/FSDP log-prob correction diagnostics

Metrics quality is not the first smoke gate. If metrics names or exact values
need debugging after the infra path runs, record that as follow-up work rather
than blocking the initial training-start smoke.

## 12. Tests

Required CPU/unit coverage:

- HF joint source switching:
  - `fused` equals weighted fused logits,
  - `sub_model_0` ignores model2 logits,
  - `sub_model_1` ignores model1 logits.
- vLLM joint source mode, with mocks if full vLLM construction is too heavy.
- Trainer selector behavior:
  - when `joint_rollout_select=sub_model_1`, only model2 batch reaches actor
    update.
- Backward compatibility:
  - when dual-rollout config is absent, the trainer uses the existing fused
    rollout path and does not generate once per source.
- Config validation:
  - invalid source,
  - empty source list,
  - selected source absent from source list,
  - unsupported `joint_rollout_train_on_selected_only=false`.
- WDL reward-label regression:
  - `wdl_sft` and `wdl_sft_is` both receive raw reward labels in `advantages`.
- Beta path:
  - unit-level coverage should show all-incorrect groups can produce reverse
    SFT signal when beta is nonzero.

The beta reverse-SFT signal check belongs in unit tests, not GPU smoke, because
the smoke data may not reliably construct all-incorrect groups.

## 13. GPU Smoke Acceptance

Smoke tests must use real GPU execution, vLLM, and FlashInfer.

Recommended 3A command shape:

```bash
TOTAL_TRAINING_STEPS=1 \
TRAIN_PROMPT_BSZ=2 \
TRAIN_PROMPT_MINI_BSZ=1 \
ROLLOUT_AGENT_NUM_WORKERS=1 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=1 \
bash recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh
```

If 1 step does not exercise enough of the checkpoint/eval/update-weight cycle,
raise to 3 steps with matching small batch settings.

Smoke should focus on the training rollout/update path. Pre-train validation
should be disabled by default for smoke unless the implementer is explicitly
testing validation behavior.

3A smoke acceptance:

- launches under vLLM with `VLLM_ATTENTION_BACKEND=FLASHINFER`;
- executes model1 rollout path;
- executes model2 rollout path;
- selects model2 for actor update;
- completes at least one actor update;
- logs or metrics prove both sources generated responses from the same prompt
  batch, for example source-switch markers plus per-source response counts;
- logs or metrics prove `selected_source=sub_model_1`;
- no shape mismatch in `old_log_probs`, `rollout_log_probs`, `advantages`, or
  `response_mask`;
- checkpoint/update-weight path does not crash.

3B smoke acceptance:

- run after 3A smoke and after unit tests;
- same launch path as 3A, but with `WDL_SFT_BETA=0.1` via
  `run_3b_model2_rollout_beta01.sh`;
- acceptance is limited to whether the training path starts and completes the
  configured smoke steps. Detailed statistics can be debugged after the infra
  path exists.

Long-running smoke or full training must be launched inside tmux.

## 14. Done Definition

This goal is done when:

- dual rollout can be enabled only by explicit recipe/config opt-in;
- old fused rollout behavior remains usable by default and is covered by a
  backward-compatibility test;
- model1 and model2 rollout source switching works;
- selected-only model2 training works;
- 3A and 3B scripts exist and follow the portable layered-launch rules;
- invalid dual-rollout config fails fast;
- coherent implementation milestones have been committed on
  `feature/on-policy-wdl-sft-dual-rollout`;
- the goal-local status file is current and includes latest commit/test/smoke
  state;
- every minimum reviewer-gated subtask has reviewer feedback recorded in the
  goal-local status file, with all `FAIL` findings fixed or explicitly accepted
  by the user;
- required unit tests pass;
- real GPU 3A smoke passes with vLLM FlashInfer;
- real GPU 3B smoke passes after 3A;
- source plan is updated with implementation notes, final config names, and any
  deviations from this goal.

## 15. Deferred Research Questions

These are intentionally not implementation blockers:

- Whether 3A beats 1A, 2A-SFT, 2Z-SFT, or MiniRL.
- Whether model1 diagnostics should drive a later selector.
- Whether to re-enable `rollout_is_weights` loss multiplication as an ablation.
- Whether to train on both sources.
- Whether the algorithm needs a public name beyond "dual-submodel rollout
  WDL-SFT".
