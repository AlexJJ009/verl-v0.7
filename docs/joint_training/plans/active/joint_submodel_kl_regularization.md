# Goal: Per-Submodel KL Regularization For Joint WDL-SFT

- Status: ACTIVE IMPLEMENTATION PLAN - reviewed READY; user-confirmed long-run acceptance
- Target branch: `feature/on-policy-wdl-sft`
- Target training family: joint Stage2 / code-task On-Policy WDL-SFT
- Target implementation areas:
  - `verl/models/joint_model/`
  - `verl/workers/actor/dp_actor.py`
  - `verl/workers/fsdp_workers.py`
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/trainer/ppo/core_algos.py`
  - `verl/workers/config/actor.py`
  - `verl/trainer/config/actor/actor.yaml`
  - `recipe/on_policy_wdl_sft/code_task/`
- Main local references:
  - Existing KL loss path: `verl/workers/actor/dp_actor.py`
  - Existing KL functions: `verl/trainer/ppo/core_algos.py`
  - Reference-policy wiring: `verl/trainer/main_ppo.py`, `verl/trainer/ppo/utils.py`
  - Joint model: `verl/models/joint_model/modeling_joint_qwen3.py`
  - Code-task Stage2 launchers: `recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh`
  - Meituan portability guide: `docs/joint_training/guides/meituan_platform.md`

## 1. Objective

Implement independent KL regularization for the two submodels inside the joint
WDL-SFT actor:

```text
L_total = L_wdl + c1 * KL(model1_current || model1_ref)
                + c2 * KL(model2_current || model2_ref)
```

The feature must be configurable per submodel. Model1 KL can be enabled without
Model2 KL, Model2 KL can be enabled without Model1 KL, both can be enabled, and
both can be disabled. When both are disabled, training behavior and loss values
must match the current code path.

This goal is motivated by the code-task Stage2 instability observed after the
P40/P60 handoff experiments: short Stage2 can improve online validation, while
longer two-model training drifts in format/length/reward quality. The KL terms
are intended as stability constraints, not as a replacement for WDL-SFT reward
selection.

## 2. Current Reality

Verl already has a single actor KL loss path:

```text
current fused actor log_prob -> ref_log_prob -> kl_penalty(...)
```

That path is controlled by `actor_rollout_ref.actor.use_kl_loss`,
`kl_loss_coef`, and `kl_loss_type`. It is insufficient for this goal because it
only carries one `ref_log_prob`, and the current joint model forward returns only
the fused logits/log-probs used by WDL-SFT. The implementation must add
per-submodel current log-probs and per-submodel reference log-probs instead of
assuming the existing one-reference KL is enough.

The existing KL type names must be reused:

| User shorthand | Existing local name |
| --- | --- |
| K1 | `kl` |
| K2 | `mse` |
| K3 | `low_var_kl` |

Other existing local options, such as `abs` and `full`, may remain accepted if
they are already accepted by `kl_penalty`.

## 3. Non-Negotiable Boundaries

### In Scope

- Add a submodel KL configuration block under actor config.
- Extend reference-policy lifecycle detection so submodel KL starts a reference
  path even when legacy `actor.use_kl_loss=false`.
- Add per-submodel current log-prob computation for joint models.
- Add per-submodel reference log-prob computation against frozen reference
  weights.
- Add KL penalties to actor loss after the WDL policy loss is computed.
- Support independent enable/disable, coefficient, and KL type for model1 and
  model2.
- Log metrics for each enabled submodel KL and for disabled/off states.
- Add unit tests, integration/plumbing tests, and one real small GPU smoke.
- Run the required 5-step local training-smoke matrix for all KL enablement modes
  before declaring the goal complete.
- Keep the code-task queue/monitor conventions and Meituan portability rules for
  any new runnable scripts.

### Out Of Scope For First Implementation

- Do not change WDL reward semantics, reward extraction, code scorer behavior, or
  positive/negative sample selection.
- Do not make KL a reward penalty in `token_level_rewards`; this goal is actor
  loss regularization only.
- Do not add adaptive KL controllers in the first pass unless the implementation
  naturally reuses the existing fixed controller without extra infra. Fixed
  coefficients are sufficient for acceptance.
- Do not run a full formal experiment queue as part of implementation acceptance.
  The required 5-step smoke matrix is to prove the infra path across KL modes, not
  to claim final benchmark improvement.

## 4. Proposed Configuration Contract

Add a nested actor config such as:

```yaml
actor_rollout_ref:
  actor:
    submodel_kl:
      enabled: false
      model1:
        enabled: false
        coef: 0.0
        kl_type: low_var_kl
        ref_path: null
      model2:
        enabled: false
        coef: 0.0
        kl_type: low_var_kl
        ref_path: null
```

Rules:

1. `enabled=false` at the top level disables all submodel KL and should avoid
   extra reference computation.
2. `model1.enabled=true` requires a valid model1 reference source.
3. `model2.enabled=true` requires a valid model2 reference source.
4. `coef=0.0` is allowed but should be treated as off for compute unless a test
   intentionally forces metric-only mode.
5. `ref_path` is explicitly user-configurable. If `ref_path=null`, the default
   reference is the model used at Stage2 start for that submodel:
   - model1 ref defaults to the Stage2 initial weak/base model;
   - model2 ref defaults to the Stage1 handoff Model2 used to initialize Stage2.
   The implementation must log the resolved path and fail fast if the source is
   ambiguous.
6. Requested reference tokenizer/config must be compatible with the active joint
   model. Incompatibility is a launch-time error, not a silent fallback.

## 4.1 User-Confirmed Decisions

Confirmed on 2026-07-01:

1. First implementation uses fixed KL coefficients only. Adaptive KL is deferred.
2. References are user-configurable; when unspecified, each submodel defaults to
   its own Stage2-start initialization source.
3. Required local training smoke uses `TOTAL_TRAINING_STEPS=5`, not 2.
4. The implementation goal must validate every KL enablement mode: both off,
   model1-only, model2-only, and both-on. Any one failing mode means the goal is
   incomplete.
5. The implementation goal must validate every required KL type used by Verl's
   local WDL-SFT KL path: `kl` (K1), `mse` (K2), and `low_var_kl` (K3). The
   real smoke matrix therefore contains one both-off baseline plus
   model1-only/model2-only/both-on runs for each of these three KL types.
6. The goal may be split into the three serial sub-goals in Section 5.1.
7. Long-running verification may take 7-10 hours and should use local GPU, disk,
   existing model weights, tmux, and monitor scripts.

## 5. Technical Milestones

Milestones are hard ordered. Do not advance to a later milestone until the
required acceptance criteria for the current milestone pass.

### M1. Design And Config Schema

- Add dataclass/YAML config fields for `submodel_kl`.
- Add config validation helpers that decide whether a submodel reference path is
  required.
- Extend `need_reference_policy(config)` or an equivalent worker-setup predicate
  so enabled submodel KL creates a reference path even when the legacy
  `actor.use_kl_loss` and `algorithm.use_kl_in_reward` are both false.
- Preserve current config behavior when the block is absent.
- Add unit tests that use config-only fixtures, not real HF checkpoints.

### M2. Current Per-Submodel Log-Prob Plumbing

- Extend the joint model/actor path so a forward pass can expose:
  - fused log-probs, unchanged, for WDL loss;
  - `model1_log_probs`, current submodel1 token log-probs;
  - `model2_log_probs`, current submodel2 token log-probs.
- Preserve the existing `CausalLMOutputWithPast(logits=fused_logits, ...)`
  contract for all callers that do not opt into submodel KL.
- The default actor update path must not compute or retain these tensors when
  submodel KL is fully disabled.

### M3. Frozen Reference Log-Prob Plumbing

- Add a reference path that can compute:
  - `model1_ref_log_probs` when model1 KL is enabled;
  - `model2_ref_log_probs` when model2 KL is enabled.
- Prefer reusing existing reference worker lifecycle when feasible, but do not
  overload the single `ref_log_prob` tensor in a way that loses submodel identity.
- Preserve the legacy `ref_log_prob` tensor for existing KL reward/loss paths.
- Update actor `select_keys` so enabled submodel KL requests only the required
  `model*_ref_log_probs`, while disabled submodel KL requests none of them.
- Record reference provenance in logs/config output.

### M4. KL Loss Integration And Metrics

- Use existing `kl_penalty(logprob, ref_logprob, kl_penalty=...)` for each
  enabled submodel.
- Aggregate with the actor's configured `loss_agg_mode` and `response_mask`.
- Add the result to `policy_loss` after WDL policy loss and entropy handling.
- Metrics must include at least:
  - `actor/submodel_kl/model1_loss`
  - `actor/submodel_kl/model1_coef`
  - `actor/submodel_kl/model1_type_code`
  - `actor/submodel_kl/model2_loss`
  - `actor/submodel_kl/model2_coef`
  - `actor/submodel_kl/model2_type_code`
  - `actor/submodel_kl/total_loss`

The human-readable KL type string must still be printed in launch/dry-run
configuration output. The training metric uses a numeric code because Verl's
metric aggregation path expects numeric values.

### M5. Tests And Real Smoke

- Add deterministic unit tests for KL math, off-equivalence, and config
  validation.
- Add a lightweight joint actor plumbing test that verifies tensor names, shapes,
  masks, and no-extra-compute off behavior.
- Run a short real training smoke in Docker on local GPU resources with rollout
  and actor update both exercised.

### M6. Runnable Script And Documentation Updates

- Add a code-task smoke wrapper or environment override for submodel KL that is
  compatible with the existing code-task Stage2 runner.
- Preserve the existing Stage2 source/provenance guards in dry-run and real-run
  paths. A submodel-KL wrapper must not bypass `STAGE1_RUN_PREFIX`,
  `EXPECTED_STAGE1_BETA`, merged Model2 provenance, or train-shard checks.
- Add Meituan routing for any new runnable script.
- Update the training script index if a runnable script is added.
- If a queue/monitor wrapper is added, keep it thin and delegate shared monitor
  behavior to `scripts/training_queue_monitor.sh`.

## 5.1 Serial Goal Decomposition

This implementation should be executed as three serial sub-goals rather than one
large undifferentiated edit:

1. **Schema and lifecycle goal**: config dataclasses/YAML, reference-policy
   predicate, validation helpers, and config/schema tests.
2. **Log-prob and loss goal**: current per-submodel log-probs, reference
   per-submodel log-probs, KL loss addition, metrics, and focused unit/integration
   tests.
3. **Runner and smoke goal**: code-task wrapper, Meituan route, docs/index
   updates, dry-run guard verification, monitor/queue support, and the local
   5-step GPU smoke matrix.

Do not start sub-goal 2 until sub-goal 1 tests pass. Do not start sub-goal 3
until sub-goal 2 tests pass.

## 6. Acceptance Criteria

AC-01 - Config Defaults Preserve Current Behavior

- Given the repo checkout before enabling submodel KL,
- When `pytest tests/on_policy_wdl_sft/test_submodel_kl_config.py -q` is run,
- Then the default parsed actor config has submodel KL disabled, no reference
  requirement is triggered by submodel KL, and all existing KL config fields keep
  their previous defaults.

Expected evidence: pytest passes and the test asserts `enabled == false` for
both submodels.

AC-01b - Enabled Submodel KL Starts Reference Lifecycle

- Given `actor_rollout_ref.actor.submodel_kl.enabled=true` and at least one
  submodel has `enabled=true` and `coef > 0`,
- When the reference-policy requirement predicate is evaluated and trainer worker
  setup is initialized with a config-only fixture,
- Then a reference policy path is required even if
  `actor_rollout_ref.actor.use_kl_loss=false` and
  `algorithm.use_kl_in_reward=false`.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_submodel_kl_config.py::test_submodel_kl_requires_reference_lifecycle -q
```

AC-02 - Existing WDL Loss Is Numerically Unchanged When Submodel KL Is Off

- Given fixed fake tensors for `old_log_prob`, fused `log_prob`, `advantages`,
  and `response_mask`,
- When the WDL loss is computed once with no `submodel_kl` block and once with
  `submodel_kl.enabled=false`,
- Then total loss and all existing WDL metrics match within `1e-7`.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_submodel_kl_loss.py::test_submodel_kl_off_matches_existing_wdl -q
```

AC-03 - K1/K2/K3 Per-Submodel KL Matches Existing Verl KL Functions

- Given deterministic current/ref token log-probs and a response mask,
- When model1 KL is computed with `kl`, model2 KL is computed with `mse`, and
  model2 KL is computed with `low_var_kl`,
- Then each result equals `verl.trainer.ppo.core_algos.kl_penalty` aggregated by
  the same actor loss aggregator.

This AC intentionally tests only `kl`, `mse`, and `low_var_kl`. The existing
`full` option is not part of the first required matrix because it has different
shape/semantics in Verl.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_submodel_kl_loss.py::test_submodel_kl_uses_existing_kl_penalty_types -q
```

AC-04 - Independent Enable/Disable Matrix Works

- Given a fake actor micro-batch containing current and reference submodel
  log-probs,
- When four configs are tested: both off, model1 only, model2 only, both on,
- Then only the enabled submodel terms contribute to `actor/submodel_kl/total_loss`
  and only enabled submodel reference tensors are required.
- Then the both-off case does not add `model1_ref_log_probs`,
  `model2_ref_log_probs`, `model1_log_probs`, or `model2_log_probs` to the actor
  selected batch keys.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_submodel_kl_loss.py::test_independent_enable_disable_matrix -q
```

AC-05 - Joint Model Can Expose Current Submodel Log-Prob Tensors

- Given a tiny joint Qwen config or monkeypatched two-submodel joint module,
- When the actor forward is requested with submodel KL enabled,
- Then the returned outputs contain fused `log_probs`, `model1_log_probs`, and
  `model2_log_probs` with identical `(batch, response_length)` shapes.
- Then the same forward without submodel KL still returns the legacy fused
  `log_probs` only and preserves the fused logits API for non-KL callers.

Expected command:

```bash
pytest tests/joint_training/test_joint_submodel_logprob_plumbing.py -q
```

AC-06 - Reference Plumbing Preserves Submodel Identity

- Given model1-only KL is enabled in a fake reference worker fixture,
- When reference log-probs are computed for a batch,
- Then the batch contains `model1_ref_log_probs` and does not require
  `model2_ref_log_probs`.
- Given both KLs are enabled,
- When reference log-probs are computed,
- Then both reference tensors exist and have the same shape as `response_mask`.
- Then the legacy `ref_log_prob` tensor remains unchanged for configs that use
  the existing single-reference KL path.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_submodel_kl_reference_plumbing.py -q
```

AC-07 - Fail Fast On Missing Or Incompatible Reference

- Given model2 KL is enabled with a missing or incompatible `model2.ref_path`,
- When config validation runs with mocked filesystem and mocked tokenizer/config
  compatibility checks,
- Then the command exits non-zero with an error naming the missing/incompatible
  model2 reference.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_submodel_kl_config.py::test_missing_enabled_reference_fails_fast -q
```

AC-08 - Dry-Run Script Shows KL Overrides And No Training Starts

- Given the code-task Stage2 submodel-KL wrapper exists,
- When it is invoked with `DRY_RUN=1`,
- Then it first runs the existing Stage2 source/provenance guards,
- Then it prints the resolved model1/model2 KL settings, resolved reference
  paths, and exits before Docker/Ray training starts.
- Given the existing Stage2 source/provenance guards fail,
- When the same dry-run is invoked,
- Then it exits non-zero before printing a training-ready command.

Expected command:

```bash
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_instruct2507_ctx8k_p40_beta01_subkl_smoke.sh
```

AC-09 - Local GPU Smoke Matrix Exercises Rollout And Actor Update

- Given local Docker image `verl-harness`, local GPU resources, and existing
  small code-task smoke data,
- When the smoke queue is run with `TOTAL_TRAINING_STEPS=5` for all required KL
  modes and required KL types,
- Then rollout generation, old-logprob/ref-logprob computation, actor backward,
  checkpoint writing, and final metrics all complete without CUDA OOM for every
  matrix item.

Required KL smoke matrix:

1. both off once, proving backward compatibility in the real runner;
2. model1 KL only with `kl`, `mse`, and `low_var_kl`;
3. model2 KL only with `kl`, `mse`, and `low_var_kl`;
4. model1 + model2 KL with `kl`, `mse`, and `low_var_kl`.

This is a 10-item matrix: `1 + 3 * 3`. Every required item is blocking. If any
item fails, this goal is not complete. The executor may additionally run
small/base-model and larger/current-model variants if needed to isolate
failures, using the same acceptance rules.

Expected command:

```bash
tmux new-session -d -s submodel_kl_smoke_queue \
  "cd /data-1/verl07/verl && ALLOW_SUBMODEL_KL_SMOKE=1 TOTAL_TRAINING_STEPS=5 bash recipe/on_policy_wdl_sft/code_task/run_submodel_kl_smoke_queue.sh"
```

Expected evidence:

- final checkpoint exists for each required mode;
- queue status marks each required matrix item complete;
- monitor logs each required matrix item to final step 5;
- logs contain rollout generation timing and actor update timing;
- enabled modes contain the expected `actor/submodel_kl/*` metrics;
- both-off mode does not require submodel KL tensors;
- no `CUDA out of memory`, `KeyError: model1_ref_log_probs`,
  `KeyError: model2_ref_log_probs`, or tokenizer mismatch appears in logs.

This AC is local-resource gated and may take many hours. It must run in tmux,
should have a thin monitor, does not require any network service, and must use
`WANDB_MODE=offline`. If GPUs are unavailable, the implementation cannot be
accepted as training-ready; it may only be marked unit-test complete.

AC-10 - Existing KL Path Still Works

- Given a vanilla or GRPO config that uses the existing single `use_kl_loss`
  path and does not enable submodel KL,
- When the relevant existing unit tests are run,
- Then they still pass and no required tensor name changes from `ref_log_prob`
  to submodel-specific names.

Expected command:

```bash
pytest tests/on_policy_wdl_sft/test_wdl_sft_is_loss.py tests/workers/config/test_actor_config_on_cpu.py -q
```

AC-11 - Documentation And Script Index Are Updated

- Given new runnable scripts or monitor entries are added,
- When `rg -n "submodel.*KL|SUBMODEL_KL|subkl" docs/joint_training/guides/training_script_index.md docs/joint_training/plans/active` is run,
- Then the new plan and runnable scripts are discoverable with launch and dry-run
  commands.
- Given a new runnable code-task script is added,
- When the Meituan code-task route is dry-tested,
- Then the new experiment name resolves to the wrapper and keeps host-specific
  overrides isolated in the Meituan env layer.

Expected commands:

```bash
rg -n "submodel.*KL|SUBMODEL_KL|subkl" docs/joint_training/guides/training_script_index.md docs/joint_training/plans/active recipe/on_policy_wdl_sft/code_task
tmp=/data-1/tmp/verl_agent_scratch/submodel_kl/meituan_route_check.log
mkdir -p "$(dirname "$tmp")"
DRY_RUN=1 EXPERIMENT=s2-code-kodcode-instruct2507-ctx8k-p40-beta01-subkl-smoke \
  bash recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh | tee "$tmp"
grep -F "run_s2_code_kodcode_instruct2507_ctx8k_p40_beta01_subkl_smoke.sh" "$tmp"
```

AC-12 - Release Gate Policy Is Not Bypassed In Offline/Mock Checks

- Given the real smoke succeeds or fails,
- When release-gate state is checked with local files and W&B stays in offline
  mode,
- Then failed or incomplete smoke attempts do not create registry import markers
  or W&B synced markers.
- Given a successful smoke reaches its configured final step,
- When the release-gate check is run locally,
- Then publication remains gated by the same monitor/release-hook logic used by
  existing code-task runs.

Expected evidence: local release-gate check output plus absence of `.synced` or
`.wandb.synced` markers for failed smoke attempts.

Expected commands:

```bash
state=/data-1/tmp/verl_agent_scratch/submodel_kl/release_gate_failed.jsonl
wandb_tmp=/data-1/tmp/verl_agent_scratch/submodel_kl/wandb_failed/offline-run-failed
rm -f "$state"
rm -rf "$wandb_tmp"
mkdir -p "$wandb_tmp"
python3 scripts/training_result_release_gate.py --state "$state" record \
  --run-name SUBMODEL-KL-SMOKE-FAILED_0 \
  --family SUBMODEL-KL-SMOKE \
  --status failed \
  --source submodel-kl-ac12 \
  --observed-step 1 \
  --final-step 2 \
  --notes "AC-12 failed-smoke publication guard"
if python3 scripts/training_result_release_gate.py --state "$state" check \
  --run-name SUBMODEL-KL-SMOKE-FAILED_0 \
  --family SUBMODEL-KL-SMOKE; then
  echo "ERROR: failed smoke was releasable" >&2
  exit 1
fi
if find "$wandb_tmp" -maxdepth 1 -type f \( -name '*.wandb.synced' -o -name '.synced' \) | grep -q .; then
  echo "ERROR: failed smoke produced W&B synced marker" >&2
  exit 1
fi
```

## 7. Preflight Checklist For Executor

1. Read this plan and confirm no code reality contradiction before editing.
2. Check `git status --short` and preserve unrelated user/agent changes.
3. Inspect current submodule state under `recipe/` before editing recipe files.
4. Confirm available disk and GPU before running AC-09.
5. Use `/data-1/tmp/verl_agent_scratch/submodel_kl/` for temporary files.
6. Run long smoke jobs in tmux.
7. Do not publish failed smoke results to DB or W&B.

## 8. Stop-And-Ask Triggers

- Existing reference-policy lifecycle cannot support frozen joint references
  without a large architecture change.
- The required reference memory footprint makes even a 2-step smoke impossible
  on local GPUs.
- A config choice would silently use fused KL instead of per-submodel KL.
- A reference path is ambiguous and cannot be resolved from launch provenance.
- The implementation would require changing WDL reward semantics to pass tests.
- Any one of the required 5-step smoke modes repeatedly fails after root-cause
  debugging; do not mark the goal complete by dropping that mode.

## 9. First Formal Experiment Candidates After Implementation

These are not part of implementation acceptance. They are the next formal
performance experiments worth considering once AC-01 through AC-12 pass:

1. Stage2 P40 beta `0.1`, lambda `0.8`, model2 KL only.
2. Stage2 P40 beta `0.1`, lambda `0.8`, model1 KL only.
3. Stage2 P40 beta `0.1`, lambda `0.8`, both KL terms enabled with a small
   shared coefficient.
4. Stage2 P40 beta `0.1`, lambda `0.8`, model2 KL coefficient sweep.
5. Stage2 P40 beta `0.1`, lambda `0.9`, model2 KL only if lambda `0.8` remains
   unstable.

Use effective step 70 and effective step 100 as the first comparison points,
because prior P40 observations suggest early gains and late collapse are both
important signals.
