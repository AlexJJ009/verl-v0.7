# Experiment Batch Orchestration

- Goal ID: `experiment-batch-orchestration`
- Plan version: `2`
- Plan status: `DRAFT`
- Prerequisite implementation: `experiment-execution-core-consolidation`
- Consumer: `stage123-primary-chain-execution` and later experiment execution Goals

## Outcome

Extend the existing Python experiment execution core and persisted-event monitor
into one deterministic, reusable batch orchestrator that runs only pre-authorized
atomic experiment bundles, routes an experiment-local terminal failure to the next
pre-authorized item, supports human pause/stop control, and never changes experiment
parameters or delegates deterministic authority to an AI agent.

## Starting Evidence

- `scripts/experiment_execution_core.py` already owns single-command child lifecycle,
  persisted state, append-only events, deadline handling, terminal status, and cleanup.
- `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh`
  already delegates Stage123 lifecycle to the execution core.
- `scripts/stage123_manifest_monitor.py` already consumes execution-core persisted
  events; it must be generalized or reduced to a thin compatibility entrypoint,
  not copied into a second monitor implementation.
- `scripts/training_queue_monitor.sh` remains an observational legacy helper that
  infers state from tmux/checkpoints/metrics. It is not a valid authority for this
  Goal and must not become the batch transition engine.
- Baseline focused validation on 2026-07-14:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q
  tests/experiment_workflow/test_stage123_core_migration.py
  tests/experiment_workflow/test_manifest_queue_monitor_contract.py
  tests/experiment_workflow/test_failure_classifier.py
  tests/experiment_workflow/test_validation_deadline_cleanup.py
  tests/experiment_workflow/test_pm2_ci_keepalive.py`
  -> `23 passed in 16.69s`.

## Scope

### Included

- Extend `scripts/experiment_execution_core.py` with deterministic batch lifecycle
  and operator-control modes while preserving the existing atomic execution API.
- Define and validate one immutable batch manifest schema binding each item to its
  Goal, Plan hash, accepted admission bundle, exact launch command, expected run set,
  authorization identity, and input hashes.
- Reuse one execution state/event schema for atomic items and batch transitions.
- Generalize the existing persisted-event monitor and update Stage123 to use it.
- Implement pure-script `pause_after_current`, `stop_now`, and human-authorized
  continuation of the remaining immutable batch.
- Implement experiment-local failure routing, shared-invariant batch stops, cleanup,
  skipped-phase evidence, and no-retry/no-parameter-change policy.
- Add focused unit, mutation, control, batch-routing, cleanup, and compatibility
  tests; update only thin experiment adapters and their docs/indexes as required.

### Excluded

- Formal GPU training, rollout benchmarking, or execution of any batch item.
- Scientific experiment design, hypotheses, metric thresholds, or result claims;
  those belong to each atomic experiment Goal.
- Automatic retry, checkpoint resume, parameter tuning, OOM remediation, or dynamic
  experiment selection.
- Codex/Agent supervision, hourly wake-up tasks, crontab-driven decisions, or any
  language-model process in the execution authority path.
- Registry, W&B, Hugging Face, GitHub, WxPusher, or other real external calls.
- A second batch queue, second persisted-state ledger, second fallback engine, or
  second monitor implementation.
- Modification, deletion, or staging of protected user assets.

## Architecture Contract

### Single Authority

- `scripts/experiment_execution_core.py` is the only state-transition authority.
- Its atomic mode remains responsible for one admitted command and its cleanup.
- Its batch mode is responsible for ordered items, item terminal classification,
  next-item routing, batch-stop decisions, operator controls, and batch events.
- Shell entrypoints are adapters only: they render/validate inputs and delegate to
  the core; they must not persist competing state or implement a second lifecycle.
- Atomic compatibility is explicitly separate from batch semantics: legacy atomic
  callers may retain their existing recovery behavior, but a batch item is always
  admitted with `max_attempts=1`, an empty resumable-failure set, no `--resume`, and
  no recovery-policy override. The Stage123 adapter rejects those options in batch
  mode and renders the batch admission bundle with the no-retry policy.
- The monitor is read-only and consumes the core's persisted state/events. It never
  infers completion or failure from tmux, checkpoint presence, metrics, W&B, or the
  registry.

### Batch Manifest

The canonical batch manifest is JSON, schema version `1`, and contains:

- `batch_id`, `batch_manifest_sha256`, `authorization_id`, and `created_at`;
- ordered `items`, each with `item_id`, `goal_id`, `plan_sha256`,
  `admission_bundle_path`, `admission_bundle_sha256`, exact launch command JSON,
  expected run IDs, and input hashes;
- explicit `failure_policy_id` and `operator_control_path`;
- no secrets, mutable shell fragments, or implicit glob patterns.

The validator recomputes the canonical manifest hash, rejects duplicate item IDs or
run IDs, rejects missing/changed admission inputs, rejects current-checkout and
protected-asset mismatches, and rejects an item whose exact admission validation
does not pass before launch.

Each item is an accepted admission bundle, not an arbitrary shell command. The
bundle records `implementation_tree_sha256`, evidence commit, recipe gitlink,
canonical command JSON and `command_sha256`, input hashes, and an adapter type from
the finite accepted-adapter registry. The validator re-renders the command from the
bundle, recomputes the implementation-tree identity from the listed tracked paths,
and rejects mutable or untracked executable paths, shell fragments, and a command
hash mismatch. A command such as `bash queue_impl.sh` is valid only when the script
contents are covered by that tree identity and the accepted adapter.

### State And Events

- Batch state records `pending`, `running`, `paused_after_current`, `stopping`,
  `stopped`, `completed`, `completed_with_failures`, and `shared_failure`.
- Item state records `pending`, `running`, `succeeded`,
  `inconclusive_operational_failure`, and `skipped_after_prior_failure`.
- Every transition is append-only and binds batch hash, item ID, Goal/Plan hash,
  admission hash, command hash, failure code, cleanup result, skipped phases, next
  item, operator control state, and timestamps.
- A malformed state/event, missing binding, or cleanup failure is a shared failure;
  the batch stops rather than guessing.

Operator controls use a canonical JSON envelope with `schema_version`, `batch_id`,
`batch_manifest_sha256`, monotonic `control_seq`, `expected_batch_revision`,
`action`, `authorization_id`, and `issued_at`; the envelope hash is persisted with
the event. The core accepts a control only when its sequence is fresh, its expected
revision compare-and-swaps against the current batch revision, its authorization
matches the admitted operator, and it has not already been consumed. A replay or
stale revision is rejected without state mutation. `pause_after_current` arriving
during cleanup pauses before the next item; `stop_now` during execution or cleanup
cleans the active item and prevents the next launch; `continue_remaining` is valid
only from a paused/stopped state with the same manifest hash and fresh authorization.

### Failure Routing

- Experiment-local terminal failures include child exit, OOM, scorer timeout,
  validation timeout, missing experiment artifact, provenance failure, and ordinary
  experiment-level release failure. The item becomes
  `inconclusive_operational_failure`; its remaining phases are skipped; owned
  resources are cleaned; the next pre-authorized item starts.
- Shared batch-stop failures include batch/admission hash mismatch, protected-asset
  mutation, checkpoint-mount loss, frozen disk gate failure, GPU/driver unavailability,
  state/event corruption, cleanup failure, and operator `stop_now`.
- Two consecutive items with the same normalized failure code stop the batch. This
  is a fixed anti-cascade policy, not an adaptive retry budget.
- The batch never changes a command, profile, dataset, seed, objective, evaluator,
  rollout count, train step count, checkpoint selection, or queue order.
- A human can stop the active item and later continue only the remaining immutable
  items after reviewing evidence. Editing a not-yet-started item requires a new
  batch manifest hash and a new authorization; the active manifest is never edited.

### Authority And Deletion Budget

The implementation must publish a Goal-local `deletion-budget.json` and an evidence
inventory. The budget is acceptance-bound and requires:

- `scripts/experiment_execution_core.py` to retain the sole persisted state,
  transition, cleanup, deadline, failure-routing, and operator-control authority;
- `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh`
  to retain only argument rendering/admission delegation and child-specific adapter
  data; it must not write queue status state, own deadlines or cleanup, classify
  failures, loop phases, route to the next phase/item, or decide fallback;
- `scripts/stage123_manifest_monitor.py` and any new monitor entrypoint to share one
  monitor implementation, with the Stage123 path reduced to a delegating shim or
  deleted; `scripts/training_queue_monitor.sh` remains observational and cannot
  transition state;
- notification and release integrations to remain read-only consumers of persisted
  events.

The evidence inventory must list each removed, reduced, or retained authority-bearing
construct and a focused test proving the forbidden duplicate path cannot transition
batch state. No deletion outside this budget is authorized.

## Acceptance Criteria

### AC-01 - Existing Core Is The Sole Transition Authority

- Given the current atomic execution core and Stage123 adapter,
- When batch mode and controls are implemented,
- Then one core owns item/batch state transitions, cleanup, failure routing, and
  append-only events; no parallel queue, fallback engine, or state ledger exists.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_experiment_batch_core.py`
- Expected evidence: core-owned lifecycle tests, authority-boundary mutation tests,
  and an inventory showing no duplicate transition implementation.

### AC-02 - Batch Manifest Is Immutable And Fully Bound

- Given an ordered batch manifest of already accepted experiment bundles,
- When validation runs before the first item,
- Then canonical hash, Goal/Plan hashes, admission hashes, exact command hashes,
  expected run sets, authorization, current checkout, and protected assets all
  match; any mutation fails closed.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json`
- Expected evidence: valid fixture, deliberate manifest/input/command/admission
  mutations rejected, and no child process started by validation.

### AC-03 - Successful Items Advance Deterministically

- Given two or more valid pre-authorized items,
- When each item reaches terminal success,
- Then the next item starts in manifest order only after prior cleanup and complete
  terminal evidence; no item is duplicated, skipped, or invented.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k success`
- Expected evidence: event ledger with exact ordered item transitions and cleanup
  before each next-item start.

### AC-04 - Local Failure Falls Back To The Next Item

- Given an item-local terminal failure,
- When batch routing handles it,
- Then the item is terminal `inconclusive_operational_failure`, remaining phases are
  skipped, resources are cleaned, and the next pre-authorized item starts without
  retry, resume, parameter change, or queue mutation.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k local_failure`
- Expected evidence: failure code, cleanup, skipped phases, next item, unchanged
  command/profile/data hashes, and no second attempt for the failed item.

### AC-05 - Shared Failures Stop The Batch

- Given a shared invariant failure or two consecutive equal normalized item failures,
- When routing evaluates the failure,
- Then the batch enters `shared_failure` or `stopped`, does not start another item,
  preserves all evidence, and reports the exact stop reason.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k shared_failure`
- Expected evidence: stop event, no next child launch, preserved failed states, and
  no cleanup bypass.

### AC-06 - Human Pause And Stop Are Deterministic

- Given a running or boundary-state batch,
- When an operator issues `pause_after_current`, `stop_now`, or authorized
  `continue_remaining`,
- Then the core applies the command at the documented boundary, records operator
  identity/control hash, cleans owned resources, and never edits the immutable batch.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py`
- Expected evidence: control event sequence, graceful/forced stop evidence, and
  rejection of stale or unauthorized controls.

### AC-07 - One Persisted-Event Monitor Serves Batch And Stage123

- Given batch and Stage123 execution events,
- When the monitor runs,
- Then it reports from the shared core state/event schema, is idempotent, and never
  infers authority from tmux/checkpoint/metrics/W&B/registry observations.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_experiment_batch_monitor.py`
- Expected evidence: shared monitor tests, mutation gates, and no duplicate monitor
  lifecycle logic; `training_queue_monitor.sh` is not used as authority.

### AC-08 - Stage123 Compatibility Remains Thin

- Given the existing Stage123 public queue and phase launchers,
- When they invoke batch/atomic execution,
- Then they delegate to the unified core, preserve manifest/admission validation,
  and do not create a second state root or fallback policy.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py`
- Expected evidence: wrapper integration, exact state-root binding, and migration
  tests for old entrypoint behavior, plus the deletion-budget inventory showing that
  batch mode rejects `--resume` and recovery-policy overrides and the adapter owns no
  lifecycle transition.

### AC-09 - No Autonomous Tuning Or AI Authority

- Given OOM, timeout, scorer, dependency, or validation failures,
- When batch routing handles them,
- Then it either falls back under AC-04 or stops under AC-05; no Codex/Agent/language
  model, retry, resume, or parameter mutation participates in the decision.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_failure_classifier.py`
- Expected evidence: forbidden-action mutation tests, unchanged command/profile/input
  hashes, and explicit absence of agent/timer launch paths.

### AC-10 - Operational Policies And Protected Assets Remain Intact

- Given the implementation and focused tests,
- When policy and protected-asset checks run,
- Then PM2-only persistent CI, tmux long-running work, explicit `REPO_HOST`, no real
  external services, and protected asset fingerprints remain valid.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_pm2_ci_keepalive.py tests/experiment_workflow/test_experiment_batch_policy.py && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl`
- Expected evidence: focused policy pass and unchanged protected baseline.

### AC-11 - Independent Acceptance Binds The Unified Implementation

- Given the committed implementation and reviewer-owned evidence,
- When a fresh independent reviewer audits the Goal,
- Then AC-01 through AC-10 are individually `PASS`, the report binds the current
  Plan/version/hash and candidate commit, and no duplicate authority or unapproved
  training has occurred.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration`
- Expected evidence: reviewer-owned acceptance report, exact focused command output,
  implementation-tree identity, deletion/compatibility audit, immutable command
  binding evidence, and protected status.

## Feasibility Probes

- Existing-core baseline probe on 2026-07-14: five focused suites passed `23 tests in
  16.69s` under the explicit container wrapper. This supports extending the current
  core/monitor rather than introducing a replacement runtime.
- The `two consecutive equal normalized failure codes` rule is a fixed safety policy,
  not a performance/resource budget. It has an explicit waiver because it bounds
  cascade damage; it must be mutation-tested and cannot be tuned from observed
  results.
- No AC promises an absolute latency, throughput, GPU, memory, disk, or wall-clock
  budget. The planned one-to-two-day batch duration is scheduling context only.

## Milestones

1. Freeze batch schema, state/event extensions, failure policy, and operator-control
   contract; independently review the Plan.
2. Extend `experiment_execution_core.py` with batch validation/routing/control and
   focused tests; no new queue authority is added.
3. Generalize the existing persisted-event monitor, migrate Stage123 adapters, and
   remove/reduce duplicate lifecycle logic under an explicit deletion budget.
4. Add mutation, fallback, shared-stop, pause/stop, no-retry, and compatibility gates.
5. Run the focused policy matrix and independently review the implementation.
6. Run the full CPU preservation gate exactly once, obtain independent final
   acceptance, and record the Goal completion ledger.

Milestones are hard ordered. No GPU, formal training, external publication, or
Stage123 experiment launch is part of this Goal.

## Runtime Contract

- Start only after this Plan is independently reviewed as `READY`.
- All project Python commands use the explicit `REPO_HOST=/data-1/code/verl`
  `/data-1/verl07/run_train.sh python ...` wrapper.
- Long-running checks use tmux. Persistent CI remains PM2-only; systemd is forbidden.
- New findings are classified before action and runtime is validated after
  classification, before transitions, before acceptance, and before completion.
- The reviewer is independent from the implementer and evaluates the frozen Plan
  rather than continuing implementation.
- If two related implementation-review rounds leave the same finding open, stop
  before a third ordinary fix and perform a convergence review.
- `IN_SCOPE` fixes may extend the existing core/monitor without changing the frozen
  outcome. `DEFERRED` findings are recorded without implementation.
- `CONTRADICTION`, `AC_CHANGE`, protected-asset risk, production/shared-live mutation,
  required GPU/external service, or convergence failure stops for user decision.
- The implementer must not self-review or self-accept.

## Progression Policy

- `AUTO_ADVANCE`: validate Plan, request independent Plan review, classify findings,
  apply reviewed in-scope fixes, run focused tests, perform milestone reviews, run
  runtime validation, and request final acceptance.
- `USER_DECISION`: start this Goal; mutate shared production runtime behavior beyond
  the frozen core/monitor scope; delete non-Goal resources; add retry/resume/tuning;
  use GPU or real external services; alter protected assets; or accept a new
  independently useful outcome.

## Reviewer Contract

- Plan review verifies one reusable batch-orchestration capability, reuse of the
  accepted execution core, one persisted-event authority, exact failure routing,
  operator controls, no AI/timer authority, and testable deletion boundaries.
- Milestone review verifies behavior from focused tests and mutation gates, not file
  names or receipts.
- Final review independently runs the focused matrix and the single full CPU gate,
  audits duplicate authorities and protected assets, and binds acceptance to the
  current implementation tree and candidate commit.
- Suggestions outside this orchestration outcome are `DEFERRED_SUGGESTION`; they do
  not expand the Goal.

## Verification Commands

- Plan: `goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration`
- Runtime: `goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration`
- Plan hash: `sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md`
- Protected status: `git status --short`

## Deferred Follow-ups

- Stage123 matched-control manifest, scientific decision, and formal training belong
  to `stage123-primary-chain-execution` after this Goal is accepted.
- Scientific experiment design, parameter tuning, automatic OOM remediation, and
  cross-Goal scheduling policy require separate Plans and fresh review.
