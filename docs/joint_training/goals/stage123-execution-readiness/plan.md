# Stage123 Execution Readiness

- Goal ID: `stage123-execution-readiness`
- Plan version: `1`
- Plan status: `DRAFT`
- Serial position: `3 of 4`
- Prerequisite Goal: `calibration-qualification` completed with
  `calibration_result.json` decision `passed`

## Outcome

Produce one immutable, independently accepted Stage123 admission bundle proving
that the frozen FRAC25/P40 primary Stage2 -> Stage3 queue is deployable but not yet
started.

## Starting Evidence

- The completed Calibration Qualification Goal supplies the unchanged primary
  manifest, resource profile, candidate commit, and authoritative
  `calibration_result.json`.
- Current authority classes are `calibration_result`, `preflight_result`, and
  `acceptance_report`; legacy receipts and adoption artifacts cannot authorize.
- The current Stage123 queue still contains substantial shell-owned lifecycle,
  deadline, Docker ownership, checkpoint polling, and cleanup behavior that must be
  reconciled with the consolidated Python execution core.

## Scope

### Included

- Consume the exact passed calibration result without changing its manifest,
  profile, workload, run set, or candidate bindings.
- Produce fresh `preflight_result.json` for models, datasets, scorer dependencies,
  storage, container runtime, source checkpoints, provenance, machine facts, and
  conflicting execution.
- Migrate queue lifecycle, deadline, cleanup, persisted state, and resume ownership
  to `scripts/experiment_execution_core.py`; keep shell launchers thin.
- Make queue and monitor consume one normalized manifest and persisted event/state
  authority.
- Validate one immutable admission bundle containing the manifest, resource profile,
  calibration result, preflight result, reviewer acceptance report, hashes, and
  exact launch command.
- Run dry-run, fake-child, mutation, and non-weight-updating initialization checks.

### Excluded

- Any formal weight-updating Stage1, Stage2, or Stage3 training.
- P60, FRAC50, a 27-run queue, broader sweep, or manifest/run-set change.
- Recalibration or modification of a passed calibration result.
- Real W&B, WxPusher, Hugging Face, GitHub, registry mutation, or publication.
- New receipts, adoption artifacts, per-AC wrappers, or review-document chains.
- Modification, deletion, or staging of the three protected user assets named in
  the predecessor Goal.

## Architecture Contract

- The manifest, resource profile, and passed calibration result are immutable inputs.
- Generic admission validation checks shared bindings and result classes without
  hard-coding Stage123 run facts.
- Experiment-specific deployability policy is versioned and manifest-owned.
- The admitted run set is exactly `frac25-stage2` and `frac25-stage3`.
- `scripts/experiment_execution_core.py` owns execution state, child lifecycle,
  deadline, cleanup, interruption, and resume. Queue shell performs configuration
  and delegates; monitor consumes persisted events/state.
- `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1` expresses human launch intent but is never
  sufficient authority without a valid immutable admission bundle.
- `acceptance_report.json` is reviewer-owned and binds the same manifest, profile,
  calibration result, preflight result, candidate commit, run set, and Plan.
- Any binding mutation, stale result, extra run, dirty protected asset, or active
  conflicting execution fails closed before Ray or training starts.

## Acceptance Criteria

### AC-01 - Calibration Input Is Unchanged

- Given the predecessor calibration result and its bound artifacts,
- When readiness validation begins,
- Then manifest, profile, workload, run set, candidate commit, and calibration-result
  hash match exactly and no recalibration or rewriting occurs.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py`
- Expected evidence: exact hash comparison and mutation failures.

### AC-02 - Fresh Preflight Covers Deployability

- Given the admitted primary chain and live host/container environment,
- When preflight runs,
- Then source models, provenance, data, scorer dependencies, checkpoint mount,
  writable physical destination, container image, GPU visibility, machine facts,
  release paths, and conflicting execution are checked with structured outcomes.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py`
- Expected evidence: one fresh `preflight_result.json` plus reviewer-owned live probes.

### AC-03 - Run Set Is Exactly The Primary Chain

- Given manifest rendering, queue dry-run, and monitor inventory,
- When the three surfaces enumerate runs,
- Then all enumerate exactly `frac25-stage2` and `frac25-stage3`; FRAC50, P60, and
  hidden environment-variable filtering are absent.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: identical run IDs and a dry-run that starts no tmux/container.

### AC-04 - Queue Lifecycle Is Python-Owned

- Given fake clock, child, Docker, tmux, GPU, filesystem, and checkpoint adapters,
- When queue start, phase success, failure, deadline, interruption, cleanup, and
  resume paths execute,
- Then persisted Python state transitions are authoritative and shell contains no
  second lifecycle/deadline/cleanup state machine.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: behavior-level adapter logs and narrow shell-policy checks.

### AC-05 - Queue And Monitor Share One Event Authority

- Given queue and monitor processes,
- When fake execution emits lifecycle, validation, checkpoint, failure, and cleanup
  events,
- Then the monitor renders persisted state without hard-coded arrays, log scraping
  policy, duplicated run facts, or independent completion inference.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: event-driven monitor tests and zero duplicated run arrays.

### AC-06 - Admission Bundle Fails Closed

- Given valid and independently mutated calibration, preflight, acceptance,
  manifest, profile, commit, run-set, freshness, and protected-status inputs,
- When admission validation runs,
- Then only the exact complete bundle authorizes and every mutation blocks before
  Ray or a training child starts with structured failure evidence.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow -k 'manifest or preflight or admission or execution_result or stage123'`
- Expected evidence: mutation matrix, zero child calls on blocked inputs, and one
  admitted bundle hash.

### AC-07 - Readiness Does Not Train Or Publish

- Given dry-run, fake-child, and initialization checks,
- When all readiness verification runs,
- Then no optimizer step, formal checkpoint, training tmux, registry mutation, W&B
  sync, or external publication occurs.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_new_experiment_gate.py`
- Expected evidence: before/after artifact hashes, empty training-child call log, and
  no active Stage123 execution.

### AC-08 - Launch Command Is Exact And Reproducible

- Given the immutable admission bundle,
- When its launch command is rendered from a clean checkout,
- Then it pins repo path, candidate commit, manifest/profile/result paths and hashes,
  tmux session, primary run set, and explicit human intent variable without secrets.
- Verification command:
  `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh`
- Expected evidence: reviewer-owned rendered command and deliberate mismatch failures.

### AC-09 - Independent Readiness Acceptance Is Bound

- Given a committed candidate and complete admission inputs,
- When a fresh independent reviewer runs all frozen checks,
- Then AC-01 through AC-08 are PASS and reviewer-owned `acceptance_report.json`
  binds the current Plan, commit, run set, and all admission hashes.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness`
- Expected evidence: `ACCEPTANCE_COMPLETED=PASS`, valid runtime, and one immutable
  admission bundle ready for the Execution Goal.

## Milestones

1. Verify predecessor bindings and inventory remaining queue/core duplication.
2. Migrate queue lifecycle and monitor authority to the consolidated Python core.
3. Produce fresh structured preflight evidence from the live environment.
4. Build and mutation-test the immutable admission bundle.
5. Run dry-run and non-weight-updating readiness checks; confirm no execution.
6. Obtain independent final readiness acceptance from committed state.

## Runtime Contract

- Start only after this Plan is `READY` and Calibration Qualification is complete
  with decision `passed`.
- One user start authorizes autonomous Milestones 1-6; no routine milestone approval
  is required.
- Every project Python command uses the explicit `REPO_HOST` container invocation.
- Long-running checks use tmux; persistent CI uses PM2, never systemd.
- Classify findings before action and validate runtime after classifications and
  before transitions, acceptance, and completion.
- Finding classes are explicit: `IN_SCOPE` is fixed within an existing AC;
  `DEFERRED` is recorded without implementation; `CONTRADICTION` stops for Plan
  amendment; and `AC_CHANGE` stops for a fresh Plan review.
- The reviewer is independent from the implementer and evaluates the frozen Plan
  rather than continuing implementation.
- If two related implementation-review rounds leave the same finding open, stop
  before a third ordinary fix and perform a convergence review.
- Routine `IN_SCOPE` repairs are autonomous if they do not alter the calibrated
  manifest/profile/workload/run set or add an execution outcome.
- Stop for `CONTRADICTION`, `AC_CHANGE`, convergence failure, protected-asset risk,
  need to recalibrate, need to run weight-updating training, or need for a real
  external service.
- The implementer cannot self-review or self-accept.

## Reviewer Contract

- Plan review verifies one readiness outcome, immutable predecessor consumption,
  exact primary run set, and no hidden training authorization.
- Milestone review checks queue/core ownership, structured preflight, mutation
  failures, no-training evidence, and reproducible launch rendering.
- Final review independently probes live deployability without starting formal
  training and reports every AC as `PASS`, `FAIL`, or `WEAKENED`.
- Outside opinions are deferred; a completion-definition change is a contradiction.
- `ACCEPTED` requires every applicable AC to be `PASS`.

## Verification Commands

- Plan: `goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness`
- Runtime: `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness`
- Plan hash: `sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md`
- Protected status: `git status --short`

## Deferred Follow-ups

- Formal primary-chain execution and release belong to
  `stage123-primary-chain-execution`.
- P60, FRAC50, broader sweeps, and scientific interpretation require later Plans
  and fresh readiness acceptance.
