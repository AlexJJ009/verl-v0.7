# Calibration Qualification

- Goal ID: `calibration-qualification`
- Plan version: `1`
- Plan status: `DRAFT`
- Serial position: `2 of 4`
- Prerequisite Goal: `experiment-execution-core-consolidation` completed at
  commit `81fdd6d1f315200981bf89e268089f3c927f366d`

## Outcome

Produce one authoritative `calibration_result.json` that independently determines
whether the frozen Qwen3-1.7B Stage123 primary workload and resource profile are
qualified for readiness assessment, using bounded GPU evidence for validation
timing, timeout, truncation, score completeness, cleanup, and resource-risk
prediction.

## Starting Evidence

- The execution core accepts only `preflight_result`, `calibration_result`, and
  `acceptance_report` as current authority through `scripts/execution_results.py`.
- Legacy calibration, producer, preflight, adoption, and document-hash receipts
  are read-only migration inputs and cannot authorize execution.
- The current Stage123 manifest renders four runs, including FRAC50, while the
  active experiment design authorizes only the FRAC25/P40 primary Stage2 -> Stage3
  chain before any ablation.
- `scripts/check_calibration_prediction_contract.py` and historical prediction
  artifacts are migration inputs, not automatically accepted policy authority.
- This Plan does not itself authorize a GPU probe. The user must separately and
  explicitly authorize the bounded probe defined below before Milestone 4 starts.

## Scope

### Included

- Freeze the tracked Stage123 manifest to the primary FRAC25/P40 Stage2 and Stage3
  run set required by this serial Goal chain; FRAC50 and P60 are absent.
- Freeze one versioned calibration policy for the exact workload identity,
  resource-profile SHA256, manifest SHA256, implementation-tree SHA256, and
  calibration evidence commit.
- Migrate useful prediction logic from legacy checkers without restoring receipt,
  adoption, or duplicated policy authority.
- Execute one bounded non-training GPU calibration envelope after explicit user
  authorization.
- Record per-phase repetitions, validation timing, timeout, truncation, score
  completeness, resource observations, predicted bounds, cleanup, and structured
  failures.
- Emit exactly one authoritative `calibration_result.json` with decision `passed`
  or `blocked` and immutable bindings needed by the next Goal.
- Complete all production implementation needed by the next two Goals before the
  bounded probe: primary manifest normalization, admission-bundle validation,
  queue-to-Python-core lifecycle migration, event-driven monitoring, deterministic
  launch rendering, and the frozen recovery-policy schema.
- Preserve historical artifacts byte-identically and keep all scratch output under
  `/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/`.

### Excluded

- Formal Stage1, Stage2, or Stage3 weight-updating training.
- Stage123 deployability acceptance, queue admission, or formal execution.
- P60, FRAC50, a 27-run queue, broader sweeps, or calibration algorithm research.
- Real W&B, WxPusher, Hugging Face, GitHub, registry mutation, or result publication.
- A new receipt, adoption artifact, per-AC wrapper, or review-document chain.
- Modification, deletion, or staging of `.claude/skills/experiment-registry`,
  `docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md`,
  or `test_data/`.

## Architecture Contract

- The tracked normalized manifest is the sole owner of experiment facts and the
  calibration policy version; generic core code contains no Stage123 run facts.
- The primary manifest contains exactly `frac25-stage2` and `frac25-stage3`.
- The authoritative path is
  `recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml`. A broader historical
  matrix may exist only in Git history or a non-authoritative archival fixture.
- The calibration policy owns qualification thresholds; workload facts remain in
  the manifest and resource facts remain in the canonical resource profile.
- `scripts/experiment_execution_core.py` owns persisted state, child lifecycle,
  deadline, cleanup, and resume behavior. Shell entrypoints remain thin delegates.
- Calibration failures use stable `code`, `message`, and `context` fields.
- Before the probe, compute `implementation_tree_sha256` over the tracked production
  paths named by this Plan: manifest and resource profile; execution-result,
  manifest, and execution-core Python; admission gate; queue; monitor; and phase
  wrappers. Readiness may add Goal evidence and scratch results, but any change to
  this production tree invalidates calibration and requires a new result.
- `calibration_result.json` includes at least: schema/result type, decision,
  manifest SHA256, resource-profile SHA256, implementation-tree SHA256, evidence
  commit, workload identity,
  policy version/hash, bounded-probe authorization identity, timestamps, phase
  evidence, prediction comparison, cleanup result, and structured failures.
- Freshness is result/policy validation, not a receipt. Legacy receipt fields may
  be parsed only to explain migration and always fail closed as authority.
- Current manifest policy uses `preflight.result_max_age_seconds` and
  `calibration_result_max_age_seconds`. Legacy `receipt_max_age_seconds` and
  `calibration_receipt_max_age_seconds` names are removed from normalized current
  authority; versioned migration may only report and reject them.
- The bounded probe may use at most 8 L40S GPUs, performs no optimizer step, writes
  no formal checkpoint, has an aggregate wall-clock ceiling of 90 minutes, uses at
  most 3 acceptance repetitions per phase, and must clean owned tmux, Docker,
  child-process, and GPU state on success, failure, or timeout.
- The exact probe interface is:

```bash
tmux new-session -d -s stage123_calibration_qualification \
  "cd /data-1/code/verl && \
   export REPO_HOST=/data-1/code/verl CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
   export CALIBRATION_STATE_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/state && \
   export CALIBRATION_DEADLINE_SECONDS=5400 && \
   export CALIBRATION_CHILD_COMMAND_JSON=\"\$(/data-1/verl07/run_train.sh python scripts/render_calibration_probe_command.py \
     --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
     --resource-profile recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh \
     --phases stage2,stage3 --repetitions 3 --training-steps 0 \
     --scratch-root /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration)\" && \
   /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py queue \
     --run-id qwen3_1p7b_stage123_calibration_v1 \
     --state-root \"\$CALIBRATION_STATE_ROOT\" --timeout-seconds 5400 \
     --command-json \"\$CALIBRATION_CHILD_COMMAND_JSON\""
```

`scripts/render_calibration_probe_command.py` is an in-scope renderer. It validates
`training_steps=0`, `optimizer_enabled=false`, scratch-only outputs, repetitions,
phase set, and manifest/profile hashes, then emits argv JSON without executing it.

## Acceptance Criteria

### AC-01 - Primary Calibration Identity Is Frozen

- Given the Stage123 manifest and canonical resource profile,
- When the manifest is normalized and the calibration workload is rendered,
- Then the run set is exactly `frac25-stage2` and `frac25-stage3`, FRAC50/P60 are
  absent, and manifest, workload, profile, and candidate identities are stable.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_calibration_workload_descriptor.py`
- Expected evidence: reviewer-owned passing tests plus rendered IDs and SHA256s.

### AC-02 - Calibration Policy Has One Owner

- Given mutations to workload facts, thresholds, policy version, and legacy receipt
  fields,
- When normalization and policy validation run,
- Then facts follow the manifest/profile, qualification follows one versioned
  policy, mutations fail with structured codes, and legacy receipts never authorize.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_dynamic_calibration_interval.py tests/experiment_workflow/test_operational_calibration_checker.py tests/experiment_workflow/test_calibration_outcomes.py`
- Expected evidence: passing mutation tests and zero duplicated Stage123 fact owners
  in generic calibration code.

### AC-03 - Bounded Probe Cannot Train Or Publish

- Given fake and real bounded-probe adapters,
- When calibration starts, succeeds, fails, or times out,
- Then no optimizer step, formal checkpoint, registry mutation, W&B sync, or other
  external publication occurs and all writes remain in the calibration scratch root.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_operational_calibration_runtime_isolation.py tests/experiment_workflow/test_validation_deadline_cleanup.py`
- Expected evidence: fake-adapter call logs, before/after protected-path hashes, and
  bounded real-probe command evidence only after explicit authorization.

### AC-04 - Phase Evidence Is Complete And Structured

- Given Stage2 and Stage3 calibration repetitions including timeout, truncation,
  missing score, dependency failure, and success cases,
- When outcomes are evaluated,
- Then each repetition records timing, completion, truncation, score completeness,
  resource observations, and stable structured failures.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_checker.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_failure_classifier.py`
- Expected evidence: passing behavioral tests and reviewer inspection of result fields.

### AC-05 - Prediction Qualification Is Evidence-Bounded

- Given historical observations, frozen policy, and bounded-probe observations,
- When prediction qualification runs,
- Then predicted limits and observed values are compared by versioned policy,
  insufficient history fails closed, and historical fit is not treated as current
  execution authority.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_dynamic_calibration_interval.py tests/experiment_workflow/test_operational_calibration_checker.py`
- Expected evidence: deterministic prediction fixtures, insufficiency mutations,
  and policy-version binding.

### AC-06 - Cleanup Is Proven

- Given owned child, tmux, Docker, and GPU process identities,
- When success, child failure, deadline, or interruption occurs,
- Then cleanup is persisted, unrelated resources are untouched, and no owned GPU
  process or container remains.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_operational_calibration_runtime_isolation.py`
- Expected evidence: fake ownership tests plus post-probe `tmux`, Docker, and
  `nvidia-smi` audit.

### AC-07 - One Calibration Result Is Authoritative

- Given passed, blocked, stale, mismatched, and legacy evidence,
- When execution-result validation runs,
- Then only one current `calibration_result.json` bound to the exact identities can
  authorize the next Goal and all legacy evidence fails closed.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow -k 'calibration or execution_result or legacy_evidence'`
- Expected evidence: one authoritative result path, mutation failures, and no new
  receipt/adoption artifact.

### AC-08 - Independent Qualification Is Bound To Committed State

- Given a committed final production tree, explicit bounded-probe authorization,
  a calibration evidence commit, and completed
  calibration evidence,
- When a fresh independent reviewer runs the frozen commands and inspects the probe,
- Then AC-01 through AC-07 are individually PASS and acceptance is bound to Plan
  version, Plan hash, implementation-tree hash, evidence commit, manifest hash, and
  result hash.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification`
- Expected evidence: reviewer-owned `acceptance.md`, `ACCEPTANCE_COMPLETED=PASS`, and
  runtime validation success.

## Milestones

1. Freeze the primary manifest/run set and replace legacy receipt freshness fields.
2. Complete admission, queue/core, monitor, launch-rendering, and recovery production
   migration required by the serial Goals.
3. Freeze result schema, policy ownership, implementation-tree identity, probe
   renderer, prediction qualification, and fake-adapter behavioral tests.
4. Commit the final production implementation candidate; no production change is
   permitted after this point without invalidating calibration.
5. After explicit user authorization, execute the bounded GPU calibration envelope.
6. Render and validate the sole authoritative `calibration_result.json`.
7. Obtain independent final acceptance from committed state.

Milestones are hard ordered. Milestone 5 is a human authorization gate because it
uses GPUs. Other normal milestone boundaries are not approval gates.

## Runtime Contract

- Implementation starts only after Plan status is `READY` and the user starts this
  Goal. That start authorizes Milestones 1-4 and 6-7, but not the GPU probe.
- Milestone 5 requires a separate explicit user authorization naming this Goal and
  bounded probe. Do not infer authorization from Goal creation or Plan review.
- After GPU authorization, proceed autonomously through the remaining milestones.
- Every project Python command uses
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python ...`.
- Long-running probe and monitoring commands run in tmux. Persistent CI uses PM2,
  never systemd.
- Classify each new finding in `findings.jsonl` before action and run
  `validate-runtime` after classification and before milestone transitions,
  acceptance, or completion.
- Finding classes are explicit: `IN_SCOPE` is fixed within an existing AC;
  `DEFERRED` is recorded without implementation; `CONTRADICTION` stops for Plan
  amendment; and `AC_CHANGE` stops for a fresh Plan review.
- The reviewer is independent from the implementer and evaluates the frozen Plan
  rather than continuing implementation.
- If two related implementation-review rounds leave the same finding open, stop
  before a third ordinary fix and perform a convergence review.
- Routine `IN_SCOPE` defects and compatible implementation choices are autonomous.
- Stop for `CONTRADICTION`, `AC_CHANGE`, convergence failure, protected-asset risk,
  need for a broader GPU envelope, or need for a real external service.
- The implementer cannot self-review or self-accept.

## Reviewer Contract

- Plan review must verify that this is one qualification outcome, the primary run
  set is calibrated before Readiness, and the bounded GPU envelope is enforceable.
- Milestone review focuses on authority ownership, no-training guarantees, result
  binding, structured failures, and cleanup behavior.
- The final reviewer independently runs required commands, audits the sole GPU probe
  and protected paths, and reports each AC as `PASS`, `FAIL`, or `WEAKENED`.
- Opinions outside frozen ACs are non-blocking `DEFERRED_SUGGESTION`s. A required
  completion-definition change is `CONTRACT_CONTRADICTION`.
- `ACCEPTED` requires every applicable AC to be `PASS` from reviewer-owned evidence.

## Verification Commands

- Plan: `goal-plan-runtime validate-plan docs/joint_training/goals/calibration-qualification`
- Runtime: `goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification`
- Plan hash: `sha256sum docs/joint_training/goals/calibration-qualification/plan.md`
- Protected status: `git status --short`

## Deferred Follow-ups

- Stage123 deployability and immutable admission bundle belong to
  `stage123-execution-readiness`.
- Formal primary-chain training and release belong to
  `stage123-primary-chain-execution`.
- P60, FRAC50, broader sweeps, calibration algorithm research, and scientific result
  interpretation require later Plans.
