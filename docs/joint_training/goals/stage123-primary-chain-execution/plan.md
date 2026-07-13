# Stage123 Primary Chain Experiment Execution

- Goal ID: `stage123-primary-chain-execution`
- Plan version: `1`
- Plan status: `DRAFT`
- Serial position: `4 of 4`
- Prerequisite Goal: `stage123-execution-readiness` completed with an immutable
  independently accepted admission bundle

## Outcome

Execute, monitor, verify, and conditionally release the single admitted FRAC25/P40
Stage2 -> extracted-model2 -> Stage3 primary chain without changing its approved
configuration.

## Starting Evidence

- The Readiness Goal supplies one immutable admission bundle containing the exact
  manifest, profile, calibration result, preflight result, acceptance report,
  implementation-tree SHA256, Readiness evidence commit, run set, hashes, and launch
  command.
- The admitted run set is exactly `frac25-stage2` and `frac25-stage3`.
- Execution is operational work, not implementation: a code/config/manifest/profile
  change invalidates admission and returns to Readiness.
- The accepted bundle includes `implementation_tree_sha256`; a production-tree
  change returns first to Calibration Qualification, while evidence-only expiry or
  live-preflight change returns to Readiness.

## Scope

### Included

- Revalidate the immutable admission bundle immediately before launch.
- Launch the exact admitted queue in tmux and monitor persisted Python-owned state.
- Complete Stage2, verify checkpoint and validation completion, extract model2, and
  bind provenance before Stage3.
- Complete Stage3 and verify required validation, checkpoints, metrics, provenance,
  cleanup, and final execution state.
- Apply only the frozen automatic recovery/resume policy and record structured
  failures and attempts.
- Run the training-result release gate after successful local completion.
- Publish eligible successful results to the local registry and existing W&B project
  only if separately authorized by the user and the release gate passes.
- Obtain independent final acceptance of execution truth and release correctness.

### Excluded

- Any change to code, manifest, resource profile, calibration/preflight/acceptance
  artifacts, implementation tree, Readiness evidence commit, run set,
  hyperparameters, source checkpoints, data,
  validation breadth, or recovery policy.
- P60, FRAC50, a 27-run queue, additional seeds, retries outside frozen policy,
  broader sweeps, or a replacement experiment.
- Scientific interpretation or a claim that the plateau-breakthrough hypothesis is
  supported; this Goal verifies execution, not research conclusions.
- Publication of failed, incomplete, release-ineligible, or unapproved runs.
- Modification, deletion, or staging of protected user assets.

## Execution Contract

- Human launch intent and a valid admission bundle are both required; neither is
  sufficient alone.
- Before Ray starts, exact Plan, commit, manifest, profile, calibration, preflight,
  acceptance, run-set, source-artifact, and protected-status bindings are rechecked.
- Stage2 must complete its admitted final step and required validation before model2
  extraction. Extraction must produce content/provenance hashes bound to Stage2.
- Stage3 starts only from the admitted extracted model2 and verified provenance.
- Python persisted state is execution authority. Monitor output, tmux presence, W&B,
  and registry state are observations, not substitute authority.
- Automatic recovery is limited to one resume attempt for a failure explicitly
  classified as resumable by the admitted policy. The only resumable codes are
  `host_interruption`, `container_runtime_interruption`, and
  `checkpoint_available_child_exit`. Persisted attempt records contain `attempt`,
  `max_attempts=2`, `resume_from_checkpoint`, `failure_code`, `manifest_sha256`,
  `implementation_tree_sha256`, `bundle_sha256`, `started_at`, and `completed_at`.
  Configuration, dependency,
  provenance, OOM, repeated deadline, data, scorer, or binding failures are not
  automatically altered or downscoped.
- Any code/config/artifact binding change is `AC_CHANGE`; stop and return to a new
  Readiness candidate rather than patching during execution.
- Failed or incomplete runs remain local diagnostic evidence. Release requires the
  repository release gate to pass.

## Acceptance Criteria

### AC-01 - Launch Uses The Exact Admission Bundle

- Given the accepted immutable bundle and explicit user launch authorization,
- When pre-launch validation runs,
- Then every binding matches, protected assets remain untouched, no conflicting run
  exists, and only the exact primary launch command can start Ray.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --repo-root /data-1/code/verl`
- Expected evidence: pre-launch validation report and immutable bundle hash.

### AC-02 - Only The Primary Run Set Executes

- Given persisted execution events and system process evidence,
- When the queue runs,
- Then only `frac25-stage2` and `frac25-stage3` execute; no P60, FRAC50, 27-run,
  hidden ablation, or additional run starts.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --format json`
- Expected evidence: admitted run IDs matched to state, tmux, container, checkpoint,
  and metrics records.

### AC-03 - Stage2 Completes With Required Evidence

- Given the admitted Stage2 configuration,
- When Stage2 reaches terminal state,
- Then its final step, full required validation, checkpoint, metrics, resource
  profile, manifest binding, and cleanup state are complete and structured.
- Verification command:
  `STAGE2_RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage2 --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$STAGE2_RUN_NAME"`
- Expected evidence: persisted completed state, checkpoint/metrics paths, validation
  evidence, and release-gate diagnostic result without premature publication.

### AC-04 - Model2 Extraction And Provenance Are Exact

- Given the completed Stage2 checkpoint,
- When model2 is extracted,
- Then the extracted artifact and provenance bind source run, source checkpoint,
  final Stage2 step, submodel index, manifest/profile/commit hashes, output hash, and
  release eligibility; any mismatch blocks Stage3.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_stage123_preflight_model_identity.py`
- Expected evidence: runtime provenance validation plus artifact hashes.

### AC-05 - Stage3 Completes From The Admitted Extraction

- Given the verified extracted model2 and provenance,
- When Stage3 runs,
- Then it uses only that input, reaches its admitted final step, completes required
  validation, writes checkpoint/metrics/provenance, and reaches a clean terminal state.
- Verification command:
  `STAGE3_RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage3 --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$STAGE3_RUN_NAME"`
- Expected evidence: completed persisted state and matching runtime artifacts.

### AC-06 - Recovery Obeys The Frozen Policy

- Given any interruption or failure,
- When failure classification and recovery policy run,
- Then only an explicitly resumable failure receives at most one same-binding resume;
  all other failures stop without configuration changes, test weakening, benchmark
  downscope, or unapproved reruns.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_failure_classifier.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: attempt history, structured classification, and unchanged hashes.

### AC-07 - Monitoring And Cleanup Reflect Execution Truth

- Given queue, monitor, container, GPU, checkpoint, and persisted-state evidence,
- When each phase terminates,
- Then monitor status matches Python-owned state, owned resources are cleaned, and no
  stale tmux/container/GPU process falsely represents completion or remains leaked.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_validation_deadline_cleanup.py`
- Expected evidence: live `tmux`, Docker, `nvidia-smi`, state, log, checkpoint, and Ray
  failure-marker audit.

### AC-08 - Release Is Conditional And Accurate

- Given successful or failed execution outcomes,
- When release checks run,
- Then failed/incomplete outcomes are not published; successful outcomes are eligible
  only after release-gate PASS; registry/W&B publication occurs only under explicit
  publication authorization and is verified without checkpoint/artifact payload sync.
- Verification command:
  `for RUN_ID in frac25-stage2 frac25-stage3; do RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id "$RUN_ID" --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$RUN_NAME"; done`
- Expected evidence: release decisions, registry rows and W&B sync markers only for
  eligible authorized success, or structured local-only completion if publication
  was not authorized/available.
- Publication transport failure after successful local execution is recorded as
  `local_execution_complete_publication_blocked`; it never rewrites training state
  as failed.

### AC-09 - Independent Acceptance Verifies Execution, Not Hypothesis

- Given committed admitted configuration and terminal execution evidence,
- When a fresh independent reviewer audits the run from local authoritative state,
- Then AC-01 through AC-08 are individually PASS, no unapproved run/config change
  occurred, and acceptance makes no unsupported scientific-performance claim.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution`
- Expected evidence: reviewer-owned acceptance bound to Plan, admission bundle,
  execution state, artifacts, release decision, and terminal execution-evidence
  commit.

## Milestones

1. Revalidate admission bundle and record explicit launch authorization.
2. Launch and monitor admitted Stage2 through terminal validation and cleanup.
3. Extract model2 and independently verify provenance before Stage3.
4. Launch and monitor admitted Stage3 through terminal validation and cleanup.
5. Verify full local artifact/result integrity and recovery history.
6. Apply release gate and, only if separately authorized, publish eligible results.
7. Obtain independent final execution acceptance.

Milestones are hard ordered. Normal successful transitions are autonomous. Stage3
cannot start before Milestone 3 passes.

## Runtime Contract

- Start only after this Plan is `READY`, Readiness is complete, and the user
  explicitly authorizes formal primary-chain execution using the admitted bundle.
- One launch authorization covers Milestones 1-5 and 7. Registry/W&B publication in
  Milestone 6 requires separate explicit publication authorization unless the launch
  authorization explicitly includes it.
- Long-running training and monitoring run in tmux. Persistent CI uses PM2 only.
- Every project Python command uses the explicit `REPO_HOST` container invocation.
- Findings are classified before action and runtime is validated before transitions,
  acceptance, and completion.
- Finding classes are explicit: `IN_SCOPE` is fixed within an existing AC;
  `DEFERRED` is recorded without implementation; `CONTRADICTION` stops for Plan
  amendment; and `AC_CHANGE` stops for a fresh Plan review.
- The reviewer is independent from the implementer and evaluates the frozen Plan
  rather than continuing implementation.
- If two related implementation-review rounds leave the same finding open, stop
  before a third ordinary fix and perform a convergence review.
- Operational recovery within the frozen one-resume policy is autonomous.
- Stop for any binding/config/code change, non-resumable failure, repeated failure,
  convergence failure, protected-asset risk, new run/sweep, required benchmark
  downscope, or need for an unapproved external service.
- The implementer cannot self-review or self-accept.

## Reviewer Contract

- Plan review verifies this is one admitted execution outcome and does not smuggle in
  ablations, configuration work, or scientific interpretation.
- Milestone reviews use persisted state plus live tmux/process/container/GPU/log/
  checkpoint/Ray evidence and verify exact bindings and phase ordering.
- Final review independently audits execution truth, recovery, provenance, release,
  protected assets, and absence of unapproved runs.
- Outside scientific opinions are deferred and cannot weaken execution acceptance.
- `ACCEPTED` requires every applicable AC to be `PASS`.

## Verification Commands

- Plan: `goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution`
- Runtime: `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution`
- Plan hash: `sha256sum docs/joint_training/goals/stage123-primary-chain-execution/plan.md`
- Protected status: `git status --short`

## Deferred Follow-ups

- P60, FRAC50, broader sweeps, additional seeds, scientific interpretation, and paper
  claims require separate Plans and fresh readiness acceptance.
