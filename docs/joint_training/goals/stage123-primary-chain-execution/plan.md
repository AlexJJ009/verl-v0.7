# Stage123 Primary Chain Experiment Execution

- Goal ID: `stage123-primary-chain-execution`
- Plan version: `15`
- Plan status: `DRAFT - V15 UNIFIED VALIDATION PROTOCOL RERUN`
- Serial position: `4 of 4`
- Prerequisite Goal: `stage123-execution-readiness` completed with an immutable
  independently accepted admission bundle

## Outcome

Execute one fresh matched FRAC25/P40 control-versus-Stage2->Stage3 experiment under
one frozen validation-decoder protocol, preserve its exact execution truth, and
produce a frozen decision that either supports or rejects the specified
plateau-breakthrough hypothesis for this configuration, or declares the experiment
operationally inconclusive.

## Starting Evidence

- The Readiness Goal supplies one immutable admission bundle containing the exact
  manifest, profile, calibration result, preflight result, acceptance report,
  implementation-tree SHA256, Readiness evidence commit, run set, hashes, and launch
  command.
- The predecessor Readiness bundle admits only `frac25-stage2` and
  `frac25-stage3`; Plan v2 invalidates that bundle because a scientifically valid
  experiment also requires one matched pure-Stage1 control.
- Execution is operational work, not implementation: a code/config/manifest/profile
  change invalidates admission and returns to Readiness.
- The accepted bundle includes `implementation_tree_sha256`; a production-tree
  change returns first to Calibration Qualification, while evidence-only expiry or
  live-preflight change returns to Readiness.
- A historical pure-Stage1 step100 aggregate exists in the offline W&B output log,
  but it is not the control for this experiment: it used response length `4096`, the
  retained step100 checkpoint no longer exists, and current Stage123 uses response
  length `8192`. It is context only and cannot support or reject the hypothesis.
- A legacy Stage2 attempt with suffix `1783777744` reached only step0 validation and
  was deliberately stopped. Its logs and empty checkpoint root are diagnostic
  evidence only and must never be resumed, merged, or counted as an experimental arm.
- The first formal control completed all 60 steps on July 15, 2026, but the
  subsequent Stage2 wrapper rejected the same accepted bundle because the
  3600-second preflight freshness window elapsed during the 77-minute control.
  This is an operational contradiction, not a scientific result. This amendment
  preserves fresh admission at atomic-item start while preventing an already
  admitted running item from invalidating itself between phases.
- The completed V14 control, Stage2, and Stage3 artifacts are historical local
  diagnostic evidence only. Their online validation decoder drifted: Control and
  Stage3 used sampled `pass@1` at temperature `1.0`, while Stage2 used temperature
  `0.2`. They cannot support or reject the scientific hypothesis and must not be
  reused as a control, Stage2 handoff, Stage3 source, or final decision input for
  this V15 experiment.

## Required Prerequisite Outcomes

This Goal remains non-executable until both prerequisite outcomes are independently
accepted:

1. **Matched Stage123 Readiness**: a fresh Readiness contract has implemented and
   independently accepted the three-run manifest, control wrapper, Stage2/Stage3
   wrappers, admission bundle, preflight, calibration applicability, and exact
   launch command. The predecessor two-run bundle cannot authorize this Goal.
2. **Experiment Batch Orchestration**: a separate reusable Goal has extended the
   already accepted experiment execution core and persisted-event monitor with
   deterministic batch routing and operator pause/stop control, and independently
   accepted the focused tests described below. It must consolidate existing code,
   not introduce a parallel queue/monitor authority. This is a cross-experiment
   capability and is intentionally not implemented inside this scientific Goal.

The Stage123 Plan Reviewer must reject `READY` when either prerequisite is missing,
when any required script is only described rather than present and tested, or when
the queue/monitor would require an AI agent to make a deterministic phase transition.

## Batch Orchestration Contract

- An **atomic experiment** is one hypothesis, one matched control/treatment matrix,
  and one support/reject/inconclusive decision. This Goal is one atomic experiment.
- An **experiment batch** is an ordered list of already-`READY`, already-authorized
  atomic experiment bundles. The batch may run for one or two days and may contain
  this Goal plus later independently accepted Goals; it must never invent an
  experiment, alter a bundle, or bypass a Goal's launch authorization.
- `scripts/experiment_execution_core.py` remains the only Python execution-state and
  transition authority. The prerequisite Goal extends this file with a batch mode
  and operator-control mode instead of adding another batch queue implementation.
  Batch mode consumes a signed/hashed batch manifest, starts only the exact admitted
  command for the next item, records the active item, and supports
  `pause_after_current` and `stop_now`.
- The existing persisted-event monitor implementation is generalized for batch use
  rather than copied. `scripts/stage123_manifest_monitor.py` may be renamed to a
  generic path or retained as a thin compatibility entrypoint, but there must be one
  shared monitor implementation and one event schema. It remains read-only with
  respect to execution state and does not infer completion from tmux, checkpoints,
  W&B, or registry state.
- The existing Stage123 public queue shell remains a thin experiment adapter that
  validates its manifest/admission and delegates lifecycle to
  `experiment_execution_core.py`. Experiment-specific phase launchers remain thin
  commands; they do not create a second persisted-state authority.
- Qualification is valid by training-plane identity, not a wall-clock TTL. A batch
  validates each immutable item completely before it starts a phase child. Direct
  phase invocation performs the same complete admission. Batch environment values,
  persisted state, hashes, and receipts are consistency inputs only: they are not a
  hostile same-root authorization boundary. The local operator running the batch is
  trusted for reliability purposes; a binding mismatch still fails the launch.
- Operator pause/resume/stop is implemented as a subcommand of the same execution
  core or a thin CLI that calls its shared control API. A human can stop the active
  run at any time; resume means continuing the remaining batch after human review,
  not resuming a failed experiment from a checkpoint.
- `scripts/training_queue_monitor.sh` is not reused as execution authority because it
  observes tmux/checkpoint/metrics paths. It may remain for legacy observational and
  release workflows, but the Stage123/batch monitor must remain Python-event-owned.
- The prerequisite Goal must delete or reduce any superseded Stage123-specific
  implementation according to an explicit deletion budget. Compatibility wrappers
  may delegate, but duplicate lifecycle, fallback, or monitor logic is forbidden.
- No Codex, Agent, cron job, or language-model process participates in normal batch
  execution, monitoring, failure classification, phase transition, or fallback.
  Deterministic work is performed entirely by scripts.

### Failure And Fallback Policy

- An experiment-local terminal failure marks that atomic experiment
  `INCONCLUSIVE_OPERATIONAL_FAILURE`, preserves its logs/checkpoints/state, cleans
  resources owned by that item, skips all remaining phases of that atomic
  experiment, and starts the next pre-authorized batch item.
- There is no automatic parameter correction and no automatic same-run retry or
  checkpoint resume. OOM, timeout, scorer failure, child exit, provenance failure,
  or validation failure cannot change microbatch, vLLM utilization, tensor
  parallelism, max length, workers, timeout, batch size, optimizer, steps, data,
  seed, objective, or evaluator.
- The batch stops instead of falling forward when a shared invariant fails:
  admission/batch hash mismatch, protected-asset mutation, checkpoint-mount loss,
  insufficient frozen disk gate, GPU/driver unavailability, execution-state
  corruption, operator `stop_now`, or two consecutive experiments ending with the
  same normalized failure code. This prevents a shared defect from consuming the
  whole queue.
- Fallback always means the next already-reviewed and already-authorized atomic
  experiment in the immutable batch manifest. It never means an improvised run,
  changed parameter set, replacement checkpoint, extra seed, or hidden ablation.
- A human may pause or stop the batch, amend only not-yet-started future experiment
  Goals, obtain new review/admission/authorization, and then submit a new batch
  manifest hash. The active immutable batch is never edited in place.

## Experiment Design

### Scientific Question

Does a 20-step joint Stage2 intervention at the FRAC25 beta `0.1` Stage1 step40
plateau, followed by a 40-step Stage1-like Stage3 continuation from extracted
model2, produce a better effective-step100 model than a matched 60-step pure
Stage1 continuation from the same step40 weights?

### Frozen Hypothesis

- `H1`: the admitted `Stage2(20) -> Stage3(40)` arm has a higher final macro
  `pass@1` across HumanEval+, MBPP+, and LiveCodeBench than the matched pure-Stage1
  control and is not worse on at least two of the three datasets.
- `H0`: the chain does not satisfy `H1`. A valid completed experiment that does not
  satisfy `H1` rejects the plateau-breakthrough claim for this exact P40,
  beta `0.1`, lambda `0.8`, 20+40-step configuration; it does not prove that every
  Stage2 intervention is ineffective.

### Frozen Experimental Matrix

| Arm | Initial weights | Objective | Training budget | Training data | Final effective step |
| --- | --- | --- | ---: | --- | ---: |
| `frac25-stage1-control` | FRAC25 beta `0.1` step40 actor | Stage1-like single-model beta `0.1` | 60 steps | exact ordered union of the frozen Stage2 and Stage3 rows | 100 |
| `frac25-stage2` | same FRAC25 beta `0.1` step40 actor | fixed-model2 rollout, joint fused loss, beta `0.1`, lambda `0.8` | 20 steps | frozen Stage2 rows | 60 |
| `frac25-stage3` | model2 extracted from final admitted Stage2 checkpoint | Stage1-like single-model beta `0.1` | 40 steps | frozen Stage3 rows | 100 |

All arms use the same Qwen3-1.7B base identity, prompt/scorer/timeout semantics,
resource profile, response length `8192`, full HumanEval+/MBPP+/LiveCodeBench
validation, sampled-decoding semantic hash, validation dataset hashes, and final
evaluation command. The control and chain consume the same total 3840 training
rows in the same order; only the objective/topology and the Stage2-to-Stage3
handoff differ. Optimizer state is reinitialized for both arms because the retained
step40 artifact contains model state but no optimizer shards.

The manifest, admission bundle, preflight, calibration applicability, run-set
checks, queue, monitor, release checks, and launch command must be regenerated and
independently accepted for all three run IDs before this Goal can start. No training
may start against the predecessor two-run bundle.

### Primary Estimand And Decision Rule

- The primary metric is the unweighted macro mean of final full-validation
  `val-core/<dataset>/acc/pass@1` over HumanEval+, MBPP+, and LiveCodeBench.
- Secondary evidence is the three-dataset delta vector, Stage2-to-Stage3 retention,
  extraction/timeout rates, and execution/recovery integrity. Secondary evidence
  explains the result but cannot override the frozen primary rule.
- `HYPOTHESIS_SUPPORTED`: both arms and all evaluations are valid; chain macro
  `pass@1` is strictly greater than control macro `pass@1`; and chain final
  `pass@1` is greater than or equal to control on at least two datasets.
- `HYPOTHESIS_REJECTED_FOR_CONFIGURATION`: both arms and all evaluations are valid,
  but the support rule is not satisfied. Mixed gains, a macro tie/loss, or a gain
  concentrated in one dataset all reject this exact configuration.
- `INCONCLUSIVE`: an arm, handoff, evaluator, binding, or required artifact is
  invalid or incomplete. Operational failure never becomes scientific rejection.
- The decision is computed once from the frozen final checkpoints. No best-step
  selection, threshold adjustment, dataset removal, rerun, or interpretation-driven
  configuration change is allowed after metrics are visible.

### Outcome And Failure Matrix

| Observed outcome | Required action | Scientific decision |
| --- | --- | --- |
| Pre-launch binding, control construction, storage, dependency, or scorer failure | Do not start Ray; return to Readiness or classify the blocking finding | `INCONCLUSIVE` |
| Valid control completes; Stage2 or extraction fails | Preserve control and failure evidence; do not substitute another chain | `INCONCLUSIVE` |
| Valid Stage2 completes; Stage3 fails before wrapper/training work solely at a certified admission boundary | Preserve the old terminal item and artifacts; create a new Stage3-only identity only through the V14 certified handoff protocol | `INCONCLUSIVE` until the new Stage3 completes |
| Valid Stage2 completes; Stage3 fails after wrapper/training work, uses invalid provenance, or lacks V14 certificate proof | Preserve all artifacts; no continuation or replacement checkpoint | `INCONCLUSIVE` |
| Both final arms valid and support rule passes | Produce bound decision/report; release remains separately gated | `HYPOTHESIS_SUPPORTED` |
| Both final arms valid and support rule fails | Produce bound decision/report without adding P60/FRAC50 | `HYPOTHESIS_REJECTED_FOR_CONFIGURATION` |
| Metrics are mixed but valid | Apply the frozen rule; record the full delta vector | support or reject by rule, never `INCONCLUSIVE` merely because the result is inconvenient |
| Local experiment valid but release/publication fails | Preserve local scientific decision; record release/publication state separately | unchanged |
| Legacy `1783777744` artifacts conflict with a new run | Fail closed; quarantine by identity, never delete automatically | `INCONCLUSIVE` until conflict is resolved |

## Scope

### Included

- Revalidate the immutable admission bundle immediately before launch.
- Launch the exact admitted queue in tmux and monitor persisted Python-owned state.
- Complete Stage2, verify checkpoint and validation completion, extract model2, and
  bind provenance before Stage3.
- Complete Stage3 and verify required validation, checkpoints, metrics, provenance,
  cleanup, and final execution state.
- Complete the matched pure-Stage1 control and verify that its data, compute,
  evaluator, model, and resource identities match the chain contract.
- Produce `experiment_decision.json` and `experiment_report.md` containing the
  frozen hypothesis, both final metric vectors, macro values, exact deltas, decision
  code, execution validity, artifact hashes, and explicit limits of inference.
- Apply the frozen failure-to-next-experiment policy and record structured failures,
  cleanup, skipped phases, and batch transitions.
- Run the training-result release gate after successful local completion.
- Publish eligible successful results to the local registry and existing W&B project
  only if separately authorized by the user and the release gate passes.
- Obtain independent final acceptance of execution truth, decision correctness, and
  release correctness.

### Excluded

- Any post-admission change to code, manifest, resource profile,
  calibration/preflight/acceptance artifacts, implementation tree, Readiness
  evidence commit, run set, hyperparameters, source checkpoints, data, validation
  breadth, decision rule, or recovery policy.
- Implementation of the matched-control manifest/launcher and production changes
  needed to obtain the new three-run admission bundle; those changes belong to a
  renewed Readiness contract before this execution Goal starts.
- P60, FRAC50, a 27-run queue, additional seeds, retries outside frozen policy,
  broader sweeps, or a replacement experiment.
- Generalization beyond the exact P40 configuration, causal claims beyond the
  matched control, paper claims, or claims about P60/FRAC50/other seeds.
- Publication of failed, incomplete, release-ineligible, or unapproved runs.
- Modification, deletion, or staging of protected user assets.

## Execution Contract

- Human launch intent and a valid admission bundle are both required; neither is
  sufficient alone.
- The accepted admission bundle must be a new three-run bundle produced after Plan
  v2; the predecessor two-run bundle is invalid for this experiment.
- Before Ray starts, exact Plan, commit, manifest, profile, calibration, preflight,
  acceptance, run-set, source-artifact, and protected-status bindings are rechecked.
- The implementation binding is independently recomputed with
  `stage123-implementation-boundary-v1` over complete Git tree objects for
  `config/experiment_execution`, `scripts`, and `verl`, plus the entire recipe
  gitlink commit, before admission validation. A boundary-manifest change,
  covered-root tree change, gitlink/recipe HEAD mismatch, tracked or untracked dirty
  path, missing canonical record, or SHA mismatch blocks launch.
- Stage2 must complete its admitted final step and required validation before model2
  extraction. Extraction must produce content/provenance hashes bound to Stage2.
- Stage3 starts only from the admitted extracted model2 and verified provenance.
- Python persisted state is execution authority. Monitor output, tmux presence, W&B,
  and registry state are observations, not substitute authority.
- Failed items are terminal for the active batch. Persisted records bind the failure
  code, cleanup result, skipped phases, next batch item, manifest SHA256,
  implementation-tree SHA256, bundle SHA256, batch-manifest SHA256, start/end time,
  and operator-control state. No item receives an automatic retry or resume.
- Any code/config/artifact binding change is `AC_CHANGE`; stop and return to a new
  Readiness candidate rather than patching during execution.
- Failed or incomplete runs remain local diagnostic evidence. Release requires the
  repository release gate to pass.
- Scientific decision and release decision are separate authorities. A valid local
  experiment can support or reject the hypothesis even when publication is not
  authorized or transport fails; an invalid experiment cannot be made conclusive by
  release metadata.

## Acceptance Criteria

### AC-01 - Launch Uses The Exact Admission Bundle

- Given the accepted immutable bundle and explicit user launch authorization,
- When pre-launch validation runs,
- Then every binding matches, protected assets remain untouched, no conflicting run
  exists, and only the exact primary launch command can start Ray. Full admission is
  required before `item_started`; later phases use only the live single-use capability
  plus a fast static-binding and host-health check.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --repo-root /data-1/code/verl`
- Additional verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl`
- Expected evidence: pre-launch canonical tree comparison, admission validation
  report, and immutable bundle hash.

### AC-02 - Only The Frozen Three-Run Matrix Executes

- Given persisted execution events and system process evidence,
- When the queue runs,
- Then only `frac25-stage1-control`, `frac25-stage2`, and `frac25-stage3` execute;
  no P60, FRAC50, 27-run, hidden ablation, or additional run starts.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --format json`
- Expected evidence: admitted run IDs matched to state, tmux, container, checkpoint,
  and metrics records.

### AC-03 - Matched Stage1 Control Completes With Required Evidence

- Given the admitted step40 source and frozen control contract,
- When the pure-Stage1 control reaches effective step100,
- Then it has exactly 60 training steps, the exact ordered union of Stage2 and Stage3
  rows, the shared 8192 response limit and validation semantics, complete final
  checkpoint/metrics, and no chain-specific objective or model2 extraction.
- Verification command:
  `CONTROL_RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage1-control --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$CONTROL_RUN_NAME"`
- Expected evidence: completed persisted control state, data-order/provenance hashes,
  final checkpoint, full validation vector, and release-gate diagnostic result.

### AC-04 - Stage2 Completes With Required Evidence

- Given the admitted Stage2 configuration,
- When Stage2 reaches terminal state,
- Then its final step, full required validation, checkpoint, metrics, resource
  profile, manifest binding, and cleanup state are complete and structured. A V14
  Stage3-only recovery may consume it only after a certificate binds all of that
  evidence and proves that the prior Stage3 failure was pre-training work.
- Verification command:
  `STAGE2_RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage2 --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$STAGE2_RUN_NAME"`
- Expected evidence: persisted completed state, checkpoint/metrics paths, validation
  evidence, and release-gate diagnostic result without premature publication.

### AC-05 - Model2 Extraction And Provenance Are Exact

- Given the completed Stage2 checkpoint,
- When model2 is extracted,
- Then the extracted artifact and provenance bind source run, source checkpoint,
  final Stage2 step, submodel index, manifest/profile/commit hashes, output hash, and
  release eligibility; any mismatch blocks Stage3. A V14 handoff certificate must
  bind the extracted-model2 tree hash and the Stage2 provenance/metrics hashes before
  a new Stage3-only identity can reference the artifact.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_stage123_preflight_model_identity.py`
- Expected evidence: runtime provenance validation plus artifact hashes.

### AC-06 - Stage3 Completes From The Admitted Extraction

- Given the verified extracted model2 and provenance,
- When Stage3 runs,
- Then it uses only that input, reaches its admitted final step, completes required
  validation, writes checkpoint/metrics/provenance, and reaches a clean terminal state.
- A V14 Stage3-only identity is permitted only when its independent admission binds a
  certified completed Stage2 handoff, a new state root/monitor/provenance root, and
  the exact original Stage3 scientific configuration. It is not a retry or resume of
  the terminal Stage2->Stage3 item.
- Verification command:
  `STAGE3_RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage3 --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$STAGE3_RUN_NAME"`
- Expected evidence: completed persisted state and matching runtime artifacts.

### AC-07 - Failure Routing Obeys The Frozen Policy

- Given any interruption or failure,
- When failure classification and batch routing run,
- Then the failed atomic experiment becomes terminal and inconclusive, its remaining
  phases are skipped, owned resources are cleaned, and the next pre-authorized item
  starts unless a frozen batch-stop invariant applies; no parameter change, retry,
  checkpoint resume, test weakening, or benchmark downscope occurs.
- Wall-clock age after `item_started` is not a phase failure. A changed training-plane
  identity, bundle mismatch, certified-control mismatch, treatment-admission mismatch,
  binding mismatch, or failed required host-health check fails closed before the
  affected training wrapper is invoked. A child process that only performs admission
  or pre-wrapper filesystem checks is not training work and must be reported as such.
- The sole V14 exception to ordinary terminal-item routing is a certified completed
  Stage2 handoff after a provably pre-training Stage3 admission failure. It creates a
  new Stage3-only execution identity; it never mutates, resumes, retries, or rewrites
  the old item, and all other terminal Stage3 failures remain terminal.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_failure_classifier.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: terminal failure record, cleanup, skipped-phase records,
  deterministic next-item or batch-stop decision, and unchanged hashes.

### AC-08 - Monitoring And Cleanup Reflect Execution Truth

- Given queue, monitor, container, GPU, checkpoint, and persisted-state evidence,
- When each phase terminates,
- Then monitor status matches Python-owned state, owned resources are cleaned, and no
  stale tmux/container/GPU process falsely represents completion or remains leaked.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_validation_deadline_cleanup.py`
- Expected evidence: live `tmux`, Docker, `nvidia-smi`, state, log, checkpoint, and Ray
  failure-marker audit.

### AC-09 - Scientific Decision Is Frozen And Reproducible

- Given valid final control and chain checkpoints with complete full-validation
  metrics,
- When the experiment decision is rendered,
- Then `experiment_decision.json` and `experiment_report.md` bind the Plan hash,
  three-run manifest/bundle, source step40 artifact, final checkpoints, evaluator
  semantics, metric vectors, macro values, delta vector, decision rule, one of the
  three allowed decision codes, and limits of inference.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-primary-chain-execution/tools/render_experiment_decision.py validate --decision docs/joint_training/goals/stage123-primary-chain-execution/experiment_decision.json --plan docs/joint_training/goals/stage123-primary-chain-execution/plan.md --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml`
- Expected evidence: independently recomputed per-dataset values, macro/delta math,
  exact decision code, rejection of metric mutation or semantic mismatch, and no
  best-step/threshold selection after results are known.

### AC-10 - Release Is Conditional And Accurate

- Given successful or failed execution outcomes,
- When release checks run,
- Then failed/incomplete outcomes are not published; successful outcomes are eligible
  only after release-gate PASS; registry/W&B publication occurs only under explicit
  publication authorization and is verified without checkpoint/artifact payload sync.
- Verification command:
  `for RUN_ID in frac25-stage1-control frac25-stage2 frac25-stage3; do RUN_NAME=$(REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py run recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id "$RUN_ID" --field run_prefix) && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/training_result_release_gate.py check --run-name "$RUN_NAME"; done`
- Expected evidence: release decisions, registry rows and W&B sync markers only for
  eligible authorized success, or structured local-only completion if publication
  was not authorized/available.
- Publication transport failure after successful local execution is recorded as
  `local_execution_complete_publication_blocked`; it never rewrites training state
  as failed.

### AC-11 - Independent Acceptance Verifies Execution And Decision

- Given committed admitted configuration and terminal execution evidence,
- When a fresh independent reviewer audits the run from local authoritative state,
- Then AC-01 through AC-10 are individually PASS, no unapproved run/config change
  occurred, the scientific decision is recomputed from authoritative evidence, and
  acceptance does not generalize beyond the frozen configuration.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution`
- Expected evidence: reviewer-owned acceptance bound to Plan, admission bundle,
  execution state, artifacts, experiment decision/report, release decision, and
  terminal execution-evidence commit.

### AC-12 - Unified Validation Decoder And Fresh Rerun Identity

- Given the V15 validation-protocol amendment,
- When the new three-phase chain is admitted and executed,
- Then every main online validation point for Control Stage1, Stage2, and Stage3
  uses exactly `do_sample=True`, `temperature=0.2`, `top_p=0.95`, and `n=1`; the
  training rollout remains explicitly `temperature=1.0` and `top_p=1.0`; the shared
  profile serializes and hashes all decoder fields; and a CPU-only phase-contract
  audit fails closed on any unapproved drift. The V15 chain has one fresh root, fresh
  run names, fresh manifest/admission/profile identities, and executes Control ->
  Stage2 -> extraction -> Stage3 without V13/V14 reuse.
- Verification commands:
  `bash -lc 'source recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh && stage123_profile_snapshot'`
  and
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_validation_protocol.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: one emitted profile serialization containing the exact frozen
  decoder values; a rendered fresh three-run manifest; a passing phase-contract
  audit; and no referenced source path beneath the historical V13/V14 runtime roots.

## Milestones

1. Revalidate a newly and independently accepted three-run Readiness bundle,
   quarantine the legacy attempt by identity, and record explicit launch authorization.
2. Launch and monitor the matched pure-Stage1 control through terminal validation.
3. Launch and monitor admitted Stage2 through terminal validation and cleanup.
4. Extract model2 and independently verify provenance before Stage3.
5. Launch and monitor admitted Stage3 through terminal validation and cleanup.
6. Verify full local artifact/result integrity and recovery history, then render and
   independently recompute the frozen experiment decision.
7. Apply release gate and, only if separately authorized, publish eligible results.
8. Obtain independent final execution-and-decision acceptance.

Milestones are hard ordered. Normal successful transitions are autonomous. Stage2
may start only after the control has completed so metric visibility cannot trigger a
control change; Stage3 cannot start before model2 provenance passes.

## Feasibility Probes

- Batch-core reuse probe: `scripts/experiment_execution_core.py` already owns atomic
  state persistence, append-only execution events, child start/poll/cleanup, deadline
  handling, and terminal states; the public Stage123 queue already delegates to it,
  and `scripts/stage123_manifest_monitor.py` already consumes its persisted events.
  The missing capability is ordered multi-experiment routing and operator control in
  that same authority, not another queue/monitor implementation.
- AC-02 run-count probe: the predecessor manifest currently validates with exactly
  two runs and therefore cannot satisfy Plan v2. The required three-run manifest and
  fresh admission are explicit pre-launch work, not assumed evidence.
- AC-03 source probe: the retained FRAC25 Stage1 root contains model-only
  `global_step_40/actor` shards and `data.pt`, but no optimizer shards. Plan v2
  therefore freezes optimizer reinitialization for both control and chain rather
  than claiming exact optimizer-state continuation.
- AC-01/07 recovery probe: the July 15, 2026 control required 77 minutes, so an
  already admitted atomic item must not fail solely because a phase begins after an
  arbitrary wall-clock freshness interval. The preserved `frac25-stage2` evidence
  records `child_id=1766588` and `pending -> running -> failed`; the queue log proves
  that this child stopped in `stage123_phase_adapter.py` at the pre-wrapper existing-
  checkpoint guard, before `subprocess.run()` could invoke the Stage2 wrapper. This
  is a `child-started, pre-training-work` failure, not a pre-child failure. The
  recovery certificate must bind that immutable evidence, prove the absence of new
  Stage2 provenance/metrics/extraction/checkpoint artifacts, and preserve the legacy
  checkpoint as excluded diagnostic evidence. Control-plane changes use CPU tests;
  only a changed frozen training-plane identity requires renewed GPU qualification.
- AC-09 historical-baseline probe: the prior Stage1 step100 output log SHA256 is
  `7702c4c25464403e8db6b04d24fe726fa626d440d3d88b89ddaba777fd1df7fd`
  and records HumanEval+ `0.5`, MBPP+ `0.48677248677248675`, LiveCodeBench
  `0.1875`, macro `0.3914241622574956`; its config SHA256 is
  `1289234de19981d47fc287941866dd4dd663d9ed07d85566d8604ba3b48b38c0`
  and uses response length `4096`. This measured semantic mismatch is why it is
  excluded as the control rather than silently reused.
- AC-09 threshold waiver: the decision rule is comparative and deterministic; it
  does not promise an absolute score, throughput, latency, memory, or storage budget.
  Strict macro improvement plus the two-dataset non-regression condition is the
  frozen scientific estimand, not an environment performance budget.
- AC-03/04/06 step counts and 3840-row total are experiment assignments derived from
  batch size 64 and the frozen 20+40-step chain. They are not performance budgets.

## Progression Policy

- `AUTO_ADVANCE`: Plan validation, independent Plan review, verification of the
  separately completed three-run Readiness and batch-orchestration prerequisites,
  preflight/admission validation, control launch after explicit launch authorization,
  normal hard-ordered phase transitions, deterministic failure-to-next-experiment
  routing, evidence collection, frozen-rule decision rendering, focused validation,
  milestone review, runtime validation, release-gate checks, and final acceptance
  request.
- `USER_DECISION`: starting or amending the prerequisite Readiness work; starting
  this Goal; formal GPU training launch; any production,
  manifest, run-set, data, evaluator, resource-profile, fallback, or decision-rule
  change; destructive cleanup of legacy/user artifacts; a frozen shared batch-stop
  invariant; any request to repair/retry/requeue a failed experiment; protected-asset
  risk; publication to registry/W&B; adding P60, FRAC50, seeds, reruns, or broader
  benchmarks; contradiction, AC change, or convergence failure.
- The launch decision authorizes only the frozen three-run local experiment and
  Milestones 1-6 and 8. Milestone 7 publication remains separately gated.

## Runtime Contract

- Start only after this Plan is `READY`, a new three-run Readiness contract is
  complete, and the user explicitly authorizes formal primary-chain execution using
  that accepted bundle. The completed predecessor two-run Readiness Goal is not
  sufficient.
- One launch authorization covers local execution Milestones 1-6 and final review
  Milestone 8. Registry/W&B publication in Milestone 7 requires separate explicit
  publication authorization unless the launch authorization explicitly includes it.
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
- Deterministic failure-to-next-experiment routing within the frozen batch manifest
  is autonomous; failed experiments are never modified or resumed.
- Stop the batch for any shared batch-stop invariant, binding/config/code change,
  convergence failure, protected-asset risk, new run/sweep, required benchmark
  downscope, or need for an unapproved external service.
- The implementer cannot self-review or self-accept.

## Reviewer Contract

- Plan review verifies one matched experiment outcome, a fair control, an exact
  estimand, exhaustive outcome handling, and no hidden ablations or post-result
  threshold selection.
- Milestone reviews use persisted state plus live tmux/process/container/GPU/log/
  checkpoint/Ray evidence and verify exact bindings and phase ordering.
- Final review independently audits execution truth, recovery, provenance, the
  recomputed scientific decision, release, protected assets, and absence of
  unapproved runs.
- Scientific opinions outside the frozen hypothesis and inference limits are
  deferred and cannot replace or weaken the deterministic decision.
- `ACCEPTED` requires every applicable AC to be `PASS`.

## Verification Commands

- Plan: `goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution`
- Runtime: `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution`
- Plan hash: `sha256sum docs/joint_training/goals/stage123-primary-chain-execution/plan.md`
- Protected status: `git status --short`

## Deferred Follow-ups

- P60, FRAC50, broader sweeps, additional seeds, mechanism generalization, and paper
  claims require separate Plans and fresh readiness acceptance.

## Plan v13 Certified Control-Reuse Amendment

This amendment supersedes the v12 recovery paragraph and every earlier reference to a
Unix-socket/nonce or hostile same-root authorization claim.

### Frozen failure boundary

The preserved July 15 Stage2 state is immutable evidence that a child started:
`child_id=1766588`, `pending -> running -> failed`, and return code `1`. It must never
be described as a pre-child failure. Control reuse is eligible only for the narrower,
recomputable **child-started, pre-training-work** boundary. The certificate must prove
all of the following from preserved files and bound source hashes:

1. the old batch is terminal `completed_with_failures`, the control phase succeeded,
   and the old Stage2 phase failed exactly once with the preserved child state;
2. the preserved queue log contains the old Stage2 adapter exception from the
   existing-checkpoint guard; and the bound adapter source orders that guard before
   wrapper invocation;
3. no new Stage2 checkpoint, metrics, validation output, provenance, extraction, Ray
   workload, optimizer step, or GPU training workload was produced by that failed
   child. The pre-existing legacy Stage2 checkpoint is recorded by path and hash as
   excluded diagnostic evidence, never treated as output of the failed child; and
4. any absent or contradictory proof rejects reuse. Old batch state, control evidence,
   Stage2 failure evidence, and the legacy diagnostic checkpoint remain preserved.

### Certified control and treatment-only preparation

A certified control is a completed Stage1 control whose certificate verifies and binds:
its provenance hash, checkpoint-tree hash, metrics hash, final validation hash,
manifest hash, train-data hash, resource-profile hash, evaluator/validation identity,
and frozen training-plane identity. The certificate stores the original accepted
admission-bundle and acceptance-report hashes and the exact old failure-evidence hashes.

The recovery is not a retry and does not consume the control final weights as Stage2
input. A new treatment-only execution is prepared from the original admitted P40 source
with Stage2 then extraction then Stage3. It has a unique execution identity, treatment
manifest with unique operational run prefixes, treatment-only admission file, batch
manifest, state root, monitor path, and provenance root. Each of these binds the
certificate SHA256 and the original scientific manifest. The old root is read-only
evidence and cannot be resumed, overwritten, or adopted as the new root.

The prepared treatment admission is not launch authorization. It validates all CPU
recomputable evidence and marks itself `prepared_not_authorized`; a later explicit
training authorization must supply the fresh host/admission evidence required for an
actual GPU launch. Direct phase invocation validates this full treatment admission;
batch execution validates the same file before child start. Environment variables and
state files carry paths only and are not represented as a security primitive.

The final decision JSON and report must contain `control_reuse` with the certificate
SHA256, old control identity, old failure boundary, and a plain-language statement that
the completed matched Stage1 control was reused as baseline evidence while treatment
Stage2 -> extraction -> Stage3 executed under a new identity.

### Treatment-only requalification clarification

When a committed implementation repair changes only the Stage2/Stage3 treatment
execution wrappers, bounded zero-step requalification uses the exact ordered phase set
`stage2,stage3`. It never invokes Stage1, creates no control checkpoint, produces zero
optimizer/training steps, and cannot replace or modify the certified completed control.
The calibration renderer and result validator must reject every other subset.

## Plan v14 Certified Stage2-Handoff Amendment

This amendment resolves `F-EX-M2-25`. It is limited to the concrete terminal item
`treatment-reuse-20260716T0317Z`: its Stage2 child completed 20 admitted steps and
produced `stage2_final_joint`, `stage2_final_model2`, metrics, and provenance; its
Stage3 child exited at treatment admission before the Stage3 wrapper or training work
because the now-removed host-facts age gate rejected the bound facts. The old state,
queue log, admission, checkpoints, and artifacts remain immutable terminal evidence.

### Certified Stage2-Handoff Boundary

A handoff is eligible only when a deterministic certificate independently verifies:

1. the source item is terminal and its Stage2 state completed exactly once with no
   retry/resume;
2. the Stage2 provenance is release-eligible and its manifest hash, train-data hash,
   final step, metrics hash, extracted-model2 path, and extracted-model2 tree hash all
   match the preserved source files;
3. the source Stage3 state failed exactly once before training work, and the bound
   queue log proves failure at the treatment-admission boundary; any Stage3 checkpoint,
   metrics, validation, provenance, Ray workload, optimizer step, or GPU training
   evidence rejects handoff; and
4. the original treatment manifest, control-reuse certificate, admission, host-facts
   binding, and resource-profile/training-plane identities remain recomputable and
   mutually consistent.

The certificate must fail closed on every missing, changed, or contradictory path.
It stores the source paths and SHA256/tree hashes, the old terminal-state hashes, the
old failure boundary, and the exact extracted model2 path. It does not grant launch
authorization.

### New Stage3-Only Identity

Only a passing certificate may prepare a new execution identity containing exactly
`frac25-stage3`. Preparation must create a new manifest with the original Stage3
scientific settings unchanged, bind `STAGE2_MODEL2_PATH` and `STAGE2_PROVENANCE_FILE`
to the certified source paths, and create a separate admission file, batch manifest,
state root, monitor path, and provenance root. It must not copy, delete, or modify any
old state or checkpoint. Direct Stage3 invocation and batch execution both validate
the new admission and certificate; authorization repeats host-facts schema/`ok`,
host-facts hash, GPU inventory, and resource-profile identity checks without any
wall-clock TTL.

The final decision/report must disclose both the certified control reuse and certified
Stage2 handoff reuse, including their certificate hashes and the fact that Stage3 ran
under a new identity after a pre-training admission failure. A Stage3 training,
provenance, binding, or validation failure after this new identity starts is terminal
and cannot create another handoff.

### V14 Verification

- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_control_reuse.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py`
- A CPU fixture must prove: an eligible completed Stage2 plus pre-training Stage3
  failure creates an exact Stage3-only admission; altered Stage2 provenance/model tree,
  post-wrapper Stage3 evidence, invalid host facts, or an attempted old-root reuse all
  fail closed.
- Before GPU launch, `batch-validate` must accept the new one-phase manifest and
  reject any manifest containing Stage2 or an old state root.

## Plan v15 Unified Validation Protocol And Fresh Rerun Amendment

This amendment resolves `F-EX-PLAN-09` under the user's explicit authorization. It
supersedes V13/V14 reuse only for the new experiment: V15 does not reuse the prior
completed control, Stage2, extracted model2, or Stage3. It runs a fresh atomic
three-phase matrix after a new admission bundle is accepted. The old runtime ledgers,
metrics, provenance, certificates, and final reports remain diagnostic evidence and
must state that their cross-phase main validation decoder was inconsistent.

### Frozen Decoder Contract

The main online curve is a low-temperature one-sample code-generation measurement:

```text
training rollout: do_sample=True, temperature=1.0, top_p=1.0
main validation: do_sample=True, temperature=0.2, top_p=0.95, n=1
```

All three phases use the same full HumanEval+, MBPP+, and LiveCodeBench validation
files, prompt eligibility filter, evaluator/reward implementation, timeout policy,
maximum response length, and main validation decoder. The shared Stage123 resource
profile is the sole owner of these fields; every wrapper sources it before admission
and must not independently default a contradictory validation value. The profile
serialization, hash, manifest binding, admission bundle, preflight report, and
implementation identity all change for V15.

`do_sample=False, n=1` greedy evaluation is a required terminal confirmation for
candidate checkpoints, not the main online curve. Sampled high-temperature or
multi-sample metrics are diagnostic/offline evidence only and cannot replace the
frozen main decision vector.

### Pre-Run Consistency Audit

Before any V15 GPU work, a CPU-only audit must render the fresh manifest and verify:

1. the decoder contract above is serialized once and identical for Stage1, Stage2,
   and Stage3;
2. rollout decoder, `beta`, learning rate, data seed/shuffle policy, batch/length
   profile, validation files/hashes, validation cadence, evaluator identity, and
   timeout policy are identical unless the manifest declares the difference as the
   intended phase treatment;
3. the only allowed model-topology differences are Stage2 joint fixed-model2 rollout
   with the frozen fusion setting and Stage1/Stage3 single-model topology; and
4. no new root references any V13/V14 control, Stage2, extracted-model2, Stage3, or
   state root as an input.

Any other difference is a finding and blocks launch until classified and reviewed.

### Cleanup Boundary

After `tmux`, container, process, and GPU checks prove no active workload, V15 may
delete only agent-generated completed calibration, dry-run, probe, and obsolete
Stage123 scratch roots under `/data-1/tmp/verl_agent_scratch/experiment_workflow/`.
It must retain the append-only goal ledgers, source manifests, small provenance and
metrics evidence, protected assets, `/data-1/wandb_runs`, and all user-owned paths.
It must inventory deleted roots and reclaimed bytes in the runtime ledger. Existing
checkpoint/model artifacts remain outside this cleanup authorization until a later
path-level retention decision.

### V15 Review And Execution Order

1. Independently review this amendment before implementation.
2. Implement the centralized decoder contract and focused regression tests.
3. Validate the amended Plan, runtime ledger, rendered manifest, profile hash, and
   CPU-only phase-contract audit; obtain an independent implementation review.
4. Perform the permitted scratch cleanup and record exact paths/bytes.
5. Obtain a new V15 admission bundle and independently review the fresh pre-run
   evidence.
6. Launch exactly one fresh Control -> Stage2 -> extraction -> Stage3 chain only
   after its explicit GPU admission is recorded; no reuse, retry, resume, parameter
   tuning, external publication, or external service is allowed.
