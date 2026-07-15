# Stage123 Execution Readiness

- Goal ID: `stage123-execution-readiness`
- Plan version: `9`
- Plan status: `REVIEW_PENDING`
- Serial position: `3 of 4`
- Prerequisite Goal: `calibration-qualification` completed with
  `calibration_result.json` decision `passed`

## Outcome

Produce one immutable, independently accepted Stage123 admission bundle proving
that the matched FRAC25/P40 three-run experiment (`frac25-stage1-control`,
`frac25-stage2`, `frac25-stage3`) is deployable but not yet started, after fresh
calibration requalification of the amended production identity.

## Starting Evidence

- The completed Calibration Qualification Goal supplies the unchanged primary
  manifest, resource profile, implementation-tree SHA256, calibration evidence
  commit, and authoritative
  `calibration_result.json`.
- Current authority classes are `calibration_result`, `preflight_result`, and
  `acceptance_report`; legacy receipts and adoption artifacts cannot authorize.
- Calibration acceptance guarantees that manifest/admission/queue/monitor/recovery
  production migration was completed before the probe and is bound by the passed
  `implementation_tree_sha256`.
- Plan v8 and its accepted bundle are immutable historical evidence for the prior
  two-run chain. Primary Chain Plan v5 invalidates that bundle because a scientific
  decision requires a matched pure-Stage1 control. User decision
  `EX-MATCHED-CONTROL-READINESS-01` authorizes this bounded Plan v9 amendment and a
  fresh three-run admission; the old bundle cannot authorize formal training.

## Scope

### Included

- Preserve the accepted Plan v8 result as historical evidence while changing only
  the manifest/run-set and production surfaces required to add the frozen matched
  pure-Stage1 control from Primary Chain Plan v5.
- Produce one canonical host-owned `host_facts.json`, then one fresh
  `preflight_result.json` for models, datasets, scorer dependencies,
  storage, container runtime, source checkpoints, provenance, machine facts, and
  conflicting execution.
- Verify without production changes that queue lifecycle, deadline, cleanup,
  persisted state, and resume are Python-owned and the monitor consumes the same
  normalized manifest and persisted event/state authority.
- Validate one immutable admission bundle containing the manifest, resource profile,
  calibration result, preflight result, reviewer acceptance report, hashes, and
  exact launch command.
- Run dry-run, fake-child, mutation, and non-weight-updating initialization checks.
- Implement only the matched-control changes required by AC-01 through AC-06:
  control manifest entry and wrapper binding, three-run calibration/preflight and
  admission identity, unified batch binding, monitor inventory, and focused tests.
- After those production changes, require a fresh bounded zero-training-step
  calibration result before admission. GPU requalification is a `USER_DECISION`.

### Excluded

- Any formal weight-updating Stage1, Stage2, or Stage3 training.
- P60, FRAC50, a 27-run queue, broader sweep, or any run-set change beyond adding
  the exact `frac25-stage1-control` frozen by Primary Chain Plan v5.
- Rewriting the completed predecessor calibration result or treating it as current
  after the implementation identity changes.
- Trainer, resource profile, recovery policy, hyperparameter, Stage2/Stage3 wrapper,
  or workload changes beyond the exact matched-control construction.
- Real W&B, WxPusher, Hugging Face, GitHub, registry mutation, or publication.
- New receipts, adoption artifacts, per-AC wrappers, or review-document chains.
- Modification, deletion, or staging of the three protected user assets named in
  the predecessor Goal.

## Architecture Contract

- The manifest, resource profile, passed calibration result, and
  `implementation_tree_sha256` are immutable inputs.
- `implementation_tree_sha256` always means the predecessor's frozen
  `stage123-implementation-boundary-v1` canonical JSONL over the complete Git tree
  objects for `config/experiment_execution`, `scripts`, and `verl`, plus the entire
  recipe gitlink commit. Readiness uses the exact boundary manifest at
  `config/experiment_execution/stage123_implementation_boundary_v1.json`, reruns
  `scripts/implementation_tree_identity.py`, and compares canonical JSONL and SHA256.
  A covered-root, recipe-commit, or boundary-manifest change returns to Calibration.
- Generic admission validation checks shared bindings and result classes without
  hard-coding Stage123 run facts.
- Experiment-specific deployability policy is versioned and manifest-owned.
- The admitted run set is exactly `frac25-stage1-control`, `frac25-stage2`, and
  `frac25-stage3`, in that order.
- `scripts/experiment_execution_core.py` owns execution state, child lifecycle,
  deadline, cleanup, interruption, and resume. Queue shell performs configuration
  and delegates; monitor consumes persisted events/state.
- `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1` expresses human launch intent but is never
  sufficient authority without a valid immutable admission bundle.
- `scripts/execution_results.py admission validate` is the named generic interface.
  It consumes `--manifest`, `--resource-profile`, `--calibration-result`,
  `--preflight-result`, `--protected-baseline`, and optionally
  `--acceptance-report`; verifies shared
  bindings, freshness, implementation tree, evidence commits, and run set; and
  renders canonical `admission_bundle.json` plus `bundle_sha256`.
- Before final review the bundle is a candidate without acceptance. The independent
  reviewer writes `acceptance_report.json`; final validation includes it and renders
  the immutable accepted bundle consumed by Execution.
- `acceptance_report.json` binds the same manifest, profile, implementation tree,
  calibration result, preflight result, protected-baseline SHA256, Readiness evidence
  commit, run set, and Plan.
- Any binding mutation, stale result, extra run, dirty protected asset, or active
  conflicting execution fails closed before Ray or training starts.
- Preflight is one fast current-environment invocation. It consumes the already
  qualified timing policy and does not repeat calibration timing repetitions.
  `scripts/stage123_host_facts.sh` atomically writes one canonical
  `host_facts.json` before the mandatory project wrapper. It records schema/artifact
  type, timestamp, exact `REPO_HOST`, Docker image reference and immutable ID, tmux
  session snapshot, calculated Stage123 conflicts, mount/storage facts, and producer
  command outcomes. Container preflight accepts exactly one `--host-facts` argument,
  validates freshness/schema/bindings, records `host_facts_sha256`, and never invokes
  Docker or tmux. No environment variable or second host-fact path is authority.
- One common admission validator is used by candidate construction,
  `admission validate --bundle`, accepted-bundle validation, and `render-launch`.
  Every path recomputes the current implementation tree, rejects dirty covered roots
  or recipe mismatch, compares current `HEAD` with `readiness_evidence_commit`,
  recomputes every referenced input hash, performs the canonical protected-baseline
  comparison, reapplies freshness/run-set checks, and
  rejects before launch rendering.
- `acceptance_report.json` uses schema v1 and binds Goal/Plan ID, Plan version/hash,
  independent reviewer identity, model `GPT-5.5`, reasoning effort `medium`, candidate
  and Readiness commits, exact run IDs, candidate bundle hash, all input hashes, and
  protected-baseline SHA256, and AC-01 through AC-08 verdicts. Every verdict must be
  exactly `PASS`.
- `bundle_sha256` hashes only the canonical candidate bundle, excluding itself,
  acceptance data, and `acceptance_report_sha256`. The acceptance report binds that
  candidate hash; `acceptance_report_sha256` binds its exact bytes. Both hashes are
  independently recomputed before accepted-bundle authorization or launch rendering.

## Acceptance Criteria

### AC-01 - Re-Aligned Production Identity Is Freshly Qualified

- Given the named compatibility repair changes the frozen production identity,
- When the user-authorized bounded zero-training-step requalification completes,
- Then a new authoritative `calibration_result.json` is `passed` and binds the exact
  post-repair manifest, profile, primary run set, implementation tree, evidence
  commit, and authorization identity; the predecessor result remains immutable
  historical evidence and cannot authorize admission.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py`
- Additional verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl`
- Bounded requalification producer command, executable only after the matching
  `USER_DECISION_RECORDED` event:
  `tmux new-session -d -s stage123_matched_control_requalification "cd /data-1/code/verl && export REPO_HOST=/data-1/code/verl CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CALIBRATION_STATE_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01/state CALIBRATION_DEADLINE_SECONDS=5400 && export CALIBRATION_CHILD_COMMAND_JSON=\"\$(/data-1/verl07/run_train.sh python scripts/render_calibration_probe_command.py --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --resource-profile recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh --phases stage1,stage2,stage3 --repetitions 3 --training-steps 0 --scratch-root /data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01 --execution-run-id stage123_matched_control_requalification_RD-MATCHED-CONTROL-REQUAL-01 --authorization-decision-id RD-MATCHED-CONTROL-REQUAL-01)\" && /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py queue --run-id stage123_matched_control_requalification_RD-MATCHED-CONTROL-REQUAL-01 --state-root \"\$CALIBRATION_STATE_ROOT\" --timeout-seconds 5400 --command-json \"\$CALIBRATION_CHILD_COMMAND_JSON\""`
- Frozen requalification envelope: at most 8 L40S GPUs, at most 90 minutes aggregate,
  at most 3 repetitions per phase, zero optimizer steps, zero formal checkpoints,
  scratch-only outputs, and cleanup evidence. Authorization identity is the matching
  `USER_DECISION_RECORDED.decision_id` plus the exact Plan v9 hash and candidate
  implementation-tree SHA256.
- Authoritative result render command after the execution-core run and producer
  report both reach terminal `passed` state:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py render --run-id stage123_matched_control_requalification_RD-MATCHED-CONTROL-REQUAL-01 --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01/state --latest-probe /data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01/latest-probe.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --resource-profile recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh --implementation-tree docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl --evidence-commit "$(git rev-parse HEAD)" --runtime-ledger docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl --decision-id RD-MATCHED-CONTROL-REQUAL-01 --output docs/joint_training/goals/stage123-execution-readiness/calibration_result.json`
- Authoritative result validation command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json`
- Rendering must bind manifest/profile hashes, current implementation-tree SHA256,
  current evidence commit, exact recorded decision identity, phase/repetition evidence,
  prediction comparison, zero optimizer steps, empty formal-checkpoint lists, and
  terminal cleanup. Any absent or mismatched input renders `blocked`, never `passed`.
- The renderer resolves the report only through the producer-generated
  `latest-probe.json`; requires its `run_root`/report path and SHA256 to match the
  terminal execution-core state for
  `stage123_matched_control_requalification_RD-MATCHED-CONTROL-REQUAL-01`; requires pointer fields
  `schema_version=2`, `run_id`, `authorization_decision_id`, `report_sha256`,
  `generated_at_utc`, `report_started_at_utc`, and `report_completed_at_utc`; and
  requires the decision-specific state/scratch roots to be newly created after the
  matching decision. It rejects stale, reused, non-terminal, cross-run, or
  mismatched reports. It compares UTC wall-clock fields within the producer schema;
  it never compares producer epoch values to execution-core monotonic values.
- Expected evidence: fresh canonical-tree JSONL and SHA256, new passed calibration
  result, and mutation failures for stale predecessor evidence.

### AC-02 - Fresh Preflight Covers Deployability

- Given the admitted primary chain and live host/container environment,
- When preflight runs,
- Then source models, provenance, data, scorer dependencies, checkpoint mount,
  writable physical destination, container image, GPU visibility, machine facts,
  release paths, and conflicting execution are checked with structured outcomes.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py`
- Live command:
  `bash scripts/stage123_host_facts.sh --repo-host /data-1/code/verl --image verl-harness:latest --output docs/joint_training/goals/stage123-execution-readiness/host_facts.json && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python recipe/on_policy_wdl_sft/code_task/stage123_preflight.py --host-facts docs/joint_training/goals/stage123-execution-readiness/host_facts.json --calibration-result docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --output docs/joint_training/goals/stage123-execution-readiness/preflight_result.json`
- Expected evidence: exactly one fresh `preflight_result.json`; no repeated timing
  probe, optimizer step, or formal checkpoint.
- Mandatory wrapper integration command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py`
- Integration evidence must prove host facts are produced outside the container,
  preflight consumes exactly `--host-facts`, container code invokes neither Docker
  nor tmux, no Ray/training child or optimizer step starts, no formal checkpoint or
  Stage123 training session appears, no external service is called, and calibration
  timing repetitions are not repeated.

### AC-03 - Run Set Is Exactly The Primary Chain

- Given manifest rendering, queue dry-run, and monitor inventory,
- When the three surfaces enumerate runs,
- Then all enumerate exactly `frac25-stage1-control`, `frac25-stage2`, and
  `frac25-stage3`; FRAC50, P60, and hidden environment-variable filtering are absent.
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
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py`
- Expected evidence: mutation matrix, zero child calls on blocked inputs, and one
  admitted bundle hash. Candidate, `--bundle`, accepted-bundle, and render-launch
  paths all invoke the same current-checkout validator.

### AC-07 - Readiness Does Not Train Or Publish

- Given dry-run, fake-child, and initialization checks,
- When all readiness verification runs,
- Then no optimizer step, formal checkpoint, training tmux, registry mutation, W&B
  sync, or external publication occurs.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_new_experiment_gate.py`
- Expected evidence: before/after artifact hashes, empty training-child call log, and
  no active Stage123 execution. The mandatory wrapper integration test is required
  evidence for this AC, not merely an optional reviewer probe.

### AC-08 - Launch Command Is Exact And Reproducible

- Given the immutable admission bundle,
- When its launch command is rendered from a clean checkout,
- Then it pins repo path, implementation tree, Readiness evidence commit,
  manifest/profile/result paths and hashes,
  tmux session, primary run set, and explicit human intent variable without secrets.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission render-launch --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-host /data-1/code/verl`
- Expected evidence: reviewer-owned rendered command and deliberate mismatch failures.
  The command is printed only after current checkout/commit, candidate bundle hash,
  acceptance report hash, freshness, input hashes, and all acceptance bindings pass.

### AC-09 - Independent Readiness Acceptance Is Bound

- Given a committed candidate and complete admission inputs,
- When a fresh independent reviewer runs all frozen checks,
- Then AC-01 through AC-08 are PASS and reviewer-owned `acceptance_report.json`
  binds the current Plan, commit, run set, and all admission hashes.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness`
- Expected evidence: schema-v1 report with independent reviewer/model/effort,
  candidate and evidence commits, exact Plan ID/version/hash, every AC-01 through
  AC-08 verdict exactly `PASS`, valid `acceptance_report_sha256`,
  `ACCEPTANCE_COMPLETED=PASS`, valid runtime, and one immutable admission bundle.

### Protected Asset Baseline

The following assets are protected and are intentionally pre-existing untracked,
unstaged baseline paths:

- `.claude/skills/experiment-registry`
- `docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md`
- `test_data/`

Their existing untracked/unstaged status is accepted baseline and is not itself a
failure. This Goal must not modify, delete, replace, or stage them. The Goal-local
fingerprint scaffolding and baseline file already exist before the Milestone 1
transition. Admission must compare path/status fingerprints against the baseline
captured before Milestone 1;
any Goal-attributable content or status change fails closed.
- Canonical baseline capture command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py capture --repo-root /data-1/code/verl --path .claude/skills/experiment-registry --path docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md --path test_data --output docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl`
- Admission comparison command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl`
- Canonical JSONL is sorted by UTF-8 relative path and records path, node type,
  symlink target when applicable, byte size, and SHA256 of regular-file bytes.
  Empty directories and intentionally absent paths receive explicit records. Added,
  deleted, replaced, type-changed, symlink-retargeted, or content-changed nodes fail.
  This is Goal evidence consumed by admission, not a receipt or authority class. The
  capture occurs as the first Milestone 1 action; no production repair starts before
  it passes.
- Candidate construction requires
  `--protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl`.
  The candidate bundle stores the canonical baseline path and SHA256. The same
  baseline path/hash and live comparison are mandatory for candidate construction,
  `admission validate --bundle`, accepted-bundle validation, and `render-launch`.
  `acceptance_report.json` must bind `protected_baseline_sha256`; mutation after
  acceptance fails before launch rendering.

## Milestones

1. Amend and independently review the matched-control Readiness contract.
2. Implement the exact three-run manifest, control wrapper/binding,
   calibration/preflight/admission/batch identity, and focused tests.
3. Commit and recompute the complete production implementation identity.
4. Execute the already authorized exact bounded zero-training-step three-phase
   calibration requalification command frozen in AC-01; render the fresh
   authoritative calibration result.
5. Run one fast preflight, build/mutation-test the candidate admission bundle, and
   confirm no formal execution or publication.
6. Obtain independent final readiness acceptance and render the accepted bundle.

## Feasibility Probes

- AC-02 storage probe on 2026-07-14: `df -h /data-1 /data-2` measured approximately
  470 GiB free on `/data-1` and 1.4 TiB free on `/data-2` after deleting only
  completed calibration scratch weights. No absolute runtime budget is frozen for
  preflight; it runs once and is not a timing benchmark.
- AC-01 GPU requalification retains the already demonstrated bounded calibration
  envelope. Raw measurement command:
  `jq '{stage2_seconds: ([.phase_evidence[]|select(.phase=="stage2")|.repetitions[].elapsed_seconds]|add), stage3_seconds: ([.phase_evidence[]|select(.phase=="stage3")|.repetitions[].elapsed_seconds]|add), total_seconds: ([.phase_evidence[].repetitions[].elapsed_seconds]|add), probe_elapsed_seconds:.probe_evidence.elapsed_seconds}' docs/joint_training/goals/calibration-qualification/calibration_result.json`.
  On the same 8-L40S `verl-harness` environment it measured Stage2
  `673.6000139713287s`, Stage3 `423.5177867412567s`, repetition total
  `1097.1178007125854s`, and end-to-end probe `1097.252543926239s`.
  The frozen `5400s` ceiling provides `4302.747456073761s` slack and approximately
  `4.92x` the measured end-to-end duration for cleanup/startup variance. Starting it
  is authorized by `EX-MATCHED-CONTROL-READINESS-01`. Stage1 uses the same
  single-model topology hash as Stage3, so adding three Stage1 repetitions is
  conservatively bounded by the measured Stage3 repetition total; the resulting
  estimated three-phase total remains below half of the frozen `5400s` ceiling.
- AC-03 numeric-budget waiver: references to Stage2/Stage3 and the three manifest,
  queue, and monitor surfaces define identity/cardinality, not a performance or
  resource budget; no feasibility measurement is applicable.
- AC-02 freshness probe: host facts use the existing manifest-owned preflight
  freshness window; no new latency or repetition budget is introduced. The producer
  runs once immediately before container preflight.

## Progression Policy

- `AUTO_ADVANCE`: Plan validation/review, finding lifecycle, named compatibility
  repair, focused CPU tests, identity recomputation, reviewer prompt generation,
  and all deterministic evidence checks inside the frozen ACs.
- `USER_DECISION`: deleting resources not created by this Goal, changing
  manifest/profile/run set beyond the exact matched control, starting any training
  before accepted admission,
  using a real external service, or resolving any new `CONTRADICTION`/`AC_CHANGE`.
- The Milestone 4 decision must name the exact Plan v9 hash, decision id, maximum
  8-GPU/5400-second/3-repetition envelope, output path, and zero-step requirement.
- No milestone may start while a matching `USER_DECISION_RECORDED` event is absent.

## Runtime Contract

- Start only after Plan v9 is independently `READY`.
- User authorization `EX-MATCHED-CONTROL-READINESS-01` covers the Plan amendment,
  matched-control implementation, CPU validation, bounded zero-step requalification,
  independent acceptance, and subsequent formal Primary Chain launch only after the
  new accepted bundle passes every gate.
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
- The matched-control repair is authorized by Plan v9 and must remain limited to
  manifest/control wrapper binding, generic calibration/preflight/admission/batch
  identity, focused tests, and active reliability documentation.
- Stop for any new `CONTRADICTION` or
  `AC_CHANGE`, convergence failure, protected-asset risk, weight-updating training,
  or need for a real external service.
- The implementer cannot self-review or self-accept.

## Reviewer Contract

- Plan review verifies one readiness outcome, honest invalidation/requalification of
  predecessor evidence, exact primary run set, one fast preflight, and no hidden
  training or GPU authorization.
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
