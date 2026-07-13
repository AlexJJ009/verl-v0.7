# Experiment Execution Reliability: Architecture Consolidation Goal

Status: active, implementation paused pending independent plan acceptance

Updated: 2026-07-13

Branch: `codex/experiment-execution-reliability`

## 1. Objective

Consolidate the experiment reliability work into a small set of generic,
behavior-tested components before any Stage123 calibration or training queue is
resumed.

This goal keeps the useful capabilities already developed:

- manifest-owned run identity;
- dataset, model, and checkpoint provenance;
- preflight resource and scorer checks;
- validation response telemetry;
- deadline cleanup and runtime port isolation;
- mocked external services;
- the training-result release gate;
- content-addressed dirty-baseline protection.

It removes or merges the Stage123-specific framework layers that duplicate those
capabilities, leak calibration protocols into generic training code, or exist only
to prove that another document or checker was adopted.

The goal is successful only if the branch becomes materially smaller and easier to
change. Adding another checker, receipt, acceptance criterion, or review document
is not progress unless it replaces more code than it adds.

## 2. Current Stop Condition

GPU calibration, GPU preflight, the 27-run queue, and formal
Stage1/Stage2/Stage3 execution remain paused until Milestones 1-6 pass from
committed state, a fresh independent reviewer reports all ACs PASS, and the user
explicitly reauthorizes GPU work.

The diagnostic probe at the following path is preserved but is not acceptance
evidence and must not be resumed during consolidation:

```text
/data-2/experiment_registry/calibration_runs/bb3f51ea_stage1_transition_probe
```

It reached `validation_ready` for 1379 prompts, then was deliberately terminated
after reward workers lacked EvalPlus/LiveCodeBench scorer imports.

No existing bootstrap artifact may be deleted merely because the architecture is
being simplified. Historical evidence is immutable input, not a requirement to
preserve the code that generated it.

## 3. Baseline

The implementation baseline for consolidation is:

```text
superproject: 499d2a1e  fix: preserve legacy validation identity fallback
recipe:       b9b9593614b0181dc0bbdcbcbb9ed31c6587ef7b
```

The following untracked paths are user-owned and outside this goal:

```text
.claude/skills/experiment-registry
docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
test_data/
```

Do not stage, modify, or delete them.

The branch delta at the pre-consolidation audit was approximately:

```text
130 changed or added files
13,277 additions, 38 deletions in the superproject
2,584 additional lines in recipe
1,897 lines in the previous Goal contract
737 lines in the calibration checker
429 lines in the shell queue
```

The exact merge base and line counts must be recorded at Milestone 1. These numbers
are context, not values to hardcode into production checks.

## 4. Non-Goals

This goal does not:

- prove that a weak 1.7B model has truncation below 1%;
- replace sampled validation with deterministic decoding;
- run the Stage123 training matrix;
- normalize every historical experiment family;
- create a general workflow platform for unrelated repositories;
- redesign VERL's trainer callback system beyond the smallest generic observer
  boundary required here;
- upload failed or incomplete runs to W&B or the experiment registry;
- delete diagnostic artifacts or user-owned files;
- use real W&B, WxPusher, Hugging Face, or GitHub calls as test evidence.

Initial Stage1 truncation is diagnostic. Timeout, fatal worker failure, CUDA OOM,
missing scorer dependencies, incomplete validation identity coverage, and failed
cleanup remain hard infrastructure failures.

## 5. Architecture Boundaries

### 5.1 Generic Trainer

`RayPPOTrainer` may emit generic validation lifecycle events and generic response
telemetry. It must not know:

- `CALIBRATION_*` environment variable names;
- calibration JSONL paths or ready-marker files;
- Stage1, Stage2, or Stage3 identities;
- receipt formats or deployability decisions.

The default validation observer is a no-op. The calibration layer owns any JSONL
or marker-file adapter.

### 5.2 Manifest

The normalized manifest is the sole source for concrete experiment facts:

- phases and their ordering;
- producer and consumer identity;
- run prefix and final step;
- model and checkpoint paths;
- dataset paths, row counts, and hashes;
- output and provenance paths;
- policy TTL values.

Validators may encode schema and cross-field invariants. They may not maintain a
second copy of the current Stage123 values.

Normalization parses and canonicalizes data. Stage123 policy belongs in a
versioned policy validator, not in a function presented as generic normalization.

### 5.3 Calibration Checker

The checker is an orchestration boundary, not a 700-line policy database. Its
responsibilities are limited to:

1. load normalized inputs;
2. call validators and decision policy;
3. render one structured result;
4. return a stable exit code.

Failures use typed codes with structured context. Do not append new free-form
strings to `failures: list[str]`.

### 5.4 Queue

Python owns the queue state machine, transitions, deadlines, cleanup decisions,
and persisted state. Shell owns only environment setup and `exec` of the Python
entry point.

Long-running execution still uses tmux. PM2, not systemd, owns persistent CI.

### 5.5 Evidence

Only these machine-generated evidence classes remain:

```text
preflight_result.json
calibration_result.json
acceptance_report.json
```

They may contain nested phase or repetition records. Separate adoption receipts,
plan receipts, dirty adoption manifests, runtime-isolation receipts, and
document-hash refresh commits are removed unless an independently reviewed legal
or security requirement is identified.

Git commits are the version evidence for code and documents.

### 5.6 Documentation

The default agent context keeps:

- this current contract;
- one append-only decision log;
- one final acceptance report after completion.

Intermediate plan reviews are archived outside active context or deleted when Git
history is sufficient. No fifth review document is created.

## 6. Milestones

Milestones are serial. The implementer must not start a later milestone before the
current milestone's focused tests pass.

### Milestone 0: Stop Expansion and Repair Generic Regression

Completed at `499d2a1e`.

The generic validation path now prefers `source_uid` when present and falls back to
legacy `uid` otherwise. Behavior tests cover both paths. The committed-state focused
gate passed 12 tests, and an independent reviewer returned `READY FOR COMMIT`.

### Milestone 1: Inventory and Deletion Map

Produce a compact inventory in the decision log containing:

- manifest facts duplicated outside the manifest;
- calibration protocol references in generic trainer code;
- shell queue responsibilities and Python replacement mapping;
- source-text tests that assert implementation strings;
- receipt/adoption artifacts and active review documents;
- files to keep, merge, archive, or delete;
- every legacy artifact and entrypoint class classified as `kept`, `migrated`,
  `compat-shimmed`, or `fail-closed with migration message`;
- baseline production, test, and documentation line counts;
- a conservative target for net line deletion.

The inventory must distinguish genuinely static policy tests, such as forbidding
`systemctl`, from behavior that needs executable fakes.

### Milestone 2: Restore Generic Boundaries

Implement the generic validation observer and move calibration file output into the
calibration layer.

Remove concrete Stage123 constants from the checker and generic manifest normalizer.
All current facts must be read from the normalized manifest or an explicit policy
document referenced by it.

No production path may silently fall back to the removed constants.

### Milestone 3: Decompose Execution

Replace the shell-owned queue state machine with a Python entry point. Keep a thin
shell launcher only where required by existing operational conventions.

Split checker behavior by responsibility only where the split reduces coupling:

- schema/invariant validation;
- calibration decision policy;
- result rendering.

Do not create one file per acceptance criterion or one wrapper per phase.

### Milestone 4: Replace Brittle Tests

Replace source-text assertions for runtime behavior with tests that execute:

- wrapper environment and argv propagation through a fake child;
- Docker and tmux cleanup through fake executables and call logs;
- deadline behavior through a fake clock/process;
- manifest validation through parser and policy APIs;
- queue transitions through temporary persisted state;
- generic validation with and without a calibration observer.

Static text tests may remain only for simple repository policy that cannot be more
meaningfully exercised, such as the absence of `systemctl` in PM2-owned CI paths.

### Milestone 5: Collapse Receipts and Documents

Migrate retained evidence into the three evidence classes. Remove generators and
tests whose only purpose is adoption or document-hash churn.

Replace the previous 1,897-line contract with this contract. Maintain decisions in
one append-only log. Archive or delete superseded active review documents.

Update the active plan index and launch prompt once, after the contract is accepted.

### Milestone 6: CPU Acceptance

Run focused tests after each milestone, then one non-duplicative full CPU gate.
Do not run `fast` followed by `full` when `full` includes `fast`.

The independent reviewer runs the final commands from committed state and verifies:

- ordinary non-calibration validation still works;
- calibration observation is opt-in;
- manifest facts have one owner;
- queue transitions and cleanup are behavior-tested;
- external services are mocked;
- no protected user asset is staged;
- production plus test plus active-document lines have a meaningful net decrease.

### Milestone 7: Reauthorize GPU Work

Only after independent CPU acceptance may a fresh machine report and preflight result
be generated. A user decision is required before GPU calibration resumes.

The first resumed GPU work is one bounded sampled-validation probe, not the complete
27-run queue. The probe has a 30-minute preflight/measurement budget unless the user
explicitly changes it.

If a hard infrastructure gate fails, stop and send one WxPusher escalation. If the
run starts, fails, or requires a user decision, send the previously agreed single
status notification. Do not send routine healthy polling notifications.

## 7. Acceptance Criteria

### AC-01: Generic Validation Identity

Given a normal validation batch with `uid` and no `source_uid`, when validation
collects sample identities, then it completes using `uid`. Given both fields, it
uses `source_uid` for stable dataset identity.

### AC-02: No Calibration Protocol in Core Trainer

Given the generic trainer source, when repository checks run, then no
`CALIBRATION_*` name, calibration ready-file protocol, or calibration-specific path
appears in `RayPPOTrainer`. Generic observer events remain behavior-tested.

### AC-03: Manifest Owns Concrete Facts

Given a valid manifest with changed dataset counts, hashes, producer identity, paths,
and TTL, when normalization and checking run, then the result follows that manifest
without source edits. Given an invalid cross-field relation, validation returns a
typed error.

### AC-04: Thin Queue Entry Point

Given fake Docker, tmux, clock, and child-process adapters, when queue scenarios run,
then transitions, deadline termination, cleanup, and resume are verified from calls
and persisted state. The shell launcher contains no state machine or policy logic.

### AC-05: Structured Failures

Given multiple invalid inputs, when checker and queue errors are rendered, then every
failure has a stable code, message, and structured context. Tests do not compare
concatenated free-form failure strings.

### AC-06: Minimal Evidence Model

Given preflight, calibration, and acceptance scenarios, when artifacts are emitted,
then all machine evidence fits the three declared result classes. No adoption receipt
must be refreshed after a documentation-only commit.

### AC-07: Behavioral Tests

Given the runtime contracts named in Milestone 4, when implementation identifiers or
shell formatting change without behavior change, then tests remain green. Tests fail
when fake call traces or persisted transitions violate the contract.

### AC-08: External Isolation

Given the full CPU gate, when tests execute, then W&B, WxPusher, Hugging Face, GitHub,
Docker, and tmux interactions are mocked or local fakes. No real account or service is
used as acceptance evidence.

### AC-09: Negative Line Budget

Given the recorded Milestone 1 baseline, when consolidation finishes, then combined
production, test, and active reliability-document lines are lower by at least 25%,
and the branch contains more deletions than new additions introduced after
`499d2a1e`. Generated fixtures and archived evidence may not be used to game the count.

If 25% cannot be reached without removing required behavior, stop and request user
approval with a file-level explanation; do not weaken this criterion silently.

### AC-10: Operational Policies Preserved

Given a long-running job, when launched, then it uses tmux. Given persistent CI, it
uses PM2 and contains no systemd path. Given a failed or incomplete training run,
release-gate checks prevent DB import and W&B cloud sync.

### AC-11: Independent Acceptance

Given a committed candidate, when a fresh GPT-5.5 medium reviewer runs the declared
commands and audits the diff, then every AC is reported PASS, FAIL, or WEAKENED with
command evidence. Only all-PASS permits the Goal to advance to GPU reauthorization.

### AC-12: Safe Compatibility Migration

Given each pre-consolidation manifest, receipt, queue entrypoint, and historical
evidence class, when consolidation removes or changes its implementation, then it is
classified as kept, migrated, compatibility-shimmed, or fail-closed with an actionable
migration message. Sandbox tests prove that historical artifacts are neither
overwritten nor deleted and that unsupported legacy inputs cannot be silently treated
as current deployability evidence.

## 8. Verification Strategy

Each milestone adds only focused behavior tests. The final CPU gate consists of:

```bash
python -m pytest -q tests/joint_training/regression/test_validation_generation_logging.py
python -m pytest -q tests/experiment_workflow
python -m pytest -q tests/joint_training/regression tests/utils/dataset
python scripts/check_pm2_ci_keepalive.py \
  --contract config/pm2/experiment-reliability-ci.json \
  --repo-root . \
  --require-no-systemd
```

If repository-native full commands differ after consolidation, update this section
before running them and explain why. Do not execute duplicate supersets.

Static checks include:

```bash
rg -n 'CALIBRATION_' verl/trainer/ppo/ray_trainer.py
git diff --check <merge-base>...HEAD
git status --short
```

Line-budget reporting must list included paths and exclusions. It must compare the
same path classes at baseline and candidate commits.

## 9. Commit and Review Rules

- Commit each independently verifiable milestone.
- Do not mix protected untracked assets into commits.
- Do not refresh adoption receipts.
- Do not add a new AC without first updating this contract and obtaining review.
- Do not loosen tests to preserve an implementation detail.
- Do not use historical bootstrap success as current authorization.
- Implementer never self-accepts.
- Reviewer must run tests from committed state.

## 10. Stop and Ask

Stop and ask the user when:

- a required behavior conflicts with the negative line budget;
- a generic VERL API change would affect callers outside this branch and no local
  compatibility path is clear;
- real network or account access appears necessary for acceptance;
- a hard preflight gate fails after CPU acceptance;
- a reviewer marks any AC WEAKENED;
- three consecutive attempts hit the same blocker.

Do not stop merely because consolidation requires deleting code or rewriting tests.

## 11. Definition of Done

The architecture consolidation phase is done when:

1. AC-01 through AC-12 pass from committed state;
2. the independent reviewer reports `ACCEPTED`;
3. the branch meets the negative line budget;
4. no calibration or training process was run during consolidation;
5. protected artifacts and diagnostic evidence remain intact;
6. the user receives a compact change and deletion report.

This does not automatically complete the broader experiment reliability objective.
It authorizes a separate user decision on the bounded GPU probe. Formal calibration,
the 27-run queue, and Stage123 training remain later execution steps.
