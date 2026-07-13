# Experiment Execution Core Consolidation

- Goal ID: `experiment-execution-core-consolidation`
- Plan version: `1`
- Plan status: `REVIEW_PENDING`
- Repository: `/data-1/code/verl`
- Branch: `codex/experiment-execution-reliability`
- Baseline commit: `390be933`

## Outcome

Deliver one maintainable, CPU-verified experiment-execution core that preserves
accepted reliability behavior while removing Stage123 policy from shared layers and
deleting redundant checker, receipt, queue, test, and review-document accretion.

GPU qualification and Stage123 execution are independently useful outcomes. They are
deferred to serial Goals and cannot expand this Goal.

## Starting Evidence

The audit at
`/root/.codex/attachments/061f43ef-df29-4b55-9f3d-4f8444aaa7ed/pasted-text.txt`
confirmed calibration coupling in `RayPPOTrainer`, duplicated manifest facts, a
shell-owned state machine, brittle source-text tests, overlapping receipts, and active
review-document sprawl. Relative to merge base
`6a86aa27591178f9acf774fe107b1219e099af99`, baseline `390be933` has approximately
131 changed files, 11,742 additions, and 38 deletions.

The `source_uid` regression was fixed and independently reviewed at `499d2a1e` and
must remain fixed.

Protected user assets are outside this Goal and must remain unstaged:

- `.claude/skills/experiment-registry`
- `docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md`
- `test_data/`

## Scope

### Included

- Classify branch-added reliability components as `KEEP`, `MERGE`, `MIGRATE`,
  `COMPAT_SHIM`, `FAIL_CLOSED`, `ARCHIVE`, or `DELETE`.
- Replace calibration file protocols in generic trainer code with a minimal generic
  validation observer; default behavior is no-op.
- Make the normalized manifest the sole owner of concrete experiment identity,
  dataset, producer, path, hash, and TTL values.
- Separate generic canonicalization from Stage123 policy validation.
- Move queue states, deadlines, child lifecycle, cleanup, and resume from shell to
  Python; retain only a thin shell entrypoint.
- Replace free-form checker failures with stable code, message, and context fields.
- Collapse runtime evidence to `preflight_result.json`, `calibration_result.json`, and
  `acceptance_report.json`; remove adoption/document-hash receipt machinery.
- Replace source-text runtime assertions with fake-executable, fake-clock, temporary
  state, and observer behavior tests.
- Preserve PM2-only CI, tmux ownership, port isolation, deadline cleanup, scorer
  dependency gates, response telemetry, release gating, provenance, and dirty-baseline
  protection.
- Preserve historical artifacts without treating stale evidence as current authority.
- Delete or archive superseded active review documents.

### Excluded

- GPU preflight, calibration, training, rollout benchmark, or the 27-run queue.
- Dynamic timeout/performance-model qualification or a truncation quality threshold.
- Stage1, Stage2, or Stage3 readiness and execution.
- Real W&B, WxPusher, Hugging Face, GitHub, Docker-daemon, or registry calls as evidence.
- Normalizing unrelated historical experiment families.
- A general callback framework beyond the smallest observer required here.
- New adoption receipts, document receipts, per-AC wrappers, or review series.
- A fixed deletion percentage. Required deletion classes and preserved behavior are the
  contract; line counts are supporting evidence, not a target to game.

## Architecture Contract

- Shared trainer code may emit generic validation events and telemetry, but must not
  contain `CALIBRATION_*`, Stage123 identity, receipt, or deployability semantics.
- Canonicalization parses and normalizes. Concrete facts come from the manifest.
  Versioned policy validators enforce cross-field invariants without copying values.
- Python owns execution state. Shell performs minimal environment setup and one `exec`.
- Git is version evidence for code/docs. Only the three result classes authorize the
  current workflow.
- Every legacy manifest, receipt, entrypoint, and evidence class receives a disposition.
  Shims warn and delegate without copying policy; unsupported inputs fail closed.

## Acceptance Criteria

### AC-01 - Shared Trainer Is Generic

- Given: validation inputs with either legacy `uid` only or stable `source_uid`.
- When: validation runs with the default or recording generic observer.
- Then: legacy validation succeeds without calibration files, while stable identity
  and generic lifecycle events are observable when explicitly requested.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/joint_training/regression/test_validation_generation_logging.py tests/joint_training/regression/test_validation_reward_metadata.py`
  and zero matches from
  `rg -n 'CALIBRATION_|stage123|deployab' verl/trainer/ppo/ray_trainer.py`.
- Expected evidence: reviewer-owned passing tests and zero forbidden-name matches.

### AC-02 - Manifest Has One Owner

- Given: valid manifests with changed concrete facts and invalid cross-field fixtures.
- When: canonicalization and versioned policy checking run.
- Then: valid results follow manifest values without source edits and invalid
  relationships return structured errors.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py`.
- Expected evidence: mutated-fact behavior tests pass without production constants
  supplying expected values.

### AC-03 - Queue State Is Python-Owned

- Given: fake Docker, tmux, clock, child adapters, and the thin shell entrypoint.
- When: start, success, deadline, failure, cleanup, resume, and entrypoint checks run.
- Then: persisted transitions and call logs match, and shell contains setup plus one
  Python `exec` rather than queue policy.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_operational_calibration_runtime_isolation.py`.
- Expected evidence: reviewer-owned fake traces cover every declared transition.

### AC-04 - Failures Are Structured

- Given: multiple invalid inputs and formatting-only message variations.
- When: validators, decision policy, and tests run.
- Then: failures expose stable `code`, `message`, and `context`, and tests do not
  depend on concatenated strings.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_checker.py tests/experiment_workflow/test_calibration_outcomes.py`.
- Expected evidence: assertions target structured fields and exit semantics.

### AC-05 - Evidence And Compatibility Are Consolidated

- Given: every inventoried legacy class, stale evidence, and a documentation-only
  change fixture.
- When: compatibility and migration tests run.
- Then: dispositions are recorded, historical files remain byte-identical, stale
  authority fails closed, and documentation changes require no adoption receipt.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow -k 'compat or migration or receipt or adoption or evidence'`.
- Expected evidence: only retained result classes authorize current execution and
  temporary historical fixtures are unchanged.

### AC-06 - Accepted Reliability Behavior Is Preserved

- Given: fixtures for missing scorer dependencies, timeout, fatal exit, CUDA OOM,
  occupied ports, incomplete identities, failed cleanup, and incomplete training.
- When: the full workflow suite runs once.
- Then: execution fails closed where required, cleanup outcomes are recorded, and
  release remains prohibited.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow tests/joint_training/regression/test_validation_generation_logging.py`.
- Expected evidence: one non-duplicative full workflow suite using mocks/local fakes.

### AC-07 - Operational Policies Remain Enforced

- Given: persistent CI configuration, a fake PM2 lifecycle, and long-running
  entrypoints.
- When: operational policy checks run.
- Then: PM2 lifecycle passes, systemd surfaces fail, and tmux ownership remains.
- Verification command:
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_pm2_ci_keepalive.py` and
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/check_pm2_ci_keepalive.py --contract config/pm2/experiment-reliability-ci.json --repo-root . --require-no-systemd`.
- Expected evidence: reviewer-owned fake lifecycle output and passing policy checks.

### AC-08 - Accretion Is Removed

- Given: baseline `390be933`, the disposition inventory, and a committed candidate.
- When: the reviewer audits forbidden classes and independently computes the three
  frozen path-class budgets below.
- Then: redundant active classes are absent; production is at most 3,000 lines from a
  3,763-line baseline, workflow tests are at most 4,000 lines from a 4,694-line
  baseline, and active reliability review/docs are at most 650 lines from a 1,638-line
  baseline. Each class must independently have more deletions than additions, so
  document deletion cannot subsidize production or test growth. Each retained file
  above 300 lines requires an inventory justification.
- Verification command:
  `git diff --numstat 390be933...HEAD -- scripts tests/experiment_workflow verl/trainer/ppo/ray_trainer.py recipe/on_policy_wdl_sft/code_task docs/joint_training/codereview/active docs/joint_training/manifests docs/joint_training/goals/experiment-execution-core-consolidation`
  plus `git diff --check 390be933...HEAD`.
- Expected evidence: file-level deletion report with protected assets and historical
  runtime evidence intact, plus separate baseline/candidate totals for:
  1. production: the 21 baseline files under `scripts/`,
     `recipe/on_policy_wdl_sft/code_task/`, and `ray_trainer.py` selected by the
     Milestone 1 reliability inventory;
  2. tests: `tests/experiment_workflow/**` plus
     `test_validation_generation_logging.py`;
  3. active review/docs: matching reliability/Stage123 adoption files under
     `docs/joint_training/codereview/active`, `docs/joint_training/manifests`, and the
     superseded active plan/launch entrypoints.

### AC-09 - Independent Acceptance Is Bound To This Plan

- Given: a committed candidate and READY Plan version 1.
- When: a fresh independent reviewer uses a generated Final Acceptance prompt.
- Then: every AC is reported PASS, FAIL, or WEAKENED with reviewer-owned evidence,
  and `ACCEPTED` is valid only for all-PASS results bound to Plan hash and commit.
- Verification command:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-execution-core-consolidation`
  and `git status --short` showing protected assets unstaged.
- Expected evidence: completed reviewer-owned `acceptance.md`, acceptance ledger event,
  and passing runtime validation.

## Milestones

1. Inventory and disposition map; review AC-05 feasibility and AC-08 deletion map.
2. Generic observer and manifest/policy ownership; review AC-01 and AC-02.
3. Python execution state and structured failures; review AC-03 and AC-04.
4. Evidence migration and deletion; review AC-05 and AC-08.
5. One non-duplicative CPU preservation gate; review AC-06 and AC-07.
6. Generated final-review prompt, independent AC-01..AC-09 acceptance, ledger update,
   and runtime validation.

Milestones are hard ordered. Starting this Goal authorizes the complete Milestone 1-6
execution envelope, not one isolated milestone. After a milestone's checks and required
independent review pass and runtime validation succeeds, the implementer automatically
records completion and starts the next milestone without requesting user approval.
A later milestone cannot start with an unresolved blocker.

## Runtime Contract

- Implementation is unauthorized until Plan v1 validates, an independent reviewer
  returns `READY`, that verdict is appended to `runtime.jsonl`, runtime validates, and
  the user explicitly starts this Goal.
- One user start authorizes autonomous execution of Milestones 1-6. It does not
  authorize deferred Goals, GPU work, real external services, protected-file changes,
  or a changed Definition of Done.
- The implementer proceeds serially and automatically transitions after the current
  milestone's checks and required independent review pass and runtime validates.
- Routine `IN_SCOPE` findings, implementation choices, test failures, and compatibility
  work are resolved autonomously inside the frozen ACs; they do not require user input.
- Validate runtime before implementation, after finding classification, before a
  milestone transition, and before acceptance.
- Classify findings before acting: `IN_SCOPE`, `DEFERRED`, `CONTRADICTION`, or
  `AC_CHANGE`. The latter two stop execution and require Plan amendment/review.
- Two related review rounds leaving the same finding open trigger convergence review
  before a third implementation attempt.
- A new independently useful outcome, subsystem, runtime environment, or acceptance
  surface triggers re-planning, not scope expansion.
- Preserve protected files, untracked user work, and diagnostic evidence. Ledgers are
  append-only; corrections are new events.
- Use `/data-1/verl07/run_train.sh` when project dependencies are unavailable on host.
  No GPU or external service is authorized. Long commands use tmux; persistent CI uses
  PM2, never systemd.
- Implementer never edits `acceptance.md` to claim acceptance and never self-grades.

## Reviewer Contract

- Generate prompts with `goal-plan-runtime build-reviewer-prompt` from the stable
  template, this contract, and task-specific focus.
- Reviewer is independent and uses GPT-5.5 medium when available.
- Plan review checks one-outcome integrity, sandbox-verifiable ACs, code contradictions,
  ordering, compatibility, and reproducible commands.
- Milestone/final review runs applicable commands from committed state and audits
  deleted, skipped, loosened, or trivialized tests.
- Outside-AC opinions are `DEFERRED_SUGGESTION`; completion-definition changes are
  `CONTRACT_CONTRADICTION`, not reviewer-authored amendments.

## Verification Commands

- Plan validation:
  `goal-plan-runtime validate-plan docs/joint_training/goals/experiment-execution-core-consolidation`
- Runtime validation:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-execution-core-consolidation`
- AC commands are frozen in AC-01 through AC-09 and must be run from committed state.
- All project Python commands are frozen with
  `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh` so review cannot silently
  execute the separate `/data-1/verl07/verl` checkout.

## Stop Conditions

- Plan/runtime validation is non-zero or review is `NOT_READY`.
- A finding is `CONTRADICTION`/`AC_CHANGE`, or survives two related review rounds.
- Deletion removes accepted behavior without a replacement behavior test.
- Compatibility would silently accept stale evidence as current authority.
- GPU, real external service, or protected-file modification appears necessary.

Only these stop conditions require user input. A normal milestone boundary, passing
review, routine in-scope defect, or choice among implementations consistent with the
frozen ACs is not a human approval gate.

## Deferred Serial Goals

1. **Calibration Qualification**: qualify sampled-validation timing, timeout,
   truncation, score, and resource-risk prediction; requires explicit authorization
   for one bounded GPU probe.
2. **Stage123 Execution Readiness**: produce fresh preflight evidence and independently
   verify one deployable queue configuration.
3. **Stage123 Experiment Execution**: run the approved queue, monitor it, and apply the
   release gate before DB/W&B publication.

The paused legacy Goal is superseded by this serial decomposition.
