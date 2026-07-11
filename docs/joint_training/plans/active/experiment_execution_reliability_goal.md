# Experiment Execution Reliability and GPU Utilization Goal

- Status: `REVISED AFTER REVIEW, PENDING RE-REVIEW`
- Created: 2026-07-11
- Branch: `feature/on-policy-wdl-sft`
- Scope: future Stage1/Stage2/Stage3 experiment families, beginning with Qwen3-1.7B Stage123
- Execution mode: serial milestones; no implementation before independent plan acceptance
- External services: forbidden as acceptance evidence; W&B and WxPusher must use local fakes/mocks

## Objective

Build a machine-checkable experiment execution workflow that:

1. predicts scorer latency, timeout rate, score distribution, host-memory risk,
   and expected GPU starvation before a formal training launch;
2. uses one experiment manifest to generate or configure new queue, monitor,
   preflight, release, and documentation records;
3. rewrites project engineering principles into trigger/action/failure rules and
   replaces machine-checkable prose with executable gates.

The goal does not promise to normalize every historical queue. Historical assets
receive a best-effort inventory. The manifest contract is mandatory for Stage123
and every newly created experiment family after this goal is accepted.

## Single Source of Truth

This file and its numbered acceptance criteria are the execution contract. During
implementation, any scope or acceptance change must update this file before code.

The implementation must introduce one versioned experiment-manifest schema under:

```text
recipe/on_policy_wdl_sft/experiment_manifest/
```

New queue, monitor, preflight, and release behavior must consume that schema rather
than maintaining independent run-prefix/final-step/train-file arrays.

## Non-Negotiable Boundaries

1. Stage1, Stage2, and Stage3 continue to use the same canonical L40S resource
   profile, including `MAX_RESPONSE_LENGTH=8192`.
2. Full HumanEval+, MBPP+, and LiveCodeBench remain the formal validation set.
   A bounded preflight sample is infrastructure evidence only, never experiment
   quality evidence.
3. The implementation must not increase reward-worker concurrency without a
   measured peak-RSS budget and a passing memory-safety gate.
4. Failed or incomplete runs must not enter SQLite or W&B.
5. Long-running GPU, Docker, Ray, or transfer commands run in tmux.
6. Existing user changes and historical scripts must not be reverted merely to
   make the new structure cleaner.
7. Acceptance must not contact real W&B, WxPusher, Hugging Face, or other remote
   services.
8. No hard or soft gate has an automatic override path. Structural, provenance,
   semantic, profile-identity, model/data hash, checkpoint-placement, and host-memory
   failures remain blocked. A soft performance-threshold failure stops the Goal,
   notifies the user through WxPusher when unattended, and requires an explicit
   user decision followed by a reviewed manifest/policy commit and a fresh preflight.
9. A single phase's real pre-training measurement or formal `val_before_train`
   validation has a hard 30-minute wall-clock budget from validation rollout readiness
   to complete validation metrics. Exceeding the budget is a blocking failure, not a
   slow-but-acceptable state.
10. WxPusher sends only event-driven notifications: one verified run-start event,
    one failure event, or one human-decision-required event. It sends no periodic
    healthy-progress messages. A run-start notification requires the first training
    step or complete formal validation metrics; tmux/container existence alone is not
    evidence that an experiment has started successfully.

## Repository Transaction Baseline

The Goal spans two Git repositories and must treat them as one ordered transaction:

| Repository | Baseline branch | Baseline commit | Goal branch |
| --- | --- | --- | --- |
| Superproject `/data-1/code/verl` | `feature/on-policy-wdl-sft` | `6a86aa27591178f9acf774fe107b1219e099af99` | `codex/experiment-execution-reliability` |
| Submodule `/data-1/code/verl/recipe` | `codex/l40s-readiness` | `763aab506cd38bc7ff9fccfd9a079840620c37c5` | `codex/experiment-execution-reliability` |

Before branch creation, write path/status snapshots for both repositories to:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline/superproject.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline/recipe.json
```

These snapshots are protected evidence of pre-existing user changes. Every affected
milestone commits the submodule first, then commits superproject files plus the exact
submodule pointer. Staging is path-scoped. If an intended file already contains
unrelated edits that cannot be separated safely, stop and ask the user.

The following pre-existing untracked recipe paths are the only initial adoption set:

```text
on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh
on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh
on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh
on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh
on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh
on_policy_wdl_sft/code_task/stage123_gpu_idle_watchdog.py
on_policy_wdl_sft/code_task/stage123_preflight.py
```

They are not treated as immutable unrelated work. AC-25 governs their controlled
adoption. Every other pre-existing dirty path in either repository remains protected.

## Milestone 0: Baseline and Fixtures

Produce a read-only inventory and deterministic local fixtures before changing
runtime behavior.

Artifacts:

```text
docs/joint_training/reports/experiment_workflow_baseline.md
tests/experiment_workflow/fixtures/
tests/experiment_workflow/test_fixture_contract.py
```

### AC-01 - Baseline Records the Current Bottleneck

- Given the current Stage123 logs and local runtime artifacts,
- When the baseline collector runs,
- Then it emits machine-readable JSON plus a Markdown summary containing scorer
  timeout count/rate, score count/distribution, elapsed scorer time, peak observed
  RewardLoopWorker RSS, GPU idle intervals when evidence exists, profile hash,
  Docker image ID, and explicit `unknown` values where evidence is unavailable.

Verification:

```bash
python3 scripts/experiment_workflow_baseline.py \
  --fixture tests/experiment_workflow/fixtures/stage123_scorer_timeout \
  --json-out /data-1/tmp/verl_agent_scratch/goal/baseline.json \
  --md-out /data-1/tmp/verl_agent_scratch/goal/baseline.md
python3 -m pytest -q tests/experiment_workflow/test_fixture_contract.py
```

Expected evidence: both commands exit `0`; JSON schema validation passes; fixture
contains no secrets or remote-service credentials.

### AC-02 - Failure Taxonomy Is Deterministic

- Given fixtures for path/config failure, model provenance failure, CUDA OOM,
  Ray host-memory kill, scorer timeout, dependency failure, early queue exit, and
  release failure,
- When the classifier runs,
- Then each fixture receives exactly one primary terminal reason and zero or more
  secondary signals in JSON.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_failure_classifier.py
```

Expected evidence: all eight primary classes are asserted by name; unknown input
returns `unknown` rather than a fabricated diagnosis.

## Milestone 1: Preflight Performance Estimator

This milestone produces a bounded, representative scorer benchmark shared by
Stage1, Stage2, and Stage3. It must complete before manifest migration begins.

Artifacts:

```text
recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_benchmark.py
recipe/on_policy_wdl_sft/code_task/preflight_benchmark_schema.json
scripts/check_code_task_preflight_budget.py
tests/experiment_workflow/test_preflight_benchmark.py
```

### AC-03 - Representative Three-Phase Benchmark

- Given one Stage1 model source, one Stage2 joint/fixed-Model2 source, one Stage3
  model source, the canonical 8K resource profile, and bounded deterministic rows
  from HumanEval+, MBPP+, and LiveCodeBench,
- When the preflight benchmark runs in fixture/mock mode,
- Then it produces comparable per-phase metrics for rollout latency/tokens per
  second, scorer latency, timeout rate, score distribution, valid-score rate,
  peak RSS, and estimated GPU-wait fraction without changing context length or
  reward semantics.

The benchmark contract pins deterministic row IDs and counts per dataset, three
measured repetitions after one excluded warmup, timeout rate denominator as all
submitted scoring items, fixed score bins, RSS sampling at no more than one second,
and GPU-wait fraction as wall time with every GPU at or below 2% utilization while
the phase is active divided by measured phase wall time. Stage2 uses the manifest's
fixed Model2 source plus joint fused-loss topology, not a generic joint-model mock.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_preflight_benchmark.py
python3 recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_benchmark.py \
  --fixture tests/experiment_workflow/fixtures/preflight_three_phase \
  --output /data-1/tmp/verl_agent_scratch/goal/preflight.json
python3 scripts/check_code_task_preflight_budget.py \
  --report /data-1/tmp/verl_agent_scratch/goal/preflight.json \
  --policy tests/experiment_workflow/fixtures/preflight_policy.json
```

Expected evidence: all commands exit `0`; report identifies all three phases and
all three validation datasets; `max_response_length` equals `8192` in every phase.

### AC-04 - Launch Budget Blocks Predictable Failure

- Given a preflight report whose timeout rate, invalid-score rate, peak RSS, or
  estimated GPU-wait fraction exceeds its manifest policy threshold,
- When the launch-budget checker runs,
- Then it exits nonzero, names every exceeded threshold, and prevents formal queue
  launch. Hard-gate failures remain blocked. Soft-threshold failures enter the
  user-decision workflow in AC-20; neither `--force` nor an approval file may bypass
  the failed report.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_preflight_budget_gate.py
```

Expected evidence: passing, failing, missing-metric, hard/soft force-flag rejection,
and delegated AC-20 decision-notification cases are covered; no bypass exists.

### AC-05 - Performance Changes Require Before/After Evidence

- Given a proposed scorer/rollout/validation optimization,
- When its benchmark is compared with the frozen baseline fixture,
- Then the report records before/after timeout rate, valid scores per minute,
  p50/p95 scorer latency, peak RSS, and estimated GPU-wait fraction, and rejects
  a change that improves throughput by weakening dataset, context, timeout-result,
  or scoring semantics.

The primary optimization metric is valid scores per wall-clock minute. A performance
optimization is accepted only when its repeated median improves by at least 10%,
timeout rate does not regress by more than one absolute percentage point, peak RSS
does not increase by more than 10%, and semantic hashes are identical. Results below
that threshold are `neutral`: acceptable only for a separately justified correctness
or maintainability change, never evidence that performance was optimized.
In addition, every Stage1/Stage2/Stage3 formal `val_before_train` path must produce
complete full-validation metrics within 30 minutes after validation rollout readiness.
The historical 76-minute Stage2 pre-validation stall is a required regression fixture.
An implementation that improves relative throughput but still exceeds 30 minutes is
not accepted.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_preflight_comparison.py
```

Expected evidence: semantic-downscope fixtures fail even when their throughput is
higher; neutral results cannot satisfy a performance-optimization claim.

### AC-06 - Real GPU Smoke Is Explicitly Approval-Gated

- Given the local L40S host and a plan-approved benchmark manifest,
- When an operator requests the representative GPU smoke,
- Then the command refuses to start without `ALLOW_CODE_PREFLIGHT_GPU_SMOKE=1`,
  runs in tmux when approved, writes only under declared scratch/report roots,
  and never labels bounded scores as formal experiment results.

Verification:

```bash
bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_gpu_smoke.sh
python3 -m pytest -q tests/experiment_workflow/test_gpu_smoke_guard.py
```

Expected evidence: guard test proves the unapproved path exits nonzero without
starting Docker/Ray. Actual GPU execution is operational evidence, not required
for sandbox acceptance.

## Milestone 2: Experiment Manifest and Generated Workflow

Artifacts:

```text
recipe/on_policy_wdl_sft/experiment_manifest/schema.json
recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
scripts/experiment_manifest.py
tests/experiment_workflow/test_experiment_manifest.py
docs/joint_training/reports/legacy_queue_monitor_inventory.md
```

### AC-07 - Manifest Owns Run Identity and Lifecycle

- Given a valid experiment manifest,
- When the manifest tool renders its normalized form,
- Then run prefix, phase, source model/checkpoint provenance, train/validation
  files, final step, resource-profile hash, preflight policy, tmux name, artifact
  root, release policy, and monitor policy come from the manifest exactly once.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_experiment_manifest.py
python3 scripts/experiment_manifest.py validate \
  recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
python3 scripts/experiment_manifest.py render \
  recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
  --format json > /data-1/tmp/verl_agent_scratch/goal/stage123.normalized.json
```

Expected evidence: duplicate run identities, missing provenance, invalid phase
order, mismatched profile hashes, and checkpoint roots outside the declared mount
fail validation.

### AC-08 - Queue and Monitor Consume the Same Manifest

- Given the Stage123 manifest,
- When queue and monitor dry-runs execute,
- Then both derive the same ordered run list, final steps, train files, tmux names,
  and release metadata without independent hard-coded arrays.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py
bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
bash -n recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh
```

Expected evidence: a fixture mutation changes queue and monitor output together;
grep/static test rejects reintroduction of Stage123 run-prefix arrays.

### AC-09 - Release Gate Uses Manifest Evidence

- Given a completed fixture run and its manifest,
- When the release hook is invoked against local fake SQLite/W&B directories,
- Then it verifies final checkpoint, metrics, manifest run identity, profile hash,
  dataset hashes, normalized-manifest hash, and content-addressed preflight report
  copied into immutable run provenance before recording release success. Every
  launched run must reference a passing preflight generated after the latest reviewed
  manifest/policy commit; there is no forced-run release path.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_manifest_release_gate.py
```

Expected evidence: incomplete, hash-mismatched, failed-preflight, and wrong-run
fixtures cannot be imported or marked W&B-synced.

### AC-10 - Historical Migration Is Bounded

- Given the existing queue/monitor/script index,
- When the inventory command runs,
- Then it classifies each discoverable family as `manifest-native`,
  `legacy-traceable`, or `legacy-unresolved`, records evidence paths, and does not
  block the goal on unresolved historical archaeology.

Verification:

```bash
python3 scripts/experiment_manifest.py inventory \
  --root recipe/on_policy_wdl_sft \
  --output /data-1/tmp/verl_agent_scratch/goal/legacy_inventory.json
python3 -m pytest -q tests/experiment_workflow/test_legacy_inventory.py
```

Expected evidence: Stage123 is `manifest-native`; unresolved legacy entries are
explicit rather than silently omitted.

### AC-11 - New Experiment Gate Rejects Legacy Structure

- Given version-controlled legacy inventories for both repositories containing
  schema version, the baseline commits above, and exact legacy runnable paths,
- When a runnable queue/monitor is added, changed, or renamed relative to those
  baselines,
- Then the gate fails unless the path is unchanged and allowlisted legacy, or is
  manifest-native. Renames are treated as delete plus add, and submodule paths are
  evaluated against the submodule baseline rather than filesystem dates or mtimes.

Verification:

```bash
bash scripts/check_experiment_workflow_fast.sh
python3 -m pytest -q tests/experiment_workflow/test_new_experiment_gate.py
```

Expected evidence: compliant and intentionally noncompliant fixture trees prove
the cutoff rule without rewriting historical scripts; tampered baseline/allowlist
hashes fail closed.

## Milestone 3: Project Principles and Engineering Experience Audit

Artifacts:

```text
docs/joint_training/constraints/principles/engineering_rule_catalog.md
scripts/check_engineering_rule_catalog.py
docs/joint_training/reports/engineering_rule_migration.md
tests/experiment_workflow/test_engineering_rule_catalog.py
```

### AC-12 - Rules Use Trigger, Action, Failure Structure

- Given project-level principles and operational rules selected for migration,
- When the catalog checker runs,
- Then every prose rule contains an explicit `When`, `Do`, and `Otherwise`
  failure field, a stable rule ID, scope, evidence source, and enforcement tier.

Verification:

```bash
python3 scripts/check_engineering_rule_catalog.py \
  docs/joint_training/constraints/principles/engineering_rule_catalog.md
python3 -m pytest -q tests/experiment_workflow/test_engineering_rule_catalog.py
```

Expected evidence: malformed, triggerless, actionless, and consequence-free rule
fixtures fail with line-specific diagnostics.

### AC-13 - Machine-Checkable Rules Point to Gates

- Given a rule whose condition can be checked deterministically,
- When the audit classifies it,
- Then the prose catalog contains only a one-line pointer to its checker, command,
  test ID, failure code, and fast/full gate reachability; duplicated normative prose
  is removed from the canonical source or clearly marked noncanonical.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_rule_enforcement_tiers.py
```

Expected evidence: structural, machine-check, and judgment-only examples are all
covered; a machine-checkable prose-only rule or unreachable checker fails the audit.

### AC-14 - Existing Principles Are Migrated Without Invented History

- Given `CLAUDE.md`, `AGENTS.md`, `docs/joint_training/constraints/`, active
  runbooks, and project memory indexes,
- When the migration report is generated,
- Then each selected rule is marked `retained`, `reworded`, `replaced-by-gate`,
  `project-local`, `global-candidate`, or `dropped-unverifiable`, with source-path
  evidence and no fabricated incident claims.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_rule_migration_report.py
```

Expected evidence: every migrated rule resolves to an existing source path; rules
without recoverable trigger/failure evidence may be dropped explicitly.

### AC-15 - Startup Context Remains Small

- Given the revised project entrypoint and rule catalog,
- When the documentation-size gate runs,
- Then startup-required files remain within their declared size budget and bulk
  inventories/results live in indexed Markdown or JSON rather than `AGENTS.md` or
  `CLAUDE.md`.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_context_size_budget.py
```

Expected evidence: oversized startup-document fixture fails; the real project
passes its documented budget. An exception requires owner, reason, plan/issue path,
reviewer approval, expiry, maximum permitted lines and bytes, and fails automatically
after expiry. No exception may exceed 400 lines or 128 KiB.

## Milestone 4: Integrated Fast and Full Gates

### AC-16 - Fast Gate Is Cheap and Deterministic

- Given a normal developer checkout with no Docker/GPU requirement,
- When `bash scripts/check_experiment_workflow_fast.sh` runs,
- Then it validates Python/shell syntax, manifest schema, generated contracts,
  fixture tests, release classification, and rule catalog in at most 120 seconds
  on the current L40S host.

Verification:

```bash
/usr/bin/time -f '%e' bash scripts/check_experiment_workflow_fast.sh
```

Expected evidence: exit `0`; elapsed time is recorded and no remote network call
occurs. Network isolation is proven with a network-disabled namespace/container or
an injected socket-deny layer whose attempted connection fixture fails the gate.

### AC-17 - Full Gate Preserves Exit Codes and External Isolation

- Given local Docker access and deterministic fixtures,
- When `bash scripts/check_experiment_workflow_full.sh` runs,
- Then it includes the fast gate plus container imports, reward mock execution,
  queue/monitor/release integration, preserves failing child exit codes, and uses
  fake W&B/WxPusher endpoints only.

Verification:

```bash
bash scripts/check_experiment_workflow_full.sh
python3 -m pytest -q tests/experiment_workflow/test_gate_exit_propagation.py
```

Expected evidence: both pass; an injected failing child command makes the full
gate nonzero even when output is piped or summarized.

### AC-18 - Stage123 End-to-End Dry Run Is Manifest-Native

- Given the accepted Stage123 manifest and local model/data artifacts,
- When the formal queue is run with `DRY_RUN=1`,
- Then preflight, phase order, handoff provenance, queue output, monitor output,
  checkpoint placement, and mocked release behavior all agree with the normalized
  manifest and write no durable experiment result. Dry-run must not start real
  Docker, Ray, or tmux; before/after snapshots compare both Git repositories plus
  SQLite row count/hash, release JSONL size/hash, W&B sync markers, checkpoint roots,
  monitor status roots, and other declared durable paths.

Verification:

```bash
STAGE123_MANIFEST=recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
python3 -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py
git status --short --ignored
```

Expected evidence: commands pass; dry-run artifacts stay under declared scratch;
no SQLite/W&B success record is created.

## Milestone 5: Operational Calibration and Repository Acceptance

Milestone 5 runs only after sandbox AC-01 through AC-18 pass. It uses the local L40S
host but no remote service. Its metrics are infrastructure evidence, not experiment
quality evidence.

### AC-19 - Operational Calibration Produces a Deployment Decision

- Given sandbox acceptance is green, the canonical profile/image/model/data hashes
  are pinned, and the user explicitly approves bounded local L40S calibration,
- When Stage1, exact Stage2 fixed-Model2 joint topology, and Stage3 calibration run
  in tmux using the same deterministic benchmark contract as AC-03,
- Then the report records full provenance, before/after observed metrics, estimator
  prediction error, and one decision: `deployable`, `blocked`, or `inconclusive`.
  `Deployable` requires the AC-05 improvement/safety budget and prediction error no
  greater than 20% for elapsed scorer time, peak RSS, and GPU-wait fraction. Each
  phase has a 30-minute hard timeout and must produce complete full-validation metrics;
  any timeout, incomplete metric set, or scorer stall returns `blocked`.

Verification:

```bash
python3 scripts/check_code_task_operational_calibration.py \
  --report /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/report.json \
  --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
```

Expected evidence: checker exits `0` only for `deployable`; `blocked` and
`inconclusive` preserve reports and keep formal deployment disabled. If GPU execution
is unavailable, sandbox implementation may be accepted but workflow deployment stays
`PENDING OPERATIONAL CALIBRATION`, not complete.

### AC-20 - Soft Threshold Failure Stops and Requests User Decision

- Given every hard gate passes but a report exceeds one or more soft performance
  thresholds during an unattended Goal,
- When the launch gate evaluates that report,
- Then it exits nonzero, records `USER_DECISION_REQUIRED`, and sends one deduplicated
  WxPusher notification containing the experiment/run identity, relevant background,
  failed thresholds and observed metrics, estimated GPU/time cost, recommended action,
  and local report/manifest paths. WxPusher is notification-only: phone delivery,
  clicks, or replies never authorize execution. The Goal resumes only after the user
  returns to the interactive task, explicitly decides, the agent updates the
  manifest/policy in Git with the decision rationale, a fresh independent review
  accepts that commit, and a newly generated preflight passes the revised policy.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_user_decision_notification.py
```

Expected evidence: hard and soft failures both block launch; notification payload is
complete, secret-free, deduplicated, and uses a local fake in acceptance; no response
channel or force flag can authorize launch; only a passing post-commit preflight can
clear the block.

The shared notification policy is event-only: `run_started`, `run_failed`, and
`user_decision_required`. `run_started` requires first-step or complete-validation
evidence; `run_failed` and `user_decision_required` include background, evidence,
cost, recommendation, and local paths. Healthy polling never sends WxPusher.

### AC-21 - Dual-Repository Change Isolation

- Given the recorded baseline commits and dirty-path snapshots for the superproject
  and recipe submodule,
- When each milestone is committed,
- Then only declared milestone paths differ from the snapshots, each affected
  submodule commit precedes its superproject pointer commit, unrelated dirty paths
  are unchanged, and final reviewer evidence covers both commit ranges.

Verification:

```bash
python3 scripts/check_goal_git_isolation.py \
  --superproject /data-1/code/verl \
  --submodule /data-1/code/verl/recipe \
  --baseline-root /data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline
git -C /data-1/code/verl diff --check 6a86aa27591178f9acf774fe107b1219e099af99..HEAD
git -C /data-1/code/verl/recipe diff --check 763aab506cd38bc7ff9fccfd9a079840620c37c5..HEAD
```

Expected evidence: path-scoped commits and pointer order pass; modification of a
protected pre-existing dirty path or an unrecorded commit fails.

### AC-22 - Dirty Baselines Are Content-Addressed

- Given both repositories contain tracked and untracked pre-existing user work,
- When the Goal records its baseline before branch creation,
- Then every path entry includes repository-relative path, file type, mode, size,
  SHA-256 for file/symlink content, Git status, index blob ID when present, HEAD blob
  ID when present, and an aggregate sorted-manifest hash. Later checks detect content
  changes, deletion, type/mode changes, and replacement of protected untracked files.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_git_content_baseline.py
python3 scripts/check_goal_git_isolation.py \
  --superproject /data-1/code/verl \
  --submodule /data-1/code/verl/recipe \
  --baseline-root /data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline
```

Expected evidence: modification, deletion, replacement, chmod, symlink-target change,
and manifest truncation fixtures fail; unchanged protected work passes.

### AC-23 - Goal Completion Requires Real Deployable Calibration

- Given sandbox AC-01 through AC-18 pass,
- When real L40S calibration has not run or returns `blocked`/`inconclusive`,
- Then the highest allowed status is `READY FOR OPERATIONAL CALIBRATION`; the Goal
  remains active and cannot be marked complete. `GOAL COMPLETE` requires AC-19 to
  return `deployable` from real local measurements and an independent reviewer to
  mark AC-19 through AC-25 `PASS`. A phase exceeding the 30-minute validation budget
  can never yield `deployable`.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_goal_completion_states.py
```

Expected evidence: missing, blocked, and inconclusive calibration cannot produce a
complete status; only reviewer-accepted deployable evidence can.

### AC-24 - Formal Execution Cannot Bypass Preflight

- Given any Stage123 formal queue or direct phase launch,
- When `SKIP_STAGE123_PREFLIGHT`, `SKIP_STAGE123_MACHINE_GATE`, a force/skip alias,
  a missing/failed/stale report, a report for another manifest/run/profile, or direct
  phase invocation attempts to bypass the launch gate,
- Then execution exits nonzero before Docker, Ray, tmux, or trainer startup. Queue and
  every phase launcher require the same content-addressed preflight receipt, whose
  report/manifest/profile hashes match and whose age is within the manifest policy.
  `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1` remains only a positive formal-launch guard
  and never substitutes for the receipt.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_no_preflight_bypass.py
bash scripts/check_experiment_workflow_fast.sh
```

Expected evidence: static checks reject skip/force variables in formal launch paths;
behavioral fixtures cover missing, failed, stale, mismatched, queue-skipped, and
direct-phase cases; all fail before side effects.

### AC-25 - Stage123 Dirty Files Use a Controlled Adoption Transaction

- Given the eight exact Stage123 recipe paths listed in the repository baseline and
  their content-addressed AC-22 records,
- When Milestone 2 adopts and migrates them to the manifest-native workflow,
- Then an adoption manifest records each original hash/type/mode/size, intended
  ownership, allowed milestone, resulting hash, and baseline-to-adopted diff. Only
  those listed files may change under adoption; all other dirty paths remain
  byte-identical. A fresh reviewer accepts the recipe diff and commit parentage before
  the superproject records the new submodule pointer.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_stage123_adoption.py
python3 scripts/check_goal_git_isolation.py \
  --superproject /data-1/code/verl \
  --submodule /data-1/code/verl/recipe \
  --baseline-root /data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline \
  --adoption-manifest docs/joint_training/manifests/stage123_dirty_adoption.json
```

Expected evidence: listed baseline-to-adopted changes pass with a reviewer record;
unlisted changes, missing original hashes, altered adoption scope, wrong recipe parent,
or a superproject pointer committed before recipe review fail.

### AC-26 - Validation Deadline Cleans Up Runtime and Releases GPUs

- Given a Stage1/Stage2/Stage3 pre-training measurement or formal
  `val_before_train` runtime whose complete metrics are absent at the 30-minute
  deadline,
- When the hard-wall controller fires,
- Then it marks the run blocked, stops the owning queue from advancing, requests
  graceful phase termination, escalates to force termination after a bounded grace
  period, removes run-owned Ray/Docker/tmux/process descendants, and verifies no GPU
  process/allocation remains attributable to that run. Cleanup is idempotent; an
  incomplete cleanup remains a blocking failure and emits evidence rather than
  claiming resources were released.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_validation_deadline_cleanup.py
```

Expected evidence: scaled-clock fixtures cover completion before deadline, graceful
shutdown, forced shutdown, orphan descendants, cleanup-command failure, repeated
cleanup, and the historical 76-minute step-0 trace. Only complete cleanup reports
`resources_released=true`; the legacy run fixture also proves release gate blocked,
zero matching SQLite rows/W&B sync markers, and no residual runtime ownership.

### AC-27 - Notification Policy Is an Event State Machine

- Given local fake WxPusher delivery and run-state fixtures,
- When the shared notification policy observes runtime transitions,
- Then it sends at most one message per deduplication key for exactly three events:
  `run_started` after first training step or complete formal validation metrics,
  `run_failed` after terminal failure/cleanup evidence, and
  `user_decision_required` after a blocking decision gate. Tmux/container creation,
  model loading, incomplete validation, healthy polling, unchanged metrics, and phone
  delivery/reply state emit no authorization and no periodic notification.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_notification_policy.py
```

Expected evidence: all three positive events, every listed non-event, deduplication,
secret redaction, local-path inclusion, and fake-delivery failure are covered. A
delivery failure is recorded locally but never changes launch/release state.

## Required Execution Order

1. Milestone 0 establishes fixtures and baseline.
2. Milestone 1 implements the preflight estimator and budget gate.
3. Milestone 2 introduces the manifest and migrates Stage123.
4. Milestone 3 audits principles using the gates now available.
5. Milestone 4 assembles fast/full integration gates.
6. Milestone 5 performs approved operational calibration and dual-repository audit.
7. A fresh reviewer executes every required AC command.

No milestone may start until all required ACs from the previous milestone pass.

## Commit and Workspace Rules

1. Create `codex/experiment-execution-reliability` in both repositories from the
   recorded baseline commits, while preserving dirty-path snapshots.
2. Commit each milestone separately after its local verification passes; commit
   recipe first when affected, then the superproject and submodule pointer.
3. Stage only declared paths and do not stage or commit unrelated dirty changes.
4. Put test scratch under `/data-1/tmp/verl_agent_scratch/experiment_workflow/`.
5. Preserve failed-run logs and release evidence; do not clean checkpoints or
   W&B staging as part of this goal.

## Stop-and-Ask Triggers

Stop implementation and ask the user when:

1. an AC would require weakening full validation or the 8K response contract;
2. measured memory suggests more reward workers but no safe RSS budget can be
   demonstrated;
3. the manifest cannot represent an existing Stage123 behavior without changing
   experiment semantics;
4. acceptance would require a real external account/service;
5. the same blocker repeats three times;
6. a required upstream artifact is missing or its provenance cannot be proven;
7. implementation would need deleting or rewriting unrelated user changes.
8. any soft preflight threshold fails. When the Goal is unattended, send one guarded
   WxPusher `needs decision` message with background, evidence, cost, recommendation,
   and local paths, then stop. Do not treat WxPusher as approval.
9. any real pre-training measurement or formal `val_before_train` phase reaches 30
   minutes without complete validation metrics. Stop the affected runtime, preserve
   evidence, send one `run_failed` notification when unattended, and do not continue
   to consume all GPUs.

## Approved Operational Decisions

The user approved these boundaries on 2026-07-11:

1. Stop the pre-Goal Stage123 run
   `CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V1_1783777744`.
   It stopped at observed step 0 after approximately 76 minutes in
   `val_before_train`; it is local diagnostic evidence only and must be rerun through
   the completed workflow.
2. L40S resources may be used for operational calibration.
3. Each Stage1/Stage2/Stage3 pre-training measurement has a maximum wall-clock budget
   of 30 minutes.
4. WxPusher sends once when a run is verified started, once when it fails, or once
   when human judgment is required; it does not send routine healthy-status updates.

## Independent Acceptance Contract

The implementer may report local verification but may not mark this goal accepted.
A fresh reviewer must:

1. run all AC verification commands in both repositories;
2. report AC-01 through AC-27 as `PASS`, `FAIL`, or `WEAKENED`;
3. inspect commits and tests for skipped, deleted, loosened, or trivial checks;
4. confirm acceptance used no real external service;
5. write the final review under `docs/joint_training/codereview/active/` and move
   it to `completed/` only after every required AC passes.

The goal is complete only when every AC is reviewer-owned `PASS` and none is
`WEAKENED`.
