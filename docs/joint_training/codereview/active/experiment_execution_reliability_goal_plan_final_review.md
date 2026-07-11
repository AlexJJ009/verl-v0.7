# Experiment Execution Reliability Goal Plan Final Review

- Reviewer role: third independent Goal Plan reviewer
- Review date: 2026-07-11
- Plan: `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- Superproject observed branch/HEAD: `feature/on-policy-wdl-sft` / `6a86aa27591178f9acf774fe107b1219e099af99`
- Recipe observed branch/HEAD: `codex/l40s-readiness` / `763aab506cd38bc7ff9fccfd9a079840620c37c5`
- Scope: plan review only; no implementation or main-plan modification

## Executive Finding

The revised plan correctly removes signed approval files and automatic soft-gate
override semantics. It also makes guarded WxPusher notification explicitly
notification-only, requires a reviewed manifest/policy commit plus a fresh passing
preflight for a user decision to take effect, and makes real L40S `deployable`
calibration necessary for Goal completion.

The plan is not executable as written because two P0 contradictions remain in the
current Stage123 migration boundary:

1. the existing queue has `SKIP_STAGE123_PREFLIGHT=1`, which bypasses all preflight,
   but no AC explicitly requires removal and regression rejection of that concrete
   bypass; and
2. the Stage123 files that Milestone 2 must modify are protected pre-existing
   untracked recipe files, while AC-21/AC-22 reject modification of protected dirty
   paths without defining a content-addressed adoption transaction.

## AC Self-Verifiability Table

| AC | Verdict | Review |
| --- | --- | --- |
| AC-01 | NEEDS SCAFFOLDING | Metrics, explicit unknowns, fixture command, and outputs are measurable. The collector and fixture do not exist yet. |
| AC-02 | NEEDS SCAFFOLDING | Deterministic failure classes and fixture-driven classification are sandbox-verifiable once implemented. |
| AC-03 | NEEDS SCAFFOLDING | Pins datasets, repetitions, denominator, score bins, RSS cadence, GPU-wait definition, 8192 context, and exact Stage2 topology. Fixture mode is valid sandbox evidence. |
| AC-04 | NEEDS SCAFFOLDING | Correctly blocks hard and soft failures and rejects force/approval-file bypasses, but it does not name the existing `SKIP_STAGE123_PREFLIGHT` entry as a mandatory migration regression case. |
| AC-05 | NEEDS SCAFFOLDING | Defines primary metric, minimum improvement, non-regression budgets, semantic hashes, and neutral-result handling. |
| AC-06 | NEEDS SCAFFOLDING | The environment variable is a start guard for an explicitly requested operational smoke, not permission to bypass a failed report. Sandbox guard tests are sufficient for this AC. |
| AC-07 | NEEDS SCAFFOLDING | Manifest ownership fields and negative cases are concrete and locally testable. |
| AC-08 | NEEDS SCAFFOLDING | Queue/monitor derivation and mutation coupling are testable, but migration touches protected untracked Stage123 files without an adoption protocol. |
| AC-09 | NEEDS SCAFFOLDING | Local fake release paths, provenance hashes, post-review preflight freshness, and no forced release are testable. |
| AC-10 | NEEDS SCAFFOLDING | Historical inventory is bounded and unresolved archaeology does not block the Goal. |
| AC-11 | NEEDS SCAFFOLDING | Commit-based cutoff, dual-repository baselines, rename semantics, and tamper tests are sound. The exact version-controlled allowlist artifact/update protocol should be named. |
| AC-12 | NEEDS SCAFFOLDING | Trigger/action/failure plus ID, scope, evidence, and tier are machine-checkable. |
| AC-13 | NEEDS SCAFFOLDING | The three-tier structural/checker/prose policy is locally testable. |
| AC-14 | NEEDS SCAFFOLDING | Source-path resolution and `dropped-unverifiable` prevent invented history. |
| AC-15 | NEEDS SCAFFOLDING | Has hard maximums and expiry behavior. The implementation must declare per-file budgets before measuring the revised files. |
| AC-16 | NEEDS SCAFFOLDING | Fast gate has a 120-second bound and local network-denial evidence. |
| AC-17 | NEEDS SCAFFOLDING | Full gate checks child exit propagation and uses fake W&B/WxPusher endpoints. |
| AC-18 | NEEDS SCAFFOLDING | Manifest-native dry-run and durable-state before/after checks are strong and avoid real Docker/Ray/tmux. |
| AC-19 | EXTERNAL | Real local L40S measurement is intentionally operational-only. Inputs, repetitions, decision states, error budget, tmux, and checker boundary are explicit enough to be independently observed. |
| AC-20 | NEEDS SCAFFOLDING | Correctly stops, deduplicates a guarded fake notification in acceptance, treats phone delivery/replies as non-approval, and requires interactive user decision, reviewed Git change, and fresh passing preflight. |
| AC-21 | NEEDS SCAFFOLDING | Commit ranges, submodule-first order, pointer relation, parentage, and reviewer evidence are testable, but conflict with required edits to protected untracked Stage123 files. |
| AC-22 | NEEDS SCAFFOLDING | Content/type/mode/blob identities and tamper fixtures are strong, but no controlled adoption rule exists for dirty files intentionally brought into Goal ownership. |
| AC-23 | NEEDS SCAFFOLDING | Completion state machine correctly prevents mock-only or inconclusive calibration from completing the Goal. Final PASS still depends on external AC-19 evidence. |

`NEEDS SCAFFOLDING` means the AC is sufficiently machine-verifiable in principle
but its checker/tests/artifacts must be implemented. `EXTERNAL` is limited to the
explicit real-L40S operational boundary and is not satisfiable by reasoning or mock.

## P0 Findings

### P0-1: Existing Stage123 preflight bypass is not explicitly eliminated

Current code permits:

```text
SKIP_STAGE123_PREFLIGHT=1
```

to skip `stage123_preflight.py` entirely in
`recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh:9-13`.
This is a concrete automatic bypass of both hard and soft preflight outcomes. The
plan says no bypass exists and tests generic force flags, but AC-04, AC-08, AC-18,
and AC-20 do not explicitly require deleting this entry and proving that the
migrated formal queue cannot launch when preflight is skipped, absent, stale, or
failed.

Required plan change: add a Given/When/Then AC or explicit AC-04/AC-18 evidence that
enumerates `SKIP_STAGE123_PREFLIGHT`, requires its removal from formal execution,
and adds static plus behavioral regression fixtures for skip flags, missing report,
stale report, failed report, and direct phase-launch attempts. Deliberate dry-run
fixtures may inject a passing local report; they may not skip the gate.

### P0-2: Protected dirty baseline conflicts with required Stage123 migration

The recipe worktree currently contains eight untracked Stage123 files, including
the queue, monitor, resource profile, phase launchers, preflight, and watchdog.
Milestone 2 requires modifying at least the queue and monitor to consume the new
manifest. Repository Baseline lines 69-79 call the snapshots protected evidence;
AC-21 says modification of a protected pre-existing dirty path fails; AC-22 requires
replacement/change detection. There is no rule that permits intentionally adopting
a baseline-hashed dirty file into Goal ownership while proving unrelated content
was preserved.

This creates an impossible choice: leave the protected files unchanged and fail
AC-08, or modify them and fail AC-21/AC-22.

Required plan change: define a content-addressed adoption transaction before
Milestone 2. It must name the exact adoptable recipe paths, record their baseline
hash/type/mode, require user confirmation or pre-declared plan ownership, require a
focused first recipe commit preserving reviewable baseline-to-adopted diffs, forbid
adoption of any unlisted dirty path, and make the isolation checker compare adopted
paths against their recorded content rather than requiring byte identity. The
superproject pointer commit must reference that exact reviewed recipe commit.

## P1 Findings

1. **Allowlist artifact is under-specified.** AC-11 requires version-controlled
   inventories and exact legacy runnable paths but does not name the two allowlist
   files, their schema, owner/update command, or whether an allowlist change itself
   requires independent review. Naming these artifacts would make the cutoff less
   implementation-dependent.
2. **Operational sample contract is indirect.** AC-19 says it uses AC-03's contract,
   which pins deterministic row IDs/counts, but the actual counts are not visible in
   the plan. They may be stored in the benchmark manifest, but the plan should require
   its normalized hash in calibration evidence and state a maximum per-phase/runtime
   budget so an unattended calibration cannot grow without review.
3. **WxPusher cooldown ownership should be explicit.** AC-20 requires deduplication,
   but the plan does not pin the leader-lock/cooldown key and durable state location.
   Define it from run identity plus failed-report hash so queue and monitor cannot
   both send the same decision alert.
4. **Reviewer prompt output path is stale for this invocation.** The checked reviewer
   prompt names `experiment_execution_reliability_goal_plan_review.md`, while this
   review was explicitly requested at the separate final-review path. This does not
   weaken the plan contract, but the reusable prompt should accept or name the current
   review destination to avoid overwriting earlier reviews.

## Missing Acceptance Criteria

### Proposed AC-24 - Formal Launch Cannot Skip Preflight

- Given a manifest-native formal queue and all supported launch entrypoints,
- When a caller supplies legacy skip/force variables, omits the report, supplies a
  stale or failed report, or invokes a phase launcher outside the manifest lifecycle,
- Then formal execution exits nonzero before Docker/Ray/tmux training starts; static
  inspection and behavioral fixtures prove `SKIP_STAGE123_PREFLIGHT` and equivalent
  bypasses are absent.

### Proposed AC-25 - Dirty Stage123 Files Use a Reviewed Adoption Transaction

- Given the content-addressed recipe baseline and an exact plan-owned adoption list,
- When Milestone 2 migrates an existing untracked Stage123 file,
- Then its baseline hash is preserved in adoption evidence, only allowlisted paths
  may change, the baseline-to-commit diff is independently reviewable, unrelated
  dirty paths remain byte-identical, and the superproject pointer targets the exact
  accepted recipe commit.

These may be folded into AC-04/AC-18 and AC-21/AC-22 respectively, but the resulting
Given/When/Then and verification evidence must remain explicit.

## One Goal or Split

Keep one hard-ordered Goal. Baseline fixtures feed the estimator; the estimator and
policy feed the manifest lifecycle; manifest-native behavior feeds rule migration
and integrated gates; sandbox gates feed operational calibration; calibration and
dual-repository evidence feed final acceptance. Splitting these into independently
completable Goals would weaken the single source of truth and make schema drift more
likely. The operational calibration is a final milestone/state boundary, not a
separate Goal that permits the first Goal to claim completion early.

## Preflight Checklist for the Next Review

- [ ] Main plan explicitly eliminates `SKIP_STAGE123_PREFLIGHT` and equivalent
      skip/force/direct-launch paths from formal execution.
- [ ] Behavioral tests prove failed, missing, and stale preflight cannot launch any
      Stage123 phase; static tests reject reintroduced bypass variables.
- [ ] Exact pre-existing Stage123 recipe files eligible for adoption are listed.
- [ ] Adoption records type, mode, size, SHA-256, index/HEAD blob identity where
      applicable, and a baseline-to-adopted diff.
- [ ] Unlisted dirty paths remain protected and byte-identical.
- [ ] Recipe milestone commit parentage starts from
      `763aab506cd38bc7ff9fccfd9a079840620c37c5` and precedes the matching superproject
      pointer commit based on `6a86aa27591178f9acf774fe107b1219e099af99`.
- [ ] Version-controlled legacy allowlist paths, schema, hashes, and review/update
      protocol are named for both repositories.
- [ ] Fast gate uses deterministic local fixtures and network denial; full gate uses
      fake W&B and fake WxPusher only and preserves child exit codes.
- [ ] WxPusher decision notices contain no secrets, use a run/report-derived
      deduplication key, and cannot be interpreted as approval.
- [ ] A user decision changes reviewed manifest/policy Git content and a newly
      generated passing preflight; no chat reply, phone action, environment flag,
      approval file, or old report authorizes launch.
- [ ] Full HumanEval+, MBPP+, LiveCodeBench, `MAX_RESPONSE_LENGTH=8192`, exact
      Stage2 fixed-Model2 joint topology, and one canonical Stage1/2/3 profile remain
      hash-checked.
- [ ] Sandbox completion stops at `READY FOR OPERATIONAL CALIBRATION`.
- [ ] Real bounded L40S calibration runs in tmux and returns reviewer-verified
      `deployable`; `blocked`, `inconclusive`, or missing evidence cannot complete.
- [ ] Independent acceptance reruns every AC command, audits both commit ranges,
      checks for skipped/loosened/trivial tests, and confirms no real remote service
      was used as acceptance evidence.

PLAN VERDICT: NOT READY
