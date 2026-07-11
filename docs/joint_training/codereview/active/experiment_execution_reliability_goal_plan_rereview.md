# Independent Goal Plan Rereview: Experiment Execution Reliability

- Reviewer: second independent reviewer
- Date: 2026-07-11
- Scope: plan review only; no implementation performed
- Plan reviewed: `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- Superproject observed HEAD: `6a86aa27591178f9acf774fe107b1219e099af99`
- Recipe observed HEAD: `763aab506cd38bc7ff9fccfd9a079840620c37c5`

## Independent Review Basis

This review was performed from the plan, reviewer prompt, Goal Plan skill, test
engineering playbook, and live status of both repositories. It does not adopt the
first reviewer's verdict. The plan correctly preserves full HumanEval+, MBPP+, and
LiveCodeBench formal validation, `MAX_RESPONSE_LENGTH=8192`, one canonical L40S
resource profile across Stage1/Stage2/Stage3, local-only acceptance, and fresh
reviewer-owned final acceptance.

The live repository state matches the two baseline commits named by the plan, but
both worktrees are dirty. The superproject has tracked and untracked changes. The
recipe submodule has eight untracked Stage123 files, including the queue, monitor,
resource profile, phase launchers, preflight, and idle watchdog. These files are
pre-existing assets, not clean-baseline content, and require content-addressed
isolation before Goal implementation starts.

## AC Self-Verifiability Table

| AC | Verdict | Independent assessment |
| --- | --- | --- |
| AC-01 | PASS | Fixture-based JSON/Markdown baseline and explicit unknown values are deterministic and reviewer-runnable. |
| AC-02 | PASS | The failure taxonomy is finite, named, fixture-driven, and has a required unknown fallback. |
| AC-03 | NEEDS SCAFFOLDING | Fixture mode verifies schema and semantic invariants, but representative topology and estimator fidelity depend on AC-19 operational evidence. This is acceptable only if AC-19 remains completion-blocking. |
| AC-04 | NEEDS SCAFFOLDING | The blocking behavior is testable, but the plan does not enumerate the canonical soft-threshold keys and limits in the acceptance contract. A manifest author could classify an unsafe performance signal as soft. |
| AC-05 | PASS | Improvement, regression budgets, semantic hashes, and neutral-result handling are numerically defined and fixture-verifiable. |
| AC-06 | PASS | The refusal path, tmux requirement, scratch roots, and non-experiment labeling are machine-testable without consuming GPU. |
| AC-07 | PASS | Required manifest ownership fields and invalid cases are explicit and locally verifiable. |
| AC-08 | PASS | Queue/monitor identity equivalence and prohibition of duplicate arrays are testable from rendered output and static checks. |
| AC-09 | NEEDS SCAFFOLDING | Local fake release is appropriate, but “fresh reviewer-owned release decision” has no schema, signer/identity rule, expiry, or exact verification command independent from AC-20. |
| AC-10 | PASS | Bounded historical classification permits unresolved history without silently omitting it. |
| AC-11 | NEEDS SCAFFOLDING | Baseline commits and rename semantics are defined, but the version-controlled allowlist's trusted creation/update protocol and exact dual-repository inventory paths are not specified. |
| AC-12 | PASS | Stable ID, scope, evidence, tier, and When/Do/Otherwise fields are locally checkable. |
| AC-13 | PASS | Structural, checker, and judgment-only tiers plus gate reachability are explicit and testable. |
| AC-14 | PASS | Source resolution and `dropped-unverifiable` prevent invented historical evidence. |
| AC-15 | NEEDS SCAFFOLDING | The gate is valid, but the plan does not state concrete initial line/byte budgets for each startup-required file; the implementation could choose permissive budgets after seeing file sizes. |
| AC-16 | PASS | The fast gate has a measurable 120-second ceiling and forbids Docker, GPU, network, and real waits. |
| AC-17 | PASS | Mock external services, child exit propagation, and refusal on real endpoints are suitable deterministic acceptance checks. |
| AC-18 | PASS | Manifest-native Stage123 dry-run, physical `/data-2` checkpoint resolution, release fixtures, and no real launch are independently verifiable. |
| AC-19 | NEEDS SCAFFOLDING | This correctly makes real L40S calibration a deployability gate and defines 20% estimator error plus AC-05 budgets. However, “same deterministic benchmark contract” does not state minimum sample counts/duration for operational calibration or a repeated-run aggregation rule beyond AC-03's fixture contract. |
| AC-20 | NEEDS SCAFFOLDING | Run/report/policy/manifest binding, 24-hour expiry, and release blocking are good. The JSONL ledger is not tamper-evident or append-only by construction, approval-file authenticity is only asserted as “user-generated,” and no monotonic sequence/hash-chain/signature or protected writer is required. |
| AC-21 | NEEDS SCAFFOLDING | Recipe-first ordering and two commit ranges are present. The baseline snapshot contract records only unspecified “path/status” data, so it cannot prove that pre-existing tracked or untracked file contents were unchanged. It also does not define how pre-existing untracked Stage123 files become owned commits without being mistaken for Goal-created content. |

## P0 Blocking Findings

### P0-1: Override authorization is hash-bound but not authorization-authentic or ledger-tamper-evident

Plan lines 534-538 require a “user-generated” approval file and append a consumed
decision to JSONL. Hashing the report, policy, manifest, and run IDs prevents rebinding
those inputs, but any local process able to write the approval or ledger can fabricate,
delete, reorder, or rewrite entries. The stated requirement includes ledger
anti-tampering, yet AC-20 tests only malformed/expired/wrong-run inputs and does not
define an authenticity mechanism or append-only integrity check.

Required correction: define one locally verifiable authorization authority and one
tamper-evident ledger protocol. For example, an approval directory writable only by a
documented operator plus a checker for ownership/mode, and a ledger with monotonic
sequence numbers and previous-entry hashes stored in immutable run provenance. Tests
must cover edited, deleted, reordered, duplicated, and truncated ledger entries. Do not
use a real remote signing service.

### P0-2: Dirty-worktree isolation cannot prove preservation of pre-existing file contents

Plan lines 68-78 and AC-21 refer to path/status snapshots, while AC-21 claims unrelated
dirty paths are unchanged. Live state contains nine tracked modifications in the
superproject and eight untracked Stage123 files in recipe. A status-only snapshot cannot
prove byte preservation, detect edits to untracked files, or distinguish pre-existing
Stage123 content from Goal implementation changes.

Required correction: require a content-addressed baseline manifest for every tracked,
untracked, and submodule-dirty path, including file type, mode, size, and SHA-256 (plus
tracked index/blob identity where applicable). Define how an intended pre-existing
Stage123 file is adopted: either obtain a user-owned baseline commit before Goal branch
creation, or record its original content hash and require a path-specific patch/commit
ledger. Final acceptance must compare protected paths against those hashes and attribute
every changed path to a declared milestone.

### P0-3: Operational calibration is required for deployment but is not unambiguously required for Goal completion

AC-19 correctly says `blocked` and `inconclusive` keep formal deployment disabled, but
line 526 says sandbox implementation “may be accepted” when GPU execution is unavailable,
while the independent acceptance contract says every AC must PASS. This creates two
completion meanings: implemented-but-not-deployable versus Goal complete. It also leaves
the user-requested real performance optimization potentially unfinished.

Required correction: choose and name two statuses. Recommended: `IMPLEMENTATION READY`
may be reached with AC-01..AC-18 and AC-20..AC-21 passing, but the Goal remains incomplete
and formal queue launch remains blocked until AC-19 returns `deployable` and a fresh
reviewer marks AC-19 PASS. If the intended Goal can complete without calibration, split
deployment into a second serial Goal instead of calling the first one complete.

## P1 Findings

1. AC-04 does not establish an immutable canonical list of hard versus soft policy
   fields. The plan's boundary lists hard failure classes, but the schema must reject
   unknown/reclassified threshold keys and pin which performance metrics are overridable.
2. AC-09's override release review is under-defined. Add a local decision schema bound to
   run ID, override ledger entry hash, final artifact hashes, reviewer identity, issue time,
   and one-use release action, with negative fixtures.
3. AC-11 needs exact version-controlled inventory/allowlist paths in both repositories,
   ownership/update rules, and a requirement that allowlist changes themselves cannot
   exempt files changed in the same commit without reviewer approval.
4. AC-15 should pin initial byte and line budgets in the plan before implementation. A
   checker whose thresholds are authored after measuring the result is circular.
5. AC-19 needs operational sample size, repetitions, warmup exclusion, aggregation, maximum
   wall time, and confidence/inconclusive rules. A single noisy bounded run is insufficient
   to validate prediction error or a 10% optimization claim.
6. AC-21 should verify parentage and pointer transaction mechanically: each superproject
   commit that changes `recipe` must point to an existing reviewed recipe commit whose
   timestamp/topological order follows the declared milestone parent, and no later recipe
   dirty state may be silently included.
7. The reviewer prompt's required output path names the first review file rather than this
   rereview path. This user-provided rereview instruction overrides it for this pass, but
   future fresh-review prompts should accept an explicit output path to avoid collision.

## Missing or Under-Defined ACs

The following should be added or folded into AC-09, AC-11, AC-19, AC-20, and AC-21.

### Proposed AC-22 - Tamper-Evident Override Chain

- Given a valid locally authorized override and an existing override ledger,
- When the override is consumed or any historical ledger entry is edited, deleted,
  reordered, duplicated, or truncated,
- Then the checker accepts exactly one append with valid local authority, monotonic
  sequence, previous-entry hash, and run binding, and fails closed for every mutation.

### Proposed AC-23 - Content-Addressed Dirty Baseline

- Given both live dirty repositories before Goal branch creation,
- When the baseline collector records tracked, untracked, ignored-in-scope, executable,
  symlink, and submodule paths,
- Then it writes deterministic content/mode hashes and the final isolation checker proves
  every protected path is unchanged or explicitly attributed to one milestone commit.

### Proposed AC-24 - Operational Completion Boundary

- Given all sandbox ACs pass but real L40S calibration is absent, blocked, or inconclusive,
- When completion status is evaluated,
- Then implementation may be labeled ready for calibration, but the Goal and formal
  Stage1/Stage2/Stage3 deployment remain incomplete until reviewer-owned AC-19 evidence is
  `deployable`.

### Proposed AC-25 - Override Release Decision

- Given a successful run launched under a soft override,
- When release to local SQLite or fake W&B is requested,
- Then a fresh local reviewer decision bound to the override entry and final provenance is
  required, one-use, auditable, and rejected when stale, mismatched, reused, or fabricated.

## Goal Decomposition Decision

Keep one hard-ordered Goal. Milestone outputs form a real dependency chain: fixtures feed
the estimator, the estimator policy feeds the manifest, the manifest feeds queue/monitor/
release generation, and all of those feed operational calibration and final acceptance.
Splitting now would increase schema drift risk.

Within the Goal, preserve an explicit intermediate status after deterministic gates:
`READY FOR OPERATIONAL CALIBRATION`. This is not completion. If the user later chooses to
allow completion without real L40S evidence, then split Milestone 5 into a second serial
deployment Goal and change the objective accordingly before implementation.

## Preflight Checklist Before Execution

- [ ] Revise the plan for P0-1 through P0-3 and re-review with a fresh reviewer.
- [ ] Record concrete startup-document line/byte budgets before implementation.
- [ ] Pin canonical hard/soft threshold names and reject policy reclassification.
- [ ] Define operational calibration row counts, repetitions, aggregation, timeout, and
      inconclusive rules for all three phases.
- [ ] Define local approval authenticity and tamper-evident override ledger semantics.
- [ ] Define the separate override-to-release reviewer decision schema and tests.
- [ ] Name version-controlled superproject and recipe legacy inventory/allowlist files.
- [ ] Content-hash all current dirty tracked and untracked paths before branch creation.
- [ ] Resolve ownership of the eight existing untracked recipe Stage123 files without
      overwriting or silently treating them as Goal-created files.
- [ ] Create Goal branches from the stated commits only after snapshots are complete.
- [ ] Require recipe-first commits and superproject pointer commits per affected milestone.
- [ ] Use deterministic pytest fixtures and local fake HTTP/SQLite/W&B/WxPusher only.
- [ ] Keep full formal LCB/HumanEval+/MBPP+, 8192 response length, and one resource profile.
- [ ] Run GPU calibration in tmux and write only to declared local scratch/provenance roots.
- [ ] Have a fresh final reviewer run every command, inspect both commit ranges, audit test
      weakening/skips, and report ACs individually from reviewer-owned output.

PLAN VERDICT: NOT READY
