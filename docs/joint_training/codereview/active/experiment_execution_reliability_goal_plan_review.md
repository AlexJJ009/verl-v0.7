# Independent Goal Plan Review: Experiment Execution Reliability

- Reviewed: 2026-07-11
- Reviewer role: independent plan reviewer; no implementation performed
- Plan: `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- Branch observed: `feature/on-policy-wdl-sft`
- Worktree state: dirty in both the superproject and `recipe` submodule

## Executive Finding

The plan has a strong serial structure and correctly preserves full HumanEval+,
MBPP+, LiveCodeBench, `MAX_RESPONSE_LENGTH=8192`, the common Stage1/2/3 L40S
profile, local-only acceptance fakes, and independent final acceptance. It is not
ready to launch because four execution boundaries remain under-defined:

1. fixture-only acceptance can certify the performance estimator without proving
   that a proposed optimization improves or is safe on the real L40S runtime;
2. the force-launch override has no authorization, audit destination, or
   release-gate consequence;
3. the manifest cutoff is described as a date but has no immutable machine
   definition;
4. milestone commits do not define the two-repository transaction required by
   the dirty `recipe` submodule and dirty superproject.

These are plan defects, not implementation defects. The main plan must be updated
and independently re-reviewed before Goal execution.

## AC Self-Verifiability

| AC | Verdict | Review basis |
|---|---|---|
| AC-01 | PASS | Local fixture, JSON, Markdown, schema, and secret checks can prove the stated baseline contract. |
| AC-02 | PASS | Eight deterministic local fixtures and an explicit `unknown` result are sandbox-verifiable. |
| AC-03 | NEEDS SCAFFOLDING | Fixture mode can prove schema and semantic invariants, but not real rollout latency, RSS, or GPU-wait estimates. Add calibrated synthetic fixtures and separate operational calibration evidence. |
| AC-04 | NEEDS SCAFFOLDING | Threshold failures are testable, but the force override lacks authorized actor/policy, append-only audit path, and downstream release behavior. |
| AC-05 | NEEDS SCAFFOLDING | Semantic downscope rejection is testable, but the AC accepts neutral/no-regression and never requires measured improvement for an optimization claimed as performance work. |
| AC-06 | EXTERNAL | The guard is sandbox-verifiable; actual L40S execution is explicitly operational. It needs a required post-implementation operational calibration record before the optimized workflow is declared deployable. |
| AC-07 | PASS | Schema validation and normalized rendering can prove ownership and reject invalid identities/provenance locally. |
| AC-08 | PASS | Shared generated output and mutation tests can prove queue/monitor identity is not independently duplicated. |
| AC-09 | PASS | Local fake checkpoint, metrics, SQLite, and W&B directories can prove release blocking without remote services. |
| AC-10 | PASS | A bounded filesystem inventory with explicit unresolved classification is deterministic and does not require historical reconstruction. |
| AC-11 | NEEDS SCAFFOLDING | The cutoff date/baseline is not defined as a committed manifest, Git diff base, allowlist, or schema field, so fixtures cannot prove enforcement on the real tree. |
| AC-12 | PASS | Catalog schema and malformed fixtures can prove stable IDs and When/Do/Otherwise fields. |
| AC-13 | NEEDS SCAFFOLDING | Classification fixtures are testable, but the plan does not define how a machine-checkable rule proves that its referenced gate exists and is executed by fast/full gates. |
| AC-14 | PASS | Source-path/line evidence and explicit `dropped-unverifiable` status can be audited locally without inventing history. |
| AC-15 | NEEDS SCAFFOLDING | A time-bounded exception can make any oversized real entrypoint pass unless approver, expiry, maximum size, and failure-on-expiry are specified. Current `AGENTS.md` and `CLAUDE.md` are 317 and 268 lines. |
| AC-16 | PASS | A local timed, network-denied fast gate is independently executable; the implementation must make network isolation observable. |
| AC-17 | PASS | Docker/local fakes plus injected child failure can prove integration and exit propagation without real W&B/WxPusher. |
| AC-18 | NEEDS SCAFFOLDING | The dry-run contract is testable, but `git status --short --ignored` cannot prove no durable result in an already dirty worktree; require before/after snapshots of declared durable roots and both Git repositories. |

No AC requires a real remote service. AC-06 alone has an accepted local-hardware
operational boundary, but its evidence is currently optional rather than a gated
deployment/calibration outcome.

## P0 Contradictions And Blocking Gaps

### P0-1: Goal objective requires performance optimization, but acceptance can pass without real performance evidence

The objective promises prediction of scorer latency, host-memory risk, and GPU
starvation, while AC-03 runs only in fixture/mock mode and AC-06 says actual GPU
execution is not required for sandbox acceptance
(`experiment_execution_reliability_goal.md:10-18`, `122-146`, `183-200`). A
fixture can prove calculations but cannot establish calibration error, real peak
RSS, or whether an optimization improves the L40S bottleneck.

Required plan repair: split `implemented and sandbox-accepted` from
`operationally calibrated/deployable`. Add a required operational acceptance
record for all three phase adapters on the L40S, with bounded workload, exact
profile hash, image ID, model/data provenance, before/after metrics, calibration
error or confidence bounds, and an explicit outcome (`deployable`, `blocked`, or
`inconclusive`). It must remain non-experiment evidence and must not contact a
remote service.

### P0-2: Force launch can bypass a safety gate without a defined authority or release consequence

AC-04 permits formal launch whenever a force flag has a nonempty reason
(`experiment_execution_reliability_goal.md:148-163`). Any caller could satisfy
that textual condition. The plan does not define who may authorize it, where the
append-only record lives, whether a forced run is eligible for DB/W&B release,
or which thresholds are non-overridable.

Required plan repair: define non-overridable structural/provenance/memory limits;
define allowed overridable research thresholds; require operator identity,
reason, timestamp, report hash, policy hash, and authorization token/file in an
append-only local ledger; state whether forced runs are blocked from release or
require an additional reviewer-owned release decision. Add Given/When/Then tests
for unauthorized, incomplete, tampered, expired, and authorized override cases.

### P0-3: The manifest cutoff cannot be enforced deterministically

The plan makes manifests mandatory for every new experiment after acceptance and
AC-11 refers to a cutoff date (`experiment_execution_reliability_goal.md:20-23`,
`291-306`), but neither a date, baseline commit, tracked legacy allowlist, nor
Git-diff rule is defined. Filesystem mtimes are not a valid source of truth.

Required plan repair: introduce a version-controlled legacy inventory/allowlist
with a schema version and baseline commit. The gate must classify changed/added
runnable scripts relative to an explicit Git base or reject any runnable script
not present in the immutable legacy inventory and not manifest-native. Define how
renames and submodule files are handled.

### P0-4: Commit rules assume one repository while runtime scripts live in a dirty submodule

The plan requires one milestone commit and forbids staging unrelated dirty work
(`experiment_execution_reliability_goal.md:458-465`). The manifest and Stage123
runtime artifacts are under `recipe`, which is a separate submodule currently on
`codex/l40s-readiness` with untracked Stage123 files; the superproject is also
dirty. A superproject branch cannot directly commit those files, and updating the
submodule pointer can accidentally include or obscure unrelated work.

Required plan repair: define a two-repository protocol before execution:

- exact branch names for superproject and submodule;
- baseline commits and pre-existing dirty-path manifests for both;
- submodule-first commit order, then superproject pointer commit;
- path-scoped staging and reviewer checks for unrelated changes;
- whether each milestone requires one commit per affected repository;
- stop condition when an intended file already contains unrelated user edits;
- final reviewer commands over both commit ranges.

## P1 Clarifications

1. AC-03 says one Stage2 `joint/fixed-Model2` source, but Stage2 semantics should
   name the exact source required by the active Stage123 design. Otherwise a mock
   may validate the wrong model topology.
2. AC-03 must define deterministic row count/selection, warmup exclusion,
   repetitions, timeout denominator, score distribution bins, RSS sampling
   interval, and the formula for estimated GPU-wait fraction.
3. AC-05 needs a measurable decision rule. Define which metric is primary,
   allowed regressions, uncertainty handling, and when `neutral` means
   `acceptable correctness refactor` versus `failed performance optimization`.
4. AC-09 should state whether preflight reports and manifest normalization are
   content-addressed and copied into immutable run provenance before launch.
5. AC-13 should require every `machine-check` rule to name a command/test ID and
   prove reachability from `check_experiment_workflow_fast.sh` or the full gate.
6. AC-15 exceptions need stable owner, reason, issue/plan reference, expiry,
   maximum permitted lines/bytes, and automatic failure after expiry.
7. AC-16's "no remote network call" needs an enforceable test mechanism, such as
   a network-disabled namespace/container or a socket-deny test double; absence
   of known URLs is insufficient.
8. AC-18 must specify that dry-run cannot start Docker/Ray/tmux unless explicitly
   mocked and must compare before/after hashes or row counts for SQLite,
   release-gate JSONL, W&B markers, checkpoint roots, and monitor status roots.

## Missing Or Under-Defined ACs

The following ACs should be added or folded into existing ACs before re-review.

### Proposed AC-19 - Operational Calibration Produces A Deployment Decision

- Given sandbox acceptance is green, the canonical profile and image are pinned,
  and an operator explicitly approves a bounded local L40S calibration,
- When the Stage1/Stage2/Stage3 calibration runs in tmux against declared local
  model/data artifacts,
- Then it records provenance and before/after real metrics, compares estimator
  predictions with observations, makes no formal experiment-quality claim, and
  emits `deployable`, `blocked`, or `inconclusive` without contacting a remote
  service.

### Proposed AC-20 - Override Authorization Is Auditable And Bounded

- Given a preflight policy failure,
- When formal launch is requested with an override,
- Then non-overridable failures remain blocked and an overridable failure proceeds
  only with a valid local authorization record containing actor, reason,
  timestamp, report/policy hashes, expiry, and release disposition; tampering or
  missing fields exits nonzero.

### Proposed AC-21 - Dual-Repository Change Isolation

- Given recorded baseline commits and dirty-path manifests for the superproject
  and `recipe` submodule,
- When each milestone is committed,
- Then only declared milestone paths change, the submodule commit precedes the
  superproject pointer update, pre-existing dirty paths remain untouched, and a
  reviewer can reproduce the exact two-repository commit range.

AC-11 should additionally be rewritten to define its immutable cutoff source.
AC-15 and AC-18 should be strengthened with the P1 details above.

## Decomposition Decision

Use **one serial Goal**, not independent parallel Goals. The hard dependency chain
is valid:

1. baseline/fixtures define evidence and failure classes;
2. estimator and policy produce report artifacts;
3. manifest consumes those policy/report contracts;
4. principles catalog points machine-checkable rules at the implemented gates;
5. integration and Stage123 dry-run consume every earlier artifact.

Splitting these into independently accepted Goals would create temporary duplicate
sources of truth and cross-goal schema drift. Within the single Goal, use separate
milestone commits and fresh reviewer passes. The operational calibration can be a
final gated milestone in the same Goal because it determines deployability, while
sandbox acceptance remains reproducible.

## Preflight Checklist For The Implementing Goal

- [ ] Main plan has been repaired and receives a fresh independent `READY` review.
- [ ] Superproject and `recipe` baseline commit IDs are recorded.
- [ ] Dirty paths in both repositories are recorded and protected before branch creation.
- [ ] Branch and commit protocol is defined for both repositories.
- [ ] `python3 -m pytest` is available; exact Python environment is recorded.
- [ ] YAML/JSON-schema dependencies are pinned or a standard-library fallback is defined.
- [ ] Fixture schema, secret scan, and deterministic clock/randomness strategy exist.
- [ ] W&B, WxPusher, Hugging Face, HTTP, and socket access use local fakes or are denied.
- [ ] Docker fallback distinguishes `SKIP operational calibration` from sandbox test failure.
- [ ] GPU calibration approval, tmux naming, scratch root, cleanup, and timeout are defined.
- [ ] Full HumanEval+, MBPP+, and LiveCodeBench formal-validation paths are asserted.
- [ ] Bounded preflight rows are labeled infrastructure-only and cannot enter result registries.
- [ ] `MAX_RESPONSE_LENGTH=8192` and one Stage1/2/3 profile hash are asserted at render and launch.
- [ ] Reward semantics and timeout-result semantics are content-addressed and comparison-gated.
- [ ] Reward-worker concurrency cannot increase without measured RSS and a non-overridable memory gate.
- [ ] Manifest cutoff uses a committed baseline/allowlist, not date or mtime inference.
- [ ] Fast gate runs without Docker/GPU/network and enforces its 120-second budget.
- [ ] Full gate preserves piped child exit codes and uses only deterministic local endpoints.
- [ ] Dry-run checks before/after durable roots rather than relying on global `git status`.
- [ ] Failed/incomplete fixtures cannot produce SQLite rows or W&B synced markers.
- [ ] Reviewer final commands cover both repositories and audit skipped/loosened tests.
- [ ] Implementer cannot write `ACCEPTED`; only the fresh reviewer may do so from command output.

## Boundary Audit Summary

- Full LiveCodeBench: preserved in the plan; bounded samples are correctly labeled
  infrastructure-only.
- Response length: preserved at `8192` across Stage1/2/3; current profile also
  declares this value.
- Common resource profile: explicitly required, but operational calibration must
  prove the same rendered profile hash for all phases.
- External services: remote W&B/WxPusher/HF are forbidden for acceptance; keep
  network denial machine-checkable.
- Dirty worktree: acknowledged but not adequately handled across the two Git
  repositories.
- Independent acceptance: correctly separates implementer and reviewer; retain
  reviewer-owned commands and no self-certification.

PLAN VERDICT: NOT READY
