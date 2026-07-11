# Fourth Independent Goal Plan Review

## Scope and Independence

This review covers only the plan and launch/review contracts. It does not implement
the Goal or modify the main plan. The review was performed from the current files and
Git state without relying on earlier reviewer verdicts.

Reviewed inputs:

- `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- `docs/joint_training/plans/active/experiment_execution_reliability_goal_reviewer_prompt.md`
- `/data-1/agent-tools/goal_plan/codex/skills/goal-plan/SKILL.md`
- `/root/agent-core/workflows/TEST_ENGINEERING_PLAYBOOK.md`
- `/root/agent-core/skills/wxpusher-notify/SKILL.md`
- superproject `/data-1/code/verl` and recipe submodule live Git/worktree state

## Repository Reality Check

| Item | Observed state | Plan alignment |
| --- | --- | --- |
| Superproject branch | `feature/on-policy-wdl-sft` | Matches baseline table |
| Superproject HEAD | `6a86aa27591178f9acf774fe107b1219e099af99` | Exact match |
| Recipe branch | `codex/l40s-readiness` | Matches baseline table |
| Recipe HEAD | `763aab506cd38bc7ff9fccfd9a079840620c37c5` | Exact match |
| Recipe dirty set | Exactly eight untracked Stage123 paths | Exact match to adoption set |
| Superproject dirty set | Existing tracked and untracked work | Explicitly protected by AC-21/22 |
| Existing bypass | Queue currently contains preflight skip behavior | Explicit removal and regression required by AC-24 |
| Current positive guard | `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1` | Correctly classified as non-receipt guard |

The eight recipe paths listed by the plan exactly equal `git -C recipe
ls-files --others --exclude-standard`. This removes the earlier ambiguity between
protecting unrelated dirty work and adopting the Stage123 implementation.

## Acceptance-Criterion Review

Legend:

- `PASS`: sufficiently specified and directly verifiable with the stated evidence.
- `NEEDS SCAFFOLDING`: goal-ready, but the Goal must create the named checker/tests.
- `OPERATIONAL`: intentionally requires real local L40S evidence and cannot be
  satisfied by sandbox fixtures alone.

| AC | Verdict | Review |
| --- | --- | --- |
| AC-01 | NEEDS SCAFFOLDING | Baseline fields and fixture evidence are concrete; parser and fixtures must be implemented. |
| AC-02 | NEEDS SCAFFOLDING | Failure classes and deterministic classification contract are testable without remote services. |
| AC-03 | NEEDS SCAFFOLDING | Pins phases, datasets, repetitions, denominators, bins, RSS cadence, GPU-wait definition, 8192 context, and Stage2 topology. |
| AC-04 | NEEDS SCAFFOLDING | Nonzero blocking behavior and missing/hard/soft/bypass cases are explicit. AC-20 and AC-24 close the exception paths. |
| AC-05 | NEEDS SCAFFOLDING | Before/after metrics, semantic hash equality, 10% benefit, timeout/RSS budgets, and neutral classification are objective. |
| AC-06 | NEEDS SCAFFOLDING | GPU smoke has a positive operator guard, tmux requirement, scratch boundary, and no formal-score claim. It is not a preflight override. |
| AC-07 | NEEDS SCAFFOLDING | Manifest fields, validation failures, resource profile identity, and checkpoint mount constraints establish a usable single source of truth. |
| AC-08 | NEEDS SCAFFOLDING | Queue and monitor derivation is machine-comparable and rejects duplicated lifecycle arrays. |
| AC-09 | NEEDS SCAFFOLDING | Release evidence is bound to manifest identity/provenance and remains locally testable with fake W&B. |
| AC-10 | NEEDS SCAFFOLDING | Historical work is bounded by explicit classifications instead of unbounded archaeology. |
| AC-11 | NEEDS SCAFFOLDING | Immutable baseline/cutoff and legacy allowlist make future manifest enforcement machine-decidable. |
| AC-12 | NEEDS SCAFFOLDING | Stable rule ID plus When/Do/Otherwise/scope/evidence/tier is explicit and lintable. |
| AC-13 | NEEDS SCAFFOLDING | Correctly replaces machine-checkable prose with checker pointers and tests drift. |
| AC-14 | NEEDS SCAFFOLDING | Allows `dropped-unverifiable` and forbids invented history, preserving evidence boundaries. |
| AC-15 | NEEDS SCAFFOLDING | Startup-document line and byte budgets are fixed and mechanically enforceable. |
| AC-16 | NEEDS SCAFFOLDING | Fast gate has a 120-second budget, no Docker/GPU/network dependency, and child-exit preservation. |
| AC-17 | NEEDS SCAFFOLDING | Full gate uses local fake endpoints and explicitly verifies pipeline exit propagation. |
| AC-18 | NEEDS SCAFFOLDING | Manifest-native Stage123 end-to-end dry-run covers queue/monitor/release without creating false success records. |
| AC-19 | OPERATIONAL | Real local L40S calibration is intentionally required, bounded, provenance-pinned, tmux-run, and produces deployable/blocked/inconclusive. |
| AC-20 | NEEDS SCAFFOLDING | WxPusher only alerts. Phone delivery/reply cannot approve. Resumption requires interactive decision, reviewed Git change, and fresh passing preflight. |
| AC-21 | NEEDS SCAFFOLDING | Dual-repository isolation, recipe-first commit order, pointer ordering, protected dirty paths, and two commit ranges are explicit. |
| AC-22 | NEEDS SCAFFOLDING | Content baseline includes type/mode/size/hash/status/blob identities and aggregate hash; mutation fixtures cover relevant tampering. |
| AC-23 | NEEDS SCAFFOLDING + OPERATIONAL | State-machine tests are sandbox-verifiable, while final completion correctly depends on reviewer-owned real deployable AC-19 evidence. |
| AC-24 | NEEDS SCAFFOLDING | Explicitly removes `SKIP_STAGE123_PREFLIGHT`, `SKIP_STAGE123_MACHINE_GATE`, aliases, and direct-phase bypass. Queue and every phase require the same fresh content-addressed receipt before side effects. ALLOW cannot substitute. |
| AC-25 | NEEDS SCAFFOLDING | Exact eight-path adoption records original/result hashes and diffs; all other dirty paths remain byte-identical; recipe review/parentage precedes superproject pointer. |

No AC is intrinsically untestable. AC-19 is an accepted operational-only boundary,
and AC-23 prevents sandbox completion from being mislabeled as Goal completion.

## P0 Findings

None.

The prior high-risk boundaries are now represented as executable acceptance
requirements rather than prose aspirations:

1. Formal queue and direct phase paths share one fresh content-addressed receipt.
2. Skip variables and equivalent force aliases must be removed and regression-tested.
3. `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1` cannot create or replace a receipt.
4. The exact eight recipe files use a controlled content-addressed adoption transaction.
5. Unrelated dirty work must remain byte-identical.
6. WxPusher informs the user but cannot authorize execution.
7. A user decision becomes a reviewed manifest/policy commit followed by a fresh pass.
8. Only real local L40S `deployable` evidence permits Goal completion.

## P1 Findings

1. AC-24 and AC-25 are numbered under Milestone 5 even though their implementation
   naturally occurs during Milestone 2 and their regression integration occurs during
   Milestone 4. This is not contradictory: the required execution order, launch prompt,
   and final acceptance require all AC-01 through AC-25. The implementer should track
   their implementation dependencies explicitly and must not defer bypass removal until
   after operational calibration.

2. The plan status header still says pending re-review. That is accurate before this
   report is consumed and is not a readiness blocker. The plan author may update status
   separately; this reviewer did not modify the plan.

## Missing Acceptance Criteria

None required for goal readiness.

The following concerns are already covered and do not need new ACs:

- static and behavioral bypass regression: AC-24;
- missing, failed, stale, mismatched, and wrong-run receipt: AC-24;
- direct phase launch before any Docker/Ray/tmux/trainer side effect: AC-24;
- exact dirty adoption list and original/result content hashes: AC-25;
- recipe commit parentage and superproject pointer order: AC-21/25;
- full HumanEval+/MBPP+/LiveCodeBench, response length 8192, and one profile: boundaries plus AC-03/07/19;
- no real remote acceptance evidence: boundaries plus AC-17/20;
- user decision and phone-notification separation: AC-20;
- incomplete operational calibration cannot finish the Goal: AC-23.

## Decomposition Decision

Keep one serial Goal.

Milestone outputs are hard dependencies: fixtures feed the estimator; the estimator and
policy feed the manifest receipt; the manifest feeds queue/monitor/release generation;
those gates determine which prose rules remain; the integrated gate precedes operational
calibration. Splitting these into independently accepted Goals would create avoidable
schema, policy, and provenance drift. Per-milestone commits and independent final review
provide adequate control within one Goal.

## Execution Preflight Checklist

- Confirm both baseline branches and commits still exactly match the plan before snapshots.
- Capture content-addressed superproject and recipe dirty baselines before branch creation.
- Confirm the recipe untracked adoption set is exactly the eight declared paths; stop on drift.
- Create `codex/experiment-execution-reliability` independently in recipe and superproject.
- Establish deterministic pytest fixtures and local fake W&B/WxPusher endpoints before runtime edits.
- Keep fast gate free of Docker, GPU, network, and real account dependencies.
- Preserve full HumanEval+, MBPP+, LiveCodeBench, `MAX_RESPONSE_LENGTH=8192`, and the canonical profile hash.
- Remove both named skip variables and scan for equivalent force/skip aliases.
- Require the same fresh receipt in formal queue and all Stage1/2/3 direct phase launchers before side effects.
- Test that the ALLOW guard without a valid receipt fails.
- Commit recipe changes first; independently review recipe diff and parentage before the superproject pointer commit.
- Stage only declared paths and verify every unrelated dirty path remains byte-identical.
- Use tmux for the approved bounded L40S calibration; do not contact remote services.
- Treat bounded scores as infrastructure evidence only.
- On soft threshold failure, write `USER_DECISION_REQUIRED`, send at most one guarded notification when unattended, and stop.
- Require an interactive user decision, reviewed manifest/policy commit, and newly passing receipt before resumption.
- Let a fresh acceptance reviewer run every AC command and inspect both commit ranges.
- Do not mark complete unless real calibration is `deployable` and all AC-01..AC-25 are reviewer-owned PASS.

PLAN VERDICT: READY
