# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Readiness Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `5`
- Current milestone: `none`
- Base commit: `f2b4a1bc0c87f46cbcd2974c4a8b4fb8deef4548`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; inspect scripts/experiment_execution_core.py, scripts/stage123_manifest_monitor.py, the Stage123 public queue/impl split, scripts/training_queue_monitor.sh, and deletion budgets; verify batch/control capabilities extend the existing core and one persisted-event monitor rather than adding parallel authorities; verify three-run manifest/control wrappers, fallback/stop tests, and fresh Readiness admission; verify no training was launched

## Additional Task-Specific Focus

Return NOT_READY while batch routing/control are absent, the manifest remains two-run, the matched control is absent, or fresh Readiness is missing. Reject any design that creates a second batch queue, execution-state ledger, fallback engine, or monitor implementation. Require experiment_execution_core.py to remain sole transition authority, existing Stage123 queue to remain a thin adapter, one generalized persisted-event monitor, pure-script failure-to-next-experiment routing, no Agent supervision, no retry/resume/parameter correction, and explicit deletion of superseded duplicate logic.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.
9. For a Plan review, reject `READY` when any AC declares an absolute numeric performance or resource budget that has no recorded feasibility probe in the Plan's `Feasibility Probes` section, or whose budget contradicts the probe's measured floor.
10. When your only blocking findings are purely mechanical (formatting, patch context offsets, artifact or directory placement) with no behavioral or contract impact, say so explicitly and offer a light same-reviewer re-verification scoped to those findings instead of demanding a fresh full round.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
