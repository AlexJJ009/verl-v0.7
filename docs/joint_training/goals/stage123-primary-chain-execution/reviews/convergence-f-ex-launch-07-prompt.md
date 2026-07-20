# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Convergence Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Candidate commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Applicable ACs: `AC-01, AC-07, AC-08, AC-12`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; inspect scripts/experiment_execution_core.py, scripts/execution_results.py, recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh, scripts/stage123_phase_adapter.py, and focused batch/admission tests

## Additional Task-Specific Focus

F-EX-LAUNCH-07 recurs after the prior F-EX-M2-24 convergence fix. Determine the architectural cause and freeze an IN_SCOPE repair that makes batch admission occur once per immutable item, prevents later phases from reapplying wall-clock freshness, preserves full admission for direct phase invocation, preserves hash/identity binding and no retry/resume, and allows certified successful Control evidence to seed a new treatment-only Stage2->extraction->Stage3 execution identity. Do not implement.

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
