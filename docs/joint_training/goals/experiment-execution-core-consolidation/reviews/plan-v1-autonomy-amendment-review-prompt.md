# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Amendment Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `2f743322`
- Candidate commit: `WORKTREE`
- Applicable ACs: `AC-09 and Runtime Contract`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/experiment-execution-core-consolidation; goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-execution-core-consolidation

## Additional Task-Specific Focus

Review F-PLAN-03 only: confirm one user start authorizes autonomous Milestones 1-6, automatic transitions remain gated by checks/review/runtime, routine IN_SCOPE work does not require user approval, and stop conditions still protect AC changes, contradictions, convergence, protected assets, GPU/external services, and new outcomes.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
