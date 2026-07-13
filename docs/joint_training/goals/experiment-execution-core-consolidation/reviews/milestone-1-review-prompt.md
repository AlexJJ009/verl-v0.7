# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `Milestone 1`
- Base commit: `390be933`
- Candidate commit: `1cbea348`
- Applicable ACs: `AC-05,AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/experiment-execution-core-consolidation && goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-execution-core-consolidation && git show --stat --oneline 1cbea348 && git diff --check 390be933...1cbea348

## Additional Task-Specific Focus

Independently review the inventory completeness, exact 21-file production budget set, every legacy class disposition, AC-05 CPU feasibility, AC-08 independent deletion maps, protected asset exclusion, and >300-line justifications. Do not review implementation not yet authorized by Milestone 1.

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
