# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `plan`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `Milestone 1`
- Base commit: `390be933`
- Candidate commit: `WORKTREE`
- Applicable ACs: `AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/experiment-execution-core-consolidation; independently sum the exact 15 active reliability docs paths at 390be933; confirm production 21 paths total 3763 and workflow tests total 4694

## Additional Task-Specific Focus

Review only the F-M1-02 Plan amendment: active reliability docs frozen baseline changes to exact 2091 lines, while <=650 ceiling, independent net-negative rule, outcome, exclusions, and all other ACs remain unchanged. Also verify inventory.md uses the same selector and value.

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
