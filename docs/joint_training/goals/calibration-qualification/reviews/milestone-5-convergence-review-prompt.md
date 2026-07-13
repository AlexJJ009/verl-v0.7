# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Convergence Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `71944750`
- Candidate commit: `f489ad2e`
- Applicable ACs: `AC-03, AC-04, AC-06, AC-08`

## Required Verification

Inspect Plan, F-M5-01 lifecycle, candidate diff 71944750..f489ad2e, and reviewer result. Run goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification.

## Additional Task-Specific Focus

F-M5-01 has reached two implementation-review rounds without closure. Determine whether the loop is caused by duplicated sources of truth, experiment-specific policy in a shared layer, reviewer scope expansion, or more than one independently useful outcome. Classify the correct convergence path as one IN_SCOPE architectural fix, DEFERRED work, CONTRACT_CONTRADICTION, or split Goal. Do not implement. Specifically assess scratch-owned Ray state, ownership-scoped cleanup, structured failure codes, and Stage3 pending artifact calibration proxy.

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
