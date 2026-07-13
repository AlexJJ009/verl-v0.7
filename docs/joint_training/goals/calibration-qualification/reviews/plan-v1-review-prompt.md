# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `81fdd6d1f315200981bf89e268089f3c927f366d`
- Candidate commit: `WORKTREE`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/calibration-qualification; goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification; inspect predecessor plan, scripts/execution_results.py, scripts/experiment_manifest.py, scripts/experiment_execution_core.py, Stage123 manifest/queue/monitor/tests

## Additional Task-Specific Focus

Audit whether the bounded GPU envelope is enforceable; whether primary manifest normalization belongs before calibration; whether legacy receipt/prediction authority is fully excluded; and whether the result is sufficient but not over-authorizing for Readiness. Review all three serial Plans together for producer-consumer consistency, but issue an independent READY or NOT_READY verdict for this Plan.

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
