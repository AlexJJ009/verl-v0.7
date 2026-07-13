# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Finding Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `90b72ebb`
- Candidate commit: `7df4d318`
- Applicable ACs: `AC-04, AC-08`

## Required Verification

Read F-M5-11; inspect candidate diff and checker; run operational checker plus Milestone3 focused tests; create reviewer-owned Stage3 repetitions 1,2,3 with rep3 timeout and verify every failure context says repetition 3; recompute identity and exact freeze binding; validate-plan/runtime; no GPU and no modifications.

## Additional Task-Specific Focus

Review only the 1-based structured repetition identity fix and AC-08 binding. Also independently calculate cumulative authorized GPU probe wall time across every probe-report.json under calibration scratch, not only the latest report, and state whether another GPU run is authorized under the original 90-minute aggregate ceiling.

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
