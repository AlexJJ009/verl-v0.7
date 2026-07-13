# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Plan Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `ebe0fcaa7f8f8ba88e8eb81735a834e0619cce1e`
- Candidate commit: `220aa8aa550c4bbfa74fec7ba94c3a508a894d59`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/calibration-qualification; goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification; git diff ebe0fcaa7f8f8ba88e8eb81735a834e0619cce1e..220aa8aa550c4bbfa74fec7ba94c3a508a894d59 -- docs/joint_training/goals/calibration-qualification; git ls-files --stage; git ls-files --stage recipe; git -C recipe rev-parse HEAD

## Additional Task-Specific Focus

Final re-review F-CQ-RR-01 only. Verify stage123-implementation-tree-v1 fully freezes ordering, JSONL serialization, file mode/blob identity, symlink behavior, recipe gitlink and checked-out HEAD, selected recipe blobs, dirty-path failure, output retention, and reviewer recomputation. Confirm prior findings remain closed. If this finding closes and no new blocker exists, verdict must be READY.

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
