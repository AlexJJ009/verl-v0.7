# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Convergence Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `58cab953`
- Candidate commit: `558baa6d`
- Applicable ACs: `AC-03, AC-04, AC-08`

## Required Verification

Read F-M5-CONV-06 and implementation-freeze; run validate-plan/runtime; recompute implementation identity; run full tests/experiment_workflow/test_calibration_milestone3.py; inspect trainer val_only early return and save_freq gate; verify freeze exact candidate/recipe/hash; no GPU and no modifications.

## Additional Task-Specific Focus

Close only if val-only/no-checkpoint behavior remains PASS and committed freeze exactly binds 58cab953/888d8e1/cd081fcb. Confirm prior failed probe is diagnostic and a fresh full 2x3 probe is still required.

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
