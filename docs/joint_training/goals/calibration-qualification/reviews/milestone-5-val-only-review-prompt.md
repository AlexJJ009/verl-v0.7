# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Finding Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `5b04265a`
- Candidate commit: `HEAD`
- Applicable ACs: `AC-03, AC-04, AC-08`

## Required Verification

Read F-M5-09 and probe-20260713T200729Z Stage3 rep1 evidence; inspect candidate and trainer.fit val_only semantics; run focused Milestone3 tests; verify calibration phase sets trainer.val_only=true and trainer.save_freq=-1 after wrapper defaults, remains zero-step/file-only, and normal wrappers unchanged; recompute identity; no GPU and no modifications.

## Additional Task-Specific Focus

Review whether the calibration-only overrides guarantee exactly one initial validation, immediate return before global_steps increment/training loop, and no formal checkpoint for both Stage2 and Stage3. Treat prior Stage2 evidence as diagnostic only; do not accept without code-path proof.

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
