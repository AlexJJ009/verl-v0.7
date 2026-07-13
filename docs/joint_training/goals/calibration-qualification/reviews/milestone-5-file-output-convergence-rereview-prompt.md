# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Convergence Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `13297a9e`
- Candidate commit: `e3ae6d7b`
- Applicable ACs: `AC-03, AC-04, AC-08`

## Required Verification

goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/calibration-qualification; inspect candidate diff and FileLogger semantics; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py -k 'phase_runner or file_logger or zero_step'; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --format json; verify protected assets remain untracked and untouched; do not run GPU or modify files.

## Additional Task-Specific Focus

Re-review F-M5-CONV-04. Verify calibration-only phase runner exports VERL_FILE_LOGGER_ROOT under repetition-owned CALIBRATION_OUTPUT_ROOT/logs/metrics, creates it, preserves file-only logger and cwd behavior, does not re-enable external services, and the removed repo-root OnPolicyWDLSFT-CodeTask directory was generated untracked output rather than protected/tracked content. Return PASS only from reviewer-owned evidence.

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
