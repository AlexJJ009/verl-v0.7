# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Convergence Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 3`
- Base commit: `aae4c756`
- Candidate commit: `HEAD`
- Applicable ACs: `AC-02, AC-07`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_evidence_compatibility.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_dynamic_calibration_interval.py tests/experiment_workflow/test_operational_calibration_checker.py tests/experiment_workflow/test_calibration_outcomes.py; goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification

## Additional Task-Specific Focus

Final convergence review for F-M3-CONV-02. Independently prove calibration_result without expected_bindings fails, every expected binding mutation fails, and admission builder receives tree/evidence/authorization expectations from preflight rather than the calibration result. Do not recompute identity inside generic validator. Determine whether F-M3-CONV-02 closes and Milestone 3 passes.

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
