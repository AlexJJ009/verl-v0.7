# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 3`
- Base commit: `f116ea1a`
- Candidate commit: `HEAD`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_evidence_compatibility.py tests/experiment_workflow/test_dynamic_calibration_interval.py tests/experiment_workflow/test_operational_calibration_checker.py tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_operational_calibration_runtime_isolation.py tests/experiment_workflow/test_validation_deadline_cleanup.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/reviewer-tree.jsonl; goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification

## Additional Task-Specific Focus

Review Milestone 3 only. Verify literal boundary and canonical committed-tree hashing, zero-step scratch-only probe renderer, one versioned calibration policy/result schema, evidence-bounded prediction, complete calibration_result authority, fake-adapter cleanup/no-training behavior, and no GPU/external calls. Determine whether F-M3-01 and F-M3-02 may close.

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
