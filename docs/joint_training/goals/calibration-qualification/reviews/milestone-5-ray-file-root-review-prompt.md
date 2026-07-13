# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Finding Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `e3ae6d7b`
- Candidate commit: `5b04265a`
- Applicable ACs: `AC-03, AC-04, AC-08`

## Required Verification

Read plan.md and F-M5-08 ledger events; inspect candidate and recipe submodule diff; verify Hydra quoting produces a Ray runtime_env env var equal to the scratch path; run REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py -k 'phase_runner or file_logger or zero_step'; recompute implementation identity; verify normal training wrappers were not modified; do not run GPU or modify files.

## Additional Task-Specific Focus

Review F-M5-08 only. The previous live probe showed shell VERL_FILE_LOGGER_ROOT did not reach Ray TaskRunner. Verify the calibration phase adds the existing root to ray_kwargs.ray_init.runtime_env.env_vars with correct Hydra quoting, remains file-only/zero-step, and leaves external services and normal wrappers unchanged.

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
