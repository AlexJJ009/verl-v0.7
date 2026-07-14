# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Re-Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 2`
- Base commit: `not supplied`
- Candidate commit: `worktree-before-Milestone-3-commit`
- Applicable ACs: `F-M2-01,F-M2-02 / AC-01,AC-06,AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_calibration_milestone3.py; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness

## Additional Task-Specific Focus

Focused same-reviewer re-verification only. Confirm render-launch always invokes current-checkout/protected-baseline/strict-acceptance validation and rejects bundles without canonical inputs; confirm calibration renderer rejects fake producer reports without exact Stage2/Stage3 three-repetition evidence, zero optimizer steps, empty checkpoints, qualified prediction, and cleanup. Do not review unrelated scope or run GPU/training/external services.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.
9. For a Plan review, reject `READY` when any AC declares an absolute numeric performance or resource budget that has no recorded feasibility probe in the Plan's `Feasibility Probes` section, or whose budget contradicts the probe's measured floor.
10. When your only blocking findings are purely mechanical (formatting, patch context offsets, artifact or directory placement) with no behavioral or contract impact, say so explicitly and offer a light same-reviewer re-verification scoped to those findings instead of demanding a fresh full round.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
