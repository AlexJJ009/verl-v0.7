# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone-rereview`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `2`
- Current milestone: `Milestone 4`
- Base commit: `ea0a7d12be47518253788c0f5258ddd4de68d6e4`
- Candidate commit: `b9e4660e653f0951c5408d5467127be41c3ba196`
- Applicable ACs: `AC-05,AC-07`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_monitor.py tests/experiment_workflow/test_experiment_batch_control.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_notification_policy.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_failure_classifier.py tests/experiment_workflow/test_pm2_ci_keepalive.py

## Additional Task-Specific Focus

Focused same-reviewer rereview of F-M4-01. Verify mixed batch events carrying batch_id plus run_id are validated as batch events, malformed atomic run_id events fail closed, evidence is preserved, and all prior AC behavior remains passing. Do not implement.

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
