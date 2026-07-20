# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Candidate commit: `45802ffa7492e09c95d7d7fa7fbdc3cc35b79383`
- Applicable ACs: `AC-01, AC-07, AC-08, AC-12`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_control_reuse.py tests/experiment_workflow/test_stage123_calibration_applicability.py; bash -n recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution

## Additional Task-Specific Focus

Review the committed F-EX-LAUNCH-07 shared-layer repair. Verify direct phase invocation still performs full freshness admission; a live item_started batch record enables only no-TTL static phase validation with exact item/run/bundle/command/state bindings and machine-health check; fabricated/incomplete/terminal records fail closed; treatment-reuse adapters remain on their existing validator; no retry/resume, parameter, data, evaluator, or protected-asset change occurred.

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
