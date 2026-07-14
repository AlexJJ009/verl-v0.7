# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Adversarial Plan Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md; git diff --check -- docs/joint_training/goals/experiment-batch-orchestration/plan.md; inspect scripts/experiment_execution_core.py, scripts/stage123_manifest_monitor.py, scripts/training_queue_monitor.sh, recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh, recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh, config/experiment_execution/stage123_recovery_policy_v1.json, and tests; run the frozen 23-test baseline with REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_failure_classifier.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_pm2_ci_keepalive.py; do not run GPU or training

## Additional Task-Specific Focus

Be actively adversarial. Try to falsify the Plan rather than confirming its prose. Check for one-outcome violations; hidden implementation scope; a second authority accidentally created through shell wrappers, nested core invocations, legacy monitor reuse, or separate state files; ambiguity in item-local versus shared failures; whether two consecutive equal codes can be classified deterministically; cleanup-before-next-item guarantees; pause/stop race conditions; stale operator controls; manifest command injection or mutable paths; admission validation sequencing; whether failed experiments can be incorrectly treated as scientific results; whether AC commands are executable after implementation; whether deletion/compatibility boundaries are measurable; whether the existing core recovery policy conflicts with the no-retry/no-resume contract; and whether this Plan silently requires GPU, external services, or user input at an unlisted gate. Reject READY for any contradiction, missing feasibility evidence, untestable AC, or missing producer-consumer milestone. Do not implement or amend the Plan.

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
