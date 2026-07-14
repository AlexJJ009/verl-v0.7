# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md; inspect scripts/experiment_execution_core.py, scripts/stage123_manifest_monitor.py, scripts/training_queue_monitor.sh, recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh, recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh, current recovery policy, tests, and protected baseline; rerun the frozen 23-test focused baseline with explicit REPO_HOST

## Additional Task-Specific Focus

Review whether this is one reusable orchestration outcome and whether every AC is implementable without a second authority. Verify that experiment_execution_core.py remains sole transition authority, one persisted-event monitor is generalized rather than copied, Stage123 remains a thin adapter, legacy training_queue_monitor.sh is not promoted to authority, local failure deterministically advances to the next pre-authorized item, shared failures stop, there are no retries/resumes/parameter changes/Agent decisions, operator pause/stop is exact, the batch manifest is immutable and admission-bound, deletion boundaries are testable, numeric policy waivers are honest, and no GPU/training/external service is authorized.

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
