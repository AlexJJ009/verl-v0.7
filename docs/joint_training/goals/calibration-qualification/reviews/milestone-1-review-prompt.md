# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 1`
- Base commit: `424e6ad1ffc9a5e35f4a3d8e23d5c0c131136f27`
- Candidate commit: `536761ed`
- Applicable ACs: `AC-01, AC-02`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_calibration_workload_descriptor.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh bash -lc "cd /workspace/verl && DRY_RUN=1 STAGE123_SCRATCH_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/m1-review STAGE123_MANIFEST_PYTHON=/opt/venv/bin/python bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh"; goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification; git status --short

## Additional Task-Specific Focus

Verify primary manifest contains exactly frac25-stage2 and frac25-stage3; legacy receipt freshness fields fail closed; dry-run performs no training/GPU/external mutation; protected untracked assets remain untouched. Review only Milestone 1 and do not implement.

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
