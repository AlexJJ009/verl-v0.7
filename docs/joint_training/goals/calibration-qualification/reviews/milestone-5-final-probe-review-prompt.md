# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `15d5f3a289a59ad689569a48ddea7fbfb058c7ad`
- Candidate commit: `7bf56398dd76cac69dc60d6491f465153c8db004`
- Applicable ACs: `AC-03, AC-04, AC-05, AC-06, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/check_code_task_operational_calibration.py --report /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/probe-20260714T012119Z/probe-report.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/reviewer-m5-implementation-tree.jsonl
sha256sum /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/reviewer-m5-implementation-tree.jsonl
find /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration -name probe-report.json -type f -print0 | xargs -0 jq -s '{report_count:length,total_elapsed_seconds:([.[].elapsed_seconds//0]|add),total_elapsed_minutes:([.[].elapsed_seconds//0]|add/60)}'
tmux ls; docker ps --format '{{.ID}} {{.Names}} {{.Status}}'; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader; git status --short

## Additional Task-Specific Focus

Independently review the completed authorized Milestone 5 bounded GPU calibration. The authoritative fresh probe is /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/probe-20260714T012119Z/probe-report.json. Confirm all six sequential repetitions (stage2 and stage3, repetitions 1-3) passed; each has training_steps=0, optimizer_enabled=false, zero formal checkpoints, zero truncations, complete scores/metrics, structured resource observations, and owned cleanup with no Ray descendants. Run the operational checker yourself. Audit live tmux/process/Docker/GPU cleanup and protected paths. Recompute cumulative elapsed_seconds across every probe-report.json below the calibration scratch root: it must be below the user-corrected cumulative authorization ceiling of 120 minutes recorded by runtime EVENT_CORRECTED seq 45; max 8 L40S and max 3 repetitions per phase remain unchanged. Confirm production candidate commit 15d5f3a289a59ad689569a48ddea7fbfb058c7ad, recipe gitlink 888d8e1a979070013ffc9ccca401ea17c73f26d6, and implementation identity ccab8cac8c4909288a2d322cc914a100d72d2837fe13f0784673008457d0fcab remain reproducible and no covered production path changed after freeze. Do not implement or modify files. Report each applicable AC as PASS, FAIL, or WEAKENED with commands and evidence, then a Milestone 5 verdict.

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
