# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 6`
- Base commit: `c2fc0a1ce5576618be14c9d4093ea8b86c5005e8`
- Candidate commit: `685a41ddf641d7e19704e50f6cb8dc758e0ff62b`
- Applicable ACs: `AC-03, AC-04, AC-05, AC-06, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/check_code_task_operational_calibration.py --report /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/probe-20260714T015835Z/probe-report.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/reviewer-m6-fresh-probe-tree.jsonl
find /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration -name probe-report.json -type f -print0 | xargs -0 jq -s '{report_count:length,total_elapsed_seconds:([.[].elapsed_seconds//0]|add),total_elapsed_minutes:([.[].elapsed_seconds//0]|add/60)}'
goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification
tmux ls; docker ps --format '{{.ID}} {{.Names}} {{.Status}}'; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader; git status --short

## Additional Task-Specific Focus

Independently review the fresh post-F-M6-01 bounded probe at probe-20260714T015835Z. It must bind the refrozen production commit 9a8d78edc06a9a63a409704c7aab4b221dc6a15a, recipe 888d8e1a979070013ffc9ccca401ea17c73f26d6, and implementation identity 453b60bf1b626934a20916a1ca1aa9c90cc9de2b028d19b5ad875471d3c90ead recorded by committed Goal evidence 685a41ddf641d7e19704e50f6cb8dc758e0ff62b. Confirm 6/6 repetitions pass with zero training/optimizer/checkpoints/truncation, complete scores/resources, and owned cleanup. Recompute cumulative all-probe elapsed below the corrected 120-minute authorization. Audit live cleanup and protected paths. Do not modify files. Report AC verdicts and whether the fresh probe can support Milestone 6 result rendering.

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
