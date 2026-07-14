# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 4`
- Base commit: `62f6c5c8d34007d71b5269b795e8f7c72db92752`
- Candidate commit: `73f973c6592eea850f0ceb91d35d7091a4c0b9d4`
- Applicable ACs: `AC-01, AC-07`

## Required Verification

goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output /data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-GPU-REQUAL-01/reviewer-tree.jsonl; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; git status --short

## Additional Task-Specific Focus

Independently inspect the fresh producer pointer/report, execution-core terminal state, and authoritative calibration_result.json for RD-GPU-REQUAL-01. Recompute the four prediction comparisons (validation_elapsed_seconds, phase_elapsed_seconds, peak_rss_gib, gpu_wait_fraction) from the predecessor accepted calibration_result historical arrays/predicted bounds and the fresh six repetitions, applying policy maximum ratio 1.25 and minimum history count 3. Decide explicitly whether calibration_result.prediction_comparison={qualified:true,verification:{}} satisfies frozen AC-01 requirement to bind a real prediction comparison; do not infer PASS merely from schema validation. Confirm exactly Stage2/Stage3 x3, zero optimizer steps, empty formal checkpoints, scratch-only outputs, deadline/envelope, terminal cleanup, current implementation identity, and no formal training/publication. Report PASS/FAIL/WEAKENED per AC and classify any defect.

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
