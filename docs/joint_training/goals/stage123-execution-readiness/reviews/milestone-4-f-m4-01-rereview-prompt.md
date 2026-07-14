# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 4`
- Base commit: `73f973c6592eea850f0ceb91d35d7091a4c0b9d4`
- Candidate commit: `2501a8d37a1ea8635ce0070e57d73963ef9612f9`
- Applicable ACs: `AC-01, AC-07`

## Required Verification

goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_evidence_compatibility.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; git status --short

## Additional Task-Specific Focus

Focused behavioral re-review of F-M4-01 after production fix commit 0d57e58f and fresh requalification. Independently verify the producer report SHA256 05b544aa..., current implementation tree 8b2ceb90..., authorization rebind to the latest RD-GPU-REQUAL-01 decision, exact Stage2/Stage3 x3 evidence, zero steps/checkpoints, cleanup, and the four persisted comparisons. Recompute every ratio from history/predicted/observed and confirm policy id/hash, minimum history 3, maximum ratio 1.25, and semantic mutation fail-closed behavior. Confirm the old attempt-1 cannot authorize and no formal training/publication occurred. Return PASS only if F-M4-01 is closed and AC-01/AC-07 pass.

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
