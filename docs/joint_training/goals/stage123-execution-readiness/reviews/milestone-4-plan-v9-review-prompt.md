# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `9`
- Current milestone: `Milestone 4`
- Base commit: `9632ebdc`
- Candidate commit: `a7ca8ee2`
- Applicable ACs: `AC-01, AC-07`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_execution_results.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl

## Additional Task-Specific Focus

Milestone 4 Plan v9 requalification. Independently inspect the archived first producer-only child_exit and the successful second exact zero-step three-phase x3 probe. Confirm no training steps/formal checkpoints/external service, 8-L40S/5400-second envelope preservation, current Plan v9 authorization identity, current implementation-tree binding 0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc, current policy prediction comparison, all cleanup evidence, and fail-closed historical policy compatibility fix. Determine whether F-M4-MC-01 is closed. Write reviewer-owned report to docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-4-plan-v9-review.md.

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
