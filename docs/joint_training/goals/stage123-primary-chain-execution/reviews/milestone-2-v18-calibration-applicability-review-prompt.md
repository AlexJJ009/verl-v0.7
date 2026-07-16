# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `31b1c31b`
- Candidate commit: `458aa212067ba04e9bcea441ae6224d902b81836`
- Applicable ACs: `AC-01, AC-12`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_calibration_applicability.py tests/experiment_workflow/test_stage123_validation_protocol.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_stage123_preflight_model_identity.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -c 'import json; from pathlib import Path; from scripts.calibration_result import validate; value=json.loads(Path("docs/joint_training/goals/stage123-execution-readiness/calibration_result.json").read_text()); schema=json.loads(Path("config/experiment_execution/calibration_result_schema_v1.json").read_text()); result=validate(value,schema); print(json.dumps(result,sort_keys=True)); raise SystemExit(0 if result["ok"] else 1)'; jq '{decision,capacity_differences,source_capacity_sha256,candidate_capacity_sha256,implementation_tree_sha256,evidence_commit,plan_sha256}' docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Independently review F-EX-IMPL-02 and F-EX-IMPL-03. Verify fresh V16 output identities cannot reuse V13/V14 artifacts; the checker reconstructs the accepted source profile/manifest from recipe gitlink 6fcccb..., fails closed for changed or missing capacity fields, proves an exact empty diff for the current candidate, accepts only a passed Stage1/2/3 zero-training calibration with released resources, and derives a valid current-binding calibration result. Confirm no GPU calibration/training/external service occurred. Do not edit implementation or protected assets. Write the review report to docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-calibration-applicability-review.md.

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
