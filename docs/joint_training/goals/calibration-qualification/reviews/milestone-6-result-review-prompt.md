# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 6`
- Base commit: `7cc302aa4ec9ae3efd8729749342a83f441753d7`
- Candidate commit: `cd47d93a3f5561cc9492da7c082ca1a6e9736950`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_evidence_compatibility.py tests/experiment_workflow/test_execution_results.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/check_code_task_operational_calibration.py --report /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/probe-20260714T015835Z/probe-report.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/reviewer-m6-result-tree.jsonl
goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification
sha256sum docs/joint_training/goals/calibration-qualification/plan.md docs/joint_training/goals/calibration-qualification/calibration_result.json config/experiment_execution/calibration_result_schema_v1.json
git status --short

## Additional Task-Specific Focus

Independently review the sole authoritative calibration_result.json at candidate cd47d93a3f5561cc9492da7c082ca1a6e9736950. Re-run calibration_result.validate and execution_results.validate_result with explicit expected bindings. Confirm exact immutable bindings: manifest e665049c..., resource profile d9b6a36d..., implementation identity 453b60bf..., evidence commit 7cc302aa..., workload c3eaf337..., policy 6a4966d4..., run IDs frac25-stage2/frac25-stage3, and authorization identity seq 45. Independently recompute each prediction comparison from legacy report.json migration input and the fresh probe, confirm all history counts >=3 and ratios <=1.25, and confirm historical evidence is not current authority. Audit complete per-repetition evidence, cleanup, cumulative authorization usage, no new receipt/adoption artifact, and only one calibration_result.json current authority path. Report each applicable AC PASS/FAIL/WEAKENED and overall Milestone 6 verdict. Do not modify files.

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
