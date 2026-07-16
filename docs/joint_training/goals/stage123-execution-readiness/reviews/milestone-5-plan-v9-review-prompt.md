# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `9`
- Current milestone: `Milestone 5`
- Base commit: `9c736bc0`
- Candidate commit: `9c736bc0`
- Applicable ACs: `AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_monitor.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_new_experiment_gate.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl

## Additional Task-Specific Focus

Milestone 5 Plan v9. Independently review current fresh host_facts.json, preflight_result.json, and candidate admission_bundle.json bound to committed code candidate 9c736bc0. These result artifacts are intentionally uncommitted until final acceptance because the bundle requires current HEAD to equal readiness_evidence_commit; this is not a production code change. Verify exact three-run matrix, preflight freshness, scorer/mount/GPU/conflict checks, shared batch core/monitor authority, candidate mutation gates, no training/Ray/publication, protected baseline, and reproducible launch readiness. Write reviewer-owned report to docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-5-plan-v9-review.md.

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
