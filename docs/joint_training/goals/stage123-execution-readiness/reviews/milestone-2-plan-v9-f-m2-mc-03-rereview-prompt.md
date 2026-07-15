# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Milestone Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `9`
- Current milestone: `Milestone 2`
- Base commit: `0540935b`
- Candidate commit: `1d288529`
- Applicable ACs: `AC-01`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_calibration_outcomes.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -c 'from pathlib import Path; import importlib.util; p=Path("scripts/render_calibration_result.py"); s=importlib.util.spec_from_file_location("renderer",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); d=m.matching_authorization(Path("docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl"),"RD-MATCHED-CONTROL-REQUAL-01"); assert d["plan_version"] == 9; print(d["plan_sha256"])'
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json

## Additional Task-Specific Focus

Focused re-review of F-M2-MC-03 only. Confirm the renderer no longer hard-codes Plan v8, derives plan_version and plan_sha256 from the exact matching USER_DECISION_RECORDED event, validates that hash against current Goal plan.md, fails closed on stale/malformed authorization identity, and preserves the prior report's PASS conclusions for F-M2-MC-01/F-M2-MC-02. Do not reopen unrelated accepted scope. Update the reviewer-owned report at docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-f-m2-mc-03-rereview.md.

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
