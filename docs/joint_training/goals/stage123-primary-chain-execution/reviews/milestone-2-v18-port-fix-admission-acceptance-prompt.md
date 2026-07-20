# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Implementation Review and Admission Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `425f844734607b6e02bcd83a1de702d6e3239a30^`
- Candidate commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Applicable ACs: `AC-01, AC-02, AC-07, AC-08, AC-12; readiness AC-01 through AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/utils/test_net_utils_on_cpu.py tests/joint_training/feat/test_vllm_joint_rollout.py tests/experiment_workflow/test_stage123_calibration_applicability.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl; jq '{decision,capacity_differences,source_capacity_sha256,candidate_capacity_sha256,implementation_tree_sha256,evidence_commit}' docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Review F-EX-LAUNCH-05 and the refreshed admission. Verify the observed EADDRINUSE root cause, that get_free_port now reserves ports exclusively without SO_REUSEADDR/SO_REUSEPORT, the regression fails on the old behavior and passes now, existing vLLM reserved-socket lifecycle tests pass, the change is capacity-neutral with empty applicability diff, failed attempts remain local diagnostic evidence, and no training step/external publication occurred. If and only if all applicable ACs pass, rewrite readiness acceptance_report.json for candidate/readiness commit 425f8447 and current unsigned bundle SHA, include execution Plan v18/hash and applicability hash, compute acceptance_report_sha256, rebuild canonical admission_bundle.json with acceptance, validate --require-accepted, render-launch but do not execute, and write report to docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-port-fix-admission-acceptance.md. Do not edit implementation, Plan, runtime, findings, batch manifest, or protected assets.

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
