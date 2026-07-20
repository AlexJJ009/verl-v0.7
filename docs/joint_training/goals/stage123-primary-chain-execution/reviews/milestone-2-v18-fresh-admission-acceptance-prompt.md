# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review and Admission Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `6a069213f3467529530217fa14a473d0671859f6`
- Candidate commit: `6a069213f3467529530217fa14a473d0671859f6`
- Applicable ACs: `AC-01, AC-02, AC-07, AC-08, AC-12; readiness AC-01 through AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl; jq '{decision,capacity_differences,source_capacity_sha256,candidate_capacity_sha256,implementation_tree_sha256,evidence_commit,plan_sha256}' docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Independently accept or reject the fresh V18/V16 no-training admission artifacts currently uncommitted at HEAD 6a069213. Verify canonical host facts, protected baseline, preflight, derived calibration applicability, exact three-run manifest, current implementation tree, current HEAD, recipe gitlink aa972ba, fresh V16 outputs, and zero GPU/training/external activity. If and only if every applicable AC passes, you own and must rewrite docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json as a schema-v1 accepted report bound to Readiness Plan v9/hash, candidate/readiness commit 6a069213, candidate unsigned bundle SHA, all current input hashes, protected baseline, exact run IDs, GPT-5.5 medium identity, AC-01..08 PASS, and additionally record execution_plan_version=18, execution_plan_sha256=471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852, calibration_applicability_sha256, and review_id. Compute acceptance_report_sha256 with scripts.execution_results.acceptance_report_sha256. Then rebuild the canonical admission_bundle.json with --acceptance-report, validate it with --require-accepted, render-launch but do not execute it, and write your review report to docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-fresh-admission-acceptance.md. Do not edit implementation, Plan, runtime, findings, batch manifest, or protected assets.

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
