# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 6`
- Base commit: `73f973c6592eea850f0ceb91d35d7091a4c0b9d4`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow tests/joint_training/regression/test_validation_generation_logging.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json; git status --short; tmux list-sessions; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Perform independent final acceptance from committed candidate 2020531b and candidate bundle SHA256 97e87458. Run the full CPU preservation gate exactly once and do not repeat it. Audit every AC-01 through AC-08, protected assets, no deleted/skipped/loosened/trivial tests, no real external services, no formal training/checkpoints/publication, current implementation tree 91b8e419, calibration 80d09f5a, preflight 23dac2c8, exact run IDs, and all mutation/freshness/current-checkout bindings. If every AC passes, write reviewer-owned acceptance_report.json schema v1 with decision accepted, reviewer model GPT-5.5 reasoning_effort medium, goal_id and plan_id stage123-execution-readiness, plan_version 8 and exact plan SHA256, candidate_commit and readiness_evidence_commit 2020531b470ec932d7b00afd13080e1318fc8429, run_ids, bundle_sha256, top-level manifest/resource-profile/implementation-tree/calibration-result/preflight-result/protected-baseline bindings, input_hashes, AC-01 through AC-08 verdicts exactly PASS, command evidence, and a correctly recomputed acceptance_report_sha256 excluding that field. Then rebuild admission_bundle.json with --acceptance-report, validate it with --require-accepted, render the exact launch command but DO NOT execute it, and complete acceptance.md with PASS, Plan hash, candidate commit, reviewer identity, per-AC verdicts, full-gate output, accepted bundle/report hashes, and the rendered unexecuted command. If any AC fails or weakens, do not fabricate accepted artifacts; report REJECTED/PENDING and classify the blocker.

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
