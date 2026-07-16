# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `9`
- Current milestone: `Milestone 6`
- Base commit: `9c736bc0`
- Candidate commit: `9c736bc0`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_experiment_batch_control.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_monitor.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_new_experiment_gate.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl

## Additional Task-Specific Focus

Final Plan v9 acceptance. You alone own docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json, acceptance.md, reviews/final-plan-v9-acceptance.md. First independently verify all frozen ACs. If and only if AC-01 through AC-08 are PASS, replace the stale Plan v8 acceptance JSON with a schema-v1 acceptance_report bound to plan_id stage123-execution-readiness, Plan version 9/hash 29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c, candidate/readiness evidence commit 9c736bc029f4da16e5932a16b3f8bdf49dba57f1, raw candidate bundle sha256 5578540d602ae8ba01e4c79ee7b2c6ac1cdaae87b5b41f29620cce18b8f21b44, current manifest/profile/calibration/preflight/protected hashes, exact three run IDs, reviewer GPT-5.5 medium, and AC-01..08 PASS. Compute acceptance_report_sha256 using scripts.execution_results.acceptance_report_sha256. Then rebuild docs/.../admission_bundle.json with --acceptance-report, validate it with --require-accepted, and render-launch but do not execute it. Update acceptance.md and the final review report with exact command evidence and PASS/FAIL. Do not modify any implementation/tests/Plan/runtime/findings/protected assets or start training/GPU.

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
