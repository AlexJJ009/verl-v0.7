# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 5`
- Base commit: `14250ae14354d45de35d0bc19daec82aceb21a51`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Applicable ACs: `AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_new_experiment_gate.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_notification_policy.py; git status --short; tmux list-sessions; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Second and final ordinary implementation-review round for F-M5-01/F-M5-02. Verify fresh preflight_result has exactly one structured scorer_dependencies check with all official imports, release-v5 index, and required PYTHONPATH, and that dependency failure blocks preflight before Ray. Verify stage123_manifest_monitor and its shell wrapper have no tmux/checkpoint/metrics/deadline runtime probes or CLI inputs; monitor replays experiment_execution_core events for the single stage123-primary-queue authority, carries manifest run IDs only as immutable context, maps running/failure statuses through notification policy, and never invents per-run completion/failure. Re-run behavioral event replay and mutation gates. Confirm fresh calibration result 80d09f5a, preflight 23dac2c8, bundle 97e87458, implementation tree 91b8e419, no training/publication, and candidate launch remains blocked without acceptance. PASS closes both findings; any remaining failure must be stated clearly because convergence review is required before another fix.

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
