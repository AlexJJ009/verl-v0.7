# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 2`
- Base commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Candidate commit: `worktree-before-Milestone-3-commit`
- Applicable ACs: `AC-01,AC-02,AC-03,AC-04,AC-05,AC-06,AC-07,AC-08 implementation readiness`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_new_experiment_gate.py tests/experiment_workflow/test_calibration_milestone3.py

## Additional Task-Specific Focus

Independently review Milestone 2 implementation only. Inspect behavior and tests for host-owned atomic host_facts, container preflight consuming exactly --host-facts without Docker/tmux calls, FRAC25-only run set, producer pointer schema v2, post-probe renderer fail-closed bindings, one common current-checkout admission validator on candidate/--bundle/accepted/render-launch paths, protected baseline binding, strict acceptance schema, no training/publication/external calls, and preservation of Python lifecycle/event authority. GPU probe and formal training are out of scope and must not run.

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
