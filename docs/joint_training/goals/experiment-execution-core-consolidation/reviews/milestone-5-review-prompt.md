# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `600c2506`
- Candidate commit: `819df295`
- Applicable ACs: `AC-06,AC-07`

## Required Verification

Do not rerun the full suite. Inspect docs/joint_training/goals/experiment-execution-core-consolidation/full-cpu-gate-summary.json showing the sole full run: 146 passed, 3 stale source-text failures. Run focused replacements: REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_gate_exit_propagation.py tests/experiment_workflow/test_initial_validation_timing.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py; run bash scripts/check_experiment_workflow_fast.sh; run AC-07 PM2 commands from Plan.

## Additional Task-Specific Focus

Verify the full gate was run exactly once and not repeated; confirm all three failures were stale source-text assertions replaced with stronger behavior tests, not skipped/trivialized. Audit accepted reliability fixtures for scorer dependency, timeout, fatal exit, CUDA OOM, occupied ports, incomplete identities, cleanup failure, incomplete training/release blocking. Verify PM2-only CI, no systemd, tmux ownership. Use focused tests only.

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
