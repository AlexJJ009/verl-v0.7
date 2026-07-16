# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `13`
- Current milestone: `Milestone 2`
- Base commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Candidate commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Applicable ACs: `AC-01, AC-03, AC-04, AC-07, AC-08, AC-09`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_control_reuse.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260715T154800Z/treatment-manifest.yaml --format json; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260715T154800Z/treatment-batch-manifest.json --repo-root /data-1/code/verl

## Additional Task-Specific Focus

Independently review the current worktree, including untracked scripts/stage123_control_reuse.py and tests/experiment_workflow/test_stage123_control_reuse.py, for F-EX-M2-10 and F-EX-M2-11. Confirm a treatment-only manifest cannot use original three-run preflight as authorization; it must require a fresh dedicated full treatment admission. Confirm prepare fails closed unless artifact-root is /data-2, the Stage3 pending producer output/provenance linkage is rewritten consistently, direct phase and batch paths validate the same authorized admission, and no training parameter/run-set/old evidence mutation occurred. Do not implement changes.

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
