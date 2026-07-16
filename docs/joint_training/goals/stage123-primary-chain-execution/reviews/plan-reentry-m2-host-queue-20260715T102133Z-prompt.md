# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Re-entry Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `9`
- Current milestone: `Milestone 2`
- Base commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Candidate commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-07, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-refresh-20260715T102133Z/state --repo-root /data-1/code/verl; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Re-entry review for F-EX-M2-02 and F-EX-M2-03. Confirm the renewed Readiness acceptance is fresh and independently bound, Primary batch manifest changes only its admission-bundle receipt hash and self hash, and host-level invocation of the existing queue adapter is mandatory so each admitted phase invokes run_train.sh exactly once rather than nesting Docker. Confirm RAY_ADDRESS=local changes only Ray topology and no scientific input. This review must explicitly decide whether Plan v9 may return READY and the two findings can close. Do not edit files or launch queue/GPU training. Return READY/NOT_READY/CONTRADICTION.

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
