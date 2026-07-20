# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Mechanical Rereview`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Candidate commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Applicable ACs: `AC-01, AC-02, AC-07, AC-08, AC-12`

## Required Verification

goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state --repo-root /data-1/code/verl; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Scoped rereview of the sole prior blocker. Verify experiment_batch_manifest.json now binds accepted admission file SHA b0f232..., implementation tree 7b0c9449..., Plan v18 hash, unchanged command hash/run set, correct self-hash, and the supplied new empty state root. Reuse the prior PASS findings for port fix/tests/applicability/admission; do not reopen them. Do not edit or launch. Write PASS/FAIL to docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-port-fix-batch-rereview.md.

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
