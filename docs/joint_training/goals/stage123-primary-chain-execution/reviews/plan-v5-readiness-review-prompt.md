# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Readiness Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `5`
- Current milestone: `none`
- Base commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Candidate commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl

## Additional Task-Specific Focus

Review the current checkout at readiness evidence commit 9c736bc0 plus its intentionally uncommitted, reviewer-owned readiness evidence and Primary Goal artifact. Confirm the Primary experiment_batch_manifest.json is self-hashed and binds the exact accepted three-run bundle, current Plan v5 hash, stage123_queue_v1 command hash, implementation tree identity, and a single operator-control path. Verify it adds no execution authority, does not launch GPU work, and resolves F-EX-PLAN-06, F-EX-PLAN-07, F-EX-PLAN-08, and F-EX-LAUNCH-01. Treat the current checkout HEAD constraint as frozen: the accepted admission validator requires HEAD exactly equal to 9c736bc0, so do not demand a new commit merely for Goal evidence.

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
