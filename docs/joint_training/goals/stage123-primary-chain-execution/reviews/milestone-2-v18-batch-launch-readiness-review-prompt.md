# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Launch Readiness Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Current milestone: `Milestone 2`
- Base commit: `6a069213f3467529530217fa14a473d0671859f6`
- Candidate commit: `6a069213f3467529530217fa14a473d0671859f6`
- Applicable ACs: `AC-01, AC-02, AC-07, AC-08, AC-12`

## Required Verification

goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T115009Z/state --repo-root /data-1/code/verl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission render-launch --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-host /data-1/code/verl; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Final mechanical launch-readiness review only. Verify the accepted fresh admission file hash is bound into the self-hashed batch manifest, Plan v18 hash and implementation tree match, exact run order is Control -> Stage2 -> extraction -> Stage3 through stage123_queue_v1, the supplied state root is new and empty, no retry/resume or legacy output path is used, and render-launch prints but does not execute the one admitted tmux command. Confirm no active Stage123/GPU training. Do not edit any file. Write report to docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-batch-launch-readiness-review.md.

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
