# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Treatment Launch Readiness Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `13`
- Current milestone: `Milestone 2`
- Base commit: `3b8a62a245ffad53f3b7d9e8ea5227eb097c79f6`
- Candidate commit: `3b8a62a245ffad53f3b7d9e8ea5227eb097c79f6`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260716T0305Z/treatment-admission.json; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260716T0305Z/authorized-treatment-batch-manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260716T0305Z/state --repo-root /data-1/code/verl; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution

## Additional Task-Specific Focus

Confirm treatment-only execution identity is new and includes exactly frac25-stage2 then frac25-stage3; certified control is baseline evidence only and not a Stage2 input; treatment admission is bound to preserved certificate and fresh host facts; no retry/resume permitted; current zero-step treatment-only calibration result is bound to current implementation identity; all direct/batch paths use admission. Do not launch any GPU work or edit files.

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
