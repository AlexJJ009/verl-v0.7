# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone 2 Implementation Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `13`
- Current milestone: `Milestone 2`
- Base commit: `f64afb4dc972bee67dcf4af5a3e97008cf9840a0`
- Candidate commit: `3b8a62a245ffad53f3b7d9e8ea5227eb097c79f6`
- Applicable ACs: `AC-01, AC-04, AC-06, AC-07, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_operational_calibration_checker.py; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution

## Additional Task-Specific Focus

Review F-EX-M2-20 and F-EX-M2-21. Verify the exact treatment-only stage2,stage3 outcome scope is accepted end-to-end while all other subsets/orders fail closed; verify no training-plane/wrapper parameters changed; verify the test correction binds its actual selected manifest phase and retains failure context coverage. Determine whether preserved six-repetition probe evidence can be re-rendered without repeating GPU work, since the behavioral probe already completed and the fix only corrects result classification. Do not edit or launch GPU.

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
