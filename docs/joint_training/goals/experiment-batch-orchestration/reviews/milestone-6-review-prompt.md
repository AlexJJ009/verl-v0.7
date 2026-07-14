# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `2`
- Current milestone: `Milestone 6`
- Base commit: `b50c3f83cd7664f0dbb6d611324c35d21974db15`
- Candidate commit: `98fed9681265a79dbbf5b023ab08e9a2550aa914`
- Applicable ACs: `AC-01,AC-02,AC-03,AC-04,AC-05,AC-06,AC-07,AC-08,AC-09,AC-10`

## Required Verification

Do not rerun the full tests/experiment_workflow suite. Inspect full-cpu-gate-summary.json for the sole full run and run the focused replacement command recorded there plus validate-plan, validate-runtime, batch-validate, protected comparison, and git diff --check.

## Additional Task-Specific Focus

Adversarially verify final implementation after F-M6-01 repair. Confirm the single full CPU gate is honestly recorded, the focused replacement restores the accepted Stage123 dry-run contract, the compatibility renderer has no lifecycle authority, all AC-01 through AC-10 remain PASS, protected assets are unchanged, and no GPU/training/external services occurred. Do not implement or rerun the full gate.

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
