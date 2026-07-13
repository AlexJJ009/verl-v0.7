# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `convergence`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `none (Plan review)`
- Base commit: `220aa8aa550c4bbfa74fec7ba94c3a508a894d59`
- Candidate commit: `2bf183399fac669454e370507ccb6038be7b58b4`
- Applicable ACs: `AC-01, AC-08`

## Required Verification

`goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/calibration-qualification`; inspect the append-only findings ledger; `git diff 220aa8aa550c4bbfa74fec7ba94c3a508a894d59..2bf183399fac669454e370507ccb6038be7b58b4 -- /data-1/code/verl/docs/joint_training/goals/calibration-qualification`; verify every existing explicit superproject path with `git ls-files --stage`; verify every existing explicit recipe path with `git -C recipe ls-files --stage`; inspect the direct Stage123 `source`/`exec` chain; confirm the versioned list uses no glob or dynamic discovery and that planned identity/renderer paths are explicitly in-scope.

## Additional Task-Specific Focus

Convergence review for `F-CQ-RR-01` after two implementation-review fix proposals. Verify the explicit superproject_paths and recipe_paths are sufficient to reproduce the one production identity, including transitive launch surfaces and planned in-scope identity/renderer artifacts. Determine whether the loop was caused by duplicated/implicit identity authority and whether the explicit versioned no-glob path manifest is the correct in-scope architectural convergence. Confirm there is no new outcome, AC change, reviewer scope expansion, or split requirement. Recommend `CLOSE_AND_READY`, `NOT_READY`, or `CONTRACT_CONTRADICTION`. A third ordinary patch is prohibited.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Report commands, relevant output, reviewed Plan version, and reviewed commit.

## Required Output

- Review identity
- Overall verdict: `READY`, `NOT_READY`, or `CONTRACT_CONTRADICTION`
- Finding disposition for `F-CQ-RR-01`
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- One-line `PLAN_REVIEWED` ledger summary
