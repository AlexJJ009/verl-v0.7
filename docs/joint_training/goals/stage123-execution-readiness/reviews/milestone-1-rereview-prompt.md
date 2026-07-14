# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Re-Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 1`
- Base commit: `not supplied`
- Candidate commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Applicable ACs: `F-M1-01 protected baseline ordering`

## Required Verification

git status --short -- scripts tests/experiment_workflow recipe docs/joint_training/guides/training_script_index.md; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness

## Additional Task-Specific Focus

Light re-verification of F-M1-01 only. Confirm all covered production/test/doc repair paths were restored to candidate HEAD before Milestone 2, the protected baseline remains identical and passes, and any saved patch is outside the repository scratch area. No implementation repair may remain in the worktree.

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
