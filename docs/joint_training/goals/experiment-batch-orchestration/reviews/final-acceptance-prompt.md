# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `final-acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `2`
- Current milestone: `none`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `ce827f2a9d631197a5dac97484a7f5f6543f4535`
- Applicable ACs: `AC-01,AC-02,AC-03,AC-04,AC-05,AC-06,AC-07,AC-08,AC-09,AC-10,AC-11`

## Required Verification

Run validate-plan and validate-runtime; inspect full-cpu-gate-summary.json and do not rerun the full suite; run the focused replacement command recorded there, batch-validate, protected_asset_fingerprint compare, git diff --check, and reviewer-owned source/mutation audits. Verify candidate commit ce827f2a9d631197a5dac97484a7f5f6543f4535 and recipe commit 9b83f9f488ac4b34e87a36a40d969d98d7d025f7.

## Additional Task-Specific Focus

Fresh independent final acceptance from committed state. AC-01 through AC-11 must each be PASS. Bind acceptance to Plan v2 SHA256 15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb, candidate commit, recipe gitlink, implementation identity, deletion-budget inventory, full CPU gate evidence and focused replacement. Reject if any duplicate authority, retry/resume/tuning/AI supervision, protected asset mutation, uncommitted required artifact, or unresolved finding remains. Write acceptance.md and an acceptance report; do not implement.

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
