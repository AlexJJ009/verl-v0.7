# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Convergence Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `79bf9a0e7f7f14ec0069f3ef9f06a08872c28fd1`
- Candidate commit: `88ac17c2181d22f1e33373c54763ea8e6351bf74`
- Applicable ACs: `AC-05, AC-09, AC-10, AC-12`

## Required Verification

Hub full pytest; hubctl check; git cat-file/show conflict commit; git ls-remote private conflict ref; pristine local-CI verdict; runtime sync event/cursor; goal-plan-runtime validate-runtime after convergence event

## Additional Task-Specific Focus

Classify F-M7-R04 convergence only. Determine whether a normal two-parent retention merge commit and private named ref structurally close the reproducibility defect without duplicate sources of truth or a second outcome. Verify conflict commit and pre-edit snapshot are reachable from a fresh private main clone, ref target matches, no private token/OAuth/URL leaks, tests/gate/test_tracked_common_snapshot.py fails when the object is absent and passes now, 154 tests/root gate/local CI/no-op pass. Do not implement, push, call Feishu, update parent/ledger/acceptance. Return PASS/FAIL and convergence classification.

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
