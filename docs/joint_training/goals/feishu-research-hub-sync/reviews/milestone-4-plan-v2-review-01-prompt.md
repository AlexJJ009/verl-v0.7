# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 4`
- Base commit: `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72`
- Candidate commit: `efee75e560e44f9484ce2dc7a7e3d85a757f4fb8`
- Applicable ACs: `AC-03, AC-04, AC-05, AC-06, AC-07`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && uv run --with '.[test]' pytest -q && uv run --with '.[test]' hubctl check --root . && git diff --check 90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72 efee75e560e44f9484ce2dc7a7e3d85a757f4fb8

## Additional Task-Specific Focus

Plan v2 re-entry after D-AC06-01. Independently verify AC-06 end-to-end and regression-check AC-03/04/05/07. Required evidence: recursive pagination; prior user-approved disposable live root -> Archive -> root yields in_root_move; out-of-root metadata yields detached; 970003, 970005, empty metadata, denied/type failures, and malformed pagination cause byte-for-byte zero entry/content/sync-state/audit/tombstone mutation; trusted confirmed-deletion fixture creates a recoverable tombstone retaining adapter version; no remote create/delete/recreate. Do not infer deletion from polling. Do not perform any Feishu delete or create; any live checks must stay within D-04 and only use the existing disposable objects. Bind verdict to Plan v2 SHA256 ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba and candidate efee75e560e44f9484ce2dc7a7e3d85a757f4fb8.

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
