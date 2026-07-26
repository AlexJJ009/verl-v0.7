# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Convergence Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `1`
- Current milestone: `Milestone 4`
- Base commit: `1eb7b3215f903129e048c2d8b616be5bd40ba010`
- Candidate commit: `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72`
- Applicable ACs: `AC-06`

## Required Verification

Read reviews/milestone-4-review-01.md through -03.md, findings F-M4-R01 through R04, lark-cli schema drive.metas.batch_query, src/hubctl/adapters/lark_cli.py, src/hubctl/sync.py, and tests; run no Feishu write

## Additional Task-Specific Focus

Convergence review before any fourth ordinary fix round. Determine whether the repeated AC-06 loop is caused by duplicated source of truth, experiment-specific policy in shared layer, reviewer scope expansion, multiple outcomes, a frozen-contract contradiction, or a bounded implementation gap. Critical fact: drive.metas.batch_query code 970005 conflates token/type mismatch with nonexistence, so it is not alone safe deletion proof. The desired invariant is: confirmed deletion produces RemoteDocument(deleted=True) and recoverable tombstone; permission/type/ambiguous absence fails closed. Recommend an exact evidence contract and testable adapter design without requiring an actual remote delete, or return CONTRACT_CONTRADICTION/AC_CHANGE if impossible. Output report only; do not implement, edit ledgers, delete, or write Feishu.

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
