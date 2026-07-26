# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `1`
- Current milestone: `Milestone 4`
- Base commit: `1eb7b3215f903129e048c2d8b616be5bd40ba010`
- Candidate commit: `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72`
- Applicable ACs: `AC-06`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && uv run --with '.[test]' pytest -q && uv run --with '.[test]' hubctl check --root .; inspect and run live-adapter pagination/metadata failure canaries; read-only inspect D-04 subtree if useful, but perform no Feishu write

## Additional Task-Specific Focus

Fresh independent behavioral review of Milestone 4 AC-06 after F-M4-R01/R02/R03 fixes. Verify recursive managed-root inventory, root/Archive child path classification, outside-root metadata fallback, strict rejection of failed_list/ambiguous empty metadata so no false tombstone occurs, complete next_page_token traversal with repeated/missing token fail-closed behavior, deterministic tombstone fixtures retained, and no remote create/delete. D-04 scope is root FEISHU_DISPOSABLE_ROOT_TOKEN, Docx FEISHU_DISPOSABLE_DOCX_TOKEN, Markdown FEISHU_DISPOSABLE_MARKDOWN_TOKEN, Archive FEISHU_DISPOSABLE_ARCHIVE_TOKEN. Resolve aliases only from protected runtime input; do not write exact tokens into tracked evidence. This review is read-only on Feishu: no move, rename, update, delete, seed publication, sharing, or permissions. Output report only; do not implement or edit ledgers.

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
