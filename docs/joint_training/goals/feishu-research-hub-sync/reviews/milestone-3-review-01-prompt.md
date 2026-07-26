# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `1`
- Current milestone: `Milestone 3`
- Base commit: `653fb6c022397f0765ed9864e85a1a07fdbd2cf4`
- Candidate commit: `94a80708`
- Applicable ACs: `AC-01`

## Required Verification

gh repo view AlexJJ009/feishu-research-hub --json nameWithOwner,visibility,defaultBranchRef; git config --file .gitmodules --get-regexp ^submodule\\.feishu-research-hub\\.(path|url)$; git ls-files --stage research/feishu-research-hub; fresh authenticated clone + submodule init at parent 94a80708; hubctl setup in fresh Hub clone with isolated GIT_CONFIG_GLOBAL; uv run --with .[test] pytest -q; uv run --with .[test] hubctl check --root .

## Additional Task-Specific Focus

Independently verify AC-01 only. Parent candidate 94a80708 must change exactly .gitmodules and the mode-160000 gitlink. Hub candidate is 2a7ac5c6abcfbe79516364e0e9f2a2abc1dd2aa7 on private GitHub main. Confirm PRIVATE visibility, reproducible authenticated initialization, repo-local hooks/name/email after setup, unchanged parent/global identity, agent-authored Hub commits with standard human co-author trailer, shared_writes_authorized=false, and protected unrelated-worktree digest 6ecb2a933250739fb09ab28f4bda7ca6612f310726f45ab6524016b9af891606. Do not implement or modify files.

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
