# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone 7 Fix Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`
- Candidate commit: `66d6f5a`
- Applicable ACs: `AC-03, AC-04, AC-09, AC-10, AC-12`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && .venv/bin/python -m pytest -q && .venv/bin/python -m hubctl check --root . && git diff --check 593b4bac..66d6f5a && git show --stat --oneline 66d6f5a; goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync

## Additional Task-Specific Focus

Review F-M7-R01 and F-M7-02 only. Prove real object/folder tokens cannot enter tracked entry.yaml, Git diff/history, CLI stdout, or audit; runtime-private bindings are ignored and enforce directory 0700/file 0600; malformed/wide/missing live binding fails closed; fixture adapters remain deterministic; live publish/pull/reconcile/inventory resolve only runtime-private identifiers; revision/editor IDs remain in structured Git trailers as required by AC-04 while tracked reviewer evidence stays redacted; full suite and root gate pass. Do not run live Feishu writes, push Git, edit ledgers, or implement fixes. Return PASS/FAIL with exact evidence.

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
