# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `3`
- Current milestone: `none`
- Base commit: `dcb978bd1a241578a6b4fa61c05b81913c1da230`
- Candidate commit: `6eab4d08ddf320499a9992254d8ef43f22e0ab5a`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11, AC-12`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync; goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync; sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md; cd /data-1/code/verl/research/feishu-research-hub && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root .; git -C /data-1/code/verl status --short; git -C /data-1/code/verl/research/feishu-research-hub status --short

## Additional Task-Specific Focus

Plan v3 hash: cac8ebb81a776d9f72e40732ed3857adbfb49a54e4db49f5740b1c3280f87ddb
Hub candidate/origin: 88ac17c2181d22f1e33373c54763ea8e6351bf74
Parent current candidate: 6eab4d08ddf320499a9992254d8ef43f22e0ab5a (base dcb978bd1a241578a6b4fa61c05b81913c1da230); staged gitlink already pins Hub candidate; final R3 commit follows acceptance.
Protected unrelated-worktree baseline: protected_entries=63 protected_files=1457 sha256=f28aa7d602b623c2dc128417fb46d3e583c4f7db308fdfb23f18ba95001fe520 (reproduced consecutively before prompt construction).
Reuse completed Milestones 1-7 evidence, especially milestone-7-conflict-retention-convergence-01.md. Do not repeat prior successful live Feishu operations. Minimum missing R2 evidence is reviewer-owned current deterministic/root checks, current Git/identity/runtime state, prior authorized live-cycle evidence audit, and final exact binding checks.
Current PM2 records contain exactly feishu-hub-local-ci and feishu-hub-sync, both stopped; restart only if reviewer determines deployed-path evidence is genuinely missing.
Current lark-cli 1.0.76 auth is identity=user, verified=true, default-as user. Never emit its token/scope payload or private binding values in tracked evidence.
Write the final report to docs/joint_training/goals/feishu-research-hub-sync/reviews/final-acceptance-05.md and complete docs/joint_training/goals/feishu-research-hub-sync/acceptance.md only if AC-01..AC-12 all PASS. Do not edit ledgers. Preserve unrelated dirty paths. No delete/trash/public share/permission expansion/owner transfer/force push/non-disposable object access/seed publication.


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
11. Audit the Authorization Policy: silence means default authorization; milestone boundaries and risk notices are not approval gates. Flag a `USER_DECISION` outside the narrow stop classes as a Plan defect. Preserve any current-turn confirmation explicitly enforced by a tool.
12. Confirm every pre-authorized stop-class action names its exact action, target, boundary, and milestone. Broadened facts require a new decision.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
