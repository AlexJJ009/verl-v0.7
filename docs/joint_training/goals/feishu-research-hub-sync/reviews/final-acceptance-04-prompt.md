# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `653fb6c022397f0765ed9864e85a1a07fdbd2cf4`
- Candidate commit: `dcb978bd1a241578a6b4fa61c05b81913c1da230`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11, AC-12`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync; goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync; sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md; cd /data-1/code/verl/research/feishu-research-hub && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root .; git -C /data-1/code/verl status --short; git -C /data-1/code/verl/research/feishu-research-hub status --short

## Additional Task-Specific Focus

Current immutable bindings:

- Plan v2 SHA256: `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Hub candidate and private `origin/main`: `c024c980e3a4773de7bd1361853a70eea8074b3b`.
- Parent candidate: `dcb978bd1a241578a6b4fa61c05b81913c1da230`, whose only commit change advances the Hub gitlink from `7078fdd` to `c024c98`.
- Protected unrelated-worktree digest immediately before this review: `50664ac51c939bf751676587bd14f1c26b78c9b0c27c9e6c4fcb342d749d6550` (`63` entries, `1440` files), reproduced twice. Use this as the no-touch baseline.
- Local CI and fixture sync both recorded terminal PASS at `c024c98`; dedicated PM2 jobs ended stopped/exit 0.
- `F-M7-R02` is closed by independent focused PASS: guarded `binding-refresh` refreshed only runtime-private CAS metadata after proving canonical remote content equals the protected recorded common edition. Missing common or remote divergence fail closed.

Authorization and privacy:

- `D-07` and `D-M7-REMOTE-01` are recorded APPROVED. Re-run runtime validation before live work.
- Only the existing D-04 disposable `FS03-DOC` Docx may be published, read, manually edited, pulled, committed, and pushed.
- Resolve exact token/revision/editor values only from mode-0700/mode-0600 runtime/reviewer scratch state. Never place them in tracked prompts/reports/acceptance or normal command excerpts.
- Object tokens/private URLs never enter Git. Revision/editor IDs must remain in the actual live-pull Git trailers/private audit, but tracked reports use aliases.
- No delete, trash, seed publication, existing research-document access, public sharing, permission expansion, owner transfer, event subscription, or extra PM2 process.

Execution sequence:

1. Act as a fresh independent final reviewer. Do not implement fixes or edit Goal ledgers.
2. Independently verify validators, Plan hash, exact Hub/private remote/parent bindings, protected digest, 147-test full suite, root gate, required known-bad canaries, identity/attribution, local CI/fixture runtime state, binding permissions/privacy, and current lark-cli user auth.
3. Read the installed version-matched lark-doc fetch/Markdown/style/update/update-workflow references using `lark-cli skills read` before docs operations.
4. Perform the authorized synthetic publish/read-back on only `FS03-DOC`, then an idempotent publish rerun. Keep exact values private.
5. Return the private disposable Docx link only to the primary agent and stop. Ask the user to append exactly one short visible marker chosen by you. Never simulate the human edit.
6. After the primary agent confirms the user edit, resume this same reviewer context. Run the periodic-job-shaped live pull, require remote wins, human Author `GongxunLi <lgxma01@buaa.edu.cn>`, service Committer, structured Feishu revision/editor trailers, push success, terminal status PASS, matching cursor, and a second no-op run.
7. Report the resulting Hub SHA privately and stop for the primary agent to update only the parent gitlink. After receiving the final parent SHA, re-verify bindings and protected digest, then write `acceptance.md` with AC-01 through AC-12 individually PASS/FAIL/WEAKENED. Overall ACCEPTED requires every AC PASS.

Stop without further mutation on any auth/binding/privacy/AC failure and report a redacted defect to the primary agent for ledger classification.

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
