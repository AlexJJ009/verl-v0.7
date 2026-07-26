# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `653fb6c022397f0765ed9864e85a1a07fdbd2cf4`
- Candidate commit: `8a589d0f85f8d93fdc487da0ee003881c17e1ea3`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11, AC-12`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync; goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync; sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md; cd /data-1/code/verl/research/feishu-research-hub && uv run --with .[test] pytest -q && uv run --with .[test] hubctl check --root .; git -C /data-1/code/verl status --short; git -C /data-1/code/verl/research/feishu-research-hub status --short

## Additional Task-Specific Focus

Plan/runtime bindings and immutable candidates:

- Plan version 2; SHA256 `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Hub candidate and private `origin/main`: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`.
- Parent candidate: `8a589d0f85f8d93fdc487da0ee003881c17e1ea3`, whose only commit change is the Hub gitlink pinned to `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`.
- Protected unrelated-worktree digest before the acceptance story: `c1f55704c14dd32d15e754ca948af0e6be65f7705f71cf40decbdf924602716e`.

Authorization and live-test envelope:

- Do not start any Feishu mutation until runtime ledger decision `D-07` has a matching `USER_DECISION_RECORDED=APPROVED` and `validate-runtime` passes.
- The only live Feishu targets are the pre-existing Goal-created disposable capability-probe root, Archive child, FS03 Docx, and FS03 Markdown from D-04.
- Resolve `FEISHU_DISPOSABLE_ROOT_TOKEN`, `FEISHU_DISPOSABLE_ARCHIVE_TOKEN`, `FEISHU_DISPOSABLE_DOCX_TOKEN`, and `FEISHU_DISPOSABLE_MARKDOWN_TOKEN` from protected mode-0600 scratch/runtime input. Never copy exact tokens, revisions, editor IDs, URLs, or OAuth material into tracked prompts, reports, acceptance, Git messages, or command transcripts.
- One synthetic local publish/read-back and one human edit -> live pull -> attributable Git commit/push -> status/no-op cycle are permitted only after D-07 approval.
- No delete, trash, tombstone-from-absence, seed publication, existing research-document access, public sharing, permission expansion, owner transfer, webhook/event subscription, or additional PM2 process is authorized.
- If a manual browser edit is required, pause only for that exact user action; do not broaden the target.

Independent acceptance requirements:

1. Work as a fresh independent reviewer. Do not implement fixes and do not edit `runtime.jsonl` or `findings.jsonl`.
2. Run every Plan verification command yourself. Run the complete deterministic suite and `hubctl check`; confirm required tests are collected and no relevant test was skipped, deleted, weakened, or trivialized.
3. Re-run the known-bad gate canaries required by the Plan and show they fail for the intended reason before trusting green gate output.
4. Verify exact Plan, Hub, private remote, parent gitlink, PM2 provenance, protected-path digest, repo-local identity `GongxunLi <lgxma01@buaa.edu.cn>`, and Agent attribution contract.
5. Use current lark-cli user auth only within the approved disposable envelope. Record exact identifiers only in the protected reviewer scratch directory; tracked evidence uses stable aliases.
6. Have the user make one visible, synthetic edit in the disposable Docx. Capture pre/post normalized content hashes and redacted revision/editor evidence. Run the PM2-shaped live pull and require remote-wins preservation, correct human Author, service Committer, push success, terminal status PASS, matching cursor, and a second no-op run.
7. Confirm no unrelated Feishu object or protected parent-worktree path changed. Do not infer deletion from missing metadata, polling absence, `970005`, or permission errors.
8. Write `acceptance.md` with AC-01 through AC-12 individually classified `PASS`, `FAIL`, or `WEAKENED`, exact commands and relevant redacted output, Plan/Hub/parent bindings, and overall `ACCEPTED` only if every AC is `PASS` from reviewer-owned evidence. Otherwise return `REJECTED` or `PENDING REVIEW`.

Stop and report without mutating further if D-07 is absent, auth cannot be verified safely, an exact identifier would enter tracked evidence, a protected digest changes unexpectedly, or any operation would exceed the disposable envelope.


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
