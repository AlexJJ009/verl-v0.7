Plan/runtime bindings and immutable candidates:

- Plan version 2; SHA256 `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Hub candidate and private `origin/main`: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`.
- Parent candidate: `8a589d0f85f8d93fdc487da0ee003881c17e1ea3`, whose only commit change is the Hub gitlink pinned to `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`.
- Protected unrelated-worktree digest before the acceptance story: `c1f55704c14dd32d15e754ca948af0e6be65f7705f71cf40decbdf924602716e`.

Authorization and live-test envelope:

- Do not start any Feishu mutation until runtime ledger decision `D-07` has a matching `USER_DECISION_RECORDED=APPROVED` and `validate-runtime` passes.
- The only live Feishu targets are the pre-existing Goal-created disposable capability-probe root, Archive child, FS03 Docx, and FS03 Markdown from D-04.
- Resolve `FEISHU_DISPOSABLE_ROOT_TOKEN`, `FEISHU_DISPOSABLE_ARCHIVE_TOKEN`, `FEISHU_DISPOSABLE_DOCX_TOKEN`, and `FEISHU_DISPOSABLE_MARKDOWN_TOKEN` from protected mode-0600 scratch/runtime input. Never copy exact object/folder tokens, private URLs, or OAuth material into tracked files, Git messages, reports, acceptance, or command transcripts. The frozen attribution contract still requires Feishu revision/editor IDs in structured Git trailers and private runtime audit; redact those IDs only from tracked reviewer reports and acceptance prose.
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
