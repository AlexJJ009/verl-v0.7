Plan/runtime bindings and immutable candidates:

- Plan version 2; SHA256 `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Hub candidate and private `origin/main`: `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`.
- Parent candidate: `22499d3b9e6b2e7d1f34f57c2d33e67a56740f31`; its only commit change advances the Hub gitlink to `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`.
- Protected unrelated-worktree digest immediately before this acceptance story: `7f356bc33a378284825a77a0fa4d03e7ebaedcc675d0f2d2976c768d68161b39` (`61` entries, `1436` files), reproduced twice. This supersedes the older point-in-time digest because unrelated user work changed before this final review; it is the no-touch baseline for this live story.
- Runtime local CI has a truthful `PASS` verdict for `7078fdd`; the fixture sync cursor also reached `7078fdd` with terminal `PASS`. Both dedicated PM2 entries are stopped with exit code 0 before live work.

Authorization and live-test envelope:

- Runtime ledger decision `D-07` is recorded `APPROVED`; re-run `validate-runtime` before mutation.
- The only live target for publish/edit/pull is the existing Goal-created disposable `FS03-DOC` Docx from D-04. Its binding is already present in `/data-1/feishu-research-hub-runtime/repo/.hub/live-bindings.json` with directory mode 0700 and file mode 0600.
- Resolve the exact token/revision/editor values only inside protected scratch/runtime state. Do not copy them into tracked prompts, reviews, acceptance, stdout excerpts, or normal shell transcripts.
- Object tokens and private URLs never enter Git. Revision/editor IDs are required in the live-pull Git trailers and private audit, but their values must be replaced with stable aliases in tracked review/acceptance prose.
- One synthetic local publish/read-back and one human edit -> periodic-job-shaped live pull -> attributable Git commit/push -> status/no-op cycle are permitted.
- No delete, trash, seed publication, existing research-document access, public sharing, permission expansion, owner transfer, webhook/event subscription, or additional PM2 process is authorized.

Required execution sequence:

1. Work as a fresh independent reviewer. Do not implement fixes. Do not edit `runtime.jsonl` or `findings.jsonl`.
2. Run both Goal validators, verify the Plan hash/candidates/digests, then run the complete deterministic suite, `hubctl check`, required known-bad canaries, private-repo/submodule checks, and protected-state checks yourself.
3. In this same execution session run `lark-cli auth status --json --verify`; require `identity=user`, `verified=true`, and default identity `user`.
4. Before any docs fetch/update, use version-matched embedded guidance via `lark-cli skills read lark-doc ...` for fetch, Markdown, style, update, and update workflow. Do not rely on missing filesystem reference files.
5. Perform the authorized synthetic publish/read-back on only the bound disposable FS03 Docx. Keep exact identifiers in a fresh mode-0700 reviewer scratch directory with mode-0600 files. Confirm read-back and idempotent rerun.
6. Return only the disposable document's private link to the primary agent and stop for the user to make a visible synthetic edit. The human edit must be authored by the user in Feishu; do not simulate it. Tell the primary agent exactly what short marker the user should add, without putting the private link/token into any tracked file.
7. After the primary agent confirms that the user edited it, resume from the same reviewer context: run the approved periodic-job-shaped live pull, require remote-wins preservation, human Author `GongxunLi <lgxma01@buaa.edu.cn>`, service Committer, structured Feishu revision/editor trailers, push success, terminal status `PASS`, matching cursor, and a second no-op run.
8. The live pull will create a new Hub commit. Report its SHA privately to the primary agent. Do not update the parent gitlink yourself. Pause so the primary agent can make the exact parent gitlink update and supply the final parent SHA; then re-verify the final bindings before writing acceptance.
9. Write `acceptance.md` with AC-01 through AC-12 individually marked `PASS`, `FAIL`, or `WEAKENED`, exact commands with redacted relevant output, current Plan/Hub/parent bindings, and overall `ACCEPTED` only if every AC is reviewer-owned `PASS`.

Stop without further mutation if auth is not verified, the binding store is invalid, an identifier would enter tracked evidence, the protected digest changes unexpectedly, any operation exceeds D-07, or an AC fails. Report any new defect to the primary agent for ledger classification.
