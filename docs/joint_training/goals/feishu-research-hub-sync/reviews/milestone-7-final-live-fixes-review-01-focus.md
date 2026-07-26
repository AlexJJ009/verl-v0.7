Review only `F-M7-R03` and `F-M7-R04` against frozen Plan v2.

Candidate Hub history is the normal, non-rewritten chain from base `c024c980e3a4773de7bd1361853a70eea8074b3b` through:

- `51d73f34924ee525328a8475adc097618f328470`: accept the installed current `lark-cli auth status` envelope only when `identity=user` and `verified=true`, while preserving legacy compatibility and rejecting bot/none/unverified identities;
- `ee1fe0bf428edebe715de98c320d588a7d31b479`: the user-authored `FINAL-AC04` Feishu edit pulled as a human-authored/service-committed Git commit with required revision/editor trailers;
- `79bf9a0e7f7f14ec0069f3ef9f06a08872c28fd1`: normal Agent repair commit that resolves the false conflict without rewriting the live-pull commit, retains the reachable conflict snapshot/ref as resolved evidence, and adds a clean-binding invariant test.

Verify independently:

1. full Hub suite and root gate pass from the candidate checkout;
2. current and legacy verified-user auth envelopes pass, while bot/none/unverified envelopes fail closed;
3. `FS03-DOC` is active `clean`, its canonical content contains the synthetic `FINAL-AC04` marker, its recorded common hash matches that content, and its resolved conflict ref/commit remain reachable and preserve the pre-edit local bytes;
4. `ee1fe0b` has Author `GongxunLi <lgxma01@buaa.edu.cn>`, service Committer, and structured Feishu revision/editor trailers, without reporting exact private revision/editor values;
5. candidate `79bf9a0` equals private `origin/main`, the dedicated runtime clone is clean, pristine local-CI verdict is PASS, the latest scheduled-sync event is PASS with `changed=false`, and cursor SHA matches `79bf9a0`;
6. no token/private URL/OAuth value appears in tracked Hub or Goal evidence;
7. no test was deleted, skipped, weakened, or trivialized.

Do not call Feishu, push Git, edit files, resolve findings, or modify ledgers. Return PASS/FAIL for `F-M7-R03`, `F-M7-R04`, AC-04, AC-05, AC-09, AC-10, and AC-12 with exact commands and redacted evidence. Write the report to `reviews/milestone-7-final-live-fixes-review-01.md`.
