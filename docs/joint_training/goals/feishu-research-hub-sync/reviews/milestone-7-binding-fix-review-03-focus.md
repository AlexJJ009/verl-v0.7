Focused re-review of the sole residual F-M7-R01 defect from
`reviews/milestone-7-binding-fix-review-02.md`. Candidate
7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89 adds a complete-store preflight
before the execution/fetch loop.

Independently rerun the mixed valid-plus-invalid canary and require zero adapter
construction/fetch, zero commit/push, unchanged cursor, terminal stable binding
ERROR and matching status. Also rerun the full negative matrix and positive
valid-binding no-op, privacy, and attribution controls. Confirm no tests were
deleted, weakened, skipped, or trivialized.

Review only. Do not implement, push, call Feishu, modify ledger/parent
gitlink/PM2, or include any real object/folder token, URL, revision, or editor
value in tracked evidence.
