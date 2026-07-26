Review only the convergence-approved F-M7-R01 correction at candidate
82cabe1c361ba7141762c50b5d4da6a4bd0175f7. The independent convergence
report is `reviews/milestone-7-binding-convergence-01.md`.

Verify `.hub/live-bindings.json` is now the single runtime-private authority
for live object bindings. Independently exercise the full negative matrix:
missing store, empty entries, malformed document/schema/entry, missing or empty
token, fixture-shaped live token, unknown entry binding, and configured entry
without a valid private binding. Each must fail before adapter use, Git commit
or push, or cursor advancement; `sync_once` and the tracked launcher must record
and expose a stable terminal binding ERROR. Also run the positive valid-binding
no-op control and prove direct publish/pull privacy plus AC-04
Feishu-Revision/Feishu-Editor-Ids trailers remain intact.

Review only. Do not implement, push, call Feishu, modify the parent gitlink,
edit deployed PM2 state, or expose any real object/folder token, private URL,
revision, or editor value in tracked evidence.
