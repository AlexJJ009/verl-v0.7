# Milestone 4 Independent Review 03

## Review identity

- Reviewer: fresh independent reviewer (`/root/m4_fresh_review_03`)
- Requested model: GPT-5.5, medium. The routing layer did not expose that
  override to this reviewer; review performed with the available model.
- Plan version: 1 (frozen)
- Base commit: `1eb7b3215f903129e048c2d8b616be5bd40ba010`
- Candidate commit: `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72`
- Scope: AC-06 only
- Feishu operations: none. This review made no Feishu write, move, rename,
  update, delete, sharing, or permission call.

## Overall verdict: FAIL

The candidate correctly makes pagination and unproven metadata responses fail
closed, but the live inventory adapter has no path that represents a confirmed
remote deletion as `RemoteDocument(deleted=True)`. Therefore the real
`reconcile-inventory` path cannot invoke AC-06's tombstone branch for a deleted
object. Fixture-only tombstone coverage remains green, but does not prove the
live adapter can supply its required deletion input.

| AC | Verdict | Evidence |
|---|---|---|
| AC-06 | FAIL | `LarkCliAdapter.inventory()` either finds an object, treats one metadata row as an out-of-root move, or raises `E_ADAPTER_METADATA` for zero metadata rows. It never emits a deleted document. `reconcile_inventory()` only creates a tombstone when `remote.deleted` is true. |

## Commands and evidence

```bash
cd /data-1/code/verl/research/feishu-research-hub
uv run --with '.[test]' pytest -q
```

Result: `90 passed in 4.08s`.

```bash
cd /data-1/code/verl/research/feishu-research-hub
uv run --with '.[test]' hubctl check --root .
```

Result: `PASS`.

```bash
cd /data-1/code/verl/research/feishu-research-hub
uv run --with '.[test]' pytest -q \
  tests/feature_stories/test_fs06_remote_structure_changes.py \
  tests/adapters/test_lark_cli_live_adapter.py
git diff --check 1eb7b3215f903129e048c2d8b616be5bd40ba010 \
  90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72
```

Result: `14 passed in 0.90s`; whitespace check passed.

Candidate inspection shows the intended positive behavior:

- recursive `drive files list` traversal covers root children and descendants;
- missing/repeated page tokens raise `E_ADAPTER_PAGINATION`;
- a found metadata row becomes `outside-managed-root` and thus a detached
  entry, rather than a tombstone;
- `failed_list` and ambiguous empty metadata both raise
  `E_ADAPTER_METADATA`, avoiding a false tombstone;
- deterministic FS-06 fixture tests still verify rename, in-root move,
  detach, tombstone preservation, and no fake `publish` call.

The blocking gap is in
`src/hubctl/adapters/lark_cli.py:117-152`: after a managed-root inventory miss,
the only branches are an out-of-root metadata hit or `E_ADAPTER_METADATA`.
No branch returns a `RemoteDocument` with `deleted=True`. In contrast,
`src/hubctl/sync.py:154-194` creates the recoverable tombstone only after it
receives `remote.deleted`. Thus a confirmed live deletion is currently
indistinguishable from an error at the integration boundary, not reconciled
into the required tombstone.

## Blocking in-scope defects

- **F-M4-R04 — live confirmed deletion cannot create a tombstone**
  (`IN_SCOPE_DEFECT`, behavioral): add an explicit, documented evidence path
  for a confirmed deletion response from Drive metadata and make it return a
  deleted `RemoteDocument`; retain fail-closed behavior for permission,
  malformed, and ambiguous responses. Add a live-adapter canary that proves
  the deletion classification reaches `reconcile_inventory()` and produces a
  tombstone with no remote write.

## Deferred suggestions

None.

## Contract contradictions

None. This is an in-scope implementation defect under frozen AC-06, not a Plan
change.

## Single most likely weakness in this review

The exact lark-cli/OpenAPI deletion evidence schema was not exercised against a
live deleted object because this review is explicitly read-only. The defect does
not depend on guessing that schema: the current adapter has no output path for
`deleted=True` under any schema, so it cannot satisfy live tombstone
reconciliation as written.
