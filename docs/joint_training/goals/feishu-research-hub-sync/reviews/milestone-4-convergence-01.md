# Milestone 4 Convergence Review 01

## Review identity

- Reviewer: fresh independent convergence reviewer (`/root/m4_convergence_review`)
- Requested model: GPT-5.5, medium. The routing layer did not expose that
  override; this review used the available model.
- Frozen Plan version: `1`
- Base / candidate: `1eb7b3215f903129e048c2d8b616be5bd40ba010` /
  `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72`
- Scope: the repeated AC-06 live-inventory/tombstone finding only
- Feishu operations: none. This review made no Feishu write, move, rename,
  update, delete, permission, sharing, or history call.

## Overall verdict: CONTRACT_CONTRADICTION

The first two failures were bounded implementation defects: the adapter lacked
live reconciliation, then it did not traverse all pages and could false-positive
a tombstone. Candidate `90204fc` fixes those safely. The remaining failure is
not a fourth ordinary implementation gap: the only available metadata read
contract cannot prove a deletion. Its `970005` means either token/type mismatch
or nonexistent, and the prompt explicitly rules out treating it alone as
deletion proof. The live adapter therefore correctly fails closed, but no
documented lark-cli read result inspected in this review can produce the
required `RemoteDocument(deleted=True)` safely.

The loop is caused by a **frozen-contract contradiction at the external adapter
boundary**, not duplicated source of truth, shared-layer policy, reviewer scope
expansion, or a second outcome. A fourth ordinary fix would either invent an
unsafe deletion inference or repeat the same failure. Amend and re-review the
Plan before resuming this AC.

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-06 | WEAKENED | Rename/in-root move, out-of-root metadata, pagination, and ambiguous-metadata fail-closed behavior are implemented and tested. The live confirmed-deletion-to-tombstone input remains unprovable from the exposed CLI schema, so the complete frozen AC cannot receive PASS. |

## Commands and evidence

```bash
cd /data-1/code/verl/research/feishu-research-hub
uv run --with '.[test]' pytest -q
# 90 passed in 4.29s

uv run --with '.[test]' pytest -q \
  tests/feature_stories/test_fs06_remote_structure_changes.py \
  tests/adapters/test_lark_cli_live_adapter.py
# 14 passed in 0.96s

uv run --with '.[test]' hubctl check --root .
# PASS

git diff --check 1eb7b3215f903129e048c2d8b616be5bd40ba010 \
  90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72
# PASS

lark-cli schema drive.metas.batch_query --format json
```

Reviewer inspection of the last command's output establishes that
`failed_list.code=970005` is documented as “file token and doc_type mismatch
or file does not exist”; `970003` is a permissions failure. Candidate
`LarkCliAdapter.inventory()` correctly turns every `failed_list` response and
an empty ambiguous response into `E_ADAPTER_METADATA`. It returns a detached
document only for an actual metadata row and follows every inventory page,
rejecting malformed or repeated page tokens. `reconcile_inventory()` creates a
tombstone only when `RemoteDocument.deleted` is true, but the adapter has no
safe branch that can return that value.

The deterministic FS-06 fixture remains meaningful: it proves tombstone
preservation and no fake remote create/delete when a trusted adapter supplies
`deleted=True`. It is not proof that the live CLI can safely supply that fact.

## Exact evidence contract required after amendment

A future live adapter may emit `RemoteDocument(deleted=True)` only when all of
the following are true:

1. A documented Feishu/lark-cli response for the exact stored token and stored
   representation explicitly states deletion/nonexistence, with a response code
   that is disjoint from type mismatch, permission, token invalidity, and
   transient failure.
2. The adapter records the raw classified evidence code/schema version and the
   prior stored token/representation in `deletion.json`.
3. Any other response, including `970005`, missing metadata, malformed output,
   incomplete inventory, fetch/history error, or permission error, returns a
   fail-closed adapter error and leaves the entry untouched.
4. A fixture canary proves each ambiguous/error class cannot create a
   tombstone; a separate trusted-confirmed-deletion fixture proves the adapter
   produces `deleted=True` and `reconcile_inventory()` retains the existing
   content, history, and audit evidence without a remote write.

At present the inspected `drive.metas.batch_query` schema does not meet item 1.
A complete recursive root inventory plus a `970005` response is still not
enough under the frozen prompt, because the latter remains ambiguous. A real
remote delete would be a high-risk decision and is not needed merely to reveal
that the exposed response class is insufficient.

## Required Plan decision

This is a `CONTRACT_CONTRADICTION`, not an `IN_SCOPE_DEFECT`. Before another
ordinary review/fix round, the Plan must choose and independently re-review one
of these contract-level changes:

- identify a new documented, read-only Feishu API/CLI capability whose schema
  supplies the disjoint deletion proof above, then add that exact probe and
  adapter contract; or
- limit live AC-06 reconciliation to rename/move/detach plus fail-closed
  suspected absence, while retaining fixture-only tombstone behavior until a
  later separately authorized capability Plan.

Neither option is a safe implementation choice under the current frozen Plan.

## Blocking in-scope defects

None remaining. `F-M4-R01`, `F-M4-R02`, and `F-M4-R03` were bounded defects
and are closed by candidate behavior. `F-M4-R04` must be reclassified as the
contract contradiction described above rather than repaired through another
ordinary implementation round.

## Deferred suggestions

None.

## Contract contradictions

- `CONTRACT_CONTRADICTION`: AC-06/Milestone 4 demand live confirmed deletion
  reconciliation, while the exposed, documented metadata command has no
  deletion-exclusive result and the runtime prohibits the unnecessary
  high-risk remote-delete probe. The required safe input cannot be obtained
  from the frozen adapter contract.

## Single most likely weakness in this review

This review did not exhaustively explore every unregistered Feishu OpenAPI
endpoint. It did verify the exact public lark-cli command the candidate uses
and the specified schema. If an unregistered API has a deletion-exclusive
read response, discovering and accepting that new dependency is a Plan
amendment, not grounds for a fourth ordinary AC-06 implementation round.
