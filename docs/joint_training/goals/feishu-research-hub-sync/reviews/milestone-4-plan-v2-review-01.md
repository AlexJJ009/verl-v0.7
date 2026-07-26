# Milestone 4 Plan v2 Independent Review 01

## Review identity

- Reviewer: fresh independent reviewer (`/root/m4_plan_v2_review`)
- Requested model: GPT-5.5, medium. The spawn API did not expose GPT-5.5, so
  this review used the available model and records that fallback explicitly.
- Review type: Milestone Review
- Frozen Plan version / SHA-256: `2` /
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`
- Reviewed base / candidate Hub commits:
  `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72` /
  `efee75e560e44f9484ce2dc7a7e3d85a757f4fb8`
- Scope: AC-03 through AC-07, with AC-06 re-evaluated under the independently
  reviewed Plan v2 boundary selected by `D-AC06-01`.
- External/shared actions: read-only Drive listing and metadata queries inside
  the already approved D-04 disposable subtree only. No object was created,
  updated, moved, renamed, deleted, recreated, shared, or permission-modified.

## Overall verdict: PASS

Candidate `efee75e` satisfies the applicable Plan v2 milestone contract. The
full reviewer-owned suite and root gate pass. The focused AC-03 through AC-07
suite passes. AC-06 now proves recursive pagination, non-destructive structure
reconciliation, fail-closed suspected absence, and recoverable trusted-fixture
tombstones. It does not infer deletion from `970005` or any other polling
ambiguity. Live `drive.file.trashed_v1` consumption and live deletion-backed
tombstones are explicitly deferred by Plan v2 and therefore are not grounds for
repeating the Plan v1 live-tombstone failure.

## Per-AC verdict

| AC | Verdict | Reviewer evidence |
| --- | --- | --- |
| AC-03 | PASS | The full and focused suites cover revision preflight, one verified write, read-back, idempotent no-op, and stale-revision rejection before mutation. Earlier reviewer-owned D-04 evidence on the same named Docx observed live no-op publication/pull and unchanged revision after a stale preflight; this review's live calls were read-only. |
| AC-04 | PASS | Focused tests independently reproduce remote pull, exact content, attributable Author/Committer and Feishu trailers, local bare-remote push, status/audit binding, unknown-editor neutral identity, and no-op rerun. Existing D-04 evidence binds the named live editor IDs to the approved human identity. |
| AC-05 | PASS | The focused conflict story proves remote-active content, byte-exact local recovery through a reachable conflict ref, and writer-factory blocking while unresolved. Existing D-04 evidence previously exercised the same preservation path against the named disposable object. |
| AC-06 | PASS | Recursive inventory follows each root page exactly once and descends into folders; missing/repeated page tokens fail closed. Existing reviewer-owned D-04 root -> Archive -> root evidence returned `in_root_move` both ways and restored the object. A metadata row outside the managed inventory yields `detached`. `970003`, `970005`, and empty metadata all raise `E_ADAPTER_METADATA` before reconciliation, and the new candidate assertions prove entry/content bytes, audit, and tombstone state remain unchanged. Trusted `deleted=True` fixture evidence creates a readable tombstone retaining content, remote history, token, and adapter version without a writer call. No live deletion is claimed or required by Plan v2. |
| AC-07 | PASS | Focused canonicalization/diff/adapter-contract tests prove stable semantic and formatting-only behavior, assets and Mermaid/table surfaces, native Markdown parity, and explicit malformed/unsupported-version errors; the full root gate remains green at the reviewed commit. |

## Commands and evidence

```bash
cd /data-1/code/verl
sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md
# ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba

git -C research/feishu-research-hub rev-parse HEAD
# efee75e560e44f9484ce2dc7a7e3d85a757f4fb8

cd research/feishu-research-hub
uv run --with '.[test]' pytest -q
# 90 passed in 4.12s

uv run --with '.[test]' hubctl check --root .
# PASS

git diff --check 90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72 \
  efee75e560e44f9484ce2dc7a7e3d85a757f4fb8
# PASS

uv run --with '.[test]' pytest -q \
  tests/feature_stories/test_fs03_local_publish.py \
  tests/feature_stories/test_fs04_remote_edit_pull.py \
  tests/feature_stories/test_fs05_concurrent_remote_wins.py \
  tests/feature_stories/test_fs06_remote_structure_changes.py \
  tests/canonicalization tests/diff \
  tests/adapters/test_lark_cli_contract.py \
  tests/adapters/test_lark_cli_live_adapter.py
# 30 passed in 1.32s
```

Reviewer-owned live read-only inspection of the already authorized D-04
subtree:

```bash
lark-cli drive files list \
  --params '{"folder_token":"FEISHU_DISPOSABLE_ROOT_TOKEN","page_size":200}' \
  --as user --format json
# root contains exactly Archive, the named Docx, and the named Markdown;
# has_more=false

lark-cli drive files list \
  --params '{"folder_token":"FEISHU_DISPOSABLE_ARCHIVE_TOKEN","page_size":200}' \
  --as user --format json
# files=[]; has_more=false

lark-cli drive metas batch_query \
  --data '<exact named Docx/docx and Markdown/file tokens>' \
  --as user --format json
# two exact metadata rows; no failed_list
```

The final live state agrees with earlier reviewer-owned reversible D-04
evidence: the Docx was moved root -> Archive -> root, each Hub reconciliation
returned `in_root_move`, and the final root contains the same exact tokens with
Archive empty. This review did not repeat a write because the retained evidence
is already reviewer-owned and the current Plan requires no further mutation.

Candidate diff inspection adds the missing safety assertions rather than
weakening production behavior:

- `denied` (`970003`), `ambiguous` (`970005`), and empty metadata paths prove
  byte-for-byte unchanged `entry.yaml` and `content.md`, and prove no audit or
  tombstone directory is created;
- the trusted confirmed-deletion fixture now proves `deletion.json` retains
  the adapter version used to classify the event;
- no test was deleted or loosened between the reviewed base and candidate.

The live adapter itself has no `deleted=True` branch, which is correct under
Plan v2: polling results cannot prove deletion. Only trusted fixture input can
exercise the downstream tombstone branch in this Goal. Event subscription,
event consumption, and live deletion tombstoning remain deferred.

## Blocking in-scope defects

None.

## Deferred suggestions

None beyond the already frozen Plan v2 follow-up: a separate Goal may add
`drive.file.trashed_v1` subscription/consumption and live deletion-backed
tombstones after its own permissions, runtime, and acceptance decisions.

## Contract contradictions

None. Plan v2 removes the earlier external-adapter contradiction without
weakening the fail-closed preservation rule.

## Single most likely weakness in this review

The live root -> Archive -> root moves and out-of-root metadata classification
were not repeated in this review; the conclusion uses prior independent
reviewer-owned D-04 captures plus current read-only token/location verification
and deterministic adapter coverage. This is weaker than a same-round repeated
live move, but it avoids unnecessary remote mutation and still binds the final
state, exact objects, candidate behavior, and Plan v2 safety boundary.
