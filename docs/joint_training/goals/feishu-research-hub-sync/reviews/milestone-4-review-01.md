# Milestone 4 Independent Review

- Review identity: `m4_independent_review` (current routed model; the requested
  project reviewer model was unavailable)
- Frozen Plan version: `1`
- Base / candidate: `2a7ac5c6abcfbe79516364e0e9f2a2abc1dd2aa7` /
  `1eb7b3215f903129e048c2d8b616be5bd40ba010`
- Scope reviewed: AC-03 through AC-07 only
- Overall verdict: **FAIL**

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-03 | PASS | The reviewer ran all 83 tests and `hubctl check --root .` on the candidate. In a scratch-only copy bound to `FEISHU_DISPOSABLE_DOCX_TOKEN`, live `hubctl publish ... --adapter live` returned `changed:false, revision:FEISHU_DISPOSABLE_DOCX_REVISION`; live pull was a no-op; a reviewer-owned stale adapter publication returned `E_ADAPTER_STALE_REVISION` and a subsequent fetch remained at the same redacted revision. This specifically proves the adapter preflight, no-op, and no-mutation stale path for the named Docx. |
| AC-04 | PASS | `tests/feature_stories/test_fs04_remote_edit_pull.py` and `tests/git/test_attribution.py` passed in the 83-test run, including author/committer, trailers, local-bare-remote push, no-op, and unknown-editor handling. Live Docx history at `FEISHU_DISPOSABLE_DOCX_REVISION` returned editor `FEISHU_DISPOSABLE_DOCX_EDITOR_ID`; live Markdown history returned `FEISHU_DISPOSABLE_MARKDOWN_EDITOR_OPEN_ID`. Both map to the approved `GongxunLi <lgxma01@buaa.edu.cn>` in the candidate identity map. |
| AC-05 | PASS | `tests/feature_stories/test_fs05_concurrent_remote_wins.py` passed: it asserts the local bytes are retained on a reachable `refs/hub-conflicts/...` ref, remote content is active, and the writer factory is never constructed while conflict is unresolved. The scratch live probe also left a readable conflict ref, demonstrating the same preservation mechanism with the named object. |
| AC-06 | FAIL | The fake feature-story tests pass, and the reviewer independently moved the named Docx root -> `Archive` -> root and renamed it -> restored title; Drive listings confirmed the exact token remained inside the approved subtree and the Archive finished empty. But the candidate `LarkCliAdapter` has only `version`, `fetch`, and `publish`; it has no inventory/history/structure API, and `hubctl` exposes no live structure reconciliation command. `reconcile_inventory` is therefore reachable only with `FakeLarkAdapter` in this candidate. The implementation cannot turn an actual live rename/in-root move/out-of-root move/deletion into the required metadata update, detach, or tombstone. Manual Drive capability does not prove the Hub behavior required by AC-06. |
| AC-07 | PASS | The 83-test reviewer run includes canonicalization, diff, and lark-cli contract tests. Candidate adapter version was independently observed as `1.0.76`, matching the tested contract; malformed/unsupported-version cases remain deterministic fixtures, as required by the Plan. |

## Commands and evidence

Executed from the candidate Hub unless stated otherwise:

```text
uv run --with '.[test]' pytest -q
# 83 passed in 3.48s

uv run --with '.[test]' hubctl check --root .
# PASS

lark-cli docs +fetch --doc FEISHU_DISPOSABLE_DOCX_TOKEN ...
# Docx title/content present; revision_id=FEISHU_DISPOSABLE_DOCX_REVISION

lark-cli docs +history-list --doc FEISHU_DISPOSABLE_DOCX_TOKEN ...
# revision FEISHU_DISPOSABLE_DOCX_REVISION editor_ids=[FEISHU_DISPOSABLE_DOCX_EDITOR_ID]

lark-cli markdown +fetch --file-token FEISHU_DISPOSABLE_MARKDOWN_TOKEN ...
lark-cli drive +version-history --file-token FEISHU_DISPOSABLE_MARKDOWN_TOKEN ...
# Markdown version=FEISHU_DISPOSABLE_MARKDOWN_REVISION, edited_by=FEISHU_DISPOSABLE_MARKDOWN_EDITOR_OPEN_ID

lark-cli drive files list --params '{"folder_token":"FEISHU_DISPOSABLE_ROOT_TOKEN","page_size":200}' ...
lark-cli drive files list --params '{"folder_token":"FEISHU_DISPOSABLE_ARCHIVE_TOKEN","page_size":200}' ...
# Final root contains only Archive, FS03-DOC Capability Probe, FS03-MD Capability Probe.md;
# Archive is empty.
```

Reviewer-owned bounded live checks in
`/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/m4-final.G6vpa2/hub`:

```text
hubctl publish FS03-DOC --adapter live --json
# changed=false, revision=8
hubctl pull FS03-DOC --adapter live --json
# changed=false, state=clean
adapter.publish(... expected_revision='7' ...)
# E_ADAPTER_STALE_REVISION; read-back remained FEISHU_DISPOSABLE_DOCX_REVISION
```

The reviewer performed a reversible Docx move into the named Archive and back,
then a reversible rename and restoration. No delete, permission, sharing, or
non-named Drive object operation was performed.

## Blocking in-scope defect

`IN_SCOPE_DEFECT`: AC-06 live integration is absent. Add a live Drive inventory
adapter and a `hubctl` reconciliation entrypoint that fetches the approved
managed-root inventory and histories, then drives `reconcile_inventory` for
rename, in-root move, out-of-root detachment, and tombstone handling. Add
reviewer-repeatable live tests against only the named disposable subtree. The
existing fake-only test must remain; it cannot be replaced by the live probe.

## Deferred suggestions

None.

## Contract contradictions

None.

## Most likely weakness in this review

The live probe was deliberately bounded to the D-04 disposable objects and did
not exercise deletion (the frozen Runtime Contract requires a new current-turn
high-risk confirmation for it). The AC-06 failure does not depend on that
unperformed delete: inspection shows no live inventory/reconciliation path at
all.
