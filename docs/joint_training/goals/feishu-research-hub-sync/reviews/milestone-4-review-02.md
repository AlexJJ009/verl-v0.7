# Milestone 4 Mechanical Re-verification

- Review identity: `m4_independent_review` (same reviewer; current routed model)
- Frozen Plan version: `1`
- Reviewed base / candidate: `1eb7b3215f903129e048c2d8b616be5bd40ba010` /
  `1102bc8e1001a76423a740e5aee36e7640abb763`
- Scope: mechanical re-verification of F-M4-R01 / AC-06 only
- Overall verdict: **FAIL**

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-06 | FAIL | The new live command and current-subtree move behavior work, but the live inventory still treats any empty `metas` response as deletion and does not process Drive pagination. Either state can produce a false tombstone instead of a safe error/continuation, violating AC-06's recoverability requirement. |

## Commands and evidence

The candidate itself passed the mandatory deterministic checks:

```text
cd /data-1/code/verl/research/feishu-research-hub
uv run --with '.[test]' pytest -q
# 84 passed in 3.72s
uv run --with '.[test]' hubctl check --root .
# PASS
```

I created a new scratch-only checkout at candidate `1102bc8`, bound only its
`FS03-DOC` fixture to the approved D-04 Docx and root, regenerated its local
catalog, and ran the new live command. Initial reconciliation returned
`metadata_update`, with clean state and root path
`FS03-DOC Capability Probe`.

I then performed exactly the reviewer-authorized reversible sequence on the
named object only:

```text
drive +move FEISHU_DISPOSABLE_DOCX_TOKEN (docx) -> FEISHU_DISPOSABLE_ARCHIVE_TOKEN (Archive)
hubctl reconcile-inventory --adapter live --json
# [{"action":"in_root_move", ..., "state":"clean"}]
# remote_path: Archive/FS03-DOC Capability Probe

drive +move FEISHU_DISPOSABLE_DOCX_TOKEN (docx) -> FEISHU_DISPOSABLE_ROOT_TOKEN (managed root)
hubctl reconcile-inventory --adapter live --json
# [{"action":"in_root_move", ..., "state":"clean"}]
# remote_path: FS03-DOC Capability Probe
```

Final recursive reads showed the approved root contains only `Archive`,
`FS03-DOC Capability Probe`, and `FS03-MD Capability Probe.md`; Archive is
empty. No delete, seed publication, permission/sharing mutation, or object
outside the D-04 subtree was touched. Direct live adapter inventory also found
both named objects in-root and returned 8 Docx history rows and 2 Markdown
history rows.

## Blocking in-scope defects

1. **False deletion on metadata lookup failure.** In
   `src/hubctl/adapters/lark_cli.py`, the row-absent branch calls
   `drive metas batch_query`, then classifies `metas == []` as
   `RemoteDocument(deleted=True)`. The documented response also supplies
   `failed_list`; specifically, a permission failure (`970003`) or token/type
   mismatch (`970005`) can yield no metadata without proving deletion. The
   candidate ignores `failed_list`, so a transient/access failure would create
   an irreversible-in-practice local tombstone from unproven remote state.
   This must fail closed with an adapter error; only explicit "not found" /
   deletion evidence may reach tombstoning.

2. **Incomplete paginated inventory is rejected rather than completed.** The
   live adapter raises `E_ADAPTER_PAGINATION` whenever `files.list` reports
   `has_more=true` (lines 87-88). The project Drive guidance requires following
   `next_page_token` until completion. A managed root with more than one page
   therefore cannot reconcile remote structural changes, despite AC-06 requiring
   reconciliation within the managed root. Implement pagination and add a
   canary proving that a second page is scanned exactly once.

Both defects are within F-M4-R01's live adapter change and are blocking
`IN_SCOPE_DEFECT`s. They are behavioral/safety defects, not a formatting-only
mechanical issue.

## Deferred suggestions

None.

## Contract contradictions

None.

## Most likely weakness in this review

The live test intentionally did not move the disposable objects outside the
approved root or delete them. Those actions would enlarge the remote mutation
envelope. The reported failures do not rely on that omission: both follow
directly from the candidate's metadata and pagination branches, whose missing
failure coverage is visible in the added tests.
