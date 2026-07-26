# Milestone 7 Binding Convergence Review 01

## Review identity

- Reviewer: fresh independent convergence reviewer using the available
  `gpt-5.6-terra` medium-reasoning fallback; the project-requested Haiku model
  was unavailable.
- Review type: convergence review before a third ordinary repair round for
  `F-M7-R01`.
- Frozen Plan: version `2`, SHA-256
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Base commit: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`.
- Candidate commit: `66d6f5aaf804d17c0f37f312f28277dea88a2c98`.
- Scope: `AC-03`, `AC-04`, `AC-10`, and `AC-12`. No implementation,
  ledger edit, Git push, or Feishu call was performed.

## Overall verdict

**FAIL, with convergence disposition `IN_SCOPE architectural fix`.**

The recurrence is primarily a **duplicated-source-of-truth migration defect**.
The rejected baseline treated tracked `entry.yaml` as the live object-binding
authority. The candidate correctly introduces `.hub/live-bindings.json` as the
runtime-private authority and removes live tokens from tracked content, but the
scheduled reconcile path does not make absence or invalidity of that new
authority fatal. `load_live_bindings` converts a missing store into an empty
valid mapping, and `reconcile_live` skips invalid token values. The result is a
partial migration in which the private authority exists but is not required by
the periodic writer.

This is not experiment-specific policy in a shared layer, reviewer scope
expansion, or a second independently useful outcome. The frozen Architecture
Contract already requires private tokens outside Git, and AC-10/AC-12 already
require truthful, fail-closed periodic execution. No Plan amendment or fresh
Plan review is needed.

For live root/object bindings, `.hub/live-bindings.json` must be the **single
runtime-private authority**. Tracked `entry.yaml` may keep non-secret sync
metadata required by the frozen contract, including revision/editor attribution,
but it must neither supply nor recover a live folder/object token. A scheduled
live cycle must never interpret an absent, empty, or invalid private authority
as “nothing to synchronize.”

## Per-AC verdicts

| AC | Verdict | Reviewer evidence |
| --- | --- | --- |
| AC-03 | PASS | Direct live publish resolves the object binding with `require_private=True`, clears any tracked live token, stores the token under `.hub`, and omits it from result/audit evidence. The focused FS-03 and privacy tests pass. |
| AC-04 | PASS | Direct live pull uses the private binding and removes the token before an attributable commit. Revision/editor identifiers remain present in structured Git trailers and private audit as required; only their concrete values are redacted from this report. |
| AC-10 | FAIL | Scheduled `reconcile_live` returns a successful no-op for a missing store, empty mapping, tokenless binding, empty token, and fixture-shaped live token. A full `sync_once` can consequently proceed to push and cursor advancement instead of recording a binding ERROR. |
| AC-12 | FAIL | The periodic-job-shaped leg of the final live Feature Story is not fail closed, so the exact committed candidate is not yet safe for the authorized live acceptance cycle. |

## Commands and evidence

Required reviewer-owned verification:

```text
cd /data-1/code/verl/research/feishu-research-hub && \
  .venv/bin/python -m pytest -q \
    tests/gate/test_live_binding_privacy.py \
    tests/feature_stories/test_fs03_local_publish.py \
    tests/feature_stories/test_fs04_remote_edit_pull.py \
    tests/git/test_attribution.py \
    tests/runtime/test_sync_once.py
22 passed in 1.67s

.venv/bin/python -m pytest -q
132 passed in 14.38s

.venv/bin/python -m hubctl check --root .
PASS

git diff --check \
  593b4bac5ab46bde0b732e2b705c7aa0a79d2846 \
  66d6f5aaf804d17c0f37f312f28277dea88a2c98
exit 0

goal-plan-runtime validate-runtime \
  /data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 7;
plan_status=READY; pending_user_decisions=[]
```

The green suite does not cover the required negative binding matrix. A
reviewer-owned synthetic probe called only `reconcile_live` with a forbidden
adapter canary. It made no external call and produced this redacted matrix:

| Synthetic live case | Current outcome | Adapter calls |
| --- | --- | ---: |
| missing binding store | `PASS`, unchanged/no revision | 0 |
| empty `entries` | `PASS`, unchanged/no revision | 0 |
| malformed JSON document | `ERROR E_STATE_READ` | 0 |
| malformed entry shape | `ERROR E_BINDING_STATE` | 0 |
| missing token | `PASS`, unchanged/no revision | 0 |
| empty token | `PASS`, unchanged/no revision | 0 |
| fixture-shaped token | `PASS`, unchanged/no revision | 0 |
| binding for unknown entry | `ERROR E_BINDING_STATE` | 0 |
| configured entry without a valid binding | `PASS`, unchanged/no revision | 0 |

The source-level failure path is direct:

- `src/hubctl/bindings.py` calls `read_json` with an empty-store default, so a
  missing `.hub/live-bindings.json` becomes a valid empty document.
- `src/hubctl/runtime.py` iterates only the private mapping and explicitly
  continues past missing, empty, non-string, or fixture-shaped tokens.
- `sync_once` treats the returned `changed=false` result as success, performs
  its Git push, appends `SYNC_SUCCESS`, and writes the cursor. Its existing
  exception handler would record truthful `ERROR` and preserve the cursor if
  `reconcile_live` raised a stable binding error instead.

## Required fail-closed case matrix for the next implementation

The third repair round must add reviewer-verifiable negative cases with the
following disposition. `E_ADAPTER_BINDING` is appropriate for absent required
authority; `E_BINDING_STATE` is appropriate when authority exists but is
invalid. Equivalent stable binding-specific codes are acceptable only if used
consistently by both direct and scheduled live entrypoints.

| Live case | Required disposition |
| --- | --- |
| Binding store missing | Raise nonzero `E_ADAPTER_BINDING` before adapter construction. |
| `entries` missing or empty | Raise nonzero `E_ADAPTER_BINDING`; live scheduled mode cannot use an empty work set as a successful no-op. |
| JSON/YAML document malformed, wrong schema version, or `entries` wrong type | Normalize to nonzero `E_BINDING_STATE`; do not leak a generic state-reader outcome as the live binding contract. |
| Binding entry is not a mapping | Raise nonzero `E_BINDING_STATE`. |
| Object token missing, non-string, or empty | Raise nonzero `E_BINDING_STATE`; never `continue`. |
| Fixture-shaped token in the live store | Raise nonzero `E_BINDING_STATE`; fixture values are valid only in tracked deterministic fixtures, not live private state. |
| Binding key refers to no tracked `entry.yaml` | Raise nonzero `E_BINDING_STATE`. |
| Configured managed entry has no matching private binding | Raise nonzero `E_ADAPTER_BINDING`. Treat an ID present in the private managed-entry mapping but lacking a valid binding item as unmatched; do not introduce a second tracked live-target list merely to satisfy this test. |

Every case must additionally prove all of these invariants:

1. the adapter constructor/fetch/publish call count remains zero;
2. no Git commit or push occurs;
3. the existing cursor bytes remain unchanged, or no cursor is created;
4. `sync-events.jsonl` ends with a terminal `SYNC_ATTEMPT` whose outcome is
   `ERROR` and whose `error_code` is the stable binding code;
5. `hubctl status` reports that same terminal error rather than a PASS/no-op;
6. `scripts/sync_once.sh` exits nonzero and does not convert the error into
   success;
7. no live token, folder token, private URL, OAuth material, or concrete
   revision/editor value enters stdout, tracked tests, Git, or review evidence.

The positive control must retain the existing no-op behavior when the private
store is valid and every configured binding is valid: zero remote writes,
successful status, and no duplicate commit. The fix must also retain AC-04's
structured `Feishu-Revision` and `Feishu-Editor-Ids` trailers/private audit.

## Blocking in-scope defects

- **F-M7-R01 remains open.** The periodic live writer does not validate its
  sole private binding authority before treating the cycle as successful. This
  is the one convergence-approved architectural correction required before a
  third ordinary review and before any D-07 live mutation.

## Deferred suggestions

None.

## Contract contradictions

None. The required correction implements existing AC-10/AC-12 behavior and
does not change the accepted outcome, external operation envelope, or deletion
scope.

## Single most likely weakness in this review

The reviewer did not execute a real Feishu operation, as prohibited by this
review lane. The full `sync_once` consequences are established from its tested
control flow plus the direct negative `reconcile_live` matrix; the next review
must run the new end-to-end negative `sync_once`/launcher canaries before the
separate authorized disposable live acceptance.
