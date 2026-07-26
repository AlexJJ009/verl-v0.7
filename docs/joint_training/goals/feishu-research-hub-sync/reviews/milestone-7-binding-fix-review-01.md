# Milestone 7 Binding Fix Review 01

## Review identity

- Reviewer: fresh independent `gpt-5.6-terra` reviewer, medium reasoning
  fallback (the project-requested Haiku tier was unavailable).
- Frozen Plan: v2, SHA-256
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Base: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`.
- Candidate: `66d6f5aaf804d17c0f37f312f28277dea88a2c98`.
- Scope: F-M7-R01 and F-M7-02 only; no live Feishu call, Git push, ledger
  mutation, or implementation change was performed.

## Overall verdict

**FAIL.** The candidate fixes the tracked-token and ordinary live publish/pull
leak found in F-M7-R01, and preserves the AC-04 revision/editor trailer
contract required by F-M7-02. However, `reconcile_live` silently treats a
malformed runtime binding as an empty work set. A scheduled live sync can
therefore return a successful no-op instead of failing closed. This directly
violates the review prompt's malformed/missing live-binding requirement and
leaves F-M7-R01 open.

## Applicable AC verdicts

| AC | Verdict | Reviewer evidence |
| --- | --- | --- |
| AC-03 | PASS | Live-mode `publish_entry` resolves with `require_private=True`, stores the binding in ignored runtime state, clears tracked token state, and its response/audit omit the object token. The focused privacy and FS-03 tests pass. |
| AC-04 | PASS | Live-mode `pull_entry` uses the private binding and clears tracked token state before an optional commit. The existing attribution flow retains structured `Feishu-Revision` and `Feishu-Editor-Ids` trailers plus audit revision/editor fields; FS-04 and attribution tests pass. |
| AC-09 | PASS | Full suite passed and `hubctl check` passed. The new root-gate detector has a reviewer-run green-to-red canary. |
| AC-10 | FAIL | `reconcile_live` skips a binding whose token is missing, empty, non-string, or fixture-shaped rather than raising a stable binding error. A reviewer-owned malformed/missing-binding probe returned `{'changed': False, 'revision': None}`. That is an untruthful successful scheduled-sync result. |
| AC-12 | FAIL | AC-12 requires the periodic-job-shaped pull/status path to be independently safe. Because its live reconcile path can silently accept malformed or missing binding state, the end-to-end binding boundary is not proven fail-closed. |

## Commands and evidence

Reviewer-owned commands, all run from the candidate checkout:

```text
cd research/feishu-research-hub && .venv/bin/python -m pytest -q
132 passed in 14.39s

cd research/feishu-research-hub && .venv/bin/python -m hubctl check --root .
PASS

cd research/feishu-research-hub && git diff --check <base> <candidate>
exit 0

cd research/feishu-research-hub && git show --stat --oneline <candidate>
66d6f5a fix: keep live Feishu bindings outside Git
12 files changed, 421 insertions(+), 33 deletions(-)

goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 7; plan_status=READY
```

Focused reviewer-owned suite:

```text
tests/gate/test_live_binding_privacy.py
tests/feature_stories/test_fs03_local_publish.py
tests/feature_stories/test_fs04_remote_edit_pull.py
tests/git/test_attribution.py
tests/runtime/test_sync_once.py
22 passed
```

Additional detector-matrix canary:

```text
tests/gate/test_detector_canary_matrix.py
32 passed
```

The candidate's positive privacy evidence is behavioral, not merely a test
name: focused live publish/pull exercises assert that the synthetic private
object token is absent from the tracked entry, Git commit/history, bare remote,
JSON result, and audit; the runtime binding is ignored; and the private
directory/file modes are `0700`/`0600`. Source inspection confirms live
publish, pull, inventory, and scheduled reconcile use runtime-private
resolution, while fake adapters keep fixture-prefixed bindings deterministic.
The root gate rejects a non-fixture tracked object token with
`E_PRIVATE_BINDING_TRACKED`; its matrix canary passed green-before-red.

The candidate also correctly keeps AC-04's revision/editor data separate from
object-token secrecy: `commit_remote_entry` writes structured
`Feishu-Revision` and `Feishu-Editor-Ids` trailers, and the FS-04 assertions
verify both trailers and audit fields. This report intentionally uses no real
identifier or revision value.

The failing reviewer probe created only a temporary local Hub-shaped directory
with no binding store and invoked `reconcile_live`; it made no external call and
printed the following redacted result:

```text
missing_binding_result={'changed': False, 'revision': None}
```

Code inspection identifies the cause:

- `src/hubctl/runtime.py:116-123` iterates only existing binding entries and
  `continue`s for absent/invalid token values.
- With no binding file, `load_live_bindings` supplies an empty `entries` map,
  so `reconcile_live` returns success at line 160 instead of an explicit
  binding error.
- This differs from the direct live `publish`/`pull` path, where
  `resolve_entry_binding(..., require_private=True)` raises
  `E_ADAPTER_BINDING`.

## Blocking in-scope defects

- **F-M7-R01 remains open — scheduled live reconcile does not fail closed for
  malformed or missing binding state (AC-10, AC-12).** The correction must
  make a live reconcile with any configured target and no valid runtime-private
  binding stop with a stable nonzero binding error. Invalid binding entry
  shapes/tokens must not be silently skipped. Add reviewer-verifiable canaries
  for absent, malformed, and over-permissive binding state. The permissions
  canary already correctly returns `E_BINDING_PERMISSIONS`; this finding is
  limited to the no-op branch.

## Deferred suggestions

None.

## Contract contradictions

None. F-M7-02 is resolved within the frozen contract: object/folder tokens
remain private, while revision/editor identifiers remain structured attribution
trailers and audit fields and are redacted only in tracked reviewer evidence.

## Single most likely weakness in this review

No real Feishu operation was run by design. The report relies on deterministic
fixture behavior plus source-level control-flow review for the live adapter;
the required disposable live acceptance remains a separate authorized final
review.
