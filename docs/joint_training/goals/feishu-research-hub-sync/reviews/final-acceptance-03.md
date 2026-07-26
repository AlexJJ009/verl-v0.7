# Final Acceptance 03 — REJECTED / PENDING REPAIR

## Review identity

Fresh independent final reviewer (`final_acceptance03`, GPT-5 Codex fallback;
the requested GPT-5.5 medium route was unavailable).  Reviewed Plan version 2
at SHA-256 `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`,
Hub candidate `7078fdd`, and parent candidate `22499d3`.  This reviewer did
not edit `runtime.jsonl`, `findings.jsonl`, or `acceptance.md`.

## Overall verdict

**REJECTED / PENDING REPAIR.** Deterministic evidence is green, but the single
authorized D-07 live publication could not begin: its pre-write compare-and-swap
check returned `E_ADAPTER_STALE_REVISION`.  The runtime-private binding's stored
revision no longer matched the disposable Docx's fetched revision.  The error
occurred before any Feishu update; no retry, binding repair, human-edit
simulation, pull, commit, push, or parent-gitlink update was performed.

## Per-AC verdict

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-01 | PASS | Candidate submodule path, mode-160000 gitlink, Hub-local hook path and selected local identity were verified; parent/global identity remained distinct. |
| AC-02 | PASS | Full suite plus the complete detector-canary matrix passed; each detector has a green-before-red known-bad control. |
| AC-03 | WEAKENED | Fake Feature Story and writer/gate tests pass, but the required disposable live publish/read-back could not pass the pre-write CAS gate. The observed stale-revision rejection is fail-closed, not successful live publication evidence. |
| AC-04 | PASS | Feature Story pull and attribution tests passed in the complete deterministic suite. |
| AC-05 | PASS | Concurrent remote-wins preservation Feature Story passed. |
| AC-06 | PASS | Structure-change, suspected-absence fail-closed, and no-recreation Feature Story passed. |
| AC-07 | PASS | Canonicalization, diff, and lark-cli contract tests passed. |
| AC-08 | PASS | Pre-push and writer-gate tests, including bypass-resistant writer checks, passed. |
| AC-09 | PASS | Pristine-clone PASS/RED/ERROR runner and verdict-query tests passed. |
| AC-10 | PASS | Locking, sync-once, and status tests passed; runtime preflight showed the fixture cursor at the reviewed Hub candidate with terminal PASS. |
| AC-11 | PASS | `verl` importer and curated-seed Feature Story tests passed. |
| AC-12 | FAIL | The authorized end-to-end story is incomplete: no successful live publish/read-back, user-authored edit, periodic pull, attributable live commit/push, or no-op pull can be claimed. |

## Commands and redacted evidence

The following reviewer-owned commands completed successfully:

```text
goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync
goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md
cd research/feishu-research-hub && uv run --with .[test] pytest -q
cd research/feishu-research-hub && uv run --with .[test] hubctl check --root .
```

Results: Plan validation `PASS`; runtime validation reports Plan `READY`,
Milestone 7 active, no pending user decisions, and prior focused review `PASS`;
the Plan hash matched; `144 passed`; root gate `PASS`.

Focused frozen AC and known-bad coverage was independently rerun:

```text
uv run --with .[test] pytest -q \
  tests/gate/test_detector_canary_matrix.py \
  tests/gate/test_live_binding_privacy.py \
  tests/hooks/test_pre_push.py tests/gate/test_writer_gate.py \
  tests/runtime/test_sync_once.py
uv run --with .[test] pytest -q \
  tests/feature_stories/test_fs03_local_publish.py \
  tests/feature_stories/test_fs04_remote_edit_pull.py \
  tests/feature_stories/test_fs05_concurrent_remote_wins.py \
  tests/feature_stories/test_fs06_remote_structure_changes.py \
  tests/canonicalization tests/diff tests/adapters/test_lark_cli_contract.py \
  tests/local_ci/test_runner_e2e.py tests/local_ci/test_verdict_query.py \
  tests/runtime/test_locking.py tests/runtime/test_status.py \
  tests/importers/test_verl_candidates.py \
  tests/feature_stories/test_fs11_curated_seed.py
```

Results: `58 passed`, then `44 passed`.

The parent candidate changed only `.gitmodules` and the Hub gitlink.  The
protected-worktree helper reproduced the frozen baseline exactly:

```text
protected_entries=61 protected_files=1436
sha256=7f356bc33a378284825a77a0fa4d03e7ebaedcc675d0f2d2976c768d68161b39
```

The reviewer-session Feishu authorization check reported `identity=user`,
`verified=true`, and default identity `user`.  Before any Docx operation, the
reviewer loaded the installed version-matched lark-doc fetch, Markdown, style,
update, and update-workflow references through `lark-cli skills read`.

The runtime-private binding store was present with required mode `0700` for its
directory and `0600` for its file.  Exact object, URL, revision, and editor
identifiers were kept only in a newly created reviewer scratch directory with
the same restricted permissions and are intentionally omitted here.

The permitted live command was executed once against alias `FS03-DOC` using
the runtime-private binding and live adapter.  It returned the stable error:

```text
E_ADAPTER_STALE_REVISION
```

The Hub implementation performs this check before its writer invocation.  The
redacted command output contained no URL or object-token-shaped value; its
stderr was empty.  Immediately after failure, the runtime Hub worktree was
clean and the protected-worktree helper again reproduced the exact digest above.
Therefore the attempted operation made **zero Feishu writes**.  A second
publish was intentionally not run: it would exceed the one successful
synthetic publish/read-back story while its CAS precondition was invalid.

## Blocking in-scope defect

- **F-FA03-01 — IN_SCOPE_DEFECT.** The D-07 disposable binding is stale before
  its first authorized publication.  The implementation correctly blocks the
  write with `E_ADAPTER_STALE_REVISION`, but the acceptance fixture is not
  executable as bound.  Repair or re-establish the runtime-private binding by
  an authorized implementer path, then obtain a fresh independent final-review
  live sequence.  This reviewer did not change the binding.

## Deferred suggestions

None.

## Contract contradictions

None.  The stale binding is an in-scope acceptance-precondition defect, not a
reason to amend the frozen Plan.

## Single most likely weakness in this review

The review intentionally did not exercise the successful user-edit/pull half of
the live Feature Story because the initial publication CAS precondition failed.
The deterministic suite and negative canaries remain strong evidence for the
implementation, but they cannot substitute for AC-12's expressly required
disposable live cycle.
