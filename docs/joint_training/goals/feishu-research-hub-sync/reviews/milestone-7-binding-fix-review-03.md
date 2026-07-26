# Milestone 7 Binding Fix Review 03

## Review identity

- Reviewer: `m7_binding_review02`, the same independent reviewer that reported
  the sole residual defect in review 02.
- Model routing: the project-requested Haiku model was unavailable in the
  spawn interface. The actual fallback was the inherited GPT-5 Codex reviewer
  model; the runtime did not expose a more specific backend identifier.
- Review type: focused Milestone 7 re-review of `F-M7-R01` after the independent
  convergence review.
- Frozen Plan: version `2`, SHA-256
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Base Hub commit: `82cabe1c361ba7141762c50b5d4da6a4bd0175f7`.
- Candidate Hub commit:
  `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`.
- Applicable criteria: `AC-03`, `AC-04`, `AC-09`, `AC-10`, and `AC-12`.
- No implementation, ledger/acceptance edit, Git push, Feishu call, parent
  gitlink change, or PM2 change was performed.

## Overall verdict

**PASS.**

Candidate `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89` closes the sole
residual `F-M7-R01` defect. `reconcile_live` now preflights the complete
runtime-private binding authority and its tracked entry references before it
constructs an adapter or fetches a remote object.

The reviewer-owned mixed valid-plus-invalid `sync_once` canary now reports the
stable terminal `E_BINDING_STATE` with zero adapter constructions, zero
fetches, zero Git commits, zero pushes, unchanged cursor bytes, and the same
status error. The complete single-invalid matrix also remains fail closed. The
valid private-binding no-op, direct publish/pull privacy, attribution trailers,
local-CI controls, launcher behavior, root gate, and full suite remain green.

## Per-AC verdicts

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-03 | PASS | Direct publication/privacy tests pass. Live-mode publish resolves only the private binding, keeps the object token outside tracked output, preserves revision checking, and remains idempotent. |
| AC-04 | PASS | Remote pull/attribution tests pass. Human `Author`, service `Committer`, and structured `Feishu-Revision` / `Feishu-Editor-Ids` trailers remain intact. Concrete revision/editor values are redacted from this report. |
| AC-09 | PASS | Pristine-clone local-CI and verdict-query tests still cover honest `PASS`, `RED`, and `ERROR`; this focused candidate does not modify or weaken the CI implementation/tests. |
| AC-10 | PASS | Full binding preflight precedes adapter construction. The mixed canary proves zero adapter/fetch/commit/push, unchanged cursor, terminal `ERROR`, and matching status. The full negative matrix, launcher nonzero behavior, and positive no-op control pass. |
| AC-12 | PASS | For the scope of this focused deterministic review, the committed candidate is safe to advance to the separately authorized disposable live acceptance story. No live Feishu operation was used as substitute evidence here. |

## Commands and evidence

Required reviewer-owned verification:

```text
cd /data-1/code/verl/research/feishu-research-hub
.venv/bin/python -m pytest -q \
  tests/runtime/test_sync_once.py \
  tests/runtime/test_launchers.py \
  tests/runtime/test_status.py \
  tests/gate/test_live_binding_privacy.py \
  tests/feature_stories/test_fs03_local_publish.py \
  tests/feature_stories/test_fs04_remote_edit_pull.py \
  tests/git/test_attribution.py
42 passed in 2.24s

.venv/bin/python -m pytest -q
144 passed in 15.26s

.venv/bin/python -m hubctl check --root .
PASS

git diff --check \
  82cabe1c361ba7141762c50b5d4da6a4bd0175f7 \
  7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89
exit 0

goal-plan-runtime validate-runtime \
  /data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 7;
plan_status=READY; pending_user_decisions=[]
```

The reviewer independently reran the full negative binding matrix using
temporary synthetic state and a forbidden adapter. No Feishu call or real
identifier was used:

| Case | Stable outcome | Adapter constructions |
| --- | --- | ---: |
| Missing store | `E_ADAPTER_BINDING` | 0 |
| Empty `entries` | `E_ADAPTER_BINDING` | 0 |
| Wrong schema version | `E_BINDING_STATE` | 0 |
| Binding item is not a mapping | `E_BINDING_STATE` | 0 |
| Object token missing | `E_BINDING_STATE` | 0 |
| Object token empty | `E_BINDING_STATE` | 0 |
| Fixture-shaped live token | `E_BINDING_STATE` | 0 |
| Unknown-only entry binding | `E_BINDING_STATE` | 0 |

The independent mixed valid-plus-invalid probe exercised `sync_once` rather
than only calling the helper:

```text
error_code=E_BINDING_STATE
adapter_constructed=0
adapter_fetches=0
git_commits=0
git_pushes=0
cursor_unchanged=true
terminal_outcome=ERROR
status_error=E_BINDING_STATE
```

The tracked launcher test confirms `scripts/sync_once.sh` exits nonzero for a
live binding error and does not create/advance the cursor. The existing
`sync_once` matrix confirms terminal append-only ERROR recording and matching
`hubctl status` output.

Focused positive/privacy/attribution/local-CI controls:

```text
.venv/bin/python -m pytest -q \
  tests/runtime/test_sync_once.py::test_valid_private_binding_keeps_live_noop_successful \
  tests/gate/test_live_binding_privacy.py::test_live_publish_and_pull_never_track_or_print_object_token \
  tests/feature_stories/test_fs04_remote_edit_pull.py::test_remote_edit_creates_attributed_commit_push_and_noop_rerun \
  tests/local_ci/test_runner_e2e.py \
  tests/local_ci/test_verdict_query.py
12 passed in 1.15s
```

Diff audit:

```text
M src/hubctl/runtime.py                 +5/-0
M tests/runtime/test_sync_once.py      +31/-0
```

No test or source file was deleted. The candidate adds the preflight list and a
mixed-store canary; it does not skip, loosen, or trivialize any existing test.

## Blocking in-scope defects

None. The focused `F-M7-R01` residual is resolved by reviewer-owned evidence.

## Deferred suggestions

None.

## Contract contradictions

None.

## Single most likely weakness in this review

This review intentionally did not perform the real disposable Feishu edit/pull
cycle. Its PASS covers the focused deterministic binding correction and permits
advancement to that separately authorized final live acceptance; it is not a
substitute for the reviewer-owned D-07 live Feature Story.
