# Milestone 7 Binding Fix Review 02

## Review identity

- Reviewer: `m7_binding_review02`, a fresh independent Codex reviewer.
- Model routing: the project-requested Haiku model was unavailable in the
  spawn interface. The actual fallback was the inherited GPT-5 Codex reviewer
  model; the runtime did not expose a more specific backend model identifier.
- Review type: full independent implementation review after the required
  convergence review.
- Frozen Plan: version `2`, SHA-256
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`.
- Base Hub commit: `66d6f5aaf804d17c0f37f312f28277dea88a2c98`.
- Candidate Hub commit:
  `82cabe1c361ba7141762c50b5d4da6a4bd0175f7`.
- Applicable criteria: `AC-03`, `AC-04`, `AC-09`, `AC-10`, and `AC-12`.
- No implementation, runtime/findings/acceptance edit, Git push, parent-gitlink
  mutation, PM2 mutation, or Feishu call was performed.

## Overall verdict

**FAIL.**

The candidate closes the single-invalid-entry cases from the convergence
review: a missing/empty store, malformed store/schema/item, missing/empty
token, fixture-shaped live token, and unknown-only entry all produce stable
binding errors before adapter construction. `sync_once` records the terminal
error, leaves the cursor unchanged, and does not push. The launcher propagates
the failure, and the valid private-binding no-op control remains green.

One blocking fail-closed defect remains. `reconcile_live` validates and acts on
each binding in the same loop. It does not validate the entire private binding
authority before constructing an adapter. A reviewer-owned mixed-store canary
with a valid binding sorted before an unknown binding constructed the adapter
and fetched the first object before reporting `E_BINDING_STATE` for the later
invalid binding. This violates the convergence requirement that any invalid
private authority fail before adapter use. It also permits a remote fetch, and
potentially an attributable local commit for a changed first object, before the
cycle discovers that the binding store is invalid.

The cycle still recorded `ERROR`, performed no Git push, preserved the cursor,
and exposed the same error through status. Those later safeguards do not make
the earlier remote access fail closed.

## Per-AC verdicts

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-03 | PASS | Focused publication/privacy tests pass. Live-mode direct publish resolves only the runtime-private binding, does not persist or print the object token, remains revision checked, and keeps the idempotent no-op behavior. |
| AC-04 | PASS | Focused pull/attribution tests pass. The human `Author`, service `Committer`, `Feishu-Revision`, and `Feishu-Editor-Ids` trailers remain present. Direct live pull keeps the object token out of tracked content, audit output, commits, and the bare remote. Concrete revision/editor values are intentionally redacted from this report. |
| AC-09 | PASS | The pristine-clone local-CI runner and verdict-query tests pass for `PASS`, `RED`, and `ERROR`. The candidate does not weaken or delete those tests, and the binding failure is exercised only after a matching CI `PASS`. |
| AC-10 | FAIL | The basic negative matrix, terminal ERROR, no-push, cursor-preservation, launcher, status, and positive no-op checks pass. However, a mixed store can use the adapter for an earlier valid binding before a later invalid binding is detected. The runtime-private authority therefore is not validated atomically before remote access. |
| AC-12 | FAIL | The exact candidate is not safe for the periodic-job-shaped live acceptance cycle because an invalid binding store can cause a real remote fetch before the stable terminal binding ERROR. The reviewer did not perform the prohibited live Feishu story. |

## Commands and evidence

Required verification:

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
41 passed in 2.11s

.venv/bin/python -m pytest -q
143 passed in 14.97s

.venv/bin/python -m hubctl check --root .
PASS

git diff --check \
  66d6f5aaf804d17c0f37f312f28277dea88a2c98 \
  82cabe1c361ba7141762c50b5d4da6a4bd0175f7
exit 0

goal-plan-runtime validate-runtime \
  /data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 7;
plan_status=READY; pending_user_decisions=[]
```

Additional focused controls:

```text
.venv/bin/python -m pytest -q \
  tests/local_ci/test_runner_e2e.py \
  tests/local_ci/test_verdict_query.py
9 passed in 0.91s

.venv/bin/python -m pytest -q \
  tests/runtime/test_sync_once.py::test_valid_private_binding_keeps_live_noop_successful \
  tests/gate/test_live_binding_privacy.py::test_live_publish_and_pull_never_track_or_print_object_token \
  tests/feature_stories/test_fs04_remote_edit_pull.py::test_remote_edit_creates_attributed_commit_push_and_noop_rerun
3 passed in 0.68s
```

Reviewer-owned direct negative matrix used a temporary synthetic Hub and a
forbidden-adapter constructor. It made no Feishu call and used no real
identifier:

| Case | Outcome | Adapter constructions |
| --- | --- | ---: |
| Missing store | `E_ADAPTER_BINDING` | 0 |
| Empty `entries` | `E_ADAPTER_BINDING` | 0 |
| Missing `entries` field | `E_BINDING_STATE` | 0 |
| Wrong schema version | `E_BINDING_STATE` | 0 |
| Binding item is not a mapping | `E_BINDING_STATE` | 0 |
| Object token missing | `E_BINDING_STATE` | 0 |
| Object token empty | `E_BINDING_STATE` | 0 |
| Fixture-shaped live token | `E_BINDING_STATE` | 0 |
| Unknown-only entry binding | `E_BINDING_STATE` | 0 |

The tracked `sync_once` and launcher tests additionally prove that representative
binding failures occur before Git push, leave an existing cursor unchanged or
create no cursor, append a terminal `SYNC_ATTEMPT` with `outcome=ERROR`, expose
the same binding code through `hubctl status`, and exit nonzero through
`scripts/sync_once.sh`.

The blocking mixed-store probe used two synthetic bindings: the first pointed
to a tracked entry and the second to an unknown entry. All concrete synthetic
values were discarded and are omitted here. The result was:

```text
error_code=E_BINDING_STATE
adapter_constructed=1
adapter_fetches=1
git_pushes=0
cursor_unchanged=true
terminal_outcome=ERROR
status_error=E_BINDING_STATE
```

Source inspection explains the result. In
`src/hubctl/runtime.py:116-156`, `reconcile_live` loads the top-level document,
then validates one binding and immediately constructs/fetches its adapter before
validating the next binding. The full-store preflight required by the
convergence disposition is absent.

The candidate diff changes only two source files and adds/extends runtime tests.
It deletes no tests and contains no skipped, loosened, or trivialized gate.
Candidate commit attribution is `Codex Agent` with the approved
`Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>` trailer.

## Blocking in-scope defects

- **F-M7-R01 remains open — `IN_SCOPE_DEFECT`.** Validate the complete
  `.hub/live-bindings.json` authority, including every binding item and tracked
  entry reference, before constructing any adapter, fetching any remote object,
  mutating tracked content, or creating a Git commit. Add a mixed valid-plus-
  invalid canary so a later malformed/unknown binding proves zero adapter calls,
  zero commits/pushes, and unchanged cursor with the stable terminal binding
  error.

## Deferred suggestions

None.

## Contract contradictions

None. The required full-store preflight implements the existing convergence
disposition and frozen AC-10/AC-12 behavior; it does not require a Plan change.

## Single most likely weakness in this review

The reviewer did not run a real Feishu operation, as this review lane forbids
it. The blocking result is nevertheless a direct behavioral canary: the
synthetic adapter was constructed and fetched before the later invalid binding
was detected. A future PASS review must rerun this mixed-store canary and then
the separately authorized disposable live story.
