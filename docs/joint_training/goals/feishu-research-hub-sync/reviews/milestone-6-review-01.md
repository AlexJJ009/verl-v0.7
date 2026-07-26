# Milestone 6 Independent Review 01

## Review identity

- Review type: fresh independent Milestone Review
- Reviewer: fresh Codex subagent, independent of the implementer
- Requested reviewer model: GPT-5.5, medium reasoning
- Actual reviewer model: GPT-5.5 was unavailable as a selectable model; used the available inherited Codex model as a disclosed fallback, with reasoning setting not exposed to the reviewer
- Frozen Plan version: 2
- Applicable ACs: AC-09, AC-10
- Base commit: `6fca611d12caaec97c5efd4b72f950c6eec1a977`
- Candidate commit: `82ddd18c2dd514c64b4dd35a14d63438c33bd777`
- Reviewed repository: `/data-1/code/verl/research/feishu-research-hub`
- Deployment inspected read-only: `/data-1/feishu-research-hub-runtime`

## Overall verdict

**FAIL**

The candidate has substantial working behavior: the focused 25-test suite, full
125-test suite, and `hubctl check` pass; deployed PM2 contains exactly the two
approved one-shot processes with tracked script paths and explicit fixture mode;
the autonomous pristine-clone CI verdict and fixture sync cursor are bound to
`82ddd18`; state/application logs are mode `0600`; and no systemd, external cron,
container, listener, or additional Hub process was found.

Three reviewer-owned findings block this review:

1. An AC-09 clone-infrastructure failure is recorded as `ERROR` but falsely says
   `pristine_clone=true`, even though the clone command exited 72 and no clone
   existed. The same record has `error_code=null`. This is not a truthful
   deployed-shape verdict record.
2. After a successful AC-10 sync, `hubctl status` reports `last_attempt.outcome`
   as `RUNNING` forever. The runtime is stopped and `last_success` is PASS, but
   the operator-facing last-attempt field never settles, so it does not honestly
   expose whether the most recent attempt finished.
3. Candidate `82ddd18` was Agent-authored but records the human user as both
   Author and Committer and has no `Co-authored-by` trailer. This violates the
   Goal Reviewer Contract's attribution/provenance requirement and makes the
   exact revision reported by AC-10 misattributed. No history rewrite was
   performed during this review.

## Per-AC verdict

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-09 | **FAIL** | PASS/RED/ERROR behavior, cursor canaries, query behavior, deployed autonomous PASS, and zero-real-Feishu fixture mode are otherwise present; however, reviewer-owned clone-failure evidence records `pristine_clone=true` and `error_code=null` after clone exit 72. The verdict record is therefore not fully truthful. |
| AC-10 | **FAIL** | Locking, bounded retry/timeout, CI gating, failed-push retry, cursor non-advance/advance, PM2 provenance, protected state, and deployed fixture sync all work; however, a completed successful cycle leaves `status.last_attempt.outcome=RUNNING`, and the reviewed current revision `82ddd18` is misattributed to the human user rather than the Agent. |

## Commands and evidence

### Candidate identity, diff, and test-strength audit

```bash
cd /data-1/code/verl/research/feishu-research-hub
git rev-parse HEAD
git diff --check 6fca611d12caaec97c5efd4b72f950c6eec1a977..82ddd18c2dd514c64b4dd35a14d63438c33bd777
git diff --stat 6fca611d12caaec97c5efd4b72f950c6eec1a977..82ddd18c2dd514c64b4dd35a14d63438c33bd777
git diff --numstat 6fca611d12caaec97c5efd4b72f950c6eec1a977..82ddd18c2dd514c64b4dd35a14d63438c33bd777 -- tests
rg -n 'skip|xfail|pytestmark' tests
```

Relevant output:

```text
82ddd18c2dd514c64b4dd35a14d63438c33bd777
21 files changed, 1464 insertions(+), 17 deletions(-)
```

`git diff --check` exited 0. The candidate adds 500 lines across the seven
focused test files and does not delete or loosen prior tests. No `skip`,
`xfail`, or file-level `pytestmark` was found in the added AC-09/AC-10 tests.

### Frozen verification command

```bash
cd /data-1/code/verl/research/feishu-research-hub
.venv/bin/python -m pytest -q \
  tests/local_ci/test_runner_e2e.py \
  tests/local_ci/test_verdict_query.py \
  tests/runtime/test_sync_once.py \
  tests/runtime/test_locking.py \
  tests/runtime/test_status.py \
  tests/runtime/test_launchers.py
.venv/bin/python -m pytest -q
.venv/bin/python -m hubctl check --root .
```

Output:

```text
.........................                                                [100%]
25 passed in 1.34s
........................................................................ [ 57%]
.....................................................                    [100%]
125 passed in 13.62s
PASS
```

The shell emitted unrelated read-only sandbox profile warnings about fnm and a
missing `/root/.local/bin/env`; all three requested commands still exited 0.

### AC-09 reviewer-owned clone failure canary

The reviewer created a temporary local bare Git origin under `/tmp`, resolved a
real commit SHA, and invoked `run_local_ci_once` with a clone command that exits
72. The state directory was also under `/tmp`.

```text
AC09_CLONE_FAILURE {"error_code": null, "exit_code": 72,
 "pristine_clone": true,
 "sha": "bd3c7a8878fd946fd3c885c5eae0109cb20d9d32",
 "step": "clone", "verdict": "ERROR"}
AC09_CURSOR_EXISTS False
```

The non-advance behavior is correct. The `pristine_clone=true` field is false:
the clone step failed and the workspace was removed/nonexistent. `error_code`
is also null for this infrastructure failure. This failure mode is present in
the committed implementation: `pristine_clone` is set unconditionally in the
post-resolve record, while a nonzero clone exit does not assign an error code.

### AC-10 reviewer-owned completed-sync status canary

The reviewer created a temporary local bare origin and worker under `/tmp`,
recorded a matching PASS CI verdict, ran one fixture reconciliation and push,
then called `runtime_status` without touching the deployed runtime.

```text
SYNC_RESULT {"changed": false, "ci_verdict": "PASS",
 "event": "SYNC_SUCCESS", "outcome": "PASS",
 "remote_revision": "fixture",
 "sha": "852478177b11108c816f99aca4677864d95f4595", ...}
STATUS_LAST_ATTEMPT {"error_code": null, "event": "SYNC_ATTEMPT",
 "outcome": "RUNNING", ...}
STATUS_LAST_SUCCESS {"changed": false, "ci_verdict": "PASS",
 "event": "SYNC_SUCCESS", "outcome": "PASS", ...}
STATUS_CURRENT_REVISION 852478177b11108c816f99aca4677864d95f4595
STATUS_ERROR_CODE None
```

`runtime_status` selects only `SYNC_ATTEMPT` rows for `last_attempt`. A
successful cycle appends `SYNC_SUCCESS` but never settles or correlates the
attempt row, so the operator-facing last-attempt state permanently says
`RUNNING` after the one-shot process exits successfully.

### Read-only D-06 deployed-state inspection

```bash
PM2_HOME=/data-1/feishu-research-hub-runtime/pm2-home pm2 jlist
find /data-1/feishu-research-hub-runtime -maxdepth 3 -type f -printf '%m %p\n'
sed -n '1,240p' /data-1/feishu-research-hub-runtime/state/ci-verdicts.jsonl
sed -n '1,240p' /data-1/feishu-research-hub-runtime/state/sync-events.jsonl
sed -n '1,120p' /data-1/feishu-research-hub-runtime/state/sync-cursor.json
/data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl status \
  --root /data-1/feishu-research-hub-runtime/repo \
  --state-dir /data-1/feishu-research-hub-runtime/state --json
git -C /data-1/feishu-research-hub-runtime/repo rev-parse HEAD
git -C /data-1/feishu-research-hub-runtime/repo status --short
```

PM2 has exactly two entries:

```text
feishu-hub-local-ci
  status=stopped exit_code=0 autorestart=false cron_restart=*/5 * * * *
  pm_exec_path=/data-1/feishu-research-hub-runtime/repo/scripts/ci_once.sh
  HUB_SYNC_MODE=fixture
feishu-hub-sync
  status=stopped exit_code=0 autorestart=false cron_restart=*/10 * * * *
  pm_exec_path=/data-1/feishu-research-hub-runtime/repo/scripts/sync_once.sh
  HUB_SYNC_MODE=fixture
```

The runtime clone is clean at `82ddd18`. Its autonomous local-CI ledger records
a pristine-clone PASS for `82ddd18`, with frozen `uv sync` and full pytest gate
command identities and exit 0. The sync ledger first records an expected
`E_SYNC_CI_MISSING` while CI was pending, then records fixture-only PASS for
`82ddd18`; a later scheduled no-change cycle also passes. The cursor is:

```json
{"advanced_at":"2026-07-24T15:50:09.463865Z",
 "remote_revision":"fixture-no-feishu",
 "sha":"82ddd18c2dd514c64b4dd35a14d63438c33bd777"}
```

Deployed `hubctl status --json` returns current revision `82ddd18`, CI verdict
PASS, no conflicts, and no error. It also demonstrates the same status defect:
`last_attempt.outcome` is `RUNNING` although `last_success` is PASS and the PM2
one-shot exited 0.

All files under `state/logs`, `state/pm2-logs`, the CI/sync ledgers, cursor and
lock are mode `0600`; the PM2 application logs are inside the mode-`0700` state
directory. Legacy PM2 logs are also mode `0600`.

The deployed manifest and environment use `HUB_SYNC_MODE=fixture`; the launcher
routes that mode to `hubctl sync-fixture`, whose injected `reconcile` and
`auth_check` avoid `LarkCliAdapter` entirely. The local-CI subprocess sets
`HUB_CI_NO_FEISHU=1`, and the live adapter rejects the real `lark-cli` binary in
that environment. This supports zero Feishu calls for the deployed evidence.

Read-only system inventories returned no matching systemd unit/timer, user
crontab entry, container, TCP listener, or extra Hub process. The only persistent
match was the approved PM2 daemon at
`/data-1/feishu-research-hub-runtime/pm2-home`.

### Candidate Git attribution/provenance

```bash
git show -s --format=fuller 82ddd18c2dd514c64b4dd35a14d63438c33bd777
git show -s --format='%B' 82ddd18c2dd514c64b4dd35a14d63438c33bd777
```

Output:

```text
Author:     gongxunli <lgxma01@buaa.edu.cn>
Commit:     gongxunli <lgxma01@buaa.edu.cn>

fix: protect PM2 runtime logs
```

There is no Agent Author/Committer identity and no `Co-authored-by` trailer.
Earlier commits in the reviewed range use `Codex Agent
<codex-agent@example.invalid>`, confirming that `82ddd18` is the attribution
outlier. The review is bound to the actual existing commit; the reviewer did
not amend or force-push history.

### Goal validators

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync
goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md
```

`validate-plan` passed. `validate-runtime` passed and reported Plan v2,
Milestone 6, Goal `ACTIVE`, Plan `READY`. At review time it also reported open
finding `F-M6-10` and pending decision `D-M6-ATTR-01`; therefore this report
does not claim Milestone 6 completion. The Plan hash is:

```text
ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba
```

## Blocking in-scope defects

1. **AC-09 false clone-error metadata.** When branch resolution succeeds but
   the clone command exits nonzero, record `pristine_clone=false` and a stable
   clone infrastructure error code (or otherwise remove the false claim), while
   preserving ERROR, command/exit/log identity and cursor non-advance. Add a
   canary that asserts the corrected fields.
2. **AC-10 last-attempt never settles.** Record or derive a completed outcome
   for every attempt so a successful one-shot does not remain `RUNNING` in
   `hubctl status`. Add success and failure canaries that correlate the latest
   attempt with its terminal event while preserving append-only history.
3. **AC-10 candidate attribution/provenance.** The Agent-authored candidate must
   use the named Agent identity and standard human collaboration trailer under
   the frozen Git attribution contract. Any amend/force-with-lease is a shared
   history mutation and remains outside this review; it requires the recorded
   user decision and a fresh review bound to the resulting commit.

These are behavioral/provenance defects, not mechanical report-format issues.

## Deferred suggestions

None.

## Contract contradictions

None.

## Single most likely weakness in this review

The deployment inspection is a read-only point-in-time snapshot. I did not wait
through an additional future PM2 cron boundary or perform a live Feishu call;
the latter is intentionally outside D-06 and cannot substitute for the fixture
failure coverage required by AC-09/AC-10.
