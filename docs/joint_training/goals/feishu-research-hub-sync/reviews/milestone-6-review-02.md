# Milestone 6 Independent Review 02

## Review identity

- Review type: fresh independent Milestone Review
- Reviewer: fresh Codex subagent, independent of the implementer
- Requested reviewer model: GPT-5.5, medium reasoning
- Actual reviewer model: GPT-5.5 was not exposed as a selectable model for this
  agent; the available inherited Codex model was used as the exact disclosed
  fallback, and the harness did not expose a separately selectable reasoning
  setting
- Frozen Plan version: 2
- Frozen Plan SHA256:
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`
- Applicable ACs: AC-09, AC-10
- Base commit: `82ddd18c2dd514c64b4dd35a14d63438c33bd777`
- Candidate commit: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`
- Reviewed repository: `/data-1/code/verl/research/feishu-research-hub`
- Deployment inspected read-only:
  `/data-1/feishu-research-hub-runtime`

## Overall verdict

**PASS**

The candidate closes all three blocking defects from Milestone 6 review 01.
Reviewer-owned canaries show clone infrastructure failure as `ERROR` with
`E_CI_CLONE`, `pristine_clone=false`, exit 72, and no cursor; a successful sync
now exposes terminal `last_attempt.outcome=PASS`; and the rewritten/follow-up
commits use `Codex Agent <codex-agent@example.invalid>` for both Author and
Committer with `Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>`. The rewritten
`37077df` tree is byte-identical to the old `82ddd18` tree.

The frozen focused suite passes 26 tests, the full suite passes 126 tests, and
`hubctl check` passes. The deployed runtime is clean at `593b4ba`, records an
autonomous pristine-clone CI PASS for that exact SHA, and has a fixture-only
successful sync/cursor/status for the same SHA. PM2 contains exactly the two
approved stopped one-shot applications, both exit 0 and point to tracked
launcher scripts. Fixture mode bypasses the live adapter, and all inspected
state/application logs are protected with mode `0600` inside protected runtime
directories.

## Per-AC verdict

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-09 | **PASS** | Focused/full suites pass; a local bare-origin sequence records exact `PASS`, `RED`, and `ERROR`; query returns the matching verdict for each SHA; the clone failure is `E_CI_CLONE`, exit 72, `pristine_clone=false`, and does not advance the cursor. Deployed pristine-clone CI records PASS for exact candidate `593b4ba` with command/exit/log identity and `network_policy=git-only-no-feishu`. |
| AC-10 | **PASS** | Runtime tests cover locking, bounded retry/timeout, auth failure, failed-push retry, CI gating and cursor safety. Reviewer-owned success canary reports terminal `last_attempt=SYNC_SUCCESS/PASS`, matching `last_success`, revision and null error. Deployed fixture sync/cursor/status bind `593b4ba`; PM2 has exactly the two approved tracked launcher paths and protected state/logs. |

## Commands and evidence

### Candidate, Plan, and runtime governance

```bash
goal-plan-runtime validate-plan \
  docs/joint_training/goals/feishu-research-hub-sync
goal-plan-runtime validate-runtime \
  docs/joint_training/goals/feishu-research-hub-sync
sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md
git -C research/feishu-research-hub rev-parse HEAD
git -C research/feishu-research-hub status --short
git -C research/feishu-research-hub diff --check 82ddd18..593b4ba
git -C research/feishu-research-hub diff --stat 82ddd18..593b4ba
```

Relevant output:

```text
validate-plan: PASS
validate-runtime: Plan v2, Milestone 6, Goal ACTIVE, Plan READY,
                  no pending decisions, all listed findings CLOSED
plan sha256: ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba
Hub HEAD: 593b4bac5ab46bde0b732e2b705c7aa0a79d2846
Hub status: clean
diff --check: exit 0
10 files changed, 44 insertions(+), 11 deletions(-)
```

The parent worktree remains heavily dirty with unrelated protected user work.
This review did not modify or stage those paths.

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
..........................                                               [100%]
26 passed in 1.41s
........................................................................ [ 57%]
......................................................                   [100%]
126 passed in 13.94s
PASS
```

The shell emitted unrelated read-only profile warnings for fnm and a missing
`/root/.local/bin/env`; all requested commands exited 0. The candidate adds
behavioral assertions and no focused test deletion, `skip`, or `xfail` was
found.

### AC-09 reviewer-owned PASS/RED/ERROR and clone-failure canaries

The reviewer created temporary local source/bare-origin/state fixtures under
`/tmp`, ran `run_local_ci_once` for passing, deterministic-red, and failed-clone
commits, and queried each persisted verdict.

```text
AC09_TRIPLE
PASS  step=gate  exit_code=0  error_code=null
      pristine_clone=true
RED   step=gate  exit_code=1  error_code=E_CI_GATE
      pristine_clone=true
ERROR step=clone exit_code=72 error_code=E_CI_CLONE
      pristine_clone=false
AC09_QUERY ["PASS", "RED", "ERROR"]
AC09_ERROR_CURSOR_NONADVANCE True
```

A separate clone-failure-only run also returned:

```text
AC09_CLONE_FAILURE {"error_code":"E_CI_CLONE","exit_code":72,
 "pristine_clone":false,"step":"clone","verdict":"ERROR"}
AC09_CURSOR_EXISTS False
```

This directly reproduces and closes review-01 finding F-M6-R05.

### AC-10 reviewer-owned completed-sync status canary

The reviewer created a local bare Hub origin and worker under `/tmp`, appended
a matching PASS CI verdict, ran fixture reconciliation and push, then called
`runtime_status`.

```text
AC10_SYNC_RESULT {"event":"SYNC_SUCCESS","outcome":"PASS",
 "error_code":null,"remote_revision":"review-fixture",
 "sha":"593b4bac5ab46bde0b732e2b705c7aa0a79d2846"}
AC10_LAST_ATTEMPT {"event":"SYNC_SUCCESS","outcome":"PASS",
 "error_code":null,"sha":"593b4bac5ab46bde0b732e2b705c7aa0a79d2846"}
AC10_LAST_SUCCESS {"event":"SYNC_SUCCESS","outcome":"PASS",
 "error_code":null,"sha":"593b4bac5ab46bde0b732e2b705c7aa0a79d2846"}
AC10_CURRENT_REVISION 593b4bac5ab46bde0b732e2b705c7aa0a79d2846
AC10_ERROR_CODE None
```

This directly reproduces and closes review-01 finding F-M6-R06. Existing
focused tests also cover concurrent lock rejection, fetch timeout/retry, auth
failure and recovery, failed Git push with unchanged cursor followed by a
successful retry, non-PASS CI blocking before reconcile/push, and root-gate
rechecking.

### Attribution and exact history rewrite

```bash
git -C research/feishu-research-hub diff --exit-code \
  82ddd18^{tree} 37077df^{tree}
for c in 37077df b0b6419 593b4ba; do
  git -C research/feishu-research-hub show -s --format=fuller "$c"
  git -C research/feishu-research-hub show -s --format='%B' "$c"
done
git -C research/feishu-research-hub config --local --get user.name
git -C research/feishu-research-hub config --local --get user.email
```

Output:

```text
82ddd18 tree versus 37077df tree: exit 0, no diff
37077df Author/Committer: Codex Agent <codex-agent@example.invalid>
b0b6419 Author/Committer: Codex Agent <codex-agent@example.invalid>
593b4ba Author/Committer: Codex Agent <codex-agent@example.invalid>
each commit trailer: Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>
repo-local user.name: GongxunLi
repo-local user.email: lgxma01@buaa.edu.cn
```

The deployed clone's HEAD has the same Agent identity and collaboration trailer.

### Read-only deployed runtime inspection

```bash
git -C /data-1/feishu-research-hub-runtime/repo rev-parse HEAD
git -C /data-1/feishu-research-hub-runtime/repo status --short
PM2_HOME=/data-1/feishu-research-hub-runtime/pm2-home pm2 jlist
find /data-1/feishu-research-hub-runtime/state \
     /data-1/feishu-research-hub-runtime/pm2-home/logs \
  -maxdepth 3 -type f -printf '%m %p\n'
tail -n 20 /data-1/feishu-research-hub-runtime/state/ci-verdicts.jsonl
tail -n 30 /data-1/feishu-research-hub-runtime/state/sync-events.jsonl
cat /data-1/feishu-research-hub-runtime/state/sync-cursor.json
/data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl status \
  --root /data-1/feishu-research-hub-runtime/repo \
  --state-dir /data-1/feishu-research-hub-runtime/state --json
```

Relevant evidence:

```text
runtime clone HEAD: 593b4bac5ab46bde0b732e2b705c7aa0a79d2846
runtime clone status: clean

PM2 names: feishu-hub-local-ci, feishu-hub-sync (exactly two)
both status=stopped, exit_code=0, autorestart=false
ci script:   .../repo/scripts/ci_once.sh
sync script: .../repo/scripts/sync_once.sh
sync environment: HUB_SYNC_MODE=fixture

latest CI: sha=593b4ba verdict=PASS pristine_clone=true
           clone/install/gate exits=0 with command identities and log path
cursor:    sha=593b4ba remote_revision=fixture-no-feishu
status:    current_revision=593b4ba, ci_verdict=PASS, conflicts=[],
           last_attempt=SYNC_SUCCESS/PASS, last_success=SYNC_SUCCESS/PASS,
           error_code=null
```

The CI ledger includes a prior autonomous RED for intermediate `b0b6419` and a
later autonomous pristine-clone PASS for corrected `593b4ba`, demonstrating
that the runner did not silently accept the failing intermediate revision.

All inspected state ledgers, cursor, lock, CI logs, PM2 application logs and
legacy PM2 logs are mode `0600`. Runtime state, PM2 home and uv-cache roots are
mode `0700`; PM2 application logs are additionally under the mode-`0700`
`state/pm2-logs` directory.

The deployed shape made zero Feishu calls for this evidence: `HUB_SYNC_MODE` is
explicitly `fixture`, `scripts/sync_once.sh` dispatches that mode to
`hubctl sync-fixture`, and that command injects both fixture reconciliation and
a no-op auth check instead of instantiating the live adapter. Local CI exports
`HUB_CI_NO_FEISHU=1`; the live adapter rejects the real `lark-cli` binary in
that environment while captured fake binaries remain testable.

## Blocking in-scope defects

None.

## Deferred suggestions

None.

## Contract contradictions

None.

## Single most likely weakness in this review

The deployed inspection is a read-only point-in-time snapshot. I did not wait
through another cron boundary, mutate PM2, or perform a live Feishu operation.
That limitation does not weaken AC-09/AC-10 here because the exact deployed
candidate already has autonomous CI and fixture-sync evidence, while live
Feishu mutation is separately decision-gated for Milestone 7.
