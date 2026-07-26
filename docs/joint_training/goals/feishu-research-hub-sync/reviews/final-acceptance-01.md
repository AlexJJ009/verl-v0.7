# Final Acceptance Review 01

- Reviewer: `final_acceptance01` fresh independent reviewer
- Verdict: `REJECTED`
- Reviewed Plan: version `2`, SHA-256
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`
- Hub candidate: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`
- Parent candidate: `8a589d0f85f8d93fdc487da0ee003881c17e1ea3`
- Scope: final independent acceptance under approved decision `D-07` only.

## Reviewer-Owned Evidence

The Goal validators passed before review:

```text
goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync
PASS

goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 7; pending_user_decisions=[]
plan_status=READY; plan_version=2
```

The Plan hash, Hub candidate, private remote tip, parent candidate, and
parent gitlink were independently checked.  They resolved respectively to the
bindings above; the parent gitlink for `research/feishu-research-hub` points
to the reviewed Hub candidate.

The reviewer ran the full deterministic suite from the Hub's existing virtual
environment, avoiding an unavailable external package index:

```text
.venv/bin/python -m pytest -q
126 passed in 14.02s

.venv/bin/python -m hubctl check --root .
PASS

.venv/bin/python -m pytest -q \
  tests/gate/test_detector_canary_matrix.py \
  tests/gate/test_gate_canaries.py \
  tests/hooks/test_pre_push.py \
  tests/gate/test_writer_gate.py
42 passed in 0.95s

.venv/bin/python -m pytest --collect-only -q
126 tests collected
```

The canary subset includes the root-gate detector matrix, which verifies every
detector is green before an independent known-bad mutation and red afterwards.
It also covers the tracked pre-push hook and the writer's independent root
gate.

The reviewer reran the Feishu authentication check with elevated read-only
access because the sandbox hides the live CLI authentication store and PM2
sockets:

```text
LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 \
LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1 \
lark-cli auth status --json --verify
identity=user; verified=true; identities.user.status=ready;
identities.user.tokenStatus=valid
```

No OAuth material, object token, revision, editor ID, or URL is reproduced in
this report.  The existing PM2 process records were read-only checked through
their approved runtime home; exactly `feishu-hub-local-ci` and
`feishu-hub-sync` were present, stopped, and pointed at the tracked launchers.

## Blocking IN_SCOPE_DEFECT

`IN_SCOPE_DEFECT`: the approved disposable Docx binding cannot be used by the
current live path without placing its private remote-object token in a tracked
Git commit.

Reviewer evidence from the candidate source:

1. `src/hubctl/operations.py:publish_entry` reads
   `entry["sync"]["remote_object_token"]` and, after live publish/read-back,
   writes that field back into `entries/<entry_id>/entry.yaml`.
2. `src/hubctl/operations.py:pull_entry` delegates a changed remote edition to
   `commit_remote_entry` when `--commit` is used.
3. `commit_remote_entry` runs `git add -- entries/<entry_id>`, which includes
   that `entry.yaml`, before creating the attributable commit and permitting
   its push.

The final live Feature Story requires a real disposable binding.  The reviewer
verified in an isolated mode-0700 scratch clone that the D-04 protected binding
would therefore flow through the above tracked path.  The reviewer did not run
the publish or pull operation, so no Feishu object, Git branch, parent file,
runtime ledger, or acceptance file changed as part of this rejected live
attempt.

This contradicts the frozen Architecture Contract requiring private Feishu
tokens to remain in mode-0600 runtime state outside Git, and the final-review
contract forbidding exact identifiers in tracked evidence or Git material.
It is an in-scope implementation defect: move the live binding and revision
state to protected runtime storage, ensure generated/committed entry material
contains only a stable alias, and cover the no-token-in-commit invariant with a
failing canary before requesting a fresh review.

## AC Matrix

| Acceptance criterion | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Candidate gitlink, private remote binding, repo-local identity/hook checks completed. |
| AC-02 | PASS | Root gate passed; full detector canary matrix was reviewer-run. |
| AC-03 | PASS | Deterministic Feature Story coverage passed; live repetition was stopped before unsafe binding persistence. |
| AC-04 | PASS | Deterministic remote-pull/attribution coverage passed. |
| AC-05 | PASS | Full suite includes remote-wins conflict preservation coverage. |
| AC-06 | PASS | Full suite includes fail-closed structure/absence coverage. |
| AC-07 | PASS | Full suite includes canonicalization and adapter-contract coverage. |
| AC-08 | PASS | Reviewer-run hook and writer-gate canaries passed. |
| AC-09 | PASS | Candidate/runtime verdict evidence and deterministic suite support the prior independent milestone result. |
| AC-10 | PASS | Read-only PM2 provenance check and deterministic runtime coverage support the prior independent milestone result. |
| AC-11 | PASS | Full suite includes fixed-authority importer coverage. |
| AC-12 | FAIL | The required approved live story cannot safely commit/push without tracking a private disposable binding. |

## Additional Observations

- Deferred suggestions: none.
- Contract contradictions: none.  The defect is directly repairable within the
  existing private-runtime-state contract.
- Most likely review weakness: the real live story was deliberately not run,
  because doing so would itself create the prohibited tracked-token exposure.
  Deterministic tests cannot substitute for AC-12 until the binding storage is
  repaired and independently re-reviewed.
