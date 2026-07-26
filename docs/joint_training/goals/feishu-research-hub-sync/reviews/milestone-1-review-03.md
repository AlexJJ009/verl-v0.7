# Milestone 1 Independent Review 03

## Review identity

- Reviewer: fresh independent reviewer, `gpt-5.6-terra` / medium; requested
  project reviewer models were unavailable, so this is the documented fallback.
- Review type: full-lane Milestone 1 re-review round 03.
- Base commit: `10e2746b21d687405a3620341b754dcd96517fe3`.
- Candidate repository:
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub`.
- Candidate commit: `cf9e69f2c87566b3b49ac8b8fbb5bf4c6ac2d2ee`.
- Frozen Plan: version 1, SHA-256
  `5ffdc332b89f8de424bbff66a6bca9ffa396e8adc3506a7adf25bc9eb011789a`.
- Scope: AC-02, AC-05, AC-06, AC-07, and AC-08 in Milestone 1
  fixture/harness scope. This review made no implementation, ledger, Feishu,
  GitHub, parent-repository, or other shared-state mutation. This report is the
  only write.

## Overall verdict: PASS

Round 03 closes F-M1-R05 and F-M1-R06 with executable evidence. In-root move is
now distinct from rename, persists the new logical `remote_path`, and performs
no fake writer call. The tracked canary matrix names exactly all 30 root-gate
detector codes and independently runs every one green before red. Prior AC-05,
AC-07, and AC-08 evidence remains green. No new blocking finding surfaced in
this round.

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-02 | PASS | Full suite and root gate passed. The matrix and an independent source extraction both contained exactly 30 detector codes with empty set differences. All 30 parametrized canaries first asserted a green control and then observed their named red code. Deliberately omitting one matrix item made the reconciliation test fail. |
| AC-05 | PASS | Focused FS05 test independently recreated concurrent local/remote changes, retained local bytes in a reachable conflict ref, installed remote content, marked conflict, and blocked adapter construction for publish. |
| AC-06 | PASS | Rename and in-root move are separate cases. The move changes `remote_path` from `batch-a/fs02-doc` to `batch-b/fs02-doc`, returns `in_root_move`, persists the new path, and logs only the fake `inventory` read. Detach and tombstone cases retain their prior passing zero-writer evidence. |
| AC-07 | PASS | Captured formatting-only and semantic Docx fixtures still pass. `hubctl diff` produces structured B/L/R hashes, change flags, and a readable table/Mermaid/asset diff; malformed and unsupported payload tests remain present and green. |
| AC-08 | PASS | Writer-gate tests still prove malformed content blocks before adapter construction. A reviewer-owned real push to a local bare remote passed while green and was rejected after a malformed-manifest commit. The hook and writer gate therefore both have observed green-before-red behaviour. |

## Required commands and evidence

Executed from `/data-1/code/verl` unless the command supplies another root:

```text
REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m pytest -q tests
............................................................             [100%]
60 passed in 0.61s
exit=0

REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m hubctl check --root .
PASS
exit=0

git -C /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub status --short
<empty output; exit=0>

goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 1; plan_status=READY;
latest review=milestone-1-review-02 FAIL; pending_user_decisions=[]; exit=0
```

Focused reviewer command:

```text
REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m pytest -q -vv tests/feature_stories/test_fs05_concurrent_remote_wins.py tests/feature_stories/test_fs06_remote_structure_changes.py tests/diff/test_cli_diff.py tests/gate/test_writer_gate.py tests/gate/test_detector_canary_matrix.py
collected 40 items
40 passed in 0.50s
exit=0
```

Independent matrix reconciliation and deliberate-red evidence:

```text
source detector count=30
matrix detector count=30
source-minus-matrix=<empty>
matrix-minus-source=<empty>

# Rerun the matrix reconciliation after removing one item in memory:
AssertionError: assert set(MATRIX) == implemented
exit=1
```

Hook evidence used a fresh local clone and local bare remote only:

```text
valid git push -> hook printed PASS; exit=0
malformed-manifest git push -> hook printed E_SCHEMA_REQUIRED; push exit=1
```

The candidate `origin` remains a local bare repository under the authorized
scratch root. The only real-looking email match is the deliberately invalid
`person@example.com` negative test; fixture identities remain on
`example.invalid`, production identity remains unresolved, and shared writes
remain disabled. No real credential, external service call, or shared write was
found or performed.

## Prior finding disposition

- **F-M1-R05: closed.** `RemoteDocument.remote_path`, the initial entry path,
  persisted inventory reconciliation, and a dedicated move fixture now make an
  in-root move observable independently of rename. The reviewer ran that case.
- **F-M1-R06: closed.** The tracked 30-code matrix is mechanically reconciled
  with the implemented root-gate codes, and each mutation executes a green
  control followed by the named red result. The reviewer also proved the
  reconciliation check can fail.
- **F-M1-R01 through F-M1-R04: remain closed.** Their FS05, structured diff,
  writer boundary, hook, detach, and tombstone tests remain collected and
  passing.

## Test-strength audit

Relative to `10e2746b`, no test was deleted. The FS06 test was strengthened by
splitting rename and in-root move into separate cases; the candidate adds the
matrix fixture and 31 executable matrix tests. `git diff --check` passed. All
60 tests were collected without skips, and the 40 focused tests passed
independently. No loosened or trivialized assertion was found.

## Blocking in-scope defects

None.

## Deferred suggestions

None.

## Contract contradictions

None.

## Single most likely weakness in this review

The detector inventory is source-reconciled by literal error-code strings, so a
future detector whose code is dynamically constructed would need the matrix
reconciler to evolve. At this reviewed commit every root-gate detector uses a
literal code and the independently extracted sets match exactly, so this does
not weaken the present PASS.
