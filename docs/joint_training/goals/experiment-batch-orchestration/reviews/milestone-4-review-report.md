# Milestone 4 Independent Adversarial Review Report

## Review Identity

- Reviewer: independent GPT-5.5 medium adversarial milestone reviewer
- Review type: Milestone 4 review
- Goal: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: 2
- Base commit: `0a1d1b0696872ec88caaaa3ef7b3008380c88d0b`
- Candidate commit reviewed: `ea0a7d12be47518253788c0f5258ddd4de68d6e4`
- Recipe gitlink reviewed: `d717672fb671edb86e504ba11e15b742686d7ef8`
- Applicable ACs: AC-02, AC-03, AC-04, AC-05, AC-06, AC-08, AC-09, AC-10
- Additional exact command run: AC-07, because the prompt required every exact AC-02 through AC-10 verification command.

## Overall Verdict

**NOT_READY**

All exact Plan AC-02 through AC-10 commands passed, including protected asset comparison. However, an adversarial mutation probe falsified AC-05's event-corruption shared-stop requirement: the batch core ignores malformed atomic `run_id` events in the shared `events.jsonl` ledger, then starts and completes the next item instead of entering `shared_failure`. This violates the frozen Plan statement that malformed state/event evidence is a shared failure and the AC-05 expectation that shared invariant failures stop the batch without launching another child.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-02 | PASS | Exact `batch-validate` command returned `ok: true` for `experiment_batch_v1.json`. |
| AC-03 | PASS | Exact `test_experiment_batch_routing.py -k success` passed. |
| AC-04 | PASS | Exact `test_experiment_batch_routing.py -k local_failure` passed. |
| AC-05 | FAIL | Exact `-k shared_failure` passed, but reviewer mutation probe showed malformed atomic event schema is ignored and a child starts; see `F-M4-01`. |
| AC-06 | PASS | Exact `test_experiment_batch_control.py` passed; stale control probe stayed paused without starting a child. |
| AC-07 | PASS | Exact monitor command passed; included because the prompt required AC-02 through AC-10 commands. |
| AC-08 | PASS | Exact Stage123 core/admission command passed; recipe gitlink matches reviewed `d717672...`. |
| AC-09 | PASS | Exact policy/failure-classifier command passed; no additional no-agent/no-retry defect found in this review. |
| AC-10 | PASS | Exact PM2/policy plus protected asset compare command passed; protected compare returned `ok: true`. |

## Commands And Evidence

### Candidate and Runtime State

```bash
git rev-parse HEAD
git cat-file -t ea0a7d12be47518253788c0f5258ddd4de68d6e4
git ls-tree ea0a7d12be47518253788c0f5258ddd4de68d6e4 recipe
git -C recipe rev-parse HEAD
```

Result:

```text
ea0a7d12be47518253788c0f5258ddd4de68d6e4
commit
160000 commit d717672fb671edb86e504ba11e15b742686d7ef8 recipe
d717672fb671edb86e504ba11e15b742686d7ef8
```

```bash
goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
```

Result: `validate-plan` returned `PASS`; `validate-runtime` reported Plan status `READY`, current milestone `Milestone 4`, and prior findings closed.

### Candidate Diff Scope

```bash
git diff --stat 0a1d1b0696872ec88caaaa3ef7b3008380c88d0b ea0a7d12be47518253788c0f5258ddd4de68d6e4
git diff --name-status 0a1d1b0696872ec88caaaa3ef7b3008380c88d0b ea0a7d12be47518253788c0f5258ddd4de68d6e4
```

Result summary:

```text
7 files changed, 252 insertions(+), 8 deletions(-)
M docs/joint_training/goals/experiment-batch-orchestration/runtime.jsonl
M scripts/experiment_execution_core.py
M tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json
M tests/experiment_workflow/fixtures/experiment_batch_v1.json
A tests/experiment_workflow/test_experiment_batch_control.py
A tests/experiment_workflow/test_experiment_batch_policy.py
A tests/experiment_workflow/test_experiment_batch_routing.py
```

### Exact AC Commands

AC-02:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "8aafc2060c7ce00e5088691b1ca1dd2d7c4cf3005b5c8894a4da989f2dfff799", "items": ["stage123-primary"], "ok": true}
```

AC-03:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k success
```

Result: `1 passed, 3 deselected in 0.03s`.

AC-04:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k local_failure
```

Result: `1 passed, 3 deselected in 0.03s`.

AC-05:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k shared_failure
```

Result: `2 passed, 2 deselected in 0.04s`.

AC-06:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py
```

Result: `2 passed in 0.04s`.

AC-07:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_experiment_batch_monitor.py
```

Result: `9 passed in 16.05s`.

AC-08:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py
```

Result: `11 passed in 0.24s`.

AC-09:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_failure_classifier.py
```

Result: `7 passed in 0.04s`.

AC-10:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_pm2_ci_keepalive.py tests/experiment_workflow/test_experiment_batch_policy.py && \
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Result:

```text
10 passed in 0.74s
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

Patch hygiene:

```bash
git diff --check 0a1d1b0696872ec88caaaa3ef7b3008380c88d0b ea0a7d12be47518253788c0f5258ddd4de68d6e4
```

Result: no output, exit code 0.

### Reviewer-Owned Mutation Probes

Scratch-only probe file: `/data-1/tmp/m4_adversarial_probe.py`.

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python /data-1/tmp/m4_adversarial_probe.py
```

Result:

```json
{"detail": {"failure": null, "started": [["fixture", "one", "1"]]}, "probe": "malformed_atomic_event_status", "status": "completed"}
{"detail": {"failure": {"code": "event_corruption", "context": {}, "message": "event ledger batch binding mismatch at line 1"}, "started": []}, "probe": "batch_event_wrong_manifest_hash", "status": "shared_failure"}
{"detail": {"rejection": {"code": "control_rejected", "message": "stale batch revision"}, "started": []}, "probe": "stale_continue_control", "status": "paused_after_current"}
```

Interpretation:

- `batch_event_wrong_manifest_hash` behaved correctly: shared failure, no child start.
- `stale_continue_control` behaved correctly: stayed paused, no child start.
- `malformed_atomic_event_status` failed the contract: an existing `events.jsonl` line with `schema_version=1`, `run_id=run-one`, and invalid `status=impossible_status` did not stop the batch. The core launched the child and completed the batch.

Relevant code evidence:

- `scripts/experiment_execution_core.py:692` defines `_validate_event_ledger`.
- `scripts/experiment_execution_core.py:702` validates only `schema_version` for all events.
- `scripts/experiment_execution_core.py:704` through `scripts/experiment_execution_core.py:708` validates binding only when `batch_id` exists.
- There is no validation of atomic `run_id` event status or run binding before `run()` proceeds at `scripts/experiment_execution_core.py:770` through `scripts/experiment_execution_core.py:775`.
- The existing corruption test at `tests/experiment_workflow/test_experiment_batch_routing.py:77` through `tests/experiment_workflow/test_experiment_batch_routing.py:82` covers JSON syntax corruption, but not malformed atomic event schema.

## Blocking In-Scope Defects

### F-M4-01 — Shared event ledger accepts malformed atomic events and continues execution

- Classification: `IN_SCOPE_DEFECT`
- Affected ACs: AC-05, with supporting relevance to AC-03/AC-04 evidence integrity.
- Evidence: reviewer-owned mutation probe pre-populated `events.jsonl` with `{"schema_version": 1, "run_id": "run-one", "status": "impossible_status"}`. `BatchExecutor.run()` returned `completed` and started `[["fixture", "one", "1"]]` instead of entering `shared_failure`.
- Contract impact: Plan v2 states that malformed state/event evidence is a shared failure and AC-05 requires shared invariant failures to stop the batch without starting another item. The shared ledger contains both atomic and batch events, so ignoring malformed atomic events lets corrupted evidence coexist with further execution.
- Why in-scope: the fix is within existing event-ledger validation and test coverage; it does not require Plan changes, GPU/training, or external services.

## Deferred Suggestions

- The newly added `test_batch_manifest_fixture_is_validated_by_core_command` only checks fixture existence and schema version, not the CLI behavior named in the test. This is not a separate blocker because AC-02's exact CLI command passed independently, but the test name overstates its coverage.

## Contract Contradictions

None found. The blocker is an implementation/test-coverage defect against the existing Plan v2 contract, not a contradiction in the contract.

## Single Most Likely Weakness In This Review

The adversarial probes were CPU-only and scratch-based, so they exercised the core state machine directly rather than a long-running real Stage123 batch. This matches the no-GPU/no-training review constraint, but a real shell-launched batch could expose additional integration defects outside this focused Milestone 4 surface.
