# Milestone 4 Focused Rereview Report

## Review Identity

- Reviewer: independent GPT-5.5 medium same-reviewer focused rereview
- Review type: Milestone 4 rereview, focused on `F-M4-01` only
- Goal: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: 2
- Base commit: `ea0a7d12be47518253788c0f5258ddd4de68d6e4`
- Candidate commit reviewed: `b9e4660e653f0951c5408d5467127be41c3ba196`
- Recipe gitlink reviewed: `d717672fb671edb86e504ba11e15b742686d7ef8`
- Applicable ACs: AC-05, AC-07

## Overall Verdict

**PASS**

`F-M4-01` is closed. The candidate validates batch events before atomic events, rejects malformed atomic `run_id` events, preserves the original corrupt event in `events.jsonl`, records a `batch_shared_failure` event, and does not start a child. The full required CPU-only verification matrix also passed.

## Finding Closure Status

| Finding | Status | Evidence |
| --- | --- | --- |
| `F-M4-01` | CLOSED | Malformed atomic event now returns `shared_failure` with `event_corruption`, starts no child, and appends shared-failure evidence while preserving the original corrupt event. |

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-05 | PASS | Malformed atomic event and invalid atomic attempt both fail closed; mixed batch events are validated through the batch path. |
| AC-07 | PASS | Monitor discriminator order was updated batch-first; required monitor-related tests pass in the full matrix. |

## Commands And Evidence

### Candidate State

```bash
git rev-parse HEAD
git cat-file -t b9e4660e653f0951c5408d5467127be41c3ba196
git ls-tree b9e4660e653f0951c5408d5467127be41c3ba196 recipe
git -C recipe rev-parse HEAD
```

Result:

```text
b9e4660e653f0951c5408d5467127be41c3ba196
commit
160000 commit d717672fb671edb86e504ba11e15b742686d7ef8 recipe
d717672fb671edb86e504ba11e15b742686d7ef8
```

### Goal Validation

```bash
goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
```

Result: `validate-plan` returned `PASS`; `validate-runtime` reported Plan status `READY`, current milestone `Milestone 4`, and `F-M4-01` open pending this rereview.

### Focused Diff

```bash
git diff --name-status ea0a7d12be47518253788c0f5258ddd4de68d6e4 b9e4660e653f0951c5408d5467127be41c3ba196
```

Result:

```text
M docs/joint_training/goals/experiment-batch-orchestration/findings.jsonl
M scripts/experiment_execution_core.py
M scripts/stage123_manifest_monitor.py
M tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json
M tests/experiment_workflow/fixtures/experiment_batch_v1.json
M tests/experiment_workflow/test_experiment_batch_routing.py
```

Key implementation evidence:

- `scripts/experiment_execution_core.py:704` treats any event with `batch_id` as a batch event first.
- `scripts/experiment_execution_core.py:712` validates atomic `run_id` events only after no `batch_id` is present.
- `scripts/experiment_execution_core.py:713` rejects unknown atomic statuses.
- `scripts/experiment_execution_core.py:715` rejects non-integer atomic attempts.
- `scripts/stage123_manifest_monitor.py:58` also uses batch-first event discrimination.
- `tests/experiment_workflow/test_experiment_batch_routing.py:85` adds malformed atomic event regression coverage.

### Required Verification

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_experiment_batch_routing.py \
  tests/experiment_workflow/test_experiment_batch_core.py \
  tests/experiment_workflow/test_experiment_batch_monitor.py \
  tests/experiment_workflow/test_experiment_batch_control.py \
  tests/experiment_workflow/test_experiment_batch_policy.py \
  tests/experiment_workflow/test_stage123_core_migration.py \
  tests/experiment_workflow/test_manifest_queue_monitor_contract.py \
  tests/experiment_workflow/test_stage123_admission_bundle.py \
  tests/experiment_workflow/test_notification_policy.py \
  tests/experiment_workflow/test_operational_calibration_runner.py \
  tests/experiment_workflow/test_failure_classifier.py \
  tests/experiment_workflow/test_pm2_ci_keepalive.py
```

Result:

```text
..............................................................           [100%]
62 passed in 16.99s
```

### Reviewer-Owned Probes

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python /data-1/tmp/m4_rereview_probe.py
```

Result:

```json
{"failure": {"code": "event_corruption", "context": {}, "message": "atomic event schema mismatch at line 1"}, "probe": "malformed_atomic_run_event", "started": [], "status": "shared_failure"}
{"failure": {"code": "event_corruption", "context": {}, "message": "atomic event attempt mismatch at line 1"}, "probe": "malformed_atomic_attempt", "started": [], "status": "shared_failure"}
{"failure": null, "probe": "mixed_batch_event_valid", "started": [["fixture", "one", "1"]], "status": "completed"}
{"failure": {"code": "event_corruption", "context": {}, "message": "batch event schema mismatch at line 1"}, "probe": "mixed_batch_event_bad_state", "started": [], "status": "shared_failure"}
```

Interpretation: the original F-M4-01 malformed atomic event shape now fails closed with no child start; mixed batch events carrying `batch_id` plus `run_id` are interpreted as batch events; invalid mixed batch state fails closed.

Evidence preservation probe:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python /data-1/tmp/m4_rereview_evidence_probe.py
```

Result:

```json
{"event_count": 2, "failure": {"code": "event_corruption", "context": {}, "message": "atomic event schema mismatch at line 1"}, "first_event_preserved": true, "last_event": {"batch_id": "batch-fixture", "batch_manifest_sha256": "6666666666666666666666666666666666666666666666666666666666666666", "batch_revision": 1, "event": "batch_shared_failure", "failure": {"code": "event_corruption", "context": {}, "message": "atomic event schema mismatch at line 1"}, "schema_version": 1, "state": "shared_failure"}, "started": [], "status": "shared_failure"}
```

Additional validation:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "7d03b45a276e4acb81d9546a28eaefdeafd0ee7f9fa0d2a7e1a88d595c5ac0c7", "items": ["stage123-primary"], "ok": true}
```

Fixture hash recomputation result:

```text
impl_tree_match=True
bundle_hash_match=True
admission_file_hash_match=True
manifest_hash_match=True
recipe_gitlink=d717672fb671edb86e504ba11e15b742686d7ef8
```

Patch hygiene:

```bash
git diff --check ea0a7d12be47518253788c0f5258ddd4de68d6e4 b9e4660e653f0951c5408d5467127be41c3ba196
```

Result: no output, exit code 0.

## Blocking In-Scope Defects

None. `F-M4-01` is closed.

## Deferred Suggestions

None for this focused rereview.

## Contract Contradictions

None found.

## Single Most Likely Weakness In This Review

This rereview was intentionally scoped to `F-M4-01`; I did not re-open broader Milestone 4 design questions beyond the required full matrix and focused event-ledger probes.
