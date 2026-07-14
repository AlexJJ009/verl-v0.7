# Milestone 1 Independent Review Report

- Review identity: independent milestone reviewer, requested profile `GPT-5.5 medium`
- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Reviewed Plan version: `2`
- Reviewed Plan SHA256: `15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb`
- Current milestone: `Milestone 1`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `14b6f8081e32440a11653c7815798757cb02dfe6`
- Overall verdict: `NOT_READY`

## Summary

The required focused test suite passes, and the candidate preserves legacy atomic recovery while adding batch validation, no-retry batch execution, immutable admission binding, ordered fallback, and control CAS/replay checks. However, the implementation does not satisfy the frozen `stop_now` execution-boundary contract in AC-06. A `stop_now` control issued while an item is actively running is not read until after `ExecutionCore.run()` returns; for a single successful item, the batch completes rather than cleaning/stopping the active item.

This is an in-scope Milestone 1 defect, not a Plan contradiction.

## Per-AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | `PASS` | Batch state/events and routing live in `scripts/experiment_execution_core.py`; focused tests and `deletion-budget.json` establish the intended authority boundary for Milestone 1. |
| AC-02 | `PASS` | Admission validation binds bundle hash, command hash, implementation tree hash, evidence commit ancestry, recipe gitlink, tracked implementation paths, and input hashes; adversarial untracked implementation-path mutation is rejected. |
| AC-04 | `PASS` | Batch execution constructs `RunSpec(max_attempts=1, resumable_failure_codes=())`; local failure becomes `inconclusive_operational_failure`, records skipped phases, and starts the next pre-authorized item without retry. |
| AC-06 | `FAIL` | `stop_now` during active execution is not observed or applied at the required boundary; the active item is not terminated/cleaned as a stopped item and the batch can complete. |
| AC-09 | `PASS` | Batch CLI rejects `--resume` and recovery-policy overrides, batch execution has no retry/resume path, and legacy atomic recovery remains separate and still works. |

## Commands And Evidence

### Required Verification

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py; echo EXIT:$?
```

Result:

```text
.........................                                                [100%]
25 passed in 0.45s
EXIT:0
```

### Goal Runtime Validation

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md; echo EXIT:$?
```

Result:

```text
PASS
EXIT:0
```

Relevant runtime result:

```json
{
  "current_milestone": "Milestone 1",
  "goal_status": "ACTIVE",
  "pending_user_decisions": [],
  "plan_status": "READY",
  "plan_version": 2
}
```

Plan hash:

```text
15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb  docs/joint_training/goals/experiment-batch-orchestration/plan.md
EXIT:0
```

### Diff Hygiene

Command:

```bash
git diff --check 2020531b470ec932d7b00afd13080e1318fc8429 14b6f8081e32440a11653c7815798757cb02dfe6; echo EXIT:$?
```

Result:

```text
EXIT:0
```

### Candidate Diff Scope

Command:

```bash
git diff --name-status 2020531b470ec932d7b00afd13080e1318fc8429 14b6f8081e32440a11653c7815798757cb02dfe6
```

Relevant result:

```text
A	docs/joint_training/goals/experiment-batch-orchestration/deletion-budget.json
M	scripts/experiment_execution_core.py
A	tests/experiment_workflow/test_experiment_batch_core.py
```

The candidate also adds the Goal plan, ledgers, and review artifacts under the Goal directory.

### Adversarial Probe: `stop_now` During Active Execution

Probe summary: a fake adapter writes a valid `stop_now` control after `item_started` and while `ExecutionCore.run()` is polling the active child. The active child then succeeds. Expected contract behavior from Plan v2 AC-06 and State/Event contract: `stop_now` during execution cleans the active item and prevents completion. Observed behavior: batch status becomes `completed`, no termination is sent, and no control event is consumed.

Relevant output:

```json
{
  "state_status": "completed",
  "started": [["fixture", "one"]],
  "terminated": [],
  "items": [
    {
      "item_id": "one",
      "status": "succeeded",
      "run_id": "run-one"
    }
  ]
}
```

Relevant implementation evidence:

- `/data-1/code/verl/scripts/experiment_execution_core.py:650` enters `BatchExecutor.run()`.
- `/data-1/code/verl/scripts/experiment_execution_core.py:663` reads controls only at item-loop boundaries.
- `/data-1/code/verl/scripts/experiment_execution_core.py:691` calls blocking `ExecutionCore(...).run(spec)`.
- `/data-1/code/verl/scripts/experiment_execution_core.py:694` records success immediately when the child succeeds; controls written during active execution are not read before completion.

### Adversarial Probe: Immutable Implementation Binding

Probe summary: mutate a valid admission bundle to include an untracked implementation path and recompute manifest/bundle hashes. Expected behavior: validation rejects. Observed behavior: rejected with `invalid_batch_request`.

Relevant output:

```json
{
  "returncode": 2,
  "stdout": "{\"failure\": {\"code\": \"invalid_batch_request\", \"context\": {}, \"message\": \"implementation path is not a regular file: untracked_batch_probe.py\"}, \"ok\": false}"
}
```

### Adversarial Probe: Legacy Atomic Recovery Preserved

Probe summary: run the legacy atomic `ExecutionCore` with `max_attempts=2` and `resumable_failure_codes=('child_exit',)`. The first child fails, explicit `resume()` starts a second child and succeeds.

Relevant output:

```json
{
  "attempt": 2,
  "first": "failed",
  "second": "succeeded",
  "started": [["cmd"], ["cmd"]]
}
```

This supports the Plan v2 boundary: legacy atomic recovery remains available outside batch semantics.

## Blocking In-Scope Defects

### F-M1-01 - IN_SCOPE_DEFECT - `stop_now` is not enforced during active item execution

AC-06 requires that when an operator issues `stop_now` for a running or boundary-state batch, the core applies the command at the documented boundary, records operator/control evidence, cleans owned resources, and never edits the immutable batch. Plan v2 also states that `stop_now` during execution or cleanup cleans the active item and prevents the next launch.

The candidate only reads controls before item launch, after local failure fallback, and at item-loop boundaries. It delegates the active child to blocking `ExecutionCore.run()` and does not poll the operator-control file during active execution or cleanup. A valid `stop_now` written while the child is running is ignored until after success, and a single-item batch completes.

Impact:

- Active execution cannot be stopped deterministically by `stop_now`.
- The control event is not recorded before completion.
- Owned resources are not cleaned as a stopped item; the child can finish normally.
- A single-item batch can return `completed`, contradicting AC-06.

Required fix direction: keep implementation inside the unified core, but make active batch execution observe `stop_now` during child polling/cleanup or otherwise route child lifecycle through a control-aware loop that can terminate/clean the active child and persist the stopped event.

## Deferred Suggestions

- `DEFERRED_SUGGESTION`: Add a focused regression test where `pause_after_current` arrives during active execution and confirms it pauses after the current item but before the next item. The current blocker is `stop_now`, not pause semantics.
- `DEFERRED_SUGGESTION`: Add a malformed-control JSONL regression to ensure rejection events are idempotent across process restarts; this is useful but not required to close F-M1-01.

## Contract Contradictions

None. The failure is an implementation defect against AC-06, not a Plan contradiction.

## Single Most Likely Weakness In This Review

The probe uses a fake adapter rather than a real subprocess. That is intentional for CPU-only review and matches the existing test style, but a later fix should include a focused subprocess-level regression if practical, because the production `SubprocessAdapter` termination path is where resource cleanup matters most.
