# Milestone 2 Independent Review Report

- Review identity: independent milestone reviewer, requested profile `GPT-5.5 medium`
- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Reviewed Plan version: `2`
- Reviewed Plan SHA256: `15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb`
- Current milestone: `Milestone 2`
- Base commit: `a3812ecfc8b17f54a24a7e6eb95d535c9adc3616`
- Candidate commit: `10504ea105f4804f89356125546ccd33e58f2348`
- Overall verdict: `NOT_READY`

## Summary

The required verification command passes, and the candidate substantially improves production batch validation/routing/control: the committed fixture validates, protected-asset mutations are rejected, local fallback uses one attempt and can finish `completed_with_failures`, repeated failures stop before a third item, cleanup failure stops the batch, and pause/continue survives executor restart.

However, an adversarial `stop_now` timing still violates AC-06. If `stop_now` is written after the controlled adapter has read controls for a poll cycle but before the wrapped child poll returns success, the batch consumes the control and stops before launching the next item, but it does not terminate the active child and records no item terminal evidence. The prompt specifically required verifying that `stop_now` issued after `item_started` and during active polling terminates the active child. That remains incomplete for this race.

## Per-AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | `PASS` | Production batch validation/routing/control remains in `scripts/experiment_execution_core.py`; no second batch state authority was added in the reviewed diff. |
| AC-02 | `PASS` | `batch-validate` succeeds for the committed fixture, and protected-asset/admission mutation probes fail closed. |
| AC-03 | `PASS` | Ordered success routing is still covered by the focused suite and the core starts items in manifest order. |
| AC-04 | `PASS` | Local failure records `inconclusive_operational_failure`, skipped phases, one attempt, and starts the next item without retry/resume. |
| AC-05 | `PASS` | Repeated equal failures and cleanup failure stop the batch and prevent later item launch in reviewer probes. |
| AC-06 | `FAIL` | `stop_now` can be consumed after a child succeeds without terminating the active child or recording item terminal evidence for the active item. |
| AC-09 | `PASS` | Batch mode rejects `--resume`/recovery-policy and uses `RunSpec(max_attempts=1, resumable_failure_codes=())`; no AI/tuning path was found. |

## Commands And Evidence

### Required Verification

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py; echo EXIT:$?
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "ef1f672dececf5171ce7d1c67e32fd653835fe88b4afc03a48540f79473b342d", "items": ["stage123-primary"], "ok": true}
................................                                         [100%]
32 passed in 15.93s
EXIT:0
```

### Runtime And Diff Checks

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md; echo EXIT:$?
git diff --check a3812ecfc8b17f54a24a7e6eb95d535c9adc3616 10504ea105f4804f89356125546ccd33e58f2348; echo EXIT:$?
```

Result:

```text
PASS
EXIT:0
...
"current_milestone": "Milestone 2",
"plan_status": "READY",
"plan_version": 2
...
EXIT:0
15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb  docs/joint_training/goals/experiment-batch-orchestration/plan.md
EXIT:0
EXIT:0
```

### Candidate Diff Scope

Command:

```bash
git diff --name-status a3812ecfc8b17f54a24a7e6eb95d535c9adc3616 10504ea105f4804f89356125546ccd33e58f2348
```

Result:

```text
M	scripts/experiment_execution_core.py
A	tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json
A	tests/experiment_workflow/fixtures/experiment_batch_v1.json
M	tests/experiment_workflow/test_experiment_batch_core.py
```

### Adversarial Probe: Local Fallback

Reviewer-owned CPU probe:

```json
{
  "status": "completed_with_failures",
  "started": [["fixture", "one"], ["fixture", "two"]],
  "attempt": 1,
  "skipped": ["run-one-remaining"]
}
```

This supports AC-04: the first item fails locally, no retry occurs, skipped phases are recorded, and the second pre-authorized item starts.

### Adversarial Probe: Repeated Failure Stop

Reviewer-owned CPU probe:

```json
{
  "status": "shared_failure",
  "started": [["fixture", "a"], ["fixture", "b"]],
  "items": ["a", "b"]
}
```

This supports AC-05: two equal normalized `child_exit` failures stop the batch before item `c`.

### Adversarial Probe: Cleanup Failure Stop

Reviewer-owned CPU probe:

```json
{
  "status": "shared_failure",
  "started": [["fixture", "one"]],
  "terminated": ["1"],
  "items": [
    {
      "item_id": "one",
      "status": "inconclusive_operational_failure",
      "cleanup": {"resources_released": false, "term_sent": true, "kill_sent": false}
    }
  ]
}
```

This supports AC-05 cleanup-stop behavior: no second item starts after cleanup failure.

### Adversarial Probe: Restart-Safe Pause/Continue Cursor

Reviewer-owned CPU probe:

```json
{
  "first": "paused_after_current",
  "second": "completed",
  "started": [["fixture", "p"]],
  "events": [
    "...\"event\":\"control_consumed\"...\"action\":\"pause_after_current\"...",
    "...\"event\":\"batch_paused\"...",
    "...\"event\":\"control_consumed\"...\"action\":\"continue_remaining\"...",
    "...\"event\":\"item_started\"...",
    "...\"event\":\"batch_completed\"..."
  ]
}
```

This supports AC-06 for pause/continue across executor restart: `control_offset` and `control_seq` are persisted and the old pause is not replayed.

### Adversarial Probe: Replay/Stale CAS

Reviewer-owned CPU probe after replaying `control_seq=2`:

```json
{
  "rejection": {
    "code": "control_rejected",
    "message": "stale batch revision"
  }
}
```

This supports the stale/replay rejection path. The exact rejection reason is stale revision because the replayed control also carried an obsolete expected revision.

### Adversarial Probe: Protected Asset Mutation

Reviewer-owned mutation of `protected_asset_hashes.test_data` with recomputed bundle and manifest hashes:

```json
{
  "returncode": 2,
  "stdout": "{\"failure\": {\"code\": \"invalid_batch_request\", \"context\": {}, \"message\": \"protected asset hash mismatch: test_data\"}, \"ok\": false}"
}
```

This supports AC-02 protected binding enforcement.

### Blocking Probe: `stop_now` Race During Active Polling

Probe setup: two-item manifest. The fake adapter writes a valid `stop_now` control during the first child's `poll()` call, after `_ControlledAdapter.poll()` has already called `_read_controls()` for that cycle, and then returns success for the child.

Observed result:

```json
{
  "status": "stopped",
  "started": [["fixture", "one"]],
  "terminated": [],
  "items": [],
  "events": [
    "...\"event\":\"item_started\"...",
    "...\"status\":\"succeeded\"...",
    "...\"event\":\"control_consumed\"...\"action\":\"stop_now\"...",
    "...\"event\":\"batch_stopped\"..."
  ]
}
```

Relevant implementation evidence:

- `/data-1/code/verl/scripts/experiment_execution_core.py:685` reads controls before delegating to the inner adapter poll.
- `/data-1/code/verl/scripts/experiment_execution_core.py:687` terminates only if `stop_requested` was already observed before the inner poll.
- `/data-1/code/verl/scripts/experiment_execution_core.py:751` handles a successful atomic child.
- `/data-1/code/verl/scripts/experiment_execution_core.py:752` reads controls after success.
- `/data-1/code/verl/scripts/experiment_execution_core.py:753` stops the batch if `stop_requested`, before appending an item terminal record.

The batch does not fall forward to the next item, which is good, but it also does not terminate the active child and does not preserve an item terminal record in `state["items"]`.

## Blocking In-Scope Defects

### F-M2-01 - IN_SCOPE_DEFECT - `stop_now` race can stop the batch without terminating or recording the active item

AC-06 requires deterministic operator stop behavior: `stop_now` during execution or cleanup cleans the active item and prevents the next launch. The Milestone 2 prompt specifically asks the reviewer to verify that `stop_now` issued after `item_started` and during active polling terminates the active child, prevents the next item, records the control, and cannot fall forward or retry.

The candidate handles `stop_now` when it is visible before the controlled adapter calls the inner poll. It does not handle the race where the control is written after that read but before the child reports success in the same poll cycle. In that race:

- the child is not terminated;
- the batch stops and records `control_consumed`;
- no next item starts;
- no retry occurs;
- but `state["items"]` has no terminal record for the active item even though the atomic event ledger recorded child success.

Required fix direction: keep this in the unified core and close the post-poll race. After a successful child return, if a newly consumed `stop_now` applies, the core should preserve active-item terminal evidence and define/execute the documented cleanup boundary. If the child has already exited, termination may be impossible, but the implementation must not silently drop the active item evidence; the stopped event should bind the active item outcome/cleanup state.

## Deferred Suggestions

- `DEFERRED_SUGGESTION`: The committed fixture's protected assets are local protected workspace paths, not all present in the candidate commit tree. This is acceptable if those are intentionally environment-bound protected assets, but final acceptance should state that the fixture is not fully reproducible from a bare commit without the protected workspace assets.
- `DEFERRED_SUGGESTION`: Add a subprocess-level stop test later. The current CPU fake-adapter probes are appropriate for Milestone 2, but production cleanup semantics ultimately depend on `SubprocessAdapter`.

## Contract Contradictions

None. The issue is an implementation defect against AC-06, not a Plan contradiction.

## Single Most Likely Weakness In This Review

The blocking probe uses a narrow timing race in a fake adapter: the child reports success in the same poll call that creates the `stop_now` control. A production process may have different timing, but the frozen contract is deterministic and the core currently exposes the same read-before-poll race. The defect is therefore valid for CPU-only milestone review.
