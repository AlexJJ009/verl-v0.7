# Milestone 1 Focused Rereview Report

- Review identity: same independent milestone reviewer, requested profile `GPT-5.5 medium`
- Review type: `milestone-rereview`
- Scope: `F-M1-01` only
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Reviewed Plan version: `2`
- Base commit: `14b6f8081e32440a11653c7815798757cb02dfe6`
- Candidate commit: `d2952e7488fd4a79529e29b932638d7e92f46ee1`
- Overall verdict: `PASS`

## Summary

`F-M1-01` is closed. The candidate adds a control-aware adapter wrapper around active item execution. During child polling, it reads operator controls, consumes a valid `stop_now`, terminates the active child, returns a non-zero child exit, prevents the next item from starting, and records the control event. The original failure mode no longer reproduces.

## Per-AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-05 | `PASS` | `stop_now` during active execution now routes to a batch stop/shared failure and does not launch the next item. |
| AC-06 | `PASS` | Active polling observes `stop_now`, records `control_consumed`, terminates the child, and prevents fall-forward or retry. |

## Commands And Evidence

### Required Verification

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py; echo EXIT:$?
```

Result:

```text
..........................                                               [100%]
26 passed in 0.44s
EXIT:0
```

### Runtime And Diff Checks

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
git diff --check 14b6f8081e32440a11653c7815798757cb02dfe6 d2952e7488fd4a79529e29b932638d7e92f46ee1; echo EXIT:$?
```

Result:

```text
PASS
EXIT:0
...
"F-M1-01": {
  "classification": "IN_SCOPE",
  "review_fix_rounds": 1,
  "status": "OPEN"
}
...
EXIT:0
EXIT:0
```

The ledger still shows `F-M1-01` open because I did not modify `findings.jsonl`; this report is the reviewer evidence for closure.

### Focused F-M1-01 Probe

Probe: reproduce the original failure by issuing a valid `stop_now` after `item_started` and during active polling, using two manifest items to verify the batch cannot fall forward.

Observed result:

```json
{
  "state_status": "shared_failure",
  "started": [["fixture", "one"]],
  "terminated": ["child-1", "child-1"],
  "events": [
    "...\"event\":\"item_started\"...",
    "...\"event\":\"control_consumed\"...",
    "...\"status\":\"failed\"...",
    "...\"event\":\"batch_shared_failure\"..."
  ]
}
```

Key evidence:

- Only the first item started; the second item did not start.
- `control_consumed` was recorded at batch revision 1.
- The active child was terminated.
- The batch ended as `shared_failure`, not `completed` or `completed_with_failures`.
- The item attempt remained `1`; no retry/resume path was used.

### Implementation Evidence

- `/data-1/code/verl/scripts/experiment_execution_core.py:643` adds `_ControlledAdapter`.
- `/data-1/code/verl/scripts/experiment_execution_core.py:653` reads controls during child polling.
- `/data-1/code/verl/scripts/experiment_execution_core.py:655` checks `stop_requested` during active polling.
- `/data-1/code/verl/scripts/experiment_execution_core.py:656` terminates the active child.
- `/data-1/code/verl/scripts/experiment_execution_core.py:658` returns `143`, forcing the atomic state to terminal failure.
- `/data-1/code/verl/scripts/experiment_execution_core.py:712` routes batch item execution through `_ControlledAdapter`.
- `/data-1/code/verl/tests/experiment_workflow/test_experiment_batch_core.py:224` adds a regression for `stop_now` during active item execution.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

- `DEFERRED_SUGGESTION`: The focused probe observes duplicate `terminate` calls for the same child: once from `_ControlledAdapter.poll()` and once from `ExecutionCore` cleanup after return code `143`. This is not a blocker because termination is idempotent in the adapter contract and resources are released, but a future cleanup could avoid duplicate termination evidence.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

The rereview remains CPU-only and uses a fake adapter probe plus the focused unit suite. That is appropriate for this milestone, but a later milestone should still verify the same `stop_now` behavior with `SubprocessAdapter` if subprocess-level cleanup semantics become acceptance-critical.
