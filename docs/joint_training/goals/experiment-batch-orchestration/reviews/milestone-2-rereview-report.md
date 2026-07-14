# Milestone 2 Focused Rereview Report

- Review identity: same independent milestone reviewer, requested profile `GPT-5.5 medium`
- Review type: `milestone-rereview`
- Scope: `F-M2-01` only
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Reviewed Plan version: `2`
- Base commit: `10504ea105f4804f89356125546ccd33e58f2348`
- Candidate commit: `0e471b9efeef0e49e72c6d347696e6ae814f4d06`
- Overall verdict: `PASS`

## Summary

`F-M2-01` is closed. The candidate fixes the control-poll/child-success race by persisting the active item's terminal record before applying a newly consumed `stop_now`. In the reproduced race, the control is accepted at the running revision, the active item terminal evidence is preserved, no next item starts, and the final batch state is `stopped`.

## Per-AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-03 | `PASS` | The stopped race no longer drops the active terminal item evidence; ordered routing still prevents starting the next item after `stop_now`. |
| AC-05 | `PASS` | The batch stops before the next item and preserves evidence for the item that reached terminal status. |
| AC-06 | `PASS` | Cleanup-boundary `stop_now` is consumed against the running revision and applied only after the active terminal item record is persisted. |

## Commands And Evidence

### Required Verification

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json && REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py; echo EXIT:$?
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "aadef425f84e2c652a27f6397b56a7c559ec9375d65c4d026da4ca15d2a714f9", "items": ["stage123-primary"], "ok": true}
.................................                                        [100%]
33 passed in 15.63s
EXIT:0
```

### Runtime And Diff Checks

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
git diff --check 10504ea105f4804f89356125546ccd33e58f2348 0e471b9efeef0e49e72c6d347696e6ae814f4d06; echo EXIT:$?
```

Result:

```text
PASS
EXIT:0
...
"F-M2-01": {
  "classification": "IN_SCOPE",
  "review_fix_rounds": 1,
  "status": "OPEN"
}
...
EXIT:0
EXIT:0
```

The ledger still shows `F-M2-01` open because I did not modify `findings.jsonl`; this report is reviewer evidence for closure.

### Focused F-M2-01 Probe

Probe: two-item manifest. The fake adapter writes a valid `stop_now` during the first child's `poll()` call after the controlled adapter's pre-poll control read, then returns success for the child.

Observed result:

```json
{
  "status": "stopped",
  "started": [["fixture", "one"]],
  "terminated": [],
  "items": [
    {
      "item_id": "one",
      "status": "succeeded",
      "run_id": "run-one",
      "cleanup": {
        "resources_released": true,
        "term_sent": false,
        "kill_sent": false
      }
    }
  ],
  "events": [
    "...\"event\":\"item_started\"...",
    "...\"status\":\"succeeded\"...",
    "...\"event\":\"control_consumed\"...\"action\":\"stop_now\"...",
    "...\"event\":\"item_succeeded\"...",
    "...\"event\":\"batch_stopped\"..."
  ]
}
```

Key checks:

- `stop_now` was accepted at the running revision.
- The active item terminal record was persisted before `batch_stopped`.
- The second item did not start.
- The final state is `stopped`.
- No retry/fall-forward behavior occurred.
- The active child was already terminal success in this race, so no termination was expected or possible; the preserved cleanup record is the correct terminal evidence.

### Implementation Evidence

- `/data-1/code/verl/scripts/experiment_execution_core.py:751` handles atomic child success.
- `/data-1/code/verl/scripts/experiment_execution_core.py:752` reads cleanup-boundary controls.
- `/data-1/code/verl/scripts/experiment_execution_core.py:753` snapshots `stop_after_terminal_record`.
- `/data-1/code/verl/scripts/experiment_execution_core.py:754` appends the active item terminal record.
- `/data-1/code/verl/scripts/experiment_execution_core.py:756` persists `item_succeeded`.
- `/data-1/code/verl/scripts/experiment_execution_core.py:757` applies the stop after terminal evidence exists.
- `/data-1/code/verl/tests/experiment_workflow/test_experiment_batch_core.py:252` adds the race regression test.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

The focused probe is CPU-only and uses a fake adapter, matching the milestone's test style. It verifies the exact race from `F-M2-01`, but later acceptance should still cover subprocess-backed cleanup if production `SubprocessAdapter` behavior becomes part of final evidence.
