# Plan v2 Adversarial Rereview Report

- Review identity: independent adversarial Plan Reviewer, requested model profile `GPT-5.5 medium`
- Review type: `plan-rereview`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Reviewed Plan version: `2`
- Reviewed Plan SHA256: `15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `not supplied`
- Overall verdict: `READY`

## Summary

Plan v2 closes the four Plan v1 blocking findings. The amended contract now separates legacy atomic recovery from batch no-retry semantics, defines a measurable authority/deletion budget, freezes operator-control replay and cleanup-race behavior, and binds launch commands to accepted admission bundles plus implementation-tree identity.

The future implementation verification commands mostly do not pass yet because the batch implementation and test files are not present. I do not treat those current failures as Plan-review blockers: the rereview prompt asks whether Plan v2 closes F-PLAN-01 through F-PLAN-04 and whether the applicable ACs are testable after implementation, not whether the implementation already exists.

## Finding Closure Status

| Finding | Closure | Evidence |
| --- | --- | --- |
| F-PLAN-01 | `CLOSED` | Plan v2 lines 80-84 explicitly separate legacy atomic recovery from batch semantics and require batch admission with `max_attempts=1`, empty resumable failures, no `--resume`, and no recovery-policy override; AC-08 lines 267-270 requires deletion-budget evidence that batch mode rejects `--resume` and recovery-policy overrides. |
| F-PLAN-02 | `CLOSED` | Plan v2 lines 155-175 define an acceptance-bound `deletion-budget.json` and evidence inventory, naming what `experiment_execution_core.py`, Stage123 queue impl, monitor entrypoints, `training_queue_monitor.sh`, notification, and release integrations may retain or must not own. |
| F-PLAN-03 | `CLOSED` | Plan v2 lines 126-135 define the canonical operator-control envelope, monotonic sequence, revision compare-and-swap, authorization matching, replay/stale rejection, and cleanup race behavior for `pause_after_current`, `stop_now`, and `continue_remaining`. |
| F-PLAN-04 | `CLOSED` | Plan v2 lines 105-112 require accepted admission bundles, `implementation_tree_sha256`, evidence commit, recipe gitlink, canonical command hash, adapter registry, command re-rendering, rejection of mutable/untracked executable paths, and script-content coverage for `bash queue_impl.sh`. |

## Per-AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | `PASS` | Single transition authority remains frozen, and Plan v2 adds a deletion-budget inventory that makes duplicate lifecycle authority measurable. |
| AC-02 | `PASS` | Manifest validation now binds accepted bundles, command hash, adapter type, implementation tree identity, tracked executable paths, and protected/current-checkout inputs. |
| AC-04 | `PASS` | Batch no-retry/no-resume semantics are now scoped to batch admission and no longer conflict with legacy atomic compatibility. |
| AC-06 | `PASS` | Operator-control schema, freshness, authorization, compare-and-swap, replay rejection, and cleanup boundary semantics are now frozen and testable. |
| AC-07 | `PASS` | Plan v2 requires one shared monitor implementation and keeps notification/release integrations as read-only event consumers. |
| AC-08 | `PASS` | Stage123 compatibility is now constrained by deletion-budget evidence, batch rejection of `--resume`/recovery-policy overrides, and no adapter-owned lifecycle transition. |
| AC-09 | `PASS` | No autonomous retry/resume/tuning/AI authority is consistent with AC-04/AC-05 after the batch-only boundary is frozen. |
| AC-11 | `PASS` | Independent acceptance now has concrete evidence targets: implementation-tree identity, deletion/compatibility audit, immutable command binding, and protected status. |

## Commands And Evidence

### Plan And Runtime Validation

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
```

Result:

```text
PASS
EXIT:0
```

Command:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
```

Result:

```json
{
  "current_milestone": null,
  "goal_status": "ACTIVE",
  "latest_review": {
    "event": "PLAN_REVIEWED",
    "plan_sha256": "baec753e892f85e186be4690c2d550252758e930fb3dca82590aec286a8ca785",
    "plan_version": 1,
    "prompt": "reviews/adversarial-plan-review-prompt.md",
    "review_id": "plan-v1-adversarial-review-01",
    "reviewer": "Dirac independent GPT-5.5 medium",
    "seq": 3,
    "summary": "Retry/resume boundary, deletion and authority budget, operator-control race semantics, and mutable command/path binding are under-specified or contradictory.",
    "time": "2026-07-14T15:42:50.760141Z",
    "verdict": "NOT_READY"
  },
  "open_findings": {
    "F-PLAN-01": {
      "classification": "CONTRADICTION",
      "review_fix_rounds": 1,
      "status": "OPEN"
    },
    "F-PLAN-02": {
      "classification": "IN_SCOPE",
      "review_fix_rounds": 1,
      "status": "OPEN"
    },
    "F-PLAN-03": {
      "classification": "IN_SCOPE",
      "review_fix_rounds": 1,
      "status": "OPEN"
    },
    "F-PLAN-04": {
      "classification": "IN_SCOPE",
      "review_fix_rounds": 1,
      "status": "OPEN"
    }
  },
  "pending_user_decisions": [],
  "plan_status": "UNREVIEWED",
  "plan_version": 2
}
EXIT:0
```

The runtime ledger still marks the four findings as `OPEN` because I did not modify `findings.jsonl`; this report records reviewer closure status for the implementer/runtime owner to append later.

Command:

```bash
sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md; echo EXIT:$?
```

Result:

```text
15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb  docs/joint_training/goals/experiment-batch-orchestration/plan.md
EXIT:0
```

Command:

```bash
git diff --check -- docs/joint_training/goals/experiment-batch-orchestration/plan.md; echo EXIT:$?
```

Result:

```text
EXIT:0
```

### Applicable AC Commands

I ran the applicable AC commands to determine whether they are currently executable. Most are future implementation commands and fail because the batch mode/tests do not exist yet. These failures are not classified as Plan v2 blockers because this rereview is before implementation and is scoped to F-PLAN-01 through F-PLAN-04 closure.

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_experiment_batch_core.py
```

Result:

```text
no tests ran in 0.00s
ERROR: file or directory not found: tests/experiment_workflow/test_experiment_batch_core.py
EXIT:4
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Result:

```text
usage: experiment_execution_core.py [-h] [--run-id RUN_ID]
                                    [--state-root STATE_ROOT]
                                    [--timeout-seconds TIMEOUT_SECONDS]
                                    [--command-json COMMAND_JSON]
                                    [--recovery-policy RECOVERY_POLICY]
                                    [--resume]
                                    {queue,phase}
experiment_execution_core.py: error: argument mode: invalid choice: 'batch-validate' (choose from queue, phase)
EXIT:2
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k local_failure
```

Result:

```text
no tests ran in 0.00s
ERROR: file or directory not found: tests/experiment_workflow/test_experiment_batch_routing.py
EXIT:4
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py
```

Result:

```text
no tests ran in 0.00s
ERROR: file or directory not found: tests/experiment_workflow/test_experiment_batch_control.py
EXIT:4
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_experiment_batch_monitor.py
```

Result:

```text
no tests ran in 0.00s
ERROR: file or directory not found: tests/experiment_workflow/test_experiment_batch_monitor.py
EXIT:4
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py
```

Result:

```text
..........                                                               [100%]
10 passed in 0.21s
EXIT:0
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_failure_classifier.py
```

Result:

```text
no tests ran in 0.00s
ERROR: file or directory not found: tests/experiment_workflow/test_experiment_batch_policy.py
EXIT:4
```

Command:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration
```

Result: same as the runtime validation block above; exit code `0`.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

- `DEFERRED_SUGGESTION`: When implementation begins, create the Goal-local `deletion-budget.json` before editing the Stage123 adapter so reviewers can diff the intended authority removals against the code changes.
- `DEFERRED_SUGGESTION`: Keep the finite accepted-adapter registry small for the first implementation round; adding generic adapter extensibility would be a separate design surface.

## Contract Contradictions

None remaining in Plan v2.

## Single Most Likely Weakness In This Review

The main weakness is that this rereview validates the Plan contract, not the implementation. Current AC commands for batch mode fail because implementation files do not exist yet; I treated that as expected pre-implementation state. A later milestone reviewer must reject if those commands remain missing after implementation.
