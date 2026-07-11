# Experiment Execution Reliability Operational Delta Review

- Review date: 2026-07-12
- Reviewed delta: user-confirmed operational boundaries from 2026-07-11
- Plan: `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- Launch prompt: `docs/joint_training/plans/active/experiment_execution_reliability_goal_launch_prompt.md`
- Review mode: independent delta-only review; no implementation or plan changes

## Scope

This review checks only whether the new operational boundaries are internally
consistent with AC-01 through AC-25 and are stated strongly enough for a future
reviewer to verify them by command output. It does not reopen previously accepted
parts of the Goal except where the delta creates a contradiction.

## Summary

The plan and launch prompt now state the intended policy correctly at the prose
level: stop run `1783777744`, cap each phase at 30 minutes, treat the historical
76-minute stall as a regression, release GPUs after timeout, and make WxPusher
event-only. These additions do not weaken full HumanEval+/MBPP+/LiveCodeBench,
`MAX_RESPONSE_LENGTH=8192`, or the shared Stage1/2/Stage3 resource profile.

The delta is not yet goal-ready because two required runtime behaviors are not
owned by an acceptance criterion with executable negative-path evidence. AC-19
can reject a completed calibration report after a timeout, but it does not prove
that the watchdog terminates the whole runtime and frees all GPUs at minute 30.
AC-20 describes all three notification events, but its only test is scoped to the
user-decision path and does not prove verified start/failure semantics or absence
of periodic healthy notifications.

## P0 Blocking Findings

### P0-1: The 30-minute timeout does not have a machine-verifiable cleanup contract

Evidence:

- The boundary declares a 30-minute wall-clock budget from validation rollout
  readiness to complete metrics.
- AC-19 requires a timed-out report to become `blocked`.
- The stop trigger says to stop the affected runtime and not continue consuming
  all GPUs.
- The launch prompt says an overrun must release GPUs.

However, no verification command or fixture proves the required runtime sequence:

1. detect the exact `validation_rollout_ready` timestamp;
2. arm a monotonic 30-minute deadline;
3. terminate queue, phase launcher, Docker/Ray children, and descendants;
4. wait for bounded graceful shutdown and escalate if necessary;
5. verify no owned process/container remains;
6. verify every GPU allocation owned by the run has been released;
7. preserve timeout evidence and emit a nonzero terminal result.

`check_code_task_operational_calibration.py` only validates a report after execution.
It cannot prove that a stalled process was actually stopped at the deadline. A run
could exceed 30 minutes, continue holding all GPUs, later write `blocked`, and still
satisfy the current report-level wording.

Required plan amendment in Given/When/Then form:

- Given a fixture runtime with a validation-readiness event and deliberately stalled
  metrics, plus fake process/container/GPU ownership probes,
- When the scaled deadline expires,
- Then the watchdog records the readiness/deadline timestamps, returns nonzero,
  terminates the complete owned process tree, removes owned containers/Ray jobs,
  observes zero owned GPU allocations within a declared cleanup grace period, and
  preserves one terminal timeout report.

The test must cover graceful exit, forced escalation, orphan child, stale PID/process
identity reuse, cleanup-command failure, and idempotent repeated cleanup. The real
operational check must use the same state machine with a 30-minute value; tests may
inject a short clock but may not use a separate implementation.

### P0-2: WxPusher event semantics are not fully covered by acceptance tests

Evidence:

- Boundary 10 and AC-20 prose define exactly three events:
  `run_started`, `run_failed`, and `user_decision_required`.
- AC-20's verification command is only
  `tests/experiment_workflow/test_user_decision_notification.py`.
- No named test proves that tmux/container/process existence is rejected as start
  evidence, that first-step or complete-metrics evidence is accepted, or that
  healthy polling never emits a notification.

The current AC could pass with a correct decision-required notifier while start and
failure notifications remain over-eager, duplicated, periodic, or absent.

Required plan amendment in Given/When/Then form:

- Given local fake WxPusher plus event fixtures for tmux-only, container-only,
  validation-ready-only, complete formal metrics, first training step, timeout,
  terminal failure, healthy polls, and repeated identical events,
- When the shared notification policy evaluates them,
- Then it emits exactly one `run_started` only for complete formal metrics or first
  training step, exactly one `run_failed` for a verified terminal failure, exactly
  one `user_decision_required` for a blocking decision state, emits nothing for
  healthy polling or infrastructure existence alone, and deduplicates by run/event
  identity without contacting real WxPusher.

This should be a dedicated shared-policy regression test, not only an assertion in
the user-decision test.

## Delta Verification Table

| Delta requirement | Status | Evidence and gap |
| --- | --- | --- |
| Stop run `1783777744`; retain only as local diagnostic | NEEDS SCAFFOLDING | Approved Operational Decision 1 records the exact run, step 0, approximately 76 minutes, and local-diagnostic-only disposition. No AC command verifies terminal stop evidence, exclusion from SQLite/W&B/release, preservation of diagnostic logs, or absence of live owned processes/containers. Add a bounded historical-run disposition fixture/checker. |
| Stage1/2/3 pre-training measurement and formal `val_before_train` have a 30-minute hard wall | NEEDS SCAFFOLDING | The plan pins the start at validation rollout readiness and end at complete metrics, and AC-19 covers all phases. It does not define a machine-readable readiness event/schema, monotonic clock requirement, complete-metrics predicate, or runtime enforcement test. |
| Historical approximately 76-minute stall is a regression | NEEDS SCAFFOLDING | AC-05 names the historical stall as a required regression fixture, but its verification only promises comparison/semantic tests. The expected evidence does not require a 76-minute trace fixture to fail the hard-wall gate or prove cleanup. |
| Timeout stops runtime and releases GPUs | FAIL | Prose and launch prompt require it, but no AC verification command proves process-tree/container/Ray termination and zero run-owned GPU allocations. This is P0-1. |
| WxPusher only for verified start/failure/decision; no healthy periodic notification | FAIL | Policy prose is correct, but AC-20 tests only the decision path. There is no shared three-event behavior regression. This is P0-2. |
| Compatibility with AC-01..25, full validation, 8192, common profile | PASS | The delta adds time and notification gates without reducing validation breadth, context length, reward semantics, or phase resource-profile identity. The 30-minute requirement is stricter but not semantically contradictory. |

## Additional Required Clarifications

These can be resolved by the plan author without further user input because the
user's policy intent is already explicit:

1. Define `validation_rollout_ready` as a structured event written after rollout
   workers/models are ready and immediately before the first formal validation
   batch is submitted. Use a monotonic timestamp for enforcement and UTC wall time
   only for audit display.
2. Define `complete validation metrics` as a schema-valid terminal metric set for
   every manifest-declared full validation dataset, not merely any validation log
   line or partial aggregate.
3. Define GPU release relative to run ownership: no live owned process, Ray job,
   or container and no GPU compute process attributable to the run after a finite
   cleanup grace period. Low utilization alone is insufficient.
4. State the cleanup grace period and escalation order in policy/manifest, while
   keeping the 30-minute validation budget itself non-extendable.
5. Add a disposition record for `1783777744` containing observed evidence paths,
   terminal reason, no-release flag, and rerun-required status. The historical run
   must not be accepted as calibration or formal experiment evidence.

## Required Re-review Evidence

A fresh delta reviewer should receive:

1. the amended AC text and launch prompt;
2. the named timeout/watchdog and notification-policy test commands;
3. a scaled-clock timeout fixture proving full cleanup and nonzero exit;
4. the 76-minute historical trace fixture failing the same hard-wall state machine;
5. a local-fake WxPusher matrix covering all three events and all non-events;
6. a machine-readable disposition check for run `1783777744` proving diagnostic-only,
   no DB/W&B release, and no remaining runtime ownership.

DELTA VERDICT: NOT READY
