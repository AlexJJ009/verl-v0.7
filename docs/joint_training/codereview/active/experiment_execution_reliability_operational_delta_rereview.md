# Experiment Execution Reliability Operational Delta Re-review

- Review date: 2026-07-12
- Reviewer role: independent operational delta reviewer
- Scope: plan-only review of the 30-minute hard wall, AC-26, AC-27, and the Goal launch prompt
- Files reviewed:
  - `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
  - `docs/joint_training/plans/active/experiment_execution_reliability_goal_launch_prompt.md`
  - `docs/joint_training/codereview/active/experiment_execution_reliability_operational_delta_review.md`
- Implementation changes: none

## Review Result

The amended delta is ready for implementation. The plan now converts both prior P0
gaps into explicit, machine-verifiable acceptance criteria rather than leaving them as
operational prose.

AC-26 makes the 30-minute limit a runtime hard wall. It requires queue advancement to
stop, bounded graceful-to-force escalation, cleanup of run-owned Ray, Docker, tmux, and
process descendants, and an ownership-based proof that no GPU allocation remains. Its
scaled-clock regression matrix includes completion before deadline, graceful shutdown,
forced shutdown, orphan descendants, cleanup failure, repeated cleanup, and the
historical 76-minute step-0 trace. Cleanup failure remains blocked and cannot be
reported as resource release.

AC-27 defines one shared notification state machine. It permits exactly three event
types and gives a machine-testable start threshold: complete formal validation metrics
or the first training step. It rejects tmux/container existence, model loading,
incomplete validation, healthy polling, and unchanged metrics as notification events.
It also requires deduplication, secret redaction, local-path evidence, local fake
delivery, and proof that delivery failure cannot alter launch or release state.

The launch prompt carries the same constraints: no gate bypass, a non-extendable
30-minute deadline, verified idempotent cleanup, zero run-owned GPU resources, and
event-only WxPusher behavior. It names AC-01 through AC-27 as the source of truth and
does not weaken the existing experiment semantics.

## Delta Verification Matrix

| Requirement | Status | Plan evidence |
| --- | --- | --- |
| Scaled-clock graceful termination | PASS | AC-26 expected evidence explicitly includes graceful shutdown. |
| Scaled-clock forced termination | PASS | AC-26 requires bounded escalation and a forced-shutdown fixture. |
| Orphan descendant cleanup | PASS | AC-26 requires removal of run-owned descendants and an orphan fixture. |
| Cleanup-command failure | PASS | AC-26 requires the run to remain blocked and retain evidence when cleanup is incomplete. |
| Idempotent repeated cleanup | PASS | AC-26 states cleanup is idempotent and requires a repeated-cleanup fixture. |
| Queue does not advance after deadline | PASS | AC-26 explicitly stops the owning queue from advancing. |
| Ray/Docker/tmux/process cleanup | PASS | AC-26 names all four ownership surfaces and requires their removal. |
| Run GPU ownership reaches zero | PASS | AC-26 requires no attributable GPU process/allocation and permits `resources_released=true` only after complete cleanup. |
| Historical 76-minute step-0 trace triggers the same hard wall | PASS | AC-26 names the historical trace in the scaled-clock fixture matrix. |
| Historical run remains release-blocked | PASS | AC-26 requires the legacy fixture to prove the release gate remains blocked. |
| Historical run has zero SQLite/W&B success evidence | PASS | AC-26 requires zero matching SQLite rows and W&B sync markers. |
| Historical run leaves no runtime ownership | PASS | AC-26 requires no residual runtime ownership for the legacy fixture. |
| Exactly three WxPusher events | PASS | AC-27 limits output to `run_started`, `run_failed`, and `user_decision_required`. |
| Tmux/container existence is not start evidence | PASS | AC-27 explicitly lists both as non-events. |
| Healthy polling emits nothing | PASS | AC-27 explicitly lists healthy polling and unchanged metrics as non-events. |
| Incomplete validation emits no start event | PASS | AC-27 lists incomplete validation as a non-event. |
| Started evidence threshold is objective | PASS | AC-27 requires complete formal validation metrics or the first training step. |
| Deduplication | PASS | AC-27 requires at most one message per deduplication key and a dedicated regression assertion. |
| Secret handling | PASS | AC-27 expected evidence requires secret redaction. |
| Fake delivery failure is state-neutral | PASS | AC-27 requires local recording while forbidding delivery failure from changing launch or release state. |
| Full HumanEval+/MBPP+/LiveCodeBench preserved | PASS | The plan's non-negotiable boundaries remain unchanged; the hard wall changes runtime acceptance, not validation breadth. |
| `MAX_RESPONSE_LENGTH=8192` preserved | PASS | The canonical common profile remains mandatory in the plan and launch prompt. |
| Stage1/2/3 common resource profile preserved | PASS | AC-19 and the launch prompt retain one canonical profile across all phases. |
| No conflict with AC-01 through AC-25 | PASS | AC-26 adds runtime enforcement for AC-05/19; AC-27 expands AC-20 notification coverage without changing prior gates or approval semantics. |

## Adversarial Checks

### Deadline semantics

The deadline starts from validation rollout readiness and ends only at a schema-complete
formal metric set, as established by the amended plan context. It is not satisfied by a
partial metric, a log heartbeat, tmux existence, or model loading. A relative throughput
improvement cannot compensate for exceeding 30 minutes.

### Cleanup truthfulness

Low GPU utilization is not accepted as proof of release. AC-26 requires attribution to
the run across process, Ray, Docker, tmux, and GPU ownership. It distinguishes cleanup
success from cleanup failure and preserves a nonzero blocking outcome when ownership
cannot be cleared. This prevents the controller from claiming success after merely
issuing termination commands.

### Queue safety

The owning queue must stop before or as cleanup begins, so a timed-out Stage1/2/3 phase
cannot advance to the next phase while descendants are still being terminated. The
scaled-clock test is sufficiently specified to require this behavior as part of the
same state machine rather than as a later manual check.

### Historical regression

The stopped run ending in `_1783777744` is constrained to local diagnostic evidence.
The required fixture exercises the same deadline controller and proves all four
disposition properties: blocked release, zero SQLite success rows, zero W&B sync
markers, and zero residual runtime ownership. It cannot be reused as operational
calibration or formal experiment evidence.

### Notification authorization boundary

WxPusher remains notification-only. AC-27 tests event generation and failure behavior;
AC-20 retains the rule that phone delivery, clicks, or replies cannot authorize launch.
A soft-threshold decision still requires the user's interactive decision, a reviewed
manifest/policy commit, and a fresh passing preflight.

## AC Self-Verifiability

| AC | Status | Reason |
| --- | --- | --- |
| AC-26 | PASS | One named pytest command must exercise a deterministic scaled-clock controller and fake ownership surfaces, including every requested cleanup branch and the historical disposition fixture. No real 30-minute wait or external service is needed. |
| AC-27 | PASS | One named pytest command must exercise a local fake notification state machine across all positive events, non-events, deduplication, redaction, and delivery failure. No real WxPusher contact is needed. |

## Launch Prompt Consistency

The launch prompt is consistent with the amended plan:

1. It identifies AC-01 through AC-27 as the single source of truth.
2. It preserves milestone order and independent acceptance.
3. It requires the 30-minute hard wall to release GPUs and stop queue advancement.
4. It explicitly names Ray, Docker, tmux, process descendants, and run-owned GPU
   resources in cleanup.
5. It limits WxPusher to verified start, failure, and decision-required events.
6. It preserves full validation, 8192 response length, and the common phase profile.
7. It forbids real external services as acceptance evidence.

The prior review's two P0 findings are therefore resolved at plan level. No additional
user clarification is required before implementation of this delta.

DELTA VERDICT: READY
