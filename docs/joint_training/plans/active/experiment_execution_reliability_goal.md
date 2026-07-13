# Experiment Execution Reliability Goal - Superseded

Status: superseded by `goal-plan` protocol version 0.2.0 on 2026-07-13.

The previous contract combined architecture cleanup, calibration qualification,
queue deployability, and experiment execution. These are independently useful
outcomes with different runtime and acceptance surfaces, so they are now serial Goals.

The current Goal is:

```text
docs/joint_training/goals/experiment-execution-core-consolidation/plan.md
```

It delivers only the maintainable, CPU-verified execution core and branch cleanup.
GPU calibration and Stage123 execution are deferred.

Follow-up sequence:

1. Calibration Qualification.
2. Stage123 Execution Readiness.
3. Stage123 Experiment Execution.

Do not execute this superseded file. Use the Goal directory's append-only runtime and
findings ledgers, generated reviewer prompts, lifecycle validators, and independent
acceptance report.
