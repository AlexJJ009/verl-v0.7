# Experiment Execution Core Consolidation Launch Prompt

Do not start implementation until the user explicitly authorizes execution.

Use this Goal directory:

```text
/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation
```

Read `plan.md`, validate the Plan and runtime, and require an independently recorded
`READY` Plan review before implementation. This single user start authorizes autonomous
execution of Milestones 1-6 as one Goal envelope. Proceed serially and automatically
advance after each milestone's checks, required independent review, and runtime
validation pass. Do not pause for routine milestone boundaries, `IN_SCOPE` findings,
test failures, or implementation choices inside the frozen ACs.

This Goal is CPU-only architecture consolidation. Do not run GPU preflight,
calibration, training, or real external-service acceptance. Preserve protected user
assets and diagnostic evidence. The implementer must not self-accept.

Stop and ask the user only for a Plan `CONTRADICTION`, `AC_CHANGE`, convergence failure,
protected-asset risk, required GPU/real external-service access, or a newly discovered
independently useful outcome. These boundaries are not authorized by this Goal start.
