# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `2`
- Current milestone: `Milestone 5`
- Base commit: `1c1e56f3adde1922175f7343c5f593038ffb1dff`
- Candidate commit: `74f73f987a65e21b9d43f7d87e06b0e8c2dc03bc`
- Applicable ACs: `AC-01,AC-02,AC-03,AC-04,AC-05,AC-06,AC-07,AC-08,AC-09,AC-10`

## Required Verification

The focused AC matrix was run exactly once from the candidate. Independently rerun the exact AC-01 through AC-10 commands in Plan v2, with no GPU/training/external services.

## Additional Task-Specific Focus

Independent implementation review from committed state. Audit all AC-01 through AC-10 behavior, deletion-budget inventory, no duplicate authority, no retry/resume/tuning/agent/timer, exact manifest/implementation/protected bindings, operator race semantics, monitor idempotence, and test non-triviality. Treat source inspection plus reviewer-owned mutation probes as evidence; do not implement.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.
9. For a Plan review, reject `READY` when any AC declares an absolute numeric performance or resource budget that has no recorded feasibility probe in the Plan's `Feasibility Probes` section, or whose budget contradicts the probe's measured floor.
10. When your only blocking findings are purely mechanical (formatting, patch context offsets, artifact or directory placement) with no behavioral or contract impact, say so explicitly and offer a light same-reviewer re-verification scoped to those findings instead of demanding a fresh full round.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
