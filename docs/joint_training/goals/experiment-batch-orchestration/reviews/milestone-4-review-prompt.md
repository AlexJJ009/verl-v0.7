# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: `2`
- Current milestone: `Milestone 4`
- Base commit: `0a1d1b0696872ec88caaaa3ef7b3008380c88d0b`
- Candidate commit: `ea0a7d12be47518253788c0f5258ddd4de68d6e4`
- Applicable ACs: `AC-02,AC-03,AC-04,AC-05,AC-06,AC-08,AC-09,AC-10`

## Required Verification

Run every exact AC-02 through AC-10 verification command from Plan v2, including the protected_asset_fingerprint compare. Do not run GPU, training, or external services.

## Additional Task-Specific Focus

Adversarially mutation-test ordered phase/item routing, local fallback, repeated normalized failure across restart, cleanup/state/event corruption shared stops, protected/admission drift, control freshness/replay/CAS, no retry/resume, no agent/timer/tuning authority, and Stage123 compatibility. Audit that tests are non-trivial and do not weaken prior behavior. Do not implement.

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
