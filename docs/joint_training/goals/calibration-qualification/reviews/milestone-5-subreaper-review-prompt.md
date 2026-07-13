# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Focused Finding Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 5`
- Base commit: `000d2915`
- Candidate commit: `90b72ebb`
- Applicable ACs: `AC-03, AC-04, AC-06, AC-08`

## Required Verification

Read F-M5-10 and probe-20260713T203935Z Stage3 rep3 timeout evidence; inspect candidate diff; run all Milestone3 tests; test reap_adopted_children behavior including ECHILD; inspect Linux prctl subreaper use and call order relative to repetitions/owned cleanup; verify shared execution core and recipe unchanged; recompute identity and exact freeze binding; no GPU and no modifications.

## Additional Task-Specific Focus

Determine whether the probe driver correctly becomes a Linux child subreaper before spawning repetitions and reaps adopted exited Ray descendants after each repetition so the same container PID1 does not accumulate zombies across six runs. Verify this is bounded to calibration and preserves fail-closed cleanup. Also assess whether a final full 2x3 rerun can proceed under the remaining aggregate GPU envelope; report but do not run GPU.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
