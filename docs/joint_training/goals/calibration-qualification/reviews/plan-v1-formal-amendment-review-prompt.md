# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Review after user-authorized formal convergence amendment`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `2bf183399fac669454e370507ccb6038be7b58b4`
- Candidate commit: `f2b4a1bc0c87f46cbcd2974c4a8b4fb8deef4548`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`
- User authorization: formal Plan amendment and review to make Calibration Qualification executable

## Required Verification

- `goal-plan-runtime validate-plan docs/joint_training/goals/calibration-qualification`
- Inspect `findings.jsonl` and prior convergence result; the new review is authorized to supersede the failed candidate, not to erase ledger history.
- `git diff 2bf183399fac669454e370507ccb6038be7b58b4..f2b4a1bc0c87f46cbcd2974c4a8b4fb8deef4548 -- docs/joint_training/goals/calibration-qualification`
- `git rev-parse HEAD:scripts`
- `git rev-parse HEAD:verl`
- Verify `config/experiment_execution` is an explicitly planned covered root created before the calibration implementation commit.
- `git ls-files --stage recipe`
- `git -C recipe rev-parse HEAD`
- Audit that full Git tree roots plus the entire recipe gitlink cover all direct and transitive production dependencies without dynamic path enumeration.
- Audit all three serial Plans together and protected asset boundaries.

## Additional Task-Specific Focus

Fresh Plan review after user-authorized formal convergence amendment. Verify complete production identity is now defined by full Git tree objects for config/experiment_execution, scripts, and verl plus entire recipe gitlink commit; Goal evidence is excluded; runtime image/data identity remains separately covered; planned boundary root can be created before calibration commit; no transitive dependency can escape the boundary. Determine whether prior identity findings can be formally closed by this amendment. Issue an independent `READY`, `NOT_READY`, or `CONTRACT_CONTRADICTION` verdict. If READY, explicitly authorize closure of the remaining finding and state that the Goal may be started subject to its Runtime Contract.

## Reviewer Rules

1. Read the frozen contract and inspect the candidate diff.
2. Run required verification yourself.
3. Evaluate each applicable AC as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not create new blocking requirements outside frozen ACs.
6. Do not edit files or continue implementation.
7. Report commands, relevant output, Plan version, candidate commit, finding disposition, and one-line `PLAN_REVIEWED` ledger summary.
