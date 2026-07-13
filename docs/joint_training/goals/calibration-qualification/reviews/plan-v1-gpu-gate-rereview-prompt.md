# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Targeted Plan Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Base commit: `f2b4a1bc0c87f46cbcd2974c4a8b4fb8deef4548`
- Candidate commit: `424e6ad1ffc9a5e35f4a3d8e23d5c0c131136f27`
- Applicable ACs: `AC-03, AC-08`

## Required Verification

- `goal-plan-runtime validate-plan docs/joint_training/goals/calibration-qualification`
- Inspect every occurrence of `Milestone 4`, `Milestone 5`, `GPU`, `probe`, and `authorization` in the Plan.
- `git diff f2b4a1bc0c87f46cbcd2974c4a8b4fb8deef4548..424e6ad1ffc9a5e35f4a3d8e23d5c0c131136f27 -- docs/joint_training/goals/calibration-qualification`
- Confirm Goal start authorizes Milestones 1-4 and 6-7, while only Milestone 5 requires separate bounded-GPU authorization.
- Confirm the accepted production identity architecture and all prior finding closures remain unchanged.

## Additional Focus

Review `F-CQ-FAR-01`. Determine whether the single stale milestone reference is corrected and whether the Calibration Qualification Plan is now `READY`. If so, explicitly authorize closure of `F-CQ-FAR-01` and state that the Goal may be started for Milestones 1-4 without GPU authorization, but Milestone 5 remains blocked until separate explicit user authorization.

## Required Output

- Reviewer identity
- Verdict: `READY`, `NOT_READY`, or `CONTRACT_CONTRADICTION`
- `F-CQ-FAR-01` disposition
- AC-03 and AC-08 verdicts
- Commands and relevant evidence
- One-line `PLAN_REVIEWED` ledger summary
