# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `final-acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `Milestone 6`
- Base commit: `390be933`
- Candidate commit: `97c1c1f596da4d1e977b252a411e9444029d9599`
- Applicable ACs: `AC-01,AC-02,AC-03,AC-04,AC-05,AC-06,AC-07,AC-08,AC-09`

## Required Verification

Run frozen focused commands for AC-01 through AC-05 and AC-07 from committed state. Do not rerun the full AC-06 suite; audit the sole-run summary and Milestone 5 focused replacement evidence. Independently recompute AC-08 budgets. Run goal-plan-runtime validate-runtime and git status --short for AC-09.

## Additional Task-Specific Focus

Final acceptance must independently report every AC as PASS, FAIL, or WEAKENED. Inspect commit history/diffs for deleted, skipped, loosened, or trivialized tests. Bind acceptance.md and acceptance_report.json to Plan SHA256 b0929f0b0c910515c14140489d629c6103637c0ceb477c24e34185ddf4887ba8 and candidate commit 97c1c1f596da4d1e977b252a411e9444029d9599. Verify protected assets are untracked/unstaged and recipe submodule fb49bcc is clean. Reviewer alone must edit acceptance.md and create acceptance_report.json; ACCEPTED is valid only if all ACs PASS.

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
