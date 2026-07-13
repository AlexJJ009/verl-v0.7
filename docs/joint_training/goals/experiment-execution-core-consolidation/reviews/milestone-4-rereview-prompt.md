# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone-rereview`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `Milestone 4`
- Base commit: `59b802cf`
- Candidate commit: `HEAD`
- Applicable ACs: `AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow -k 'compat or migration or receipt or adoption or evidence'; independently recompute production including replacement files, tests, active docs; inspect every retained file above 300 lines

## Additional Task-Specific Focus

Re-review F-M4-02 and F-M4-03. Confirm deletion-budget counts replacement production at 2259, deferred preflight qualification tests/609-line fixture are removed rather than weakened, all three classes remain independently below ceilings and net-negative, and every remaining file above 300 lines has a path-specific inventory justification.

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
