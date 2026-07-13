# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `1`
- Current milestone: `none`
- Base commit: `634a0f22cb1709c45cdd9a0a5550a913f12f8235`
- Candidate commit: `ebe0fcaa7f8f8ba88e8eb81735a834e0619cce1e`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; git diff 634a0f22cb1709c45cdd9a0a5550a913f12f8235..ebe0fcaa7f8f8ba88e8eb81735a834e0619cce1e -- docs/joint_training/goals/stage123-primary-chain-execution; inspect serial producer-consumer contracts and named implementation interfaces

## Additional Task-Specific Focus

Re-review F-EX-PLAN-01..05. Verify accepted bundle producer is defined, primary-only manifest is inherited, admission validation is exact, release commands have no placeholders, recovery codes/schema are frozen, and publication failure semantics are correct. Issue a separate READY or NOT_READY verdict and explicitly state whether every open finding can close.

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
