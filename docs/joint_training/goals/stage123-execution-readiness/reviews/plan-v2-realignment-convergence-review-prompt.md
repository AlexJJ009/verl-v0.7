# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Convergence Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `2`
- Current milestone: `none`
- Base commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Candidate commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Applicable ACs: `AC-02, AC-06, AC-08, AC-09`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; inspect current worktree and both prior review reports

## Additional Task-Specific Focus

Two behavioral Plan review rounds exposed recurring host/container, identity, acceptance-authority, and coverage defects. Do not implement. Determine the smallest convergent architecture before any third ordinary fix. Specifically evaluate: one host-owned immutable host_facts.json passed through the mandatory wrapper so container Python never invokes docker/tmux; current implementation-tree recomputation for every admission path including --bundle; a versioned acceptance_report schema binding Plan v2/hash, reviewer, candidate commit, AC-01..AC-08 all PASS, bundle hash and all input hashes; one wrapper-level no-training test. Return CONVERGED with exact required changes, or CONVERGENCE_FAILED if the outcome must split/change.

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
