# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `9`
- Current milestone: `none`
- Base commit: `ec11f64e3d2bf73e2fe8b5614f151fa244ee1818`
- Candidate commit: `7099830b`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness; inspect Primary Chain Plan v5 three-run matrix and user decision EX-MATCHED-CONTROL-READINESS-01; verify the 8-GPU/5400-second/3-repetition zero-step feasibility argument

## Additional Task-Specific Focus

Adversarially verify that Plan v9 adds exactly the matched pure-Stage1 control required by Primary Chain Plan v5, preserves one readiness outcome, does not authorize training before accepted admission, uses the unified batch core, and has an honest bounded three-phase requalification path. Return READY or NOT_READY with every blocking finding.

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
