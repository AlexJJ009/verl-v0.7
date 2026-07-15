# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Plan Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `2`
- Current milestone: `none`
- Base commit: `f2b4a1bc0c87f46cbcd2974c4a8b4fb8deef4548`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-09, AC-10, AC-11`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution; sha256sum docs/joint_training/goals/stage123-primary-chain-execution/plan.md; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py validate recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml; inspect the retained Stage1 step40 artifact, historical step100 config/output hashes, legacy 1783777744 stop evidence, and current two-run admission bundle without launching training

## Additional Task-Specific Focus

Review Plan v2 as an experimental-design contract, not merely an execution checklist. Verify that the matched Stage1 control is necessary and fair, the control and chain use the same initial weights/data budget/evaluator/resource semantics, the support/reject/inconclusive rule is deterministic and non-adaptive, every operational/scientific outcome has a prescribed disposition, historical 4096-token step100 evidence is correctly excluded, the legacy step0 attempt cannot contaminate the new experiment, the renewed three-run Readiness prerequisite is explicit, no formal training is authorized by this review, and the Goal remains one independently useful outcome rather than smuggling in a sweep or paper claim.

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
