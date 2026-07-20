# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `14`
- Current milestone: `Milestone 2`
- Base commit: `7b31f23b8c1bd019fd661f5d3a339ec65ddc7262`
- Candidate commit: `4b828db9024bdeafb3f703aeec73855244166dce`
- Applicable ACs: `AC-04,AC-05,AC-06,AC-07,AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_control_reuse.py tests/experiment_workflow/test_experiment_manifest.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/stage3-handoff-reuse-20260716T0750Z/stage3-handoff-manifest.yaml --format json; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/stage3-handoff-reuse-20260716T0750Z/stage3-handoff-admission.json --allow-prepared --run-id frac25-stage3; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution

## Additional Task-Specific Focus

Audit F-EX-M2-30 and F-EX-M2-32. Verify that an admitted Stage3 handoff manifest contains exactly frac25-stage3; an absent Stage2 run is accepted only when the source is fully certificate-bound; calibration workload source is materialized with descriptor hash and cannot retain a pending producer; old terminal evidence remains untouched. Decide explicitly whether the CPU-only manifest/admission correction changes the already requalified Stage3 wrapper/GPU runtime boundary and therefore requires a new zero-step GPU probe before real Stage3 launch. Do not edit files or launch GPU.

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
