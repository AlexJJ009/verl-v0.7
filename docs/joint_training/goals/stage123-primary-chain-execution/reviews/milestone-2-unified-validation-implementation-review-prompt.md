# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `implementation`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `17`
- Current milestone: `Milestone 2`
- Base commit: `05fefe4ad43f6f16d648331b972762fd904c8c8e`
- Candidate commit: `74aa75f2c98cb80ff8d7774cc8f6af45a6ac4d04`
- Applicable ACs: `AC-12`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_validation_protocol.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_stage123_preflight_model_identity.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_phase_contract_audit.py --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml

## Additional Task-Specific Focus

Audit F-EX-IMPL-01: the all-phase Stage123 main validation decoder must be profile-owned, serialized, hashed, fail-closed, consumed by both single-model and joint Stage2 launch paths, and rejected by the CPU audit if any unauthorized drift appears. Verify the candidate contains no training launch, no legacy V13/V14 reuse, and no protected-asset change.

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
