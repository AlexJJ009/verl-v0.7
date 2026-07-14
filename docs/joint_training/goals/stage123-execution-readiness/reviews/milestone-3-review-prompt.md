# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version: `8`
- Current milestone: `Milestone 3`
- Base commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Candidate commit: `62f6c5c8d34007d71b5269b795e8f7c72db92752`
- Applicable ACs: `AC-01 implementation identity, AC-02 through AC-08 committed implementation readiness`

## Required Verification

git status --short -- config/experiment_execution scripts verl recipe; git -C recipe status --short; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_stage123_end_to_end.py

## Additional Task-Specific Focus

Independently verify Milestone 3 committed production identity. Candidate superproject commit is 62f6c5c8d34007d71b5269b795e8f7c72db92752, recipe gitlink is eeadc66e13592708b7870a93312b5ab9eb82c4a6, expected implementation_tree_sha256 is 3ff13ef9bffb2a87984cfa68284885a2cb791fafb8e4d60bd331cc07c5cad3f7. Confirm covered roots and recipe are clean, identity recomputes byte-for-byte, focused behavior passes, protected assets are unchanged/unstaged, and no GPU/training/external action occurred.

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
