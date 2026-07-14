# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Finding Re-review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 6`
- Base commit: `5e387d82a372974c9ebf3509ec178ff893c5e8a9`
- Candidate commit: `c2fc0a1ce5576618be14c9d4093ea8b86c5005e8`
- Applicable ACs: `AC-01, AC-02, AC-07, AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_evidence_compatibility.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --format json
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --output /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/reviewer-m6-profile-binding.jsonl
goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification
git status --short

## Additional Task-Specific Focus

Review F-M6-01 only. Confirm scripts/execution_results.py now consumes the normalized manifest-owned resource_profile.sha256 (d9b6a36dd9fcc4307f7b502e5511989e60c1a257f57c9ac70574acaf12eee2b5) for calibration/admission binding rather than the profile shell file byte hash (87825d59...). Confirm fail-closed behavior for missing/malformed manifest resource identity, no weakening of explicit expected bindings, and no protected asset changes. Recompute the refrozen implementation identity 453b60bf1b626934a20916a1ca1aa9c90cc9de2b028d19b5ad875471d3c90ead for production commit 9a8d78edc06a9a63a409704c7aab4b221dc6a15a and recipe gitlink 888d8e1a979070013ffc9ccca401ea17c73f26d6. Determine whether the previous GPU probe is correctly invalidated and a fresh bounded 2x3 zero-step probe is required. Do not modify files.

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
