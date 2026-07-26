# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 5`
- Base commit: `9e81e4ee9be33c7acc19c4bd88492f2d738f051e`
- Candidate commit: `6fca611d12caaec97c5efd4b72f950c6eec1a977`
- Applicable ACs: `AC-11`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && uv run --with '.[test]' pytest -q tests/importers/test_verl_candidates.py tests/feature_stories/test_fs11_curated_seed.py && uv run --with '.[test]' pytest -q && uv run --with '.[test]' hubctl check --root . && git diff --check 9e81e4ee9be33c7acc19c4bd88492f2d738f051e 6fca611d12caaec97c5efd4b72f950c6eec1a977

## Additional Task-Specific Focus

Post-convergence independent review of AC-11. Candidate 6fca611 removes all caller-selected authority paths from Python API and CLI. Project authority locations are fixed relative to repo_root: scripts/training_result_release_gate.py, .hub-authority/training_release_gate.jsonl, .hub-authority/wandb_verified.jsonl, and .hub-authority/hf_manifests. Re-run both prior FORGED canaries including attempts to pass arbitrary --authority or source paths; prove the call surface cannot replace authority sources. Verify release-gate subprocess checks the fixed state, W&B receipt requires an existing synced marker and exact URL, HF manifest requires uploaded/verified and immutable commit URL under fixed root. Confirm prior AC-11 properties remain. Review only; no implementation, push, or external writes.

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
