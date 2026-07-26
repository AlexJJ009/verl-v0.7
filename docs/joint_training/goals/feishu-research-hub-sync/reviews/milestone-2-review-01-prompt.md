# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `1`
- Current milestone: `Milestone 2`
- Base commit: `cf9e69f2c87566b3b49ac8b8fbb5bf4c6ac2d2ee`
- Candidate commit: `d924fe728002d6b76b0001bed6562823d7e109bb`
- Applicable ACs: `AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m pytest -q tests; REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m hubctl check --root .; git -C /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub status --short; git -C /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub remote get-url origin; goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync

## Additional Task-Specific Focus

Fresh Milestone 2 independent review. Adversarially verify publish first-write/no-op/stale/readback, remote pull authorship/committer/trailers/push/no-op/unknown editor, remote-wins byte recovery, rename/in-root move/detach/tombstone, canonical Docx diff/version fail-closed, complete gate canaries, real local bare-remote pre-push red/green, and zero external/shared writes. Audit skipped/weakened tests and protected parent worktree. Write only add-only report reviews/milestone-2-review-01.md; do not edit implementation or ledgers.

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
