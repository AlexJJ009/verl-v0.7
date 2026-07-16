# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `9`
- Current milestone: `Milestone 2`
- Base commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Candidate commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Applicable ACs: `AC-08`

## Required Verification

tmux has-session -t stage123_primary_monitor; timeout 3 python3 scripts/stage123_manifest_monitor.py --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-final-20260715T102742Z/rendered_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-final-20260715T102742Z/state --poll-seconds 30 --ledger /data-2/experiment_registry/stage123_notification_events.jsonl --policy /data-1/code/verl/scripts/experiment_notification_policy.py; tail -n 10 /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-final-20260715T102742Z/state/events.jsonl; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

## Additional Task-Specific Focus

Review F-EX-M2-04 only. Confirm monitor liveness is restored through the same shared persisted-state implementation, monitor remains read-only relative to execution authority, duplicate events are suppressed, and it does not claim completion. Do not edit files, stop training, or change tmux sessions. Return PASS/FAIL.

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
