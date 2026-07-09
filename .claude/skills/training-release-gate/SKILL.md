---
name: training-release-gate
description: Enforce the project release gate before publishing training results. Use when a training run finishes, fails, is monitored to completion, or when the user asks to write experiment results to the local database, upload/sync W&B logs, release training results, or spawn DB/W&B subagents after training.
---

# Training Release Gate

This skill prevents failed or incomplete training attempts from being published
as trusted experiment results.

## Required Sources

Read these before acting:

1. `docs/joint_training/constraints/experiment_tracking/training_result_release_gate_policy.md`
2. the active runbook for the queue, usually
   `docs/joint_training/guides/code_task_monitor_agent_runbook.md`

## Gate Command

Use the exact training run name, including timestamp suffix:

```bash
python3 scripts/training_result_release_gate.py check --run-name <RUN_NAME>
```

Interpretation:

- exit `0`: DB import or W&B cloud sync may proceed for that successful run;
- exit `1`: blocked or pending; stop and report the evidence;
- exit `2`: invalid input/state; fix the gate state before publishing.

Do not replace this with LLM judgement, W&B metadata, or a Codex thread note.

## Workflow

1. Identify the exact run name, checkpoint path, metrics path, and experiment
   family.
2. Run the gate check.
3. If blocked, do not write the registry and do not upload W&B. Failed-run
   logs may be inspected only as local diagnostic evidence.
4. If allowed and the user requested publication, spawn separate subagents:
   one `registry-import` worker for database import and one `wandb-sync` worker
   for W&B sync/upload. Each subagent must run the same gate check before
   changing state.
5. After publication, report the DB path, W&B target, run name, and evidence.

## Subagent Contracts

`registry-import` worker:

- run the gate check for every run it will import;
- write only releasable successful runs to the registry;
- report DB path, inserted/updated IDs, and skipped blocked runs.

`wandb-sync` worker:

- run the gate check for every run it will upload;
- sync only releasable successful runs to W&B cloud;
- preserve local offline W&B staging and report remote run URLs or sync errors.

## Monitor Hook

`scripts/training_queue_monitor.sh` supports a default-off
`TRAINING_RELEASE_SUCCESS_HOOK`. The hook runs only after a run reaches the
final checkpoint and passes the release gate. Treat the hook as an automation
entry point, not as permission to skip the subagent/script gate checks above.

## Boundaries

- Local offline W&B logging during training is allowed; W&B cloud sync is gated.
- Offline evaluation of a failed run is allowed only for diagnosis and must be
  labelled local-only unless the user explicitly overrides the policy.
- OpenAI SDK automation is optional orchestration only. It must not be required
  for the release decision.
