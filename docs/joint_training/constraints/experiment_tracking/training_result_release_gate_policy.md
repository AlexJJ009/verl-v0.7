# Training Result Release Gate Policy

This policy is mandatory for training-result publication in this project.

## Rule

Training evidence may be inspected locally at any time, but database writes and
W&B cloud uploads are release actions. A failed or incomplete training attempt
must not be released directly.

If a training attempt fails, the experiment family is blocked for publication
until a later training attempt reaches the intended final checkpoint and the
required experiment flow is complete. Only the successful completed result is
eligible for database import or W&B upload. The failed attempt remains local
diagnostic evidence.

## Scope

This applies to:

- local experiment registry writes, including scripts under `scripts/import_*`;
- W&B cloud uploads or reconstructed W&B uploads;
- monitor-driven post-training automation;
- code-task, math-task, staged, and future training queues.

Offline evaluation and local debugging may run on failed artifacts only when
the user explicitly asks for diagnosis. Diagnostic results from failed attempts
must be labelled local-only and must not be imported as final experiment
results or uploaded to W&B as successful runs.

## Deterministic Gate

Use `scripts/training_result_release_gate.py` as the local source of truth.
The default state file is:

```text
/data-1/experiment_registry/training_release_gate.jsonl
```

The generic monitor records terminal evidence:

- `success_complete`: the run reached its configured final checkpoint and has
  metrics evidence;
- `failed`: the launched run stopped before the final checkpoint.

The generic monitor may run a `TRAINING_RELEASE_SUCCESS_HOOK` after
`success_complete` is recorded and checked. For any production training queue,
the monitor is responsible for wiring this hook, either directly or through the
queue-specific monitor entry point. A successful full-flow run should therefore
automatically trigger both:

- local experiment-registry import for the exact timestamped run;
- W&B cloud sync for the matching offline run directory;
- post-release verification that the registry row exists and that W&B sync
  left a local `.wandb.synced` / `.synced` marker or equivalent explicit
  evidence.

The hook is a convenience automation point, but for production monitors it is
part of the experiment completion contract. Any DB importer or W&B sync
launched from it must still perform its own gate check before publishing. If
the release hook cannot verify both DB import and W&B sync, the training run may
remain a completed training run, but the experiment release is incomplete and
the monitor must report a release failure instead of treating publication as
done. Failed or incomplete attempts must remain local diagnostic evidence and
must not be imported or uploaded.

Before any release action, run:

```bash
python3 scripts/training_result_release_gate.py check --run-name <RUN_NAME>
```

Exit code `0` means release is allowed. Exit code `1` means release is blocked
or pending. Exit code `2` means the gate input or state is invalid.

Release scripts must not add routine bypass flags for this check. For old runs
that predate the gate, first record source-backed `success_complete` evidence
with the gate script, then run the importer or W&B sync normally.

## Subagent Workflow

Project-level subagents that write the database or upload W&B must perform this
sequence:

1. read this policy and the active runbook;
2. identify the exact run name and experiment family;
3. run the release-gate check;
4. if blocked, stop and report the blocking evidence;
5. if allowed, perform the release action and record what was released.

The subagent may inspect failed-run logs for diagnosis, but it must not publish
those failed-run results.

## OpenAI SDK / Codex Automation Boundary

The release gate is intentionally deterministic and local. It must not depend
on the OpenAI SDK, Codex thread state, W&B network state, or LLM judgement for
correctness.

OpenAI SDK or Codex-native subagents may be used later to orchestrate the
workflow, summarize evidence, or spawn DB/W&B workers. They are optional
automation on top of the gate, not the gate itself.
