# Workspace Artifact Hygiene

This policy keeps the project workspace and `/data-1` runtime area clean when
coding agents run tests, dry-runs, benchmark harnesses, scripts, or exploratory
commands.

## Why This Exists

On 2026-06-08, the repo root contained many untracked side-effect files created
by coding-agent work: `archive_test*`, `test_*`, `temp_*`, toy CSV/JSON/image
files, SQLite test databases, zip archives, extracted folders, ad-hoc scripts,
and throwaway virtualenvs. These files were not project evidence. They were
created because agents used the repo root as a scratch directory while running
general code-task tests and side-effect-heavy benchmark cases.

The failure mode was not disk usage. The failure mode was loss of trust in the
working tree: useful project files, training evidence, W&B staging data, and
garbage side effects became visually indistinguishable.

## Scope

This policy applies to:

- `/data-1/verl07/verl`, including repo root, docs, scripts, tests, and recipes.
- `/data-1` paths used by this project, including checkpoints, model weights,
  eval outputs, W&B runs, cache roots, Ray temp, and process temp.
- Coding-agent generated files, regardless of whether the current agent or a
  previous agent created them.

## Mandatory Rules

1. Never use the repo root as scratch space.
   - Do not run generated code, benchmark samples, archive/file-system tests,
     image tests, SQLite tests, or toy data scripts with cwd set to the repo
     root.
   - If a tool requires cwd, set it to an explicit scratch directory.

2. All scratch work must live under an explicit scratch root.
   - Default local scratch root:
     `/data-1/tmp/verl_agent_scratch/$USER_OR_AGENT/$TASK_SLUG/`.
   - For one-shot command probes, use a timestamped child directory.
   - Remove the scratch directory at task end unless the user asks to inspect
     it.

3. All durable experiment artifacts must live under a named artifact root.
   - Checkpoints: `/data-1/checkpoints/...`.
   - Model weights and merged model outputs: `/data-1/model_weights/...`.
   - Offline eval outputs: `/data-1/eval_outputs/...`.
   - Dataset conversion outputs: `/data-1/dataset/...`.
   - W&B offline staging: `/data-1/wandb_runs/...` or project-configured
     `WANDB_DIR`.
   - Do not place durable experiment artifacts in repo root.

4. W&B staging is not garbage.
   - `wandb/` and `/data-1/wandb_runs/...` are training evidence and upload
     staging areas.
   - Do not delete them as cleanup unless the exact runs are verified synced
     remotely, local `.wandb.synced` or equivalent evidence exists, and the user
     explicitly approves cleanup.

5. Runtime temp is a separate cleanup class.
   - `/data-1/tmp` and `/data-1/ray_tmp` can contain active sockets, Ray
     sessions, W&B temp, vLLM temp, and Python multiprocessing listeners.
   - Never bulk-delete these directories.
   - Before deleting old children, verify active tmux sessions, processes, Ray
     sessions, docker containers, and open file handles when available.

6. Dry-runs must not dirty project directories.
   - A dry-run may print commands and write a dry-run report only under the
     scratch root or a documented report path.
   - Queue dry-runs must not write permanent repo-local logs/status files unless
     those files are explicitly documented as run evidence.

7. Side-effect-heavy code evaluation must run in an isolated cwd.
   - EvalPlus, BigCodeBench, LiveCodeBench, local execution reward probes, and
     generated code tests may create arbitrary files.
   - Their execution cwd must be inside scratch or the evaluator's own output
     case directory, never the repo root.
   - Prefer `tempfile.TemporaryDirectory` or equivalent cleanup traps for
     per-sample execution.

8. Temporary model merges need cleanup semantics.
   - If a merged model is scratch, write it under a `tmp/` directory and install
     a trap/cleanup path.
   - If a merged model is reusable, write provenance beside it and document its
     retention status in the relevant README, manifest, registry, or training
     script index note.

9. New scripts must expose path roots as variables.
   - Use overridable variables such as `ARTIFACT_ROOT`, `SCRATCH_ROOT`,
     `TMPDIR`, `RAY_TMPDIR`, `WANDB_DIR`, `OUTPUT_ROOT`, `MODEL_WEIGHT_ROOT`,
     `CHECKPOINT_ROOT`, and `REPORT_ROOT`.
   - Do not bury local-only absolute paths inside script bodies when a root
     variable would make the path auditable.

10. Cleanup must be evidence-bound.
    - Do not delete files merely because they are untracked.
    - First classify them as scratch side effects, runtime temp, durable
      experiment evidence, W&B staging, source/docs/tests, or special backup.
    - Preserve special backups when tracked originals are currently deleted or
      moved until the user confirms intent.

## Required Pre-Run Checklist

Before running generated code, code-eval harnesses, local reward probes, dry-run
queues, dataset conversion smoke tests, or ad-hoc scripts:

1. Identify the output class: scratch, durable artifact, report, cache, or W&B
   staging.
2. Set `SCRATCH_ROOT` or `OUTPUT_ROOT`; do not default to repo root.
3. Set `TMPDIR` and `RAY_TMPDIR` for high-churn runtime work.
4. For dry-runs, confirm whether logs/status are evidence or scratch.
5. For code execution tasks, confirm the execution cwd is isolated.

## Required Post-Run Checklist

After the task:

1. Run `git status --short --ignored` from the repo root.
2. Explain any new untracked or ignored paths that remain.
3. Remove scratch directories unless retained for inspection.
4. Leave W&B, checkpoints, model weights, eval outputs, and dataset artifacts in
   place unless the user asked for verified cleanup.
5. If a script created new durable artifacts, update the relevant guide,
   manifest, registry, or training script index.

## Cleanup Classification

Use this table when deciding what to delete.

| Class | Examples | Default action |
| --- | --- | --- |
| Scratch side effects | repo-root `archive_test*`, `test_*`, `temp_*`, toy CSV/JSON/image/db/zip files, extracted toy directories, throwaway venvs | Delete after confirming no references |
| Runtime temp | `/data-1/tmp`, `/data-1/ray_tmp` children | Verify live process state first |
| W&B staging | `wandb/`, `/data-1/wandb_runs/...` | Preserve |
| Durable experiment artifacts | `/data-1/checkpoints`, `/data-1/model_weights`, `/data-1/eval_outputs`, `/data-1/dataset` | Preserve unless cleanup is explicitly requested and verified |
| Project source/docs/tests | `docs/`, `recipe/`, `scripts/`, `tests/`, `platform/hope_code_task` | Preserve unless the task is to edit them |
| Special backup | `archive_Notice.txt`, `archive_requirements*.txt` while originals are deleted | Preserve until intent is confirmed |

## Context Injection Rule

Agents must read this document before:

- running generated code or benchmark samples that may touch the filesystem;
- creating or changing training/eval/data/queue scripts;
- launching dry-runs or smoke tests that write logs, reports, status files, or
  model outputs;
- cleaning workspace files under `/data-1/verl07/verl` or project-owned
  `/data-1` paths;
- diagnosing a dirty working tree.

Agents do not need to read it for pure source edits that do not run commands
with filesystem side effects.
