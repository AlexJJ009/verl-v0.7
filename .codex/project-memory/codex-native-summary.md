# Codex Native Memory Summary

Summarized on 2026-04-27 from project-relevant hits in `~/.codex/memories/`.
This file imports durable lessons only; raw Codex generated memory and rollout
evidence stay in the user-level Codex home.

## Stable Project Lessons

- Treat `/data-1/verl07/verl` as the fixed checkout for this project family,
  not as an interchangeable example path.
- For training launches in this repo, use the Docker wrapper path rather than
  an ad hoc host Python environment. The standard launcher is
  `/data-1/verl07/run_train.sh`, with the repo mounted at
  `/workspace/verl` and `/data-1` mounted into the container.
- Python dependencies for this project live inside the `verl-harness` Docker
  image. Do not run project tests or training with the host Python/conda
  environment. Use Docker, for example:

  ```bash
  docker run --rm --gpus all --ipc=host \
    -v /data-1/verl07/verl:/workspace/verl \
    -v /data-1:/data-1 \
    verl-harness \
    bash -lc 'cd /workspace/verl && pytest tests/joint_training/ -q --tb=short'
  ```

  This is a project-level rule for `/data-1/verl07/verl`; do not apply it
  blindly to unrelated projects.
- Long-running work must run in `tmux`: training scripts, monitoring, checkpoint
  transfers, and large uploads/downloads.
- For follow-on WDL-SFT ablations derived from earlier runs, preserve deliberate
  defaults unless explicitly changed by the user. In the remembered `m5_6`
  follow-up, the important defaults were the `m5_5` schedule and `lr=5e-7`,
  while only reverse SFT was re-enabled with `WDL_SFT_BETA=0.1`.
- Treat run IDs, checkpoint paths, and "currently running" notes from memory as
  stale. Re-check tmux, logs, checkpoint directories, and active plan docs before
  acting on dated status.

## How To Use

Read this file only for history-dependent work in this repo. For current
experiment status, prefer live state plus `CLAUDE.md` / `AGENTS.md` and active
plans under `docs/joint_training/plans/active/`.
