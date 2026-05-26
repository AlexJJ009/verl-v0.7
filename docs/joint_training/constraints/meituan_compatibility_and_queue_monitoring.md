# Meituan Compatibility and Queue Monitoring Policy

This is a project-level rule for `docs/joint_training/` experiment work. It is
not branch-specific.

## Required For Every New Training Script

Every runnable training script must be compatible with the Meituan AFO launch
path before it is considered complete.

Required properties:

1. The per-experiment `run_*.sh` remains default-local and
   overridable-everything: every path must use `${VAR:-/data-1/...}` or another
   environment override pattern.
2. The script must be reachable through a Meituan entry path:
   `platform/<family>/jupyter.sh` -> `recipe/.../meituan/jupyter.sh` ->
   `recipe/.../run_*.sh`.
3. The recipe-family `meituan/env.sh` must override every local path consumed by
   the wrapper or common launcher, including data, model, checkpoint, log,
   wandb, cache, Ray tmp, vLLM config, and any family-specific train-file
   variables such as `MATH_TRAIN_FILE`.
4. The Meituan adapter must fail fast for missing init models and train data.
5. New scripts must be registered in the branch-local training script index,
   but the portability rule itself belongs here.

## Queue Monitor Rule

Sequential local training queues must use the project-level generic monitor:

```bash
scripts/training_queue_monitor.sh
```

Experiment-specific monitor files should only define:

- `RUN_PREFIXES`
- `RUN_SCRIPTS`
- `TMUX_NAMES`
- queue-local defaults such as `FINAL_STEP`, `MIN_FREE_GB`, `MAX_GPU_UTIL`,
  `POLL_SEC`, and `LOG_FILE`

They must source `scripts/training_queue_monitor.sh` and call
`training_queue_monitor_main`.

The generic monitor owns the shared behavior:

- wait for disk and GPU resource gates
- launch each run in a dedicated tmux session
- use `/data-1/verl07/run_train.sh` when available
- fall back to `docker run ... verl-harness`
- detect completion from `global_step_<N>` or `latest_checkpointed_iteration.txt`
- stop the queue if a tmux session exits before the required final step

Meituan jobs are submitted as separate AFO jobs rather than tmux queue sessions.
The same `run_*.sh` scripts must still be reachable through the Meituan entry
path.
