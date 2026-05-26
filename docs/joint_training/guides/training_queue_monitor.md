# Training Queue Monitor

Use `scripts/training_queue_monitor.sh` for sequential local training queues.
This is the project-level monitor implementation; experiment-specific monitors
should be thin wrappers that only define the queue.

## Wrapper Pattern

```bash
RUN_PREFIXES=(
    "EXPERIMENT-A"
    "EXPERIMENT-B"
)
RUN_SCRIPTS=(
    "/workspace/verl/recipe/.../run_a.sh"
    "/workspace/verl/recipe/.../run_b.sh"
)
TMUX_NAMES=(
    "experiment_a"
    "experiment_b"
)

REPO_HOST=/data-1/verl07/verl
REPO_CONTAINER=/workspace/verl
FINAL_STEP=115
LOG_FILE=/data-1/verl07/verl/logs/my_queue.log

source "$REPO_HOST/scripts/training_queue_monitor.sh"
training_queue_monitor_main
```

## Behavior

The generic monitor:

- waits for disk free-space and GPU-utilization gates before each launch
- starts each run in a dedicated tmux session
- launches through `/data-1/verl07/run_train.sh` when present
- falls back to `docker run ... verl-harness` when the host launcher is absent
- detects completion from `global_step_<FINAL_STEP>` or
  `latest_checkpointed_iteration.txt`
- stops the queue if a training tmux exits before the final step

## Current 4ABC Queue

The current 4ABC queue wrapper is:

```bash
recipe/on_policy_wdl_sft/dual_submodel_rollout/monitor_4abc_math_queue.sh
```

It queues:

1. `run_4a_model2_group_adv_is.sh`
2. `run_4b_math_base.sh`
3. `run_4c_math_sft.sh`

All three are configured for `FINAL_STEP=115`.

## Meituan Boundary

Meituan jobs are submitted as separate AFO jobs, not as tmux queue sessions.
The same training scripts must still be reachable through the Meituan entry
path documented in `docs/joint_training/guides/meituan_platform.md`.
