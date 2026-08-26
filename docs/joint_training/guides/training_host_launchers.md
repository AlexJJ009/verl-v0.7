# Training Host Launcher Guide

Use this guide before running Docker-backed VERL commands on the local L40S
server or the A800 Kubernetes development instance. The recipe-level
`run_*.sh` files stay host-neutral; the outer host launcher is selected here.
Meituan AFO remains a separate launch path documented in
`docs/joint_training/guides/meituan_platform.md`.

## Host selection

| Host | Host launcher | Repository inside container | Intended use |
|---|---|---|---|
| L40S server | `/data-1/verl07/run_train.sh` | `/workspace/verl` | Existing calibrated L40S training and evaluation |
| A800 development instance | `verl-dev-run` (installed at `/data_storage/yl_test/lgx/home/.local/bin/verl-dev-run`) | `/workspace/verl` | Development, dry-runs, evaluation, and A800-specific qualification work |

Do not call `/data-1/verl07/run_train.sh` on the A800 host and do not create a
host-level `/data-1` compatibility symlink there. The A800 launcher maps the
operation directory into the legacy `/data-1` and `/data-2` container paths so
recipe scripts do not need host-specific path edits.

## L40S launch path

The L40S launcher is repository-owned at `scripts/l40s/run_train.sh` and is
normally installed at `/data-1/verl07/run_train.sh`.

```bash
# Preflight / interactive shell
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh bash

# Run one recipe wrapper
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh \
  bash recipe/<family>/run_<experiment>.sh
```

If the checkout really is `/data-1/verl07/verl`, `REPO_HOST` may be omitted.
For any branch worktree, set `REPO_HOST` to that worktree explicitly rather
than relying on the default.

## A800 development-instance launch path

The A800 launcher is machine-local bootstrap state, not a repository script:

```bash
command -v verl-dev-run
verl-dev-run --doctor
verl-dev-run --a800-dev-profile --doctor
```

The first doctor command checks the Docker image, mounts, datasets, model
seeds, writable cache/artifact directories, root-layer read-only policy, and
CUDA visibility. The second also checks the eight-A800 development profile.
Run both after the instance or container environment has been rebuilt.

```bash
# Interactive shell using the main checkout
verl-dev-run --a800-dev-profile

# Run one recipe wrapper
verl-dev-run --a800-dev-profile -- \
  bash recipe/<family>/run_<experiment>.sh

# Select one of the installed branch worktrees
verl-dev-run --worktree dynperm --a800-dev-profile
verl-dev-run --worktree rebuttal --a800-dev-profile
```

The default repository mount and container root are read-only. Use
`--repo-rw` or `--mutable-root` only for a command that has an explicit need;
normal training must write through the mapped artifact and cache paths.

### A800 storage contract

The launcher keeps mutable state below `/data_storage/yl_test/lgx`:

- Hugging Face, Torch, Triton, CUDA, pip, and vLLM caches:
  `/data_storage/yl_test/lgx/data-1/.cache/`
- checkpoints, evaluation output, logs, generic output, and W&B runs:
  `/data_storage/yl_test/lgx/artifacts/verl/`
- `/tmp`, `/var/tmp`, Ray state/spill, and the A800 profile overlay:
  `/data_storage/yl_test/lgx/runtime/verl/`

The default `/tmp` is persistent under the operation directory. `--fast-tmp`
switches it to an ephemeral 32 GiB tmpfs and is only appropriate when losing
temporary files at container exit is acceptable.

### A800 admission boundary

`--a800-dev-profile` supplies a distinct, development-only A800 resource
profile. It intentionally has a different profile name and hash from the L40S
profile. It does not make an old L40S calibration, machine-evidence bundle, or
formal admission valid on A800. Before a formal A800 training campaign, create
new A800 calibration evidence and a new manifest/admission bound to that
profile.

## Queue and tmux rules

Start every long-running launcher command inside tmux on the host:

```bash
tmux new-session -s <run-name>
# invoke the host-appropriate launcher here
```

Legacy host-level queues that directly execute
`/data-1/verl07/run_train.sh` are L40S-specific. They cannot be run unchanged
on A800 because nesting that launcher inside `verl-dev-run` would attempt to
start Docker from inside Docker. On A800, launch the recipe-level wrapper
through `verl-dev-run`, or add an explicit A800 queue adapter before using that
queue. Never bypass this distinction with path aliases.
