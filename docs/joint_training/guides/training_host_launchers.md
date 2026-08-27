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

### GON-35 Pueue acceptance boundary

The qualified GON-35 path is narrower than the general A800 development path.
It pins Recipe candidate `df641ed14d8fb0de9942165ec5d1820d14eeb6a9`, Pueue
group `gpu8` at concurrency one, and image
`ghcr.io/alexjj009/verl-harness@sha256:d380888dc8a10796c7f841e341bd775c2d6500ede539f4ea16bb7bf0de92665d`.
The existing Standard GRPO Bash remains unchanged.

Run `scripts/a800/bootstrap_pueue.sh` on the host, then source the external
`/data_storage/yl_test/lgx/runtime/verl/pueue/pueue.env`. After the exact root
candidate passes targeted checks, CI admission, and independent review, render the
one-shot admission with `scripts/a800/render_gon35_grpo_admission.py`. The
CI admission accepts either a genuinely passing full-CI result or the approved
exact-environment Base/Candidate parity evidence with zero candidate-only
failures and zero shared-failure detail drift. The renderer fails closed on any
candidate, Recipe, image, launcher, mount, payload, or comparison drift; parity
does not claim that full CI passed. Candidate-bound P0, P1, CI-admission, and
zero-finding review JSON evidence must all be present. It writes the runtime
environment, source snapshot, and admission receipt beneath the run's external receipt directory;
it never writes queue state or training artifacts into either repository.

The task-specific `scripts/a800/gon35-bin/verl-dev-run` shim translates only
the admitted run's host output paths to the launcher's existing
`/data-1/outputs/<run-name>` mount and then executes the machine-local
`verl-dev-run`. It does not manage Pueue or add another container boundary.
For linked Git worktrees, the machine-local launcher mounts the shared Git
common directory read-only at its original absolute path. This preserves the
container-side clean-checkout and exact-SHA admission checks without granting
write access to repository metadata.
The GON-35 Math shim also selects `--no-code-overlay`; the A800 machine gate
still applies, but the unrelated code-task overlay cannot make the reviewed
Recipe checkout appear dirty inside the container.
The Recipe submitter remains the sole owner of `pueue add` and its native task
ID. A real submission must use the external values emitted by the renderer:

```bash
source /data_storage/yl_test/lgx/runtime/verl/pueue/pueue.env
source <external-receipt-root>/runtime.env
export PUEUE_GRPO_REPO_ROOT=<exact-clean-GON-35-worktree>
export PUEUE_GRPO_OUTPUT_ROOT=<external-run-output-root>
export PUEUE_GRPO_RECEIPT_ROOT=<external-run-output-root>/receipts
export PUEUE_GRPO_RUNTIME_ENV_FILE=<external-receipt-root>/runtime.env
export PUEUE_GRPO_A800_ADMISSION_RECEIPT=<external-receipt-root>/admission.json
export PUEUE_GRPO_ALLOW_SUBMIT=1
bash recipe/on_policy_wdl_sft/standard_grpo/pueue/submit_math_stage1_grpo.sh
```

All Pueue logs, task state, receipts, W&B files, caches, checkpoints, and
training outputs stay below `/data_storage/yl_test/lgx`. The tmux server socket
used to keep `pueued` detached is ordinary per-user runtime state; Pueue's own
PID, Unix socket, logs, configuration, and state stay below the operation root.
Do not reuse this admission for another candidate, image, run, or queue task.

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
