# Meituan Platform Playbook — Layered Launch & Cross-Host Compatibility

How experiments run on the Meituan AFO / Codelab platform, and how to add a new
one without breaking local-box reproducibility. **Every new runnable training
script in this project must be compatible with the Meituan launch path before it
is considered complete.** Even when the first launch is local, write the
`run_*.sh` by the rules here from day one and add or extend the relevant
Meituan adapter in the same change.

## The two-hostile-environments problem

Every experiment must run in **two places**:

- **Local dev box** (`/data-1/verl07/verl`): paths under `/data-1/`, direct
  disk, fast iteration.
- **Meituan AFO worker** (container + dolphinfs mount): paths under
  `/mnt/dolphinfs/…/lgx/`, network FS for persistent state, container-local
  `/tmp` for churn.

Any script that hard-codes `/data-1/…` fails on Meituan. Any script that
hard-codes `/mnt/dolphinfs/…` fails locally. The layered design below keeps
**every `run_2X_*.sh` portable**: same 15-line script, both hosts.

## The four layers

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Codelab `hope_dir` (one per submission)                 │
│    platform/hope_ablation/ → copy to ~/hope_2z_base/               │
│      run.hope       — platform config + `EXPERIMENT=2z-base`       │
│      jupyter.sh     — same copy for every experiment (thin shim)   │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ `hope submit` → MLP stages hope_dir → /workdir
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Platform shim (per family, not per experiment)          │
│    platform/hope_ablation/jupyter.sh                               │
│      • reads EXPERIMENT, SMOKE from hope env                       │
│      • locates the repo on dolphinfs                               │
│      • `exec bash $REPO/recipe/.../meituan/jupyter.sh`             │
│    It does NOT know what experiments exist. Never changes          │
│    when you add a new run_2X_*.sh.                                 │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ exec
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Recipe-family Meituan adapter (per family)              │
│    recipe/.../meituan/                                             │
│      env.sh        — override ALL paths to dolphinfs equivalents   │
│      jupyter.sh    — pick init model by EXPERIMENT suffix;         │
│                       resolve run_${EXPERIMENT//-/_}.sh; exec it   │
│    Knows the family (e.g. "ablation_single_model") but not each    │
│    experiment's hyperparameters.                                   │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ exec
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│  Layer 4 — Per-experiment wrapper (THE ONLY FILE YOU ADD)          │
│    recipe/.../run_2X_*.sh   (≈ 15 lines)                           │
│      sets: RUN_PREFIX, LOSS_MODE, WDL_SFT_BETA, LR                 │
│      INIT_MODEL_PATH default = LOCAL path (overridable)            │
│      source _common_*.sh                                           │
│    Portable: runs locally as-is; on Meituan, layer 3 overrides     │
│    every path via env before this script is exec'd.                │
└────────────────────────────────────────────────────────────────────┘
```

## How the four layers stay portable

The trick is **default-local, overridable-everything**:

- Every path in `run_2X_*.sh` uses `${VAR:-local-default}`. When Meituan's
  layer 3 has already exported `VAR`, the default is skipped.
- `_common_ablation.sh` same pattern: `TRAIN_FILE`, `TEST_FILES`,
  `BASE_CKPT_DIR`, `WANDB_DIR`, `HF_HOME`, `LOG_DIR`, `INIT_MODEL_PATH`,
  `MAX_ACTOR_CKPTS_TO_KEEP` — all `${VAR:-/data-1/default}`.
- Layer-3 `env.sh` pre-exports all of them to dolphinfs paths **before**
  layer 4 is exec'd.

Running locally, layer 3 is never invoked → defaults apply → local paths.
Running on Meituan, layer 3 overrides them → same script lands on dolphinfs.
Zero duplication, zero branching per host.

## Adding a new experiment — three cases

### Case A — New variant in an existing family (most common)

Example: add `2d-base` = Qwen3-4B-Base + `wdl_sft_is` + β=0.2 + lr=5e-7.

**You change exactly one file.**

```bash
cp recipe/on_policy_wdl_sft/ablation_single_model/run_2b_base.sh \
   recipe/on_policy_wdl_sft/ablation_single_model/run_2d_base.sh
# edit run_2d_base.sh — change 4 fields:
#   RUN_PREFIX       → WDL-SFT-Qwen3-4B-MATH-2D-BASE
#   LOSS_MODE        → wdl_sft_is  (if different from source)
#   WDL_SFT_BETA     → 0.2
#   LR               → 5e-7
```

You do NOT touch:
- `platform/hope_ablation/jupyter.sh` — it derives the run script name from
  `EXPERIMENT` via `run_${EXPERIMENT//-/_}.sh`.
- `recipe/.../meituan/jupyter.sh` — it routes by suffix (`-base` / `-sft`),
  not by enumerated names.
- `_common_ablation.sh` — already handles all supported `LOSS_MODE`s.

Submit on Meituan: copy `platform/hope_ablation/` to `~/hope_2d_base/`,
set `EXPERIMENT=2d-base` in `run.hope`, `hope submit run.hope`.

Run locally: `bash recipe/.../ablation_single_model/run_2d_base.sh`.

### Case B — New init-model suffix (not `-base` / `-sft`)

Example: `2a-distill` initialized from a distillation checkpoint.

Layer-3 `meituan/jupyter.sh` currently only recognizes `*-base` / `*-sft`.
Extend it:

1. In `recipe/.../meituan/env.sh`, add a new path variable:
   ```bash
   export MEITUAN_DISTILL_MODEL_PATH=${MEITUAN_DISTILL_MODEL_PATH:-$LGX/.../distill_ckpt}
   ```
2. In `recipe/.../meituan/jupyter.sh`, extend the suffix `case`:
   ```bash
   case "$EXPERIMENT" in
       *-base)    export INIT_MODEL_PATH="$MEITUAN_BASE_MODEL_PATH" ;;
       *-sft)     export INIT_MODEL_PATH="$MEITUAN_SFT_MODEL_PATH" ;;
       *-distill) export INIT_MODEL_PATH="$MEITUAN_DISTILL_MODEL_PATH" ;;
       …
   esac
   ```
3. Then follow Case A to add `run_2a_distill.sh`.

Layer 1 / 2 still untouched.

### Case C — Brand-new recipe family

Example: start `recipe/joint_training/ablation_something/`.

Clone the layout:

1. `recipe/joint_training/ablation_something/meituan/{env.sh,jupyter.sh}` —
   copy from `ablation_single_model/meituan/`, adapt path variables and the
   EXPERIMENT-suffix case for that family's init models.
2. `platform/hope_something/{jupyter.sh,run.hope,README.md}` — copy from
   `platform/hope_ablation/`, repoint `exec bash "$REPO/recipe/.../meituan/jupyter.sh"`
   at the new family.
3. Every experiment still goes in a 15-line `run_*.sh` (Case A).

## File-count rules (so `jupyter.sh` files don't proliferate)

| Layer                                    | Count                              | When to create                   |
| ---------------------------------------- | ---------------------------------- | -------------------------------- |
| `platform/<family>/jupyter.sh`           | **1 per family**                   | New family (Case C)              |
| `platform/<family>/run.hope`             | **1 per family** (template)        | New family (Case C)              |
| `recipe/.../meituan/jupyter.sh`          | **1 per family**                   | New family (Case C)              |
| `recipe/.../meituan/env.sh`              | **1 per family**                   | New family (Case C)              |
| `recipe/.../run_*.sh`                    | **1 per experiment**               | Every new experiment (Case A)    |
| `recipe/.../_common_*.sh`                | **1 per family**                   | Almost never changes             |

Concern "too many `jupyter.sh` to maintain" is unfounded: there are only two,
both per-family. Only `run_*.sh` grows with experiments, and each is 15 lines.

## Rules for writing a portable `run_*.sh`

Follow these from day one, even for local-only experiments. Migrating to
Meituan later becomes zero-cost.

1. **Default every path**: `export VAR=${VAR:-"/data-1/…"}`. Never
   unconditionally assign a local absolute path.
2. **Never hard-code dolphinfs paths.** Those live in `meituan/env.sh` only.
3. **Default `INIT_MODEL_PATH` to the local path.** Layer-3 jupyter.sh will
   override it based on the EXPERIMENT suffix before your script runs.
4. **Export, then source.** All knobs must be `export`-ed — `_common_*.sh`
   reads them from env, not as arguments.
5. **Wrapper stays thin.** Anything reusable across experiments belongs in
   `_common_*.sh`. `run_*.sh` should be only: `export` lines + `source`.
6. **Wire the Meituan entry now.** If the family already has
   `recipe/.../meituan/{env.sh,jupyter.sh}`, update it and the platform
   dispatcher in the same change. If the family is new, create the layer-3
   adapter and a platform route before marking the training script complete.

## Rules for writing a family's `meituan/env.sh`

1. **Override everything the local default assumes.** Any `${VAR:-/data-1/…}`
   in `run_*.sh` or `_common_*.sh` needs a dolphinfs-side `export VAR=…` here.
2. **High-churn temp dirs → container-local `/tmp`**, not dolphinfs. Ray tmp,
   vLLM config, ZMQ IPC dirs, `TMPDIR`. Network-FS latency on thousands of
   small files kills throughput. Dolphinfs is only for *persistent* state
   (checkpoints, logs, wandb offline dirs).
3. **Pre-create dolphinfs directories.** `mkdir -p "$HF_HOME" "$BASE_CKPT_DIR" …`
   at the end of env.sh — first-write on a new pod otherwise races with
   training setup.
4. **Use flat directory layout for model weights, NOT HF cache layout.**
   Dolphinfs disallows BOTH symlinks and hardlinks (`Operation not permitted`),
   so the HF cache layout (`models--<org>--<name>/snapshots/<hash>/<file>` →
   `../../blobs/<sha>`) cannot be materialized. Point `MEITUAN_*_MODEL_PATH`
   directly at a flat dir of real files. See "Trap: dolphinfs FS limitations"
   below for the upload workflow.
5. **Idempotent.** Safe to source multiple times — every variable is guarded
   by `${VAR:-…}` so re-sourcing doesn't clobber an explicit user override.
6. **`WANDB_MODE=offline` on Meituan.** Worker pods have no outbound internet.

## Rules for the per-family `meituan/jupyter.sh`

1. **Resolve exactly two things: init model + run script.** Everything else
   (hyperparameters, training command) is out of scope.
2. **Route by EXPERIMENT *suffix*, not full name.** `*-base | *-sft | …`
   scales to all future experiments of the same init kind without edits.
3. **Validate prerequisites fast, fail loud.** Check `[ -d "$INIT_MODEL_PATH" ]`
   and `[ -f "$TRAIN_FILE" ]` before launching — a 2-second `ENOENT` beats a
   15-minute Ray timeout.
4. **`exec bash "$RUN_SCRIPT"`** — hand off process identity so logs stay
   attached and failures propagate to AFO correctly.
5. **`cd "$REPO_ROOT"` before exec.** Some verl modules resolve paths
   relative to cwd (Hydra, reward modules).

## Rules for the `platform/<family>/jupyter.sh` shim

1. Stay **family-level**, not experiment-level. Never `case "$EXPERIMENT"`
   here — that belongs in layer 3.
2. Do three things only: **require EXPERIMENT, locate the repo, exec layer 3.**
3. Respect `SMOKE=1` — set `TOTAL_TRAINING_STEPS=10` and `SAVE_FREQ=5` for a
   sanity pass. Matches SMOKE semantics across all families.

## Migration checklist — porting a local-only recipe family to Meituan

Use this when an experiment family currently only runs locally and you've
decided to run it on Meituan:

1. **Audit `run_*.sh` and `_common_*.sh`**: every hard-coded `/data-1/…`
   becomes `${VAR:-/data-1/…}`. Every new `VAR` gets noted for env.sh.
2. **Create `recipe/<family>/meituan/env.sh`**: export every `VAR` from step 1
   to its dolphinfs equivalent. Add `TMPDIR` / `RAY_TMPDIR` / `HF_HOME`
   overrides. Add `mkdir -p` at the end.
3. **Create `recipe/<family>/meituan/jupyter.sh`**: model-suffix case,
   prerequisite validation, `exec bash $RUN_SCRIPT`.
4. **Create `platform/<family-tag>/`**: `jupyter.sh` shim + `run.hope`
   template + `README.md` cheat sheet.
5. **Upload data** to `$LGX/verl-exp/data/…` via `rsync -avP` (see
   `HF_HUB_UPLOAD` memory / env.sh comments).
6. **Upload init models** to `$LGX/huggingface.co/Qwen/…`. For private SFT
   models, use the `hf` CLI workflow.
7. **Sanity submit**: `EXPERIMENT=<any-variant>` + `SMOKE=1` → 10-step run.
   Confirms image, paths, model, data, logging all land.
8. **Full submit**: drop `SMOKE`, monitor first checkpoint + first eval.

## Trap: dolphinfs FS limitations (no symlinks, no hardlinks)

Dolphinfs rejects both `ln -s` and `ln` with `Operation not permitted`. This
breaks two common workflows you might reach for by reflex:

1. **HF cache layout doesn't work.** `hf download` (without `--local-dir`)
   creates `models--<org>--<name>/blobs/<sha256>` plus
   `snapshots/<hash>/<file>` → symlink → `../../blobs/<sha256>`. The blobs
   land fine, but symlinks fail silently or get dropped, leaving an empty
   snapshot dir. Always pass `hf download --local-dir <flat>` instead.

2. **Mac → container drag-and-drop loses symlinks too.** macOS Finder
   archives (Compress / "Archive Utility") and default `zip -r` do not
   preserve symlinks. If you `hf download` on Mac, then zip and drag the
   `models--…/` tree across, the unzipped result has populated `blobs/` but
   empty `snapshots/`. Fix by `mv`-ing each blob to its logical filename
   (size matching is unique enough for a known model; or use HF API to map
   sha → filename). `mv` works on dolphinfs because rename has weaker perms
   than link.

**Recommended upload workflow** (Mac is the only network-capable host):

```bash
# On Mac
hf download <repo> --local-dir ~/<flat-dir>
cd ~ && tar -czf <flat-dir>.tar.gz <flat-dir>
# Drag the tar.gz into the container, extract under $LGX, point env.sh at it.
```

Verify after extraction by sha256-ing one or two of the LFS files against
the `lfs.oid` returned by the HF tree API. The blob file's name in HF cache
layout *is* its content sha256 — match it against the unzipped flat file.

## Trap: image layer vs. mounted repo

`platform/.../jupyter.sh` and everything under `recipe/.../` are read from
**the dolphinfs-mounted repo**, not from the container image. This means:

- `git pull` on the dolphinfs repo is a **live update** — next job picks it up
  without rebuilding the image.
- Image rebuilds are only needed for OS-level / Python-package changes.
- `trl` removal and similar pip-level fixes **do** require an image rebuild,
  unless the code can be patched to tolerate the offending import (see the
  `AutoModelForCausalLMWithValueHead` fix in `verl/models/transformers/monkey_patch.py`).

When in doubt: if the change is in `.sh` / `.py` inside the repo, `git pull`
is enough. If the change is `pip install` / `apt` / base image, rebuild.

## Trap: one-time image rebuilds on Meituan

On Meituan, rebuilding an image is a manual, throttled operation — usually
one shot per iteration cycle. Two consequences:

- **Prefer code-level fixes over pip-level fixes** when the same outcome is
  achievable. The `trl` incident was fixed in-code with a try/except rather
  than uninstalling the package from the image.
- **Batch image-layer changes.** When you do rebuild, include every pending
  Dockerfile change you know about — new pip packages, OS libs, vLLM upgrades
  — instead of one per rebuild.

## Reference files

- `platform/hope_ablation/` — layer 1 template (hope_dir + shim + README)
- `platform/hope_ablation/jupyter.sh` — layer 2 shim
- `platform/hope_on_policy_wdl_sft/jupyter.sh` — unified layer 2 dispatcher for
  the on-policy WDL-SFT project families, including 4A/4B/4C
- `recipe/on_policy_wdl_sft/ablation_single_model/meituan/jupyter.sh` —
  layer 3 example
- `recipe/on_policy_wdl_sft/ablation_single_model/meituan/env.sh` —
  layer 3 env example (dolphinfs overrides)
- `recipe/on_policy_wdl_sft/dual_submodel_rollout/meituan/jupyter.sh` —
  layer 3 example for the dual-rollout family
- `recipe/on_policy_wdl_sft/ablation_single_model/run_2z_base.sh` —
  layer 4 example (15-line portable wrapper)
- `recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh` —
  shared launcher (never edited per-experiment)
