# hope_on_policy_wdl_sft

Unified Meituan hope_dir for On-Policy WDL-SFT **LABELFIX** reruns.

This directory is the hope submission entry point. It is checked in under
`platform/hope_on_policy_wdl_sft/` so it can be `git pull`'d directly on the
Meituan client host — no need to copy files around. The same template handles
both joint 1X and single-model 2X experiments; the dispatcher in `jupyter.sh`
picks the right `recipe/.../meituan/jupyter.sh` based on `$EXPERIMENT`.

## Files

| File | Purpose |
|---|---|
| `run.hope` | hope job spec template. Holds resource / image / env defaults. |
| `jupyter.sh` | AFO worker entry. Dispatches `1*` to joint, `2*` to ablation. |
| `submit_batch.sh` | **Use this.** Submits N experiments back-to-back, each with its own rendered run.hope and per-experiment log. |

## Quick start (陆晓东's workflow)

```bash
# 1) Pull the latest verl repo on the hope client host:
cd /path/to/your/verl-checkout
git pull

# 2) cd into the hope_dir:
cd platform/hope_on_policy_wdl_sft

# 3) Set the variables that depend on your account (see "Variables to confirm"
#    below). At minimum:
export HOPE_IMAGE='registry-offlinebiz.sankuai.com/.../verl-...:tag'
export LGX='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/<you>/lgx'
export REPO_SUBPATH='<subpath under $LGX where you cloned verl>'

# 4) Submit the joint trio (1A, 1B, 1C) one at a time:
./submit_batch.sh

# Or submit every LABELFIX rerun (joint + ablation):
./submit_batch.sh --all

# Or just one:
./submit_batch.sh 1a
```

`hope run run.hope` packs the cwd into the job, so the script renders a fresh
temp copy of `run.hope` + `jupyter.sh` per experiment, runs `hope run` from
that temp dir, and tears it down. The source template is never modified.

### Smoke first (recommended for the very first submit)

```bash
SMOKE=1 ./submit_batch.sh 1a
```

`SMOKE=1` makes `jupyter.sh` set `TOTAL_TRAINING_STEPS=10`, `SAVE_FREQ=5`,
`TEST_FREQ=5`. Use this once on `1a` to confirm the dispatch chain, image,
and dolphinfs paths are wired correctly before launching real runs.

### Inspect without submitting

```bash
./submit_batch.sh --dry-run 1a 1b 1c
```

Prints the rendered `run.hope` for each experiment and exits — does not call
the `hope` CLI.

## Variables to confirm (per account)

| Variable | Set via | Why it likely needs to change for 陆晓东 |
|---|---|---|
| `HOPE_IMAGE` | env or hand-edit `run.hope`'s `afo.docker.image.name` | The template ships with a `REPLACE_WITH_YOUR_V2_IMAGE_URL` placeholder. Use whatever verl-v2 image your account is approved for. |
| `LGX` | env or `afo.app.env.LGX` in `run.hope` | Default is `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx` — that's 杨丰恺's anchor. Point this at the dolphinfs root where 陆晓东 staged the model weights and dataset. |
| `REPO_SUBPATH` | env or `afo.app.env.REPO_SUBPATH` | Path under `$LGX` where the verl repo is checked out. Default `verl08/verl-v0.7-feature-on-policy-wdl-sft` matches 杨丰恺's layout; pick whatever directory 陆晓东 cloned into. |
| `USERGROUP` | env or `[resource] usergroup` in `run.hope` | If 陆晓东 is in a different usergroup than `hadoop-ai-search`, override. Confirm with the queue admin. |
| `SMOKE` | env | `1` runs only 10 steps. Default unset. |

The script substitutes these into the rendered `run.hope` and refuses to
submit if any `REPLACE_ME` placeholder remains.

## Experiment Names

| EXPERIMENT | Family | Init / model | Loss | β | lr | LABELFIX rerun? | Notes |
|---|---|---|---|---|---|---|---|
| `1a` | joint | Base + SFT-stage-1 | `wdl_sft_is` | 0.0 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-1A-LABELFIX_*` |
| `1b` | joint | Base + SFT-stage-1 | `wdl_sft_is` | 0.1 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-1B-LABELFIX_*` |
| `1c` | joint | Base + SFT-stage-1 | `wdl_sft_is` | 0.0 | 1e-6 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-1C-LABELFIX_*` |
| `2a-base` | single | Qwen3-4B-Base | `wdl_sft_is` | 0.0 | 5e-7 | yes | `WDL-SFT-Qwen3-4B-MATH-2A-BASE-LABELFIX_*` |
| `2a-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.0 | 5e-7 | yes | `WDL-SFT-Qwen3-4B-MATH-2A-SFT-LABELFIX_*` |
| `2b-base` | single | Qwen3-4B-Base | `wdl_sft_is` | 0.1 | 5e-7 | yes | `WDL-SFT-Qwen3-4B-MATH-2B-BASE-LABELFIX_*` |
| `2b-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.1 | 5e-7 | yes | `WDL-SFT-Qwen3-4B-MATH-2B-SFT-LABELFIX_*` |
| `2c-base` | single | Qwen3-4B-Base | `wdl_sft_is` | 0.0 | 1e-6 | yes | `WDL-SFT-Qwen3-4B-MATH-2C-BASE-LABELFIX_*` |
| `2c-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.0 | 1e-6 | yes | `WDL-SFT-Qwen3-4B-MATH-2C-SFT-LABELFIX_*` |
| `2z-base` | single | Qwen3-4B-Base | `minirl` | - | 5e-7 | optional | Baseline unaffected by label bug |
| `2z-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `minirl` | - | 5e-7 | optional | Baseline unaffected by label bug |
| `2g-base` | single | Qwen3-4B-Base | `vanilla` | - | 5e-7 | optional | Baseline unaffected by label bug |
| `2g-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `vanilla` | - | 5e-7 | optional | Baseline unaffected by label bug |

## How the dispatch chain resolves

```
platform/hope_on_policy_wdl_sft/jupyter.sh         (hope worker entry)
  └── case $EXPERIMENT in
        1a|1b|1c) → recipe/on_policy_wdl_sft/meituan/jupyter.sh
                    └── runs run_on_policy_wdl_sft_qwen3_4b_math_${exp}.sh
        2*-base|2*-sft) → recipe/on_policy_wdl_sft/ablation_single_model/meituan/jupyter.sh
                    └── runs ablation_single_model/run_${exp//-/_}.sh
```

Both adapters source their respective `meituan/env.sh` first, which sets:

- `HF_HOME`, `BASE_CKPT_DIR`, `WANDB_DIR`, `LOG_DIR` → `$LGX/verl-exp/...`
- `MEITUAN_BASE_MODEL_PATH`, `MEITUAN_SFT_MODEL_PATH` → flat dirs under `$LGX`
- `TRAIN_FILE`, `TEST_FILES` → `$LGX/verl-exp/data/...`
- `WANDB_MODE=offline`, `MAX_ACTOR_CKPTS_TO_KEEP=1`, best-checkpoint vars

The 1X joint run also auto-prepares the joint model under
`$LGX/verl-exp/models/QwenJoint-4B-WDL-SFT-...` if the directory is missing.

## LABELFIX guarantee

The 1A/1B/1C run scripts default `RUN_PREFIX=...-LABELFIX`. The auto-resume
logic searches `$BASE_CKPT_DIR` for `${RUN_PREFIX}_*` — so even if old
`WDL-SFT-Qwen3-4B-MATH-1A_*` checkpoints from the pre-fix runs are still on
dolphinfs, they will not match and will not be resumed.

If you want a clean slate, you can also `rm -rf
$LGX/verl-exp/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A-LABELFIX_*` before resubmit.

## Logs

`submit_batch.sh` writes one log per experiment to:

```
platform/hope_on_policy_wdl_sft/submission_logs/<timestamp>_<exp>.log
```

Override with `LOG_DIR_LOCAL=/some/dir`. Job IDs are auto-extracted with a
best-effort regex (`application_*`, `job id: *`, `app id: *`); if the regex
misses, search the log file by hand.

The actual training logs (per-step, per-batch) land on dolphinfs at
`$LGX/verl-exp/logs/...` per the recipe-level `LOG_DIR` defaults.
