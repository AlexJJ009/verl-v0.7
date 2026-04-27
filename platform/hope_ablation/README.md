# hope_ablation — hope_dir template for single-model ablation runs

Copy this directory, fill 3 placeholders, submit. One template covers all 8
experiments (2A/2B/2C/2Z × BASE/SFT).

## Use

```bash
cp -r /data-1/verl07/verl/platform/hope_ablation ~/hope_2z_base
# Edit ~/hope_2z_base/run.hope — replace the 3 REPLACE markers:
#   afo.app.name            → verl-2z-base
#   afo.docker.image.name   → your ~30GB verl v2 image URL
#   afo.app.env.EXPERIMENT  → 2z-base
cd ~/hope_2z_base && hope submit run.hope
```

Full 300-step run, eval every 25 steps, keeps up to 13 actor checkpoints.

## Optional: SMOKE mode (10 steps, save every 5)

Add this to run.hope `[others]` section if you want a sanity pass:

```
afo.app.env.SMOKE = 1
```

## Experiment → init model cheat sheet

| EXPERIMENT | init model (under $LGX/huggingface.co/Qwen/) | loss | β | lr |
|---|---|---|---|---|
| 2a-base | Qwen3-4B-Base | wdl_sft_is | 0.0 | 5e-7 |
| 2a-sft  | Qwen3-4B-Base-SFT-stage-1 | wdl_sft_is | 0.0 | 5e-7 |
| 2b-base | Qwen3-4B-Base | wdl_sft_is | 0.1 | 5e-7 |
| 2b-sft  | Qwen3-4B-Base-SFT-stage-1 | wdl_sft_is | 0.1 | 5e-7 |
| 2c-base | Qwen3-4B-Base | wdl_sft_is | 0.0 | 1e-6 |
| 2c-sft  | Qwen3-4B-Base-SFT-stage-1 | wdl_sft_is | 0.0 | 1e-6 |
| 2z-base | Qwen3-4B-Base | minirl | — | 5e-7 |
| 2z-sft  | Qwen3-4B-Base-SFT-stage-1 | minirl | — | 5e-7 |

-sft experiments need `$LGX/huggingface.co/Qwen/Qwen3-4B-Base-SFT-stage-1/`
uploaded; -base experiments only need the already-present Qwen3-4B-Base dir.

## Outputs (all under $LGX/verl-exp/)

- checkpoints: `checkpoints/<RUN_PREFIX>_<timestamp>/global_step_<N>/`
- stdout log:  `logs/ablation_single_model/<RUN_PREFIX>_<timestamp>.log`
- metrics:     `logs/ablation_single_model/metrics/`
- val gens:    `logs/ablation_single_model/validation/<RUN_PREFIX>_<timestamp>/`
- wandb (offline): `wandb_runs/<RUN_PREFIX>_<timestamp>/`
