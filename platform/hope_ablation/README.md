# hope_ablation — hope_dir template for single-model ablation runs

Copy this directory, fill 3 placeholders, submit. One template covers all
single-model ablations, including the standard GRPO baselines `2g-base` /
`2g-sft`, the MATH-train GRPO variants `2g-math-base` / `2g-math-sft`, and
the MATH-train `wdl_group_adv_is` variants `2h-math-base` / `2h-math-sft`.

## Use

```bash
cp -r /data-1/verl07/verl/platform/hope_ablation ~/hope_2g_base
# Edit ~/hope_2g_base/run.hope — replace the 3 REPLACE markers:
#   afo.app.name            → verl-2g-base
#   afo.docker.image.name   → your ~30GB verl v2 image URL
#   afo.app.env.EXPERIMENT  → 2g-base
cd ~/hope_2g_base && hope submit run.hope
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
| 2g-base | Qwen3-4B-Base | vanilla / standard GRPO | — | 5e-7 |
| 2g-sft  | Qwen3-4B-Base-SFT-stage-1 | vanilla / standard GRPO | — | 5e-7 |
| 2g-math-base | Qwen3-4B-Base | vanilla / standard GRPO on MATH train | — | 5e-7 |
| 2g-math-sft  | Qwen3-4B-Base-SFT-stage-1 | vanilla / standard GRPO on MATH train | — | 5e-7 |
| 2h-math-base | Qwen3-4B-Base | wdl_group_adv_is on MATH train | — | 5e-7 |
| 2h-math-sft  | Qwen3-4B-Base-SFT-stage-1 | wdl_group_adv_is on MATH train | — | 5e-7 |

For `2g-*`, the GRPO group size is `actor_rollout_ref.rollout.n=8`.
The current comparison baseline intentionally disables KL:
`algorithm.use_kl_in_reward=False`, `algorithm.kl_ctrl.kl_coef=0.0`,
`actor_rollout_ref.actor.use_kl_loss=False`, and
`actor_rollout_ref.actor.kl_loss_coef=0.0`.
The `2g-math-*` variants use `MATH_TRAIN_FILE`, defaulting on Meituan to
`$LGX/verl-exp/data/math/train_rl_format.parquet`, and default to one filtered
MATH epoch (`TOTAL_TRAINING_STEPS=115`, `TOTAL_EPOCHS=1`).
The `2h-math-*` variants use the same MATH train file and step budget, set
`VAL_N=3`, select best checkpoint by MATH-500 `mean@3`, and disable external
rollout IS weights because `wdl_group_adv_is` owns that term internally.

The `wdl_sft_is` entries (`2a`, `2b`, `2c`) now write to `RUN_PREFIX` values
ending in `-LABELFIX`, so post-2026-04-27 jobs do not auto-resume pre-fix
checkpoints. The Meituan `EXPERIMENT` names stay the same. The `2z` and `2g`
pure-RL baselines are unaffected by that label fix.

-sft experiments need `$LGX/huggingface.co/Qwen/Qwen3-4B-Base-SFT-stage-1/`
uploaded; -base experiments only need the already-present Qwen3-4B-Base dir.
The `2g-math-*` and `2h-math-*` experiments also need the MATH train parquet uploaded under
`$LGX/verl-exp/data/math/`.

## Outputs (all under $LGX/verl-exp/)

- checkpoints: `checkpoints/<RUN_PREFIX>_<timestamp>/global_step_<N>/`
- stdout log:  `logs/ablation_single_model/<RUN_PREFIX>_<timestamp>.log`
- metrics:     `logs/ablation_single_model/metrics/`
- val gens:    `logs/ablation_single_model/validation/<RUN_PREFIX>_<timestamp>/`
- wandb (offline): `wandb_runs/<RUN_PREFIX>_<timestamp>/`
