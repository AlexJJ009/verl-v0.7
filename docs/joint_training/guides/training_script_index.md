# Training Script Index

This file is the collaboration index for runnable training scripts in this
project. Keep it factual and short: script purpose, creation time, last use,
and the current status. Update it whenever a training script is created or
used.

Do not put full launch commands, monitor commands, or operational playbooks in
this index. Those belong in the relevant on-demand guide or workflow, so agents
and collaborators can load run mechanics only when they need them.

Times are local server time (`Asia/Shanghai`, CST).

## WDL Group-Advantage IS

| Script | Purpose | Created | Last used | Status / notes |
|---|---|---:|---:|---|
| `recipe/on_policy_wdl_sft/group_advantage_is/_common_group_adv_is.sh` | Shared launcher for WDL group-advantage IS runs. Owns validation defaults, reward function, checkpoint retention, and Hydra overrides. | 2026-05-21 19:18:47 | 2026-05-22 10:12 | Shared by 1A and 1D; do not edit for one-off dataset changes unless all group-advantage runs should inherit the behavior. |
| `recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh` | Original 1A run on EnsembleLLM train RL format. | 2026-05-21 00:01:47 | 2026-05-21 | Completed 300/300 as `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A_1779295502`; best/latest checkpoint is step 300. |
| `recipe/on_policy_wdl_sft/group_advantage_is/run_1d_group_adv_is_math_train.sh` | 1D run using Hendrycks MATH train converted to verl RL format; validation standards unchanged from 1A. | 2026-05-22 10:03:36 | 2026-05-22 10:20 | First launch `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1D-MATHDATA_1779415976` was stopped during initialization after filtering showed one epoch is 115 steps, not 117. Current launch is `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1D-MATHDATA-E1_1779416149`, with `TOTAL_TRAINING_STEPS=115`, `TOTAL_EPOCHS=1`. |

## Single-Model Ablation / GRPO Baselines

| Script | Purpose | Created | Last used | Status / notes |
|---|---|---:|---:|---|
| `recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh` | Shared launcher for single-model ablations and pure-RL baselines. Owns Meituan-portable paths, validation defaults, checkpoint retention, and Hydra overrides. | 2026-04-23 | 2026-05-24 | Supports `LOSS_MODE=wdl_sft_is`, `wdl_group_adv_is`, `minirl`, and `vanilla`; Meituan path is `platform/hope_ablation` -> `recipe/on_policy_wdl_sft/ablation_single_model/meituan/jupyter.sh`. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_2g_base.sh` | Standard GRPO baseline from Qwen3-4B-Base: `LOSS_MODE=vanilla`, GRPO advantage with std normalization, symmetric clip 0.2/0.2. | 2026-04-23 | 2026-04-23 | Runnable locally and via Meituan `EXPERIMENT=2g-base`; group size is `actor_rollout_ref.rollout.n=8`; KL is disabled (`use_kl_in_reward=False`, `use_kl_loss=False`, coefficients 0). |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_2g_sft.sh` | Standard GRPO baseline from Qwen3-4B-Base-SFT-stage-1 with the same GRPO settings as 2G-BASE. | 2026-04-23 | not verified locally in this audit | Runnable locally and via Meituan `EXPERIMENT=2g-sft` once the SFT model exists on the target host; group size is 8 and KL is disabled. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_2g_math_base.sh` | Standard GRPO baseline from Qwen3-4B-Base trained on Hendrycks MATH train RL-format parquet. | 2026-05-23 | 2026-05-23 13:18 | Completed 115/115 as `GRPO-Qwen3-4B-MATH-2G-MATHDATA-BASE-E1_1779502318`; best checkpoint step 115 by `val-core/HuggingFaceH4/MATH-500/acc/mean@1=0.7479838709677419`. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_2g_math_sft.sh` | Standard GRPO baseline from Qwen3-4B-Base-SFT-stage-1 trained on Hendrycks MATH train RL-format parquet. | 2026-05-23 | 2026-05-23 17:17 | Completed 115/115 as `GRPO-Qwen3-4B-MATH-2G-MATHDATA-SFT-E1_1779513533`; best checkpoint step 100 by `val-core/HuggingFaceH4/MATH-500/acc/mean@1=0.7903225806451613`. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_2h_math_base.sh` | Single-model `wdl_group_adv_is` ablation from Qwen3-4B-Base on the same Hendrycks MATH train RL-format parquet as 1D. | 2026-05-24 | 2026-05-24 13:04 | Completed 115/115 as `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-2H-MATHDATA-BASE-E1_1779587409`; best checkpoint step 100 by `val-core/HuggingFaceH4/MATH-500/acc/mean@3=0.7526881720430108`. |
| `recipe/on_policy_wdl_sft/ablation_single_model/run_2h_math_sft.sh` | Single-model `wdl_group_adv_is` ablation from Qwen3-4B-Base-SFT-stage-1 on the same Hendrycks MATH train RL-format parquet as 1D. | 2026-05-24 | 2026-05-24 20:49 | Completed 115/115 as `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-2H-MATHDATA-SFT-E1_1779612618`; best/latest checkpoint step 115 by `val-core/HuggingFaceH4/MATH-500/acc/mean@3=0.8051075268817204`. |
| `recipe/on_policy_wdl_sft/ablation_single_model/monitor_2h_math_queue.sh` | Host-side queue monitor for the 2H MATH single-model `wdl_group_adv_is` ablations. | 2026-05-24 | 2026-05-24 | Ran Base first, then SFT; uses `/data-1/verl07/run_train.sh` when present, otherwise falls back to `docker run ... verl-harness`; both queued runs reached final step 115. |

## Dataset Preparation Artifacts

| Artifact | Purpose | Created / updated | Notes |
|---|---|---:|---|
| `/data-1/dataset/math/prepare_train_rl_format.py` | Converts downloaded `ck46/hendrycks_math` train parquet into verl RL format. | 2026-05-22 10:06:34 | Stored under the dataset workspace rather than the repo to keep data preparation separate from experiment recipes. |
| `/data-1/dataset/math/train_rl_format.parquet` | Generated MATH train file consumed by `run_1d_group_adv_is_math_train.sh`. | 2026-05-22 10:06:47 | 7500 rows; columns match existing RL format: `data_source`, `ability`, `reward_model`, `prompt`, `split`, `extra_info`; empty `ground_truth` count is 0. |
