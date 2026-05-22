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

## Dataset Preparation Artifacts

| Artifact | Purpose | Created / updated | Notes |
|---|---|---:|---|
| `/data-1/dataset/math/prepare_train_rl_format.py` | Converts downloaded `ck46/hendrycks_math` train parquet into verl RL format. | 2026-05-22 10:06:34 | Stored under the dataset workspace rather than the repo to keep data preparation separate from experiment recipes. |
| `/data-1/dataset/math/train_rl_format.parquet` | Generated MATH train file consumed by `run_1d_group_adv_is_math_train.sh`. | 2026-05-22 10:06:47 | 7500 rows; columns match existing RL format: `data_source`, `ability`, `reward_model`, `prompt`, `split`, `extra_info`; empty `ground_truth` count is 0. |
