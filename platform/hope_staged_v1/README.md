# Meituan AFO: staged v1 Stage 1 On-Policy SFT beta search

This template dispatches to:

```text
recipe/on_policy_wdl_sft/staged_v1/meituan/jupyter.sh
```

Supported current-scope `EXPERIMENT` values:

```text
s1-beta-0
s1-beta-01
s1-beta-02
s1-beta-03
s1-beta-04
s1-beta-05
s1-beta-06
s1-beta-07
s1-beta-08
s1-beta-09
s1-beta-10
```

`s1-base-sft` is kept as a manual alias for direct Stage 1 smoke runs. Stage 2 is deferred and is not launched by this template's default submit helper.

Runs default to:

```text
WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1
WANDB_MODE=offline
TOTAL_TRAINING_STEPS=150
VAL_N=3
VAL_BEFORE_TRAIN=False
```

Sync W&B offline runs after training finishes using:

```bash
WANDB_SYNC_DIR=/path/to/wandb/offline-run-* bash recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh
```
