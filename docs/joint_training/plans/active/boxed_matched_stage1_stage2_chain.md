# Boxed Matched Stage1 -> Stage2 Chain

- Status: ACTIVE EXECUTION PLAN
- Created: 2026-05-31
- Branch: `feature/on-policy-wdl-sft`
- Recipe family: `recipe/on_policy_wdl_sft/staged_v1/`

## Objective

Rerun the staged v1 validation after fixing the train/validation prompt-format
mismatch. Training data now asks for a final answer in `\boxed{}` just like
validation.

Run only matched beta chains:

```text
Stage1 beta=0.0 -> fixed Model2 merge -> Stage2 beta=0.0
Stage1 beta=0.1 -> fixed Model2 merge -> Stage2 beta=0.1
```

Do not run mixed beta chains unless the user explicitly asks for a diagnostic
ablation.

## Primary Scripts

```text
recipe/on_policy_wdl_sft/staged_v1/run_boxed_matched_chain_queue.sh
recipe/on_policy_wdl_sft/staged_v1/monitor_boxed_matched_chain_notify.sh
recipe/on_policy_wdl_sft/staged_v1/merge_stage1_model2_fixed.sh
```

Fixed Model2 output directories:

```text
/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta0-best
/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta01-best
```

The merge writes and checks `stage1_source.json` provenance. Stage2 must not
silently reuse stale merged weights.

## Data

Stage1 train file:

```text
/data-1/dataset/EnsembleLLM-data-processed/staged_v1/train_rl_format_boxed_prompt.parquet
```

Stage2 non-overlap train shard:

```text
/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_boxed_after_s1_150steps_seed20260528_75steps.parquet
```

Verification recorded on 2026-05-31:

- boxed full train: 104,916 rows, user `\boxed` coverage 100%, system
  `<think>/<answer>` coverage 100%;
- boxed Stage2 shard: 4,800 rows, overlap with Stage1 first 9,600 eligible
  prompts is 0.

## Acceptance

Local acceptance requires:

- both Stage1 runs reach 150 steps;
- both fixed Model2 merges exist and have matching provenance;
- both Stage2 runs reach 75 steps;
- Stage2 final metrics are present;
- WxPusher notifications cover start, Stage1 completion, merge completion,
  Stage2 completion, queue completion, and unrecoverable failure;
- no mixed beta chain is used as acceptance evidence.

## Archived Context

The previous staged-v1 planning files are archived:

```text
docs/joint_training/plans/completed/on_policy_sft_then_wdl_sft_beta_search.md
docs/joint_training/plans/completed/stage2_model2_rollout_fused_loss_fast_validation.md
```
