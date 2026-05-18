# Math Stage1 M1 SFT Retrain Reference

Date: 2026-05-18

This note records the local investigation for reproducing a math-related
`stage1_m1` SFT checkpoint that can serve as the WDL-SFT `model2`
initialization when the historical
`/data-1/.cache/Qwen3-4B-Base-SFT-stage-1` weight is unavailable.

This is a reference/resource note, not an active experiment plan. Verify disk
space, GPU availability, and the current source tree before launching a retrain.

## Local Source Tree

The relevant local training code is outside this verl checkout:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM
```

Main Stage1 SFT entrypoints found on this host:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_qwen3_4b_math_m1_repro423.sh
/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_qwen3_4b_math_m1_repro67.sh
/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_qwen3_4b_math_m1_2ep_fixed.sh
/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_ablation_qwen4b_math_m1step.sh
```

For a clean replacement `model2` retrain, prefer a dedicated new output
directory rather than writing into the existing `weights/llmboost_ablation`
directories.

## Recommended Launch Shape

Use the 1-epoch Stage1 reproduction script as the closest local template:

```bash
cd /root/buaa/czh_rl_icml_2026/EnsembleLLM
bash scripts/run_qwen3_4b_math_m1_repro423.sh
```

The core command pattern is:

```bash
accelerate launch --config_file=scripts/accelerate_config.yaml llmboost_train.py \
  --stage 1 \
  --model-name Qwen/Qwen3-4B-Base \
  --stage1-data-path /root/buaa/czh_rl_icml_2026/dataset/am_deepseek_r1_filtered_ad.jsonl \
  --data-files /root/buaa/czh_rl_icml_2026/dataset/am_deepseek_r1_filtered_ad.jsonl \
  --output-dir <new-output-parent> \
  --wandb-run-name qwen3_4b_math_stage1_m1_retrain \
  --per-device-train-batch-size 1 \
  --grad-accum 64 \
  --max-seq-length 4096 \
  --use-chat-template True \
  --stage1-num-epochs 1 \
  --stage1-save-steps 25 \
  --lr 1e-5 \
  --entropy-results ""
```

If only the final epoch checkpoint is needed, omit `--stage1-save-steps`.
With the current Stage1 code, 1 epoch ends at step 218.

## Dataset

Training dataset:

```text
/root/buaa/czh_rl_icml_2026/dataset/am_deepseek_r1_filtered_ad.jsonl
```

Live-checked on 2026-05-18:

```text
111657 samples
2.2G file size
```

The Stage1 log for the fixed run also reported:

```text
Stage1 data loaded: 111657 samples
```

## Training Configuration

Live-checked local scripts and logs use:

```text
base_model: Qwen/Qwen3-4B-Base
distributed: Accelerate + DeepSpeed ZeRO-3
num_processes: 8
mixed_precision: bf16
CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
per_device_train_batch_size: 1
gradient_accumulation_steps: 64
effective_batch_size: 512
max_seq_length: 4096
learning_rate: 1e-5
weight_decay: 0.0
max_grad_norm: 1.0
warmup_ratio: 0.0
use_chat_template: True
stage1_num_epochs: 1 for repro423, 2 for repro67
```

Accelerate config:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/accelerate_config.yaml
```

Key settings:

```text
distributed_type: DEEPSPEED
zero_stage: 3
zero3_save_16bit_model: true
num_processes: 8
mixed_precision: bf16
```

Attention implementation is currently set in:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/utils/utils.py
```

as:

```text
attn_implementation="flash_attention_2"
```

## Runtime and Storage Estimate

Measured from:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/TrainLogs/qwen3_4b_math_m1_2ep_fixed_20260427_100205.log
```

The 8-GPU, `max_seq_length=4096`, 1-epoch run completed:

```text
218 steps
train_runtime: 15747.7024s
wall-clock: about 4h22m
step time: about 71-72s
```

Practical estimate:

```text
1 epoch / checkpoint-218: about 4.4 hours
2 epochs / checkpoint-436: about 8.8 hours
```

Storage:

```text
single stage1 checkpoint: about 7.6G
fixed stage1_m1 directory with multiple checkpoints: about 46G
```

If frequent intermediate checkpoints are enabled, reserve enough extra space for
all retained checkpoints or add a retention cleanup policy.

## Current Code and Comment Mismatch

There is a documented/commented mismatch that must be checked before retraining.

`run_qwen3_4b_math_m1_2ep_fixed.sh` comments say the fixed run removed
`completion_only_loss=True`. Current local code does not match that comment.

Live-checked on 2026-05-18:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/llmboost_train.py
/root/buaa/czh_rl_icml_2026/EnsembleLLM/Trainer/sft_runner.py
```

still set:

```text
completion_only_loss=True
```

Do not rely on the script comment alone. Before a replacement `model2` retrain,
decide whether the historical intended configuration is `completion_only_loss=True`
or whether the code should actually remove it.

## Ablation Checkpoint Comparison

The temporary local `model2` used in the 3A investigation came from an ablation
directory:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/weights/llmboost_ablation/Qwen3-4B-Base-Math-m1step_fixed/stage1_m1/checkpoint-218
```

This checkpoint is math-related Stage1 SFT. The Stage1 training itself uses the
same core SFT mechanism and broadly the same core parameters as the ordinary
math `stage1_m1` training:

```text
base model: Qwen/Qwen3-4B-Base
dataset: am_deepseek_r1_filtered_ad.jsonl
batch / grad_accum: 1 / 64
max_seq_length: 4096
lr: 1e-5
1 epoch: 218 steps
completion_only_loss: True in current code
flash_attention_2: True in current code
```

The main differences are wrapper-level and experiment-management differences:

```text
run_ablation_qwen4b_math_m1step.sh also computes entropy and later runs Stage3
ablation loops.
Stage1 save cadence differs across scripts: save every 25/50 steps or epoch end.
run_qwen3_4b_math_m1_2ep_fixed.sh says 2 epochs but passes --stage1-stop-at-step 218,
so that logged run stopped after 1 epoch.
Existing output paths live under weights/llmboost_ablation, which can confuse
provenance if reused as a clean replacement model2.
```

Therefore, the ablation Stage1 checkpoint is not a special fused-training
checkpoint; it is ordinary Stage1 SFT produced inside an ablation experiment
wrapper. For a clean replacement of `model2`, rerun Stage1 SFT into a new
non-ablation output path and record the exact code revision plus the
`completion_only_loss` decision.

## Existing Local Checkpoints Found

Primary fixed Stage1 directory:

```text
/root/buaa/czh_rl_icml_2026/EnsembleLLM/weights/llmboost_ablation/Qwen3-4B-Base-Math-m1step_fixed/stage1_m1
```

Observed checkpoints:

```text
checkpoint-50
checkpoint-100
checkpoint-150
checkpoint-200
checkpoint-218
checkpoint-218_KEEP_old
```

`checkpoint-218` is about `7.6G`.

## Suggested Retrain Checklist

Before launching:

1. Choose a clean output parent outside `weights/llmboost_ablation`.
2. Confirm whether `completion_only_loss=True` is intended.
3. Confirm `/root/buaa/czh_rl_icml_2026/dataset/am_deepseek_r1_filtered_ad.jsonl`
   exists and still has 111657 samples.
4. Confirm 8 GPUs are available for the Accelerate + DeepSpeed run.
5. Reserve at least 10G for one final checkpoint, or more if saving intermediate
   checkpoints.
6. Run the launch inside `tmux` and record the resulting checkpoint path before
   copying it into `/data-1/.cache/Qwen3-4B-Base-SFT-stage-1`.
