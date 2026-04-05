---
name: vllm-eval
description: "Run offline vLLM inference evaluation on model checkpoints. Use this skill whenever the user wants to evaluate a checkpoint, run validation/inference on a trained model, compute mean@n/pass@k/maj@k metrics, or update inference results. Triggers on mentions of: checkpoint evaluation, offline eval, vLLM validation, merge and evaluate, EVAL-XX, inference results, pass@k scores, or any reference to evaluating an EXP-XX experiment. Also triggers when the user provides a checkpoint path, model weight path, or experiment ID and wants to know how it performs."
---

# vLLM Offline Evaluation Skill

You help users evaluate model checkpoints by running vLLM offline inference on math benchmarks and recording results.

## Overview

This skill handles the full pipeline from checkpoint to evaluation results:

1. **Gather information** — use AskUserQuestion to resolve ambiguity before proceeding
2. **Identify** what the user provided (experiment ID, checkpoint path, or merged weights)
3. **Look up** context from the experiment index
4. **Merge** FSDP checkpoints to HF format if needed
5. **Extract** sub-model from joint models if needed
6. **Check** GPU availability
7. **Run** vLLM inference with the offline eval script (inside Docker container)
8. **Record** results in INFERENCE_RESULTS.md and update EXPERIMENT_INDEX.md

## Environment

All commands run inside the **`verl-harness`** Docker container. This container provides Python 3.12 (managed by uv), PyTorch 2.9.1+cu126, vLLM 0.12.0, and flash-attn 2.8.1.

**Launching the container:**

```bash
# Interactive shell (all GPUs)
bash /data-1/verl07/run_train.sh

# With specific GPUs
GPUS='"device=0,1,2,3"' bash /data-1/verl07/run_train.sh
```

Inside the container, Python is directly available as `python` (no conda, no LD_LIBRARY_PATH needed). The repo is mounted at `/workspace/verl`.

**IMPORTANT**: If you are already inside the container (check with `cat /proc/1/cmdline` or look for `/opt/venv`), do NOT launch another one. If you are outside the container, you must run commands through `run_train.sh`.

## Key Files

| File | Purpose |
|---|---|
| `recipe/joint_training/EXPERIMENT_INDEX.md` | Maps experiments to checkpoints, logs, and weights |
| `recipe/joint_training/INFERENCE_RESULTS.md` | Stores all evaluation results (EVAL-XX entries) |
| `recipe/joint_training/offline_eval.py` | vLLM inference + scoring + multi-k metrics |
| `recipe/joint_training/compute_multi_k_metrics.py` | Standalone post-hoc multi-k computation from saved parquet |
| `recipe/joint_training/extract_sub_model.py` | Extracts sub-model from joint model weights |

## Available Datasets

There are two types of datasets:

1. **With system prompt** (`_with_system_prompt`): Contains a chat template with system message. Use these when the model was trained with a system prompt (e.g., DPO/SFT models, or RL models trained with system-prompt data).
2. **Without system prompt** (plain): Raw user-assistant format. Use these for models trained without system prompts.

When the user doesn't specify, **ask which type they want**.

### Datasets with system prompt

| Dataset | Path | Samples |
|---|---|---|
| MATH-500 | `/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet` | 500 |
| AIME-2025 | `/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet` | 30 |
| AIME-2024 | `/data-1/dataset/AIME-2024/aime-2024_with_system_prompt.parquet` | 30 |
| AMC-2023 | `/data-1/dataset/AMC23/amc23-test_with_system_prompt.parquet` | 40 |
| AQUA | `/data-1/dataset/AQUA/aqua-test_with_system_prompt.parquet` | 254 |
| GSM8K | `/data-1/dataset/gsm8k/gsm8k-test_with_system_prompt.parquet` | 1319 |
| MAWPS | `/data-1/dataset/MAWPS/mawps-test_with_system_prompt.parquet` | 355 |
| SVAMP | `/data-1/dataset/SVAMP/svamp-test_with_system_prompt.parquet` | 300 |

### Datasets without system prompt (plain)

| Dataset | Path | Samples |
|---|---|---|
| MATH-500 | `/data-1/dataset/MATH-500/math500-test.parquet` | 500 |
| AIME-2025 | `/data-1/dataset/AIME-2025/aime-2025.parquet` | 30 |
| AIME-2024 | `/data-1/dataset/AIME-2024/aime-2024.parquet` | 30 |
| AMC-2023 | `/data-1/dataset/AMC23/amc23-test.parquet` | 40 |
| AQUA | `/data-1/dataset/AQUA/aqua-test.parquet` | 254 |
| GSM8K | `/data-1/dataset/gsm8k/test.parquet` | 1319 |
| MAWPS | `/data-1/dataset/MAWPS/mawps-test.parquet` | 355 |
| SVAMP | `/data-1/dataset/SVAMP/svamp-test.parquet` | 300 |

## Step 0: Gather Information (AskUserQuestion)

**Before proceeding with any evaluation, ensure you have sufficient information.** Use AskUserQuestion to clarify whenever any of the following are ambiguous or missing:

- **What to evaluate**: experiment ID, checkpoint path, or merged model path
- **Which checkpoint step**: if multiple steps are available and user didn't specify
- **Which datasets**: if user didn't specify, confirm whether to use the defaults (MATH-500 + AIME-2025) or a different set
- **n value(s)**: if user didn't specify, confirm the default n=8; if they want per-dataset n, ask for specifics
- **Which sub-model**: for joint models, if user didn't specify model1 vs model2
- **Append vs new EVAL**: if there's already an EVAL entry for this checkpoint

Examples of when to ask:

- User says "evaluate EXP-04" → Ask: "Which step? I see steps 50, 100, 150 available."
- User says "run eval on this checkpoint" → Ask: "Which datasets and n values would you like? Default is MATH-500 + AIME-2025 with n=8."
- User gives a path but it's unclear if it's a checkpoint or merged weights → Ask: "Is this a raw FSDP checkpoint that needs merging, or already-merged HF weights?"

**Do NOT guess when information is missing — ask.**

## Step 1: Understand What the User Provided

The user might give you:

- **An experiment ID** like "EXP-04" — look it up in EXPERIMENT_INDEX.md
- **A checkpoint path** like `/data-1/checkpoints/Joint-MiniRL-.../.../global_step_100/actor` — match it to an experiment
- **A merged model path** like `/data-1/model_weights/EXP-04_.../step_100_model2` — skip to GPU check
- **A training log name** — find the experiment it belongs to in EXPERIMENT_INDEX.md
- **Ambiguous input** — use AskUserQuestion to clarify

**Always start by reading** `recipe/joint_training/EXPERIMENT_INDEX.md` to understand the full context: which experiment this is, what algorithm was used, whether it's a joint model or single model, the model size, and what checkpoints are available.

### Determine n and multi-k behavior

- **Default**: `--n 8` (generates 8 responses per prompt)
- **Per-dataset n**: Use `--n_per_dataset` to override n for specific datasets (e.g., AIME with n=16 for more reliable maj@k on 30 samples)
- The script always computes exactly one `mean@n`, using all n responses per prompt
- When n is a power of 2 (1, 2, 4, 8, 16, ...), `offline_eval.py` **automatically** computes `pass@k` and `maj@k` for all k in {1, 2, 4, ..., n}
- When n is not a power of 2 (e.g., 3, 5, 6), it computes `pass@1`, `pass@n`, `maj@1`, and `maj@n`
- **pass@1 is always computed** regardless of n
- If the user asks for specific @k metrics (e.g., "@1, @2, @4, @8"), use the appropriate power-of-2 n value

### Determine append vs overwrite behavior

- **Default**: Results go to `{model_weights_path}/inference_n{n}/` (or a descriptive subdir when per-dataset n is used)
- Running with different n values **never overwrites** previous results
- If the user explicitly asks to **overwrite**, use the same output directory as the previous run
- If the user asks to **append** results to an existing EVAL entry, add a new results subsection instead of creating a new EVAL-XX entry
- When appending, do NOT create a new EVAL-ID — update the existing one with additional results

## Step 2: Merge Checkpoint (if needed)

If the user provided a checkpoint (not already-merged weights), merge it. Run inside the Docker container:

```bash
CUDA_VISIBLE_DEVICES={any_free_gpu} \
python -u -m verl.model_merger merge \
    --backend fsdp \
    --local_dir {checkpoint_path}/actor \
    --target_dir /data-1/model_weights/{EXP-ID}_{ShortName}/step_{N} \
    --trust-remote-code
```

The checkpoint path structure is: `/data-1/checkpoints/{RunName}_{timestamp}/global_step_{N}/actor`

The merge target follows: `/data-1/model_weights/{EXP-ID}_{ShortName}/step_{N}/`

## Step 3: Extract Sub-Model (if joint model)

**How to tell if it's a joint model**: After merging, check `config.json` in the merged output. If `model_type` is `"qwen_joint"` and `architectures` contains `"QwenJointForCausalLM"`, it's a joint model with two sub-models.

For joint models, extract the sub-model the user wants (default: model2 = index 1, the trainable model):

```bash
python -u recipe/joint_training/extract_sub_model.py \
    --joint_model_path /data-1/model_weights/{EXP-ID}_{ShortName}/step_{N} \
    --output_path /data-1/model_weights/{EXP-ID}_{ShortName}/step_{N}_model2 \
    --sub_model_index 1
```

- `--sub_model_index 0` = model1 (anchor/frozen model)
- `--sub_model_index 1` = model2 (trainable model, **default**)

If the user doesn't specify which sub-model, default to model2 and inform them. If they want both evaluated, run the extraction and evaluation twice.

For single models (`model_type: "qwen3"`), skip this step — the merged weights are directly usable.

## Step 4: Check GPU Availability

Before running inference, check which GPUs are free:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

A GPU is "free" if memory.used is under 1000 MiB and utilization is 0%.

**If no GPUs are free**: Tell the user clearly — "All GPUs are currently in use. Please run this evaluation later when GPUs become available." Do NOT proceed.

**If GPUs are available**: Select the free ones. For model sizing:
- 1.7B model: tp=1 is sufficient (1 GPU), but tp=2 or tp=4 is faster for batch inference
- 4B model: tp=2 minimum, tp=4 recommended
- Use all available free GPUs up to 4 for best throughput

## Step 5: Run vLLM Inference

All inference commands run inside the Docker container. Use `python` directly.

### Uniform n (all datasets same n)

```bash
CUDA_VISIBLE_DEVICES={comma_separated_gpu_ids} \
python -u recipe/joint_training/offline_eval.py \
    --model_path {model_weights_path} \
    --tensor_parallel {num_gpus} \
    --n {n} \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir {model_weights_path}/inference_n{n} \
    --test_files \
        /data-1/dataset/MATH-500/math500-test.parquet \
        /data-1/dataset/AIME-2025/aime-2025.parquet
```

### Per-dataset n (different n for different datasets)

Use `--n_per_dataset` to override n for specific datasets. Datasets not listed fall back to the global `--n`.

```bash
CUDA_VISIBLE_DEVICES={comma_separated_gpu_ids} \
python -u recipe/joint_training/offline_eval.py \
    --model_path {model_weights_path} \
    --tensor_parallel {num_gpus} \
    --n 8 \
    --n_per_dataset \
        /data-1/dataset/AIME-2025/aime-2025.parquet:16 \
        /data-1/dataset/AMC23/amc23-test.parquet:16 \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir {model_weights_path}/inference_mixed_n \
    --test_files \
        /data-1/dataset/MATH-500/math500-test.parquet \
        /data-1/dataset/AIME-2025/aime-2025.parquet \
        /data-1/dataset/AMC23/amc23-test.parquet
```

In this example MATH-500 uses n=8 (the default), while AIME-2025 and AMC-2023 use n=16.

### Tunable vLLM / generation parameters

| Parameter | Flag | Default | Notes |
|---|---|---|---|
| Tensor parallelism | `--tensor_parallel` | 4 | Match to number of GPUs used |
| Responses per prompt | `--n` | 8 | Power of 2 enables multi-k metrics |
| Per-dataset n overrides | `--n_per_dataset` | none | Format: `path:n` pairs |
| Temperature | `--temperature` | 1.0 | |
| Top-p | `--top_p` | 0.95 | |
| Max output tokens | `--max_tokens` | 4096 | |
| GPU memory utilization | `--gpu_memory_utilization` | 0.85 | Lower if OOM |
| Random seed | `--seed` | 42 | |

The script will:
- Initialize vLLM once, then run separate generation passes for each distinct n value
- Generate n responses per prompt
- Score all responses
- Compute `mean@n` once for the full run (per data source, using that source's n)
- If n is a power of 2: compute `pass@k` and `maj@k` for k=1,2,4,...,n
- Print both per-source detail and tabular summaries (grouped by n)
- Save `eval_metrics.json` (all metrics) and `eval_details.parquet` (raw per-response data)

### Re-computing metrics from saved data

If the user wants to recompute metrics without re-running inference (e.g., with different bootstrap params), use the standalone tool:

```bash
python -u recipe/joint_training/compute_multi_k_metrics.py \
    {path_to_eval_details.parquet} \
    {k_values_comma_separated} \
    {n_per_prompt}
```

Example: `compute_multi_k_metrics.py .../eval_details.parquet 1,2,4,8 8`

## Step 6: Record Results

**After evaluation completes, invoke the `eval-record` skill** (either via `/eval-record` or by calling the Skill tool) to record results into EXPERIMENT_INDEX.md and INFERENCE_RESULTS.md.

The eval-record skill handles:
- Creating new EVAL-XX entries in INFERENCE_RESULTS.md
- Creating/updating EXP-XX entries in EXPERIMENT_INDEX.md (for both verl-trained and external models)
- Updating the Cross-Experiment Comparison table
- Proper formatting and cross-referencing

Pass it the eval output directory path (containing `eval_metrics.json`) and the model context.

## Saved Data Structure

Each evaluation run saves to `{output_dir}/`:

```
inference_n8/
├── eval_metrics.json      # mean@n plus pass/maj@k for all requested k values
└── eval_details.parquet   # Raw per-response data (for re-computation)
```

`eval_metrics.json` structure:
```json
{
  "model_path": "...",
  "generation_params": {
    "temperature": 1.0,
    "top_p": 0.95,
    "n_default": 8,
    "n_per_dataset": {
      "/data-1/dataset/MATH-500/math500-test.parquet": 8,
      "/data-1/dataset/AIME-2025/aime-2025.parquet": 16
    },
    "max_tokens": 4096,
    "seed": 42
  },
  "n_values_used": [8, 16],
  "generation_time_s": 706.8,
  "metrics": {
    "HuggingFaceH4/MATH-500": {
      "n_used": 8,
      "k_values": [1, 2, 4, 8],
      "mean@8": 0.669,
      "pass@1": 0.669, "maj@1": 0.668,
      "pass@2": 0.746, "maj@2": 0.668,
      "pass@4": 0.801, "maj@4": 0.714,
      "pass@8": 0.846, "maj@8": 0.737,
      "n_prompts": 500, "extraction_fail": 0.001
    }
  }
}
```

`eval_details.parquet` columns: `data_source, ground_truth, acc, score, pred, verification_method, response_text, n`
- Rows are ordered by (data_source, prompt_index, response_index)
- n consecutive rows belong to the same prompt (n may vary across data sources)
- This file enables re-computation of any metrics without re-running inference

## Error Handling

- **Merge fails**: Check if the checkpoint path is correct (should end in `/actor`). Check if disk space is sufficient.
- **vLLM tokenizer error** (`all_special_tokens_extended`): The `offline_eval.py` script includes a compatibility patch — this should work automatically.
- **OOM during inference**: Reduce `--gpu_memory_utilization` (default 0.85) or increase tensor parallelism.
- **No checkpoints at requested step**: List available steps and use AskUserQuestion to ask the user which one to use.
- **Container not running**: If commands fail with "python not found" or similar, the user may not be inside the Docker container. Guide them to run `bash /data-1/verl07/run_train.sh`.

## Important Notes

- All commands run inside the `verl-harness` Docker container. Use `python` directly — no conda paths or LD_LIBRARY_PATH needed.
- The working directory should be `/workspace/verl` (the repo root inside the container).
- Do NOT install any packages — the container environment is shared with training.
- **Output directory convention**: Use `inference_n{n}` for uniform-n runs, or a descriptive name for per-dataset-n runs.
- **Default n=8**: Unless the user specifies otherwise, use n=8 for new evaluations (gives `mean@8` plus `pass/maj@1,@2,@4,@8`).
- The `VLLM_USE_V1=1` and `VLLM_NO_USAGE_STATS=1` environment variables are already set in the Docker image, so you don't need to set them in commands.
- **Attention backend**: vLLM 0.12.0 auto-selects the attention backend based on GPU capability. On A100 (compute capability 8.0), it selects **FLASH_ATTN**; on newer GPUs (H100/H200, cc 9.0+), it may select **FLASHINFER**. Do NOT hard-code `VLLM_ATTENTION_BACKEND` unless you have a specific reason.
