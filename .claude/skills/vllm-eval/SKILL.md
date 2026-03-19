---
name: vllm-eval
description: "Run offline vLLM inference evaluation on model checkpoints. Use this skill whenever the user wants to evaluate a checkpoint, run validation/inference on a trained model, compute mean@n/pass@k/maj@k metrics, or update inference results. Triggers on mentions of: checkpoint evaluation, offline eval, vLLM validation, merge and evaluate, EVAL-XX, inference results, pass@k scores, or any reference to evaluating an EXP-XX experiment. Also triggers when the user provides a checkpoint path, model weight path, or experiment ID and wants to know how it performs."
---

# vLLM Offline Evaluation Skill

You help users evaluate model checkpoints by running vLLM offline inference on math benchmarks (MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench) and recording results.

## Overview

This skill handles the full pipeline from checkpoint to evaluation results:

1. **Identify** what the user provided (experiment ID, checkpoint path, or merged weights)
2. **Look up** context from the experiment index
3. **Merge** FSDP checkpoints to HF format if needed
4. **Extract** sub-model from joint models if needed
5. **Check** GPU availability
6. **Run** vLLM inference with the offline eval script
7. **Record** results in INFERENCE_RESULTS.md and update EXPERIMENT_INDEX.md

## Key Files

| File | Purpose |
|---|---|
| `recipe/joint_training/EXPERIMENT_INDEX.md` | Maps experiments to checkpoints, logs, and weights |
| `recipe/joint_training/INFERENCE_RESULTS.md` | Stores all evaluation results (EVAL-XX entries) |
| `recipe/joint_training/offline_eval.py` | vLLM inference + scoring + multi-k metrics |
| `recipe/joint_training/compute_multi_k_metrics.py` | Standalone post-hoc multi-k computation from saved parquet |
| `recipe/joint_training/extract_sub_model.py` | Extracts sub-model from joint model weights |

## Step 1: Understand What the User Provided

The user might give you:

- **An experiment ID** like "EXP-04" — look it up in EXPERIMENT_INDEX.md
- **A checkpoint path** like `/data-1/checkpoints/Joint-MiniRL-.../.../global_step_100/actor` — match it to an experiment
- **A merged model path** like `/data-1/model_weights/EXP-04_.../step_100_model2` — skip to GPU check
- **A training log name** — find the experiment it belongs to in EXPERIMENT_INDEX.md
- **Ambiguous input** — use AskUserQuestion to clarify

**Always start by reading** `recipe/joint_training/EXPERIMENT_INDEX.md` to understand the full context: which experiment this is, what algorithm was used, whether it's a joint model or single model, the model size, and what checkpoints are available.

If the user says something like "evaluate the latest checkpoint" or "run eval on EXP-04", you need to figure out:
- Which specific step to evaluate (ask if multiple are available and user didn't specify)
- Whether model weights are already merged (check Model Weights Inventory in the index)

### Determine n and multi-k behavior

- **Default**: `--n 8` (generates 8 responses per prompt)
- The script always computes exactly one `mean@n`, using all n responses per prompt
- When n is a power of 2 (1, 2, 4, 8, 16, ...), `offline_eval.py` **automatically** computes `pass@k` and `maj@k` for all k in {1, 2, 4, ..., n}
- When n is not a power of 2 (e.g., 3, 5, 6), it computes `pass@n` and `maj@n` only
- If the user asks for specific @k metrics (e.g., "@1, @2, @4, @8"), use the appropriate power-of-2 n value

### Determine append vs overwrite behavior

- **Default**: Each n value gets its own output directory: `{model_weights_path}/inference_n{n}/`
  - Example: `inference_n3/`, `inference_n8/`, `inference_n16/`
- This means running with different n values **never overwrites** previous results
- If the user explicitly asks to **overwrite**, use the same output directory as the previous run
- If the user asks to **append** results to an existing EVAL entry, add a new results subsection (e.g., "Results (n=8, multi-k)") instead of creating a new EVAL-XX entry
- When appending, do NOT create a new EVAL-ID — update the existing one with additional results

## Step 2: Merge Checkpoint (if needed)

If the user provided a checkpoint (not already-merged weights), merge it:

```bash
CUDA_VISIBLE_DEVICES={any_free_gpu} \
LD_LIBRARY_PATH=/data-1/.cache/conda/envs/verl07/lib/python3.10/site-packages/torch/lib:/data-1/.cache/conda/envs/verl07/lib \
/data-1/.cache/conda/envs/verl07/bin/python -u -m verl.model_merger merge \
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
/data-1/.cache/conda/envs/verl07/bin/python -u recipe/joint_training/extract_sub_model.py \
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

```bash
CUDA_VISIBLE_DEVICES={comma_separated_gpu_ids} \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_USE_V1=1 \
VLLM_NO_USAGE_STATS=1 \
LD_LIBRARY_PATH=/data-1/.cache/conda/envs/verl07/lib/python3.10/site-packages/torch/lib:/data-1/.cache/conda/envs/verl07/lib \
/data-1/.cache/conda/envs/verl07/bin/python -u recipe/joint_training/offline_eval.py \
    --model_path {model_weights_path} \
    --tensor_parallel {num_gpus} \
    --n {n} \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir {model_weights_path}/inference_n{n} \
    --test_files \
        /data-1/dataset/MATH-500/math500-test.parquet \
        /data-1/dataset/AIME-2025/aime-2025.parquet \
        /data-1/dataset/AMC-2023/amc23-test.parquet \
        /data-1/dataset/MinervaMAth/minervamath-test.parquet \
        /data-1/dataset/OlympiadBench/olympiadbench-test.parquet
```

The script will:
- Generate n responses per prompt
- Score all responses
- Compute `mean@n` once for the full run
- If n is a power of 2: compute `pass@k` and `maj@k` for k=1,2,4,...,n
- Print both per-source detail and a tabular summary
- Save `eval_metrics.json` (all metrics) and `eval_details.parquet` (raw per-response data)

### Re-computing metrics from saved data

If the user wants to recompute metrics without re-running inference (e.g., with different bootstrap params), use the standalone tool:

```bash
/data-1/.cache/conda/envs/verl07/bin/python -u recipe/joint_training/compute_multi_k_metrics.py \
    {path_to_eval_details.parquet} \
    {k_values_comma_separated} \
    {n_per_prompt}
```

Example: `compute_multi_k_metrics.py .../eval_details.parquet 1,2,4,8 8`

## Step 6: Record Results

### 6a. Determine whether to create new EVAL-ID or append

- **New evaluation** (different checkpoint, different experiment): Create a new EVAL-{XX+1}
- **Supplementary run** (same checkpoint, different n): Append to the existing EVAL-XX entry as a new results subsection
- When the user says "don't overwrite" or "append", always add to the existing entry

### 6b. Add/update entry in INFERENCE_RESULTS.md

**For a new EVAL entry** (single-k, n is not power of 2):

```markdown
---

## EVAL-{XX}: {EXP-ID} {ShortDescription}

| Field | Value |
|---|---|
| **Source Experiment** | {EXP-ID} ({ExperimentName}) |
| **Model Weights** | `{model_weights_path}` |
| **Checkpoint Step** | {step} ({final/best/etc}) |
| **Sub-Model** | {model2 (trainable) / N/A (single model)} |
| **Inference Engine** | vLLM 0.8.5 (FLASH_ATTN backend, V1 engine, tp={N}) |
| **Benchmarks** | MATH-500, AIME-2025, AMC-2023, MinervaMAth, OlympiadBench |
| **Generation Params** | temperature=1.0, top_p=0.95, top_k=-1, max_tokens=4096 |
| **Date** | {YYYY-MM-DD} |

### Results (n={n})

| Benchmark | Samples | mean@{n} | pass@{n} | maj@{n} | extraction_fail |
|---|---|---|---|---|---|
| **MATH-500** | 500 | **{val}%** | {val}% | {val}% | {val}% |
| **AIME-2025** | 30 | **{val}%** | {val}% | {val}% | {val}% |
| **AMC-2023** | 40 | **{val}%** | {val}% | {val}% | {val}% |
| **MinervaMAth** | 272 | **{val}%** | {val}% | {val}% | {val}% |
| **OlympiadBench** | 674 | **{val}%** | {val}% | {val}% | {val}% |

### Notes

- {relevant notes}
- Raw results saved to: `{output_dir}/`
```

**For a new EVAL entry with multi-k** (n is power of 2):

```markdown
### Results (n={n}, multi-k)

| Benchmark | Samples | mean@{n} | pass@1 | maj@1 | pass@2 | maj@2 | ... | pass@{n} | maj@{n} | extraction_fail |
|---|---|---|---|---|---|---|---|---|---|
| **MATH-500** | 500 | ... |
...
```

Retain `mean@{n}` once, then include `pass@k` and `maj@k` pairs for each k.

**For appending to an existing EVAL entry**: Add a new `### Results (n={n}, multi-k)` subsection below the existing results section. Update the Generation Params field to remove the fixed `n=` value (since multiple n values are now recorded). Add a note about the new raw data path.

### 6c. Update EXPERIMENT_INDEX.md

Update the relevant experiment's entry:
- Set **Model Weights** field to the merged weight path
- Add **Inference** field pointing to the EVAL-ID
- Update the Model Weights Inventory table with the new entry

### 6d. Update Cross-Experiment Comparison (if applicable)

If INFERENCE_RESULTS.md has a "Cross-Experiment Comparison" section, update it to include the new results. If there are now 2+ evaluations, add or update the comparison table. Use `mean@n` from each evaluation run for the comparison table.

## Saved Data Structure

Each evaluation run saves to `{model_weights_path}/inference_n{n}/`:

```
inference_n8/
├── eval_metrics.json      # mean@n plus pass/maj@k for all requested k values
└── eval_details.parquet   # Raw per-response data (for re-computation)
```

`eval_metrics.json` structure:
```json
{
  "model_path": "...",
  "generation_params": {"temperature": 1.0, "top_p": 0.95, "n": 8, "max_tokens": 4096, "seed": 42},
  "k_values": [1, 2, 4, 8],
  "generation_time_s": 706.8,
  "metrics": {
    "HuggingFaceH4/MATH-500": {
      "mean@8": 0.669,
      "pass@1": 0.669, "maj@1": 0.668,
      "pass@2": 0.746, "maj@2": 0.668,
      "pass@4": 0.801, "maj@4": 0.714,
      "pass@8": 0.846, "maj@8": 0.737,
      "n_prompts": 500, "extraction_fail": 0.001
    },
    ...
  }
}
```

`eval_details.parquet` columns: `data_source, ground_truth, acc, score, pred, verification_method, response_text`
- Rows are ordered by (data_source, prompt_index, response_index)
- n consecutive rows belong to the same prompt
- This file enables re-computation of any metrics without re-running inference

## Error Handling

- **Merge fails**: Check if the checkpoint path is correct (should end in `/actor`). Check if disk space is sufficient.
- **vLLM tokenizer error** (`all_special_tokens_extended`): The `offline_eval.py` script includes a compatibility patch for transformers 5.x — this should work automatically.
- **OOM during inference**: Reduce `--gpu_memory_utilization` (default 0.85) or increase tensor parallelism.
- **No checkpoints at requested step**: List available steps and ask the user which one to use.

## Important Notes

- Always use `/data-1/.cache/conda/envs/verl07/bin/python` directly (not `conda run`) to avoid stdout buffering issues.
- Always set `LD_LIBRARY_PATH` to include torch and conda lib paths (GLIBCXX compatibility).
- The working directory should be `/data-1/verl07/verl` (the repo root).
- Do NOT install any packages — the verl07 environment is shared with training.
- **Output directory convention**: Always use `inference_n{n}` to avoid overwriting results from different n values.
- **Default n=8**: Unless the user specifies otherwise, use n=8 for new evaluations (gives `mean@8` plus `pass/maj@1,@2,@4,@8`).
