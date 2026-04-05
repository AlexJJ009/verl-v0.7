---
name: eval-record
description: "Record evaluation results into EXPERIMENT_INDEX.md and INFERENCE_RESULTS.md. Use this skill after completing a vLLM evaluation (/vllm-eval), or when the user wants to manually add, update, or query eval results. Triggers on: 'record results', 'update eval', 'add EVAL entry', 'update experiment index', 'update inference results', or after any vllm-eval run completes."
---

# Eval Record Skill

You manage two tracking files that form the project's experiment and inference ledger:

- **EXPERIMENT_INDEX.md** — `recipe/joint_training/EXPERIMENT_INDEX.md`
- **INFERENCE_RESULTS.md** — `recipe/joint_training/INFERENCE_RESULTS.md`

## When This Skill Runs

1. **Automatically after vllm-eval**: When the vllm-eval skill completes a run, invoke this skill to record the results.
2. **Manually by user**: When the user says "record results", "update eval", "add to experiment index", etc.

## Two Model Sources

### Source A: verl-trained models (RL experiments)

These have an existing `EXP-XX` entry in EXPERIMENT_INDEX.md. The eval results link back to that experiment.

- Use the existing EXP-XX ID
- Update the experiment's **Model Weights** and **Inference** fields
- Add to Model Weights Inventory if new weights were merged
- Create EVAL-XX entry in INFERENCE_RESULTS.md

### Source B: Externally-trained models (DPO, SFT, base models, etc.)

These do NOT have an EXP-XX entry because they were trained outside verl (e.g., HuggingFace TRL DPO, or a raw pretrained base model).

- Create a new `EXP-XX` entry in EXPERIMENT_INDEX.md with:
  - **Script**: `N/A (external)` or the external training script/framework used
  - **Goal**: What the model is (e.g., "DPO-trained Qwen3-4B-Base", "Pretrained Qwen3-4B-Base baseline")
  - **Algorithm**: The training method (e.g., "DPO (TRL)", "Pretrained (no training)")
  - **Model**: The model name and size
  - **Status**: `External` or `Pretrained`
- Then create the EVAL-XX entry as normal

## Step 1: Gather Information

You need the following before recording. Use AskUserQuestion if anything is missing:

1. **Model identity**: What model was evaluated? (experiment ID, checkpoint path, or model name)
2. **Eval results**: Where are the results? (eval_metrics.json path, or raw output from vllm-eval)
3. **Context**: Is this a verl experiment or external model? What training was done?

If this skill was auto-invoked after vllm-eval, the eval output and model path should already be in the conversation context.

## Step 2: Read Current State

Always read both files before making changes:

```
recipe/joint_training/EXPERIMENT_INDEX.md
recipe/joint_training/INFERENCE_RESULTS.md
```

Determine:
- The next EVAL-XX ID (look at the last EVAL entry in INFERENCE_RESULTS.md)
- Whether the model already has an EXP-XX entry
- Whether there's an existing EVAL entry for this checkpoint (append vs new)

## Step 3: Read Eval Results

If results were saved to `eval_metrics.json`, read it:

```bash
cat {output_dir}/eval_metrics.json
```

Extract per-benchmark metrics: `mean@n`, `pass@1`, `pass@n`, `maj@n`, `extraction_fail`, `n_prompts`, `n_used`.

## Step 4: Update INFERENCE_RESULTS.md

### New EVAL entry format

For uniform n (not power of 2, e.g., n=3):

```markdown
---

## EVAL-{XX}: {Short Description}

| Field | Value |
|---|---|
| **Source Experiment** | {EXP-XX} ({ExperimentName}) |
| **Model Weights** | `{model_weights_path}` |
| **Checkpoint Step** | {step} ({final/best/external}) |
| **Sub-Model** | {model2 (trainable) / N/A (single model)} |
| **Inference Engine** | vLLM 0.12.0 (FLASH_ATTN backend, V1 engine, tp={N}) |
| **Benchmarks** | {comma-separated list} |
| **Generation Params** | temperature={T}, top_p={P}, top_k=-1, n={n}, max_tokens={M} |
| **Date** | {YYYY-MM-DD} |

### Results (n={n})

| Benchmark | Samples | mean@{n} | pass@1 | pass@{n} | maj@{n} | extraction_fail |
|---|---|---|---|---|---|---|
| **{name}** | {count} | **{val}%** | {val}% | {val}% | {val}% | {val}% |
...

### Notes

- {relevant notes about model source, training, observations}
- Raw results saved to: `{output_dir}/`
```

For multi-k (n is power of 2), include all pass@k and maj@k columns.

For per-dataset n, group results by n value in separate subsections.

### Appending to existing EVAL entry

If the user ran additional n values on the same checkpoint:
- Do NOT create a new EVAL-XX
- Add a new `### Results (n={n})` subsection below the existing one
- Add a note about the additional run

## Step 5: Update EXPERIMENT_INDEX.md

### For verl-trained models (Source A)

Update the existing EXP-XX entry:
- Set/update **Model Weights** field
- Set/update **Inference** field: `EVAL-XX in INFERENCE_RESULTS.md ({benchmarks summary})`
- Add to **Model Weights Inventory** table if new weights

### For external models (Source B)

Create a new EXP-XX entry in the **Active Experiments** section:

```markdown
---

### EXP-{XX}: {Model Short Name}

| Field | Value |
|---|---|
| **Script** | N/A (external: {framework/method}) |
| **Goal** | {purpose of this model — e.g., "DPO-trained baseline for comparison"} |
| **Algorithm** | {training method — e.g., "DPO (TRL, beta=0.1)", "Pretrained (no training)"} |
| **Model** | {model name and size — e.g., "Qwen3-4B-Base"} |
| **Dataset** | {training data if known, or "N/A"} |
| **Key Params** | {key hyperparameters if known} |
| **Model Weights** | `{path}` |
| **Inference** | EVAL-{XX} in `INFERENCE_RESULTS.md` |
| **Status** | External / Pretrained |
```

Also add to **Model Weights Inventory** if applicable.

## Step 6: Update Cross-Experiment Comparison

If INFERENCE_RESULTS.md has a **Cross-Experiment Comparison** section at the bottom:
- Add the new eval results to the comparison table
- Use `mean@n` as the primary comparison metric
- Bold the best value in each row
- Update the **Observations** text

If there is no comparison section yet but there are now 2+ evaluations, create one.

## Formatting Rules

- Percentages: always display as `XX.X%` (one decimal place)
- Metric values from JSON are in 0-1 range — multiply by 100 for display
- Bold the primary metric (mean@n) in each benchmark row
- Keep field names consistent with existing entries
- Date format: YYYY-MM-DD
- The EVAL-XX ID must be sequential (no gaps, no reuse)
- Always end each EVAL section with `---` separator

## Important Notes

- **Read before write**: Always read both files before making any changes to avoid conflicts
- **Never delete existing entries**: Only add or update
- **Cross-reference**: Every EVAL-XX must reference an EXP-XX, and vice versa
- **Attention backend**: Check the actual vLLM log to determine which backend was used (FLASH_ATTN or FLASHINFER). Do not guess.
- **Maintenance spec**: Follow `docs/joint_training/constraints/experiment_tracking/experiment_index_spec.md`
