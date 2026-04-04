# Experiment Index Maintenance Specification

This document defines how and when to maintain the experiment index at `recipe/joint_training/EXPERIMENT_INDEX.md`.

## Index Location

- **Experiment index**: `recipe/joint_training/EXPERIMENT_INDEX.md`
- **Inference results**: `recipe/joint_training/INFERENCE_RESULTS.md` (cross-referenced with experiment index)

## When to Update

### 1. Before Starting a New Training Run

**Trigger**: Creating a new training script or launching a new experiment.

**Required actions**:
1. Assign the next `EXP-XX` ID.
2. Add a new experiment section with all known fields filled in (script, goal, algorithm, model, dataset, key params).
3. Set the log filename convention: `{ExperimentName}_{unix_timestamp}.log`.
4. Set status to "Planned" or "In progress".

**Why**: If you do not register the experiment before launch, the log filename and parameters become disconnected from their purpose. Retroactive reconstruction is error-prone and time-consuming.

### 2. After Training Completes or Stops

**Trigger**: A training run finishes, crashes, or is manually killed.

**Required actions**:
1. Update the experiment's log list with the actual log filename and outcome (completed / crashed / interrupted).
2. Fill in the checkpoint path if new checkpoints were saved.
3. Record the best and final validation metrics.
4. Update status (Completed / Interrupted / Concluded).
5. Sync W&B offline run data to the cloud and update the **W&B** field (see Section "W&B Tracking" below).

### 3. When Checkpoints Are Deleted

**Trigger**: Disk space cleanup or intentional removal of old checkpoints.

**Required actions**:
1. Update the Checkpoint Inventory table: mark the deleted entry with strikethrough or remove it.
2. Add an entry to the **Deletion Log** at the bottom of the index with:
   - Date
   - What was deleted (path)
   - Which experiment it belonged to
   - Reason (e.g., "disk space recovery", "superseded by EXP-XX")

**Why**: Without a deletion log, future analysis cannot distinguish "checkpoint was never created" from "checkpoint existed but was removed."

### 4. When Model Weights Are Merged

**Trigger**: Running the merge CLI to produce consolidated weights from a checkpoint.

**Required actions**:
1. Add the merged weight path to the **Model Weights Inventory** table.
2. Record the source experiment, checkpoint step, and merge date.
3. Create the corresponding directory under `/data-1/model_weights/` following the naming convention:
   ```
   /data-1/model_weights/{EXP-ID}_{ShortName}/step_{N}/
   ```

### 5. When Inference Results Are Produced

**Trigger**: Running vLLM offline inference on merged model weights.

**Required actions**:
1. Record results in `recipe/joint_training/INFERENCE_RESULTS.md`.
2. Cross-reference back to the experiment index by EXP-ID.
3. See the inference results spec section below.

## Naming Conventions

### Log Files
```
{ExperimentName}_{unix_timestamp}.log
```
Example: `Joint-MiniRL-Qwen3-1.7B-MATH_1773581076.log`

### Checkpoint Directories
```
/data-1/checkpoints/{ExperimentName}_{unix_timestamp}/
```

### Model Weights
```
/data-1/model_weights/{EXP-ID}_{ShortName}/step_{N}/
```
Example: `/data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_80/`

### Inference Results
```
/data-1/model_weights/{EXP-ID}_{ShortName}/step_{N}/inference/
```

## W&B Tracking

All training runs use `WANDB_MODE=offline` by default. After a run completes, the offline data must be synced to the cloud.

**W&B workspace**: `gongxunli-beihang-universally/JointTraining`

### How to Sync

1. Ensure you are inside the Docker container (`verl-harness`).

2. Find the sync command at the end of the training log:
   ```
   wandb: You can sync this run to the cloud by running:
   wandb: wandb sync /data-1/verl07/verl/wandb/offline-run-YYYYMMDD_HHMMSS-RUNID
   ```

3. Run the sync command:
   ```bash
   wandb sync <path-to-offline-run>
   ```

4. Update the experiment's **W&B** field in `EXPERIMENT_INDEX.md`:
   ```
   | **W&B** | Project: `JointTraining`, Run: [`RUNID`](https://wandb.ai/gongxunli-beihang-universally/JointTraining/runs/RUNID) (synced YYYY-MM-DD) |
   ```

### W&B Directory Locations

Training runs may save W&B data in one of two directories depending on the working directory at launch time:
- `/data-1/verl07/verl/wandb/` (when launched from the repo root)
- `/data-1/verl07/verl/recipe/joint_training/wandb/` (when launched from the recipe directory)

Check both locations when looking for offline runs to sync.

### Required Field in Experiment Index

Every experiment entry must include a **W&B** row:
- Before sync: `| **W&B** | (pending sync) |`
- After sync: `| **W&B** | Project: \`JointTraining\`, Run: [\`RUNID\`](URL) (synced DATE) |`
- If no W&B data exists (e.g., crash before W&B init): `| **W&B** | N/A (run crashed before W&B initialization) |`

## Experiment ID Assignment

- IDs are sequential: EXP-00, EXP-01, EXP-02, ...
- EXP-00 is reserved for reference scripts (not actual joint-training experiments).
- A retry of the same experiment keeps the same EXP-ID; add additional log entries.
- A variation with different hyperparameters gets a new EXP-ID.

## Inference Results File

`recipe/joint_training/INFERENCE_RESULTS.md` tracks offline evaluation results and cross-references them with experiments.

Structure:
```markdown
## EVAL-{XX}: {description}

| Field | Value |
|---|---|
| Source Experiment | EXP-{XX} |
| Model Weights | /data-1/model_weights/{path} |
| Checkpoint Step | {N} |
| Inference Engine | vLLM |
| Benchmarks | {list} |
| Date | {YYYY-MM-DD} |

### Results
| Benchmark | Metric | Value |
|---|---|---|
| ... | ... | ... |
```
