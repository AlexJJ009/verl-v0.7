# Plan: Baseline MiniRL Training Script (Non-Joint)

**Status:** Draft — awaiting Alex's review
**Date:** 2026-03-16
**Goal:** Create a standalone MiniRL baseline script using Qwen3-1.7B Base (no joint training), for fair comparison against the joint-training MiniRL script.

---

## 1. Background

The joint-training experiment uses `run_joint_minirl_qwen3_1.7b_math.sh` with:
- Joint model (`QwenJoint-1.7B`, `+joint_training=True`)
- MiniRL loss, Dr.GRPO advantage, token-level IS correction
- MATH dataset (7,500 training examples), MATH-500 + AIME-2025 validation
- `train_batch_size=32`, `n_resp_per_prompt=8`, `total_training_steps=100`
- 8× H800 GPUs (81.6 GB VRAM each)

We need a **baseline** that is identical except: no joint model, just plain Qwen3-1.7B Base.

---

## 2. Key Decisions

### 2.1 Batch Size & Group Size — Keep Identical

**Keep `train_batch_size=32` and `n_resp_per_prompt=8`.**

- Dr.GRPO advantage estimation depends on group size `G=8`. Changing it alters signal quality.
- Batch size `B=32` determines per-step optimization dynamics.
- The only controlled variable should be: joint model vs single model.

### 2.2 Use 4 GPUs — Confirmed Feasible

**4 GPUs with unchanged batch parameters works cleanly:**

```
per_gpu = (mini_bsz × n_resp) / n_gpus = (4 × 8) / 4 = 8   ✓ integer
grad_accum = train_batch_size / mini_bsz = 32 / 4 = 8        ✓ integer
```

All divisibility constraints satisfied. No hyperparameter changes needed.

#### Memory Analysis (4 GPUs vs 8 GPUs)

**Hardware:** 8× NVIDIA H800, 81.6 GB VRAM each.

**Observed 8-GPU joint-model peak memory:**
- Max allocated: 39.85 GB/GPU
- Max reserved: 47.80 GB/GPU
- Headroom: ~33.8 GB/GPU (58% utilization)

**4-GPU baseline memory estimate:**

| Component | 8-GPU Joint | 4-GPU Baseline | Notes |
|-----------|------------|---------------|-------|
| FSDP model shard | ~3.4B params / 8 = 0.85 GB | ~1.7B params / 4 = 0.85 GB | Joint=2× params, 2× GPUs → same per-GPU |
| Optimizer states (Adam) | ~13.6 GB / 8 = 1.7 GB | ~6.8 GB / 4 = 1.7 GB | Same ratio |
| Gradients | ~6.8 GB / 8 = 0.85 GB | ~3.4 GB / 4 = 0.85 GB | Same ratio |
| **FSDP subtotal** | **~3.4 GB** | **~3.4 GB** | **Identical** |
| vLLM model (TP=1) | Joint weights ~6.8 GB | Base weights ~3.4 GB | **Baseline saves ~3.4 GB** |
| vLLM KV cache (0.6 util) | ~49 GB budget | ~49 GB budget | Same `gpu_memory_utilization` |
| Activations (dynamic bsz) | Adapts to available mem | Adapts to available mem | `ppo_max_token_len` controls cap |

**Conclusion:** Per-GPU FSDP memory is identical (half the params, half the GPUs cancel out). vLLM model weights are ~3.4 GB smaller (single model vs joint). Gradient checkpointing is enabled. Dynamic batching adapts to available memory.

**Estimated peak reserved: ~50-55 GB/GPU** — well within 81.6 GB H800. Safe margin of ~27-32 GB.

**Potential adjustment:** If memory is tighter than expected, can reduce `ROLLOUT_GPU_MEMORY_UTILIZATION` from 0.60 to 0.50 or lower `actor_ppo_max_token_len`.

### 2.3 Other 4-GPU Adjustments

| Parameter | 8-GPU Joint | 4-GPU Baseline | Reason |
|-----------|------------|---------------|--------|
| `NGPUS_PER_NODE` | 8 | **4** | Hardware |
| `ROLLOUT_AGENT_NUM_WORKERS` | 8 | **4** | 1 worker per GPU |
| `fsdp_size` | -1 (all GPUs) | -1 (all GPUs) | No change needed |

### 2.4 Maximum Training Steps

```
steps_per_epoch = floor(7500 / 32) = 234   (with drop_last=True)
total_epochs = 3 → max steps = 234 × 3 = 702
```

**Requested 200 steps:** Well within budget (~85% of epoch 1).

Even with 10% overlong filtering: `floor(6750/32) = 210` steps/epoch → 200 still fits in epoch 1.

---

## 3. Script Diff: Joint → Baseline

Changes from `run_joint_minirl_qwen3_1.7b_math.sh`:

| Section | Joint Script | Baseline Script |
|---------|-------------|----------------|
| **Script header** | Joint Training description | Baseline MiniRL description |
| **RUN_PREFIX** | `Joint-MiniRL-Qwen3-1.7B-MATH` | `Baseline-MiniRL-Qwen3-1.7B-MATH` |
| **WANDB_PROJECT** | `JointTraining` | `JointTraining` (same project) |
| **NGPUS_PER_NODE** | 8 | **4** |
| **MODEL_PATH** | `QwenJoint-1.7B` | `Qwen3-1.7B-Base` (direct, no prep) |
| **Joint model prep block** | Lines 75-88 (auto-prepare) | **Remove entirely** |
| **`+joint_training=True`** | Present | **Remove this line** |
| **ROLLOUT_AGENT_NUM_WORKERS** | 8 | **4** |
| **total_training_steps** | 100 | **200** |
| **Log tag** | `joint-minirl` | `baseline-minirl` |

Everything else stays **identical**: MiniRL loss, Dr.GRPO advantage, IS correction, clipping params (`clip_ratio_low=0.2`, `clip_ratio_high=0.27`), learning rate (`1e-6`), dataset, reward function, sequence lengths (`500/4096`), sampling params, `train_batch_size=32`, `n_resp_per_prompt=8`, `train_prompt_mini_bsz=4`.

---

## 4. Fairness Assessment

| Dimension | Same? | Notes |
|-----------|-------|-------|
| Dataset (train + val) | ✓ | Identical files |
| Batch size (B=32) | ✓ | |
| Group size (G=8) | ✓ | Same advantage estimation |
| Mini-batch size | ✓ | Same gradient accumulation |
| Learning rate & schedule | ✓ | |
| Loss function (MiniRL) | ✓ | |
| Clipping params | ✓ | |
| Sequence lengths | ✓ | |
| Reward function | ✓ | Same LaTeX semantic verifier |
| Model architecture | ✓ | Same Qwen3-1.7B backbone |
| Number of GPUs | ✗ | 4 vs 8, but no impact on optimization — same global batch, same grad accum |
| Model init weights | ✗ | Base vs Joint (this IS the controlled variable) |

**GPU count difference does NOT affect training semantics** because:
- FSDP is data-parallel with gradient allreduce — final gradients are mathematically identical regardless of shard count
- Dynamic batching adjusts micro-batches per GPU, but total batch is the same
- vLLM rollout quality is independent of GPU count (same sampling params)

---

## 5. Action Items

- [ ] Create `recipe/joint_training/run_baseline_minirl_qwen3_1.7b_math.sh`
- [ ] Verify `Qwen3-1.7B-Base` exists at `/data-1/.cache/huggingface/Qwen3-1.7B-Base`
- [ ] Test-launch to validate configuration parses correctly

---

## 6. File Location

New script: `recipe/joint_training/run_baseline_minirl_qwen3_1.7b_math.sh`
