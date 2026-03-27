# Sample Efficiency Optimization — Iteration Log

> **Created**: 2026-03-27
> **Last updated**: 2026-03-27
> **Duration**: ~3 hours (5 iterations, each 5 training steps)
>
> **Hardware**: 8x NVIDIA H800 (80 GiB each, 640 GiB total)
> **Model**: Qwen3-4B SFT-Stage-1 (bf16, ~8 GiB total weights)
> **Framework**: verl (FSDP + vLLM 0.12.0 colocated), PyTorch 2.9.1
> **Script**: `recipe/joint_training/run_baseline_minirl_qwen3_4b_math.sh`
>
> **Background**: Before starting full-length baseline RL training (700 steps, ~27 hours),
> we ran a series of short (5-step) experiments to find the optimal parameter configuration
> for sample efficiency and GPU utilization. Each iteration tests a specific hypothesis,
> measures wall-clock time and memory, and feeds results into the next iteration.
>
> **Goal**: Maximize GPU memory utilization (target 80-90%), minimize scheduling/communication
> overhead, and increase throughput (tokens/second). Prompt/response lengths and epoch count
> are fixed constraints — only batch, memory, and rollout parameters are tunable.
>
> **Companion doc**: Parameter definitions and tuning guide at
> `docs/joint_training/guides/sample_efficiency_tuning.md`

---

## Iter0: Baseline (FAILED — OOM)

### Parameters
| Parameter | Value |
|-----------|-------|
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.70 |
| `actor_ppo_max_token_len` | 18,384 |
| `LOG_PROB_MICRO_BATCH_SIZE` | 4 |
| `train_prompt_bsz` | 64 |
| `n_resp_per_prompt` | 8 |
| `train_prompt_mini_bsz` | 8 |
| `ROLLOUT_MAX_NUM_SEQS` | 512 |

### Result: OOM during backward pass
```
torch.OutOfMemoryError: Tried to allocate 4.21 GiB.
GPU 0: 79.44 GiB total, 1.74 GiB free.
  vLLM process: 53.66 GiB
  FSDP process: 24.02 GiB (needs +4.21 GiB for backward)
  Total needed: 81.89 GiB > 79.44 GiB
```

### Root Cause
vLLM (0.70 × 80 = 56 GiB) and FSDP training (~28 GiB) share the same GPU simultaneously. vLLM does NOT release KV cache memory during the training phase.

### Logs
Archived: `recipe/joint_training/archive_logs/iter0-baseline-SFT-4B_*.log`

---

## Iter1: Lower vLLM Memory (SUCCESS)

### Adjustments from Iter0
| Parameter | Iter0 | Iter1 | Reason |
|-----------|-------|-------|--------|
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.70 | **0.45** | Reduce vLLM footprint (56→36 GiB) to leave room for training |
| `ROLLOUT_MAX_NUM_SEQS` | 512 | **256** | Match reduced KV cache capacity |

### Calculation
```
vLLM at 0.45: 0.45 × 80 = 36 GiB
Available for training: 80 - 36 = 44 GiB
Expected training peak: ~28-35 GiB (from iter0 data)
Headroom: ~9-16 GiB → should be safe
```

### Result: 5/5 steps completed
| Metric | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Average |
|--------|--------|--------|--------|--------|--------|---------|
| Total time (s) | 156.2 | 140.5 | 152.8 | 148.5 | 151.7 | **149.9** |
| Generation (s) | 60.8 | 53.6 | 55.7 | 53.9 | 53.0 | **55.4 (37%)** |
| Actor update (s) | 79.9 | 73.3 | 82.3 | 80.4 | 75.4 | **78.3 (52%)** |
| Old log prob (s) | 11.4 | 9.6 | 10.3 | 10.1 | 9.8 | **10.2 (7%)** |
| Weight sync (s) | 4.0 | 4.0 | 4.4 | 4.0 | 4.5 | **4.2 (3%)** |

| Memory Metric | Value |
|--------------|-------|
| GPU max_memory_allocated | **31.17 GiB** (stable) |
| GPU max_memory_reserved | **35.66 GiB** |
| Total GPU usage (vLLM + training) | ~36 + 35.66 = **71.66 GiB (89.6%)** |
| MFU (actor) | **6.59%** |
| Throughput | **1,149 tok/s** |
| Avg tokens/step | **1.38M** |

### Key Observations
1. **Training is the bottleneck** (52% of step time vs 37% for rollout)
2. **MFU is very low (6.59%)** — GPUs are underutilized during training
3. **GPU utilization is 89.6%** of total memory — close to target but MFU is poor
4. **Reference model confirmed NOT loaded** (use_kl_loss=False, use_kl_in_reward=False → need_reference_policy()=False)
5. **31 GiB training memory** is dominated by activation/logits memory, not model weights:
   - FSDP base (model shard + optimizer + grad): ~4 GiB
   - Activation + logits memory: ~27 GiB for 18K tokens
   - Logits tensor: vocab_size (151,936) × tokens per micro-batch → ~5+ GiB in bf16

### Logs
Archived: `recipe/joint_training/archive_logs/iter1-low-vllm-mem_*.log`

---

## Iter2: Larger Token Budget (NO EFFECT)

### Adjustments from Iter1
| Parameter | Iter1 | Iter2 | Reason |
|-----------|-------|-------|--------|
| `actor_ppo_max_token_len` | 18,384 | **24,000** | Hypothesis: fewer micro-batches → better kernel utilization |
| `LOG_PROB_MICRO_BATCH_SIZE` | 4 | **8** | Hypothesis: faster log-prob computation |

### Result: No measurable change
| Metric | Iter1 | Iter2 | Delta |
|--------|-------|-------|-------|
| Step time (s) | 149.9 | 149.0 | -0.6% |
| Generation (s) | 55.4 | 54.6 | -1.5% |
| Actor update (s) | 78.2 | 78.2 | -0.1% |
| Old log prob (s) | 10.2 | 10.2 | -0.2% |
| GPU mem allocated | 31.17 GiB | 31.16 GiB | ~same |
| MFU | 6.59% | 6.58% | ~same |

### Analysis
**Why no effect?** Dynamic batching packs sequences by actual token count. Average sequences are ~2600 tokens. The micro-batch token count was already well below the 18K limit, so raising it to 24K had no impact. The bottleneck is FSDP communication and compute, not batch overhead.

### Logs
Archived: `recipe/joint_training/archive_logs/iter2-larger-token-budget_*.log`

---

## Open Questions (Human Feedback)

### Q1: Why does training use 31 GiB for a 4B model?
**Answer**: Most of the 31 GiB is NOT model weights. The breakdown:
- FSDP base (model shard + optimizer + gradients, all sharded across 8 GPUs): ~4 GiB
- **Logits tensor**: vocab_size × tokens_per_micro_batch × 2 bytes (bf16) = 151,936 × 18,384 × 2 = **~5.3 GiB**
- Activation recomputation during backward (gradient checkpointing): **~5-8 GiB**
- FSDP communication buffers (all-gather, reduce-scatter): **~3-5 GiB**
- PyTorch CUDA allocator overhead / fragmentation: **~5-10 GiB**

### Q2: Can we increase vLLM utilization since rollout and training are sequential?
**Investigation needed**: In verl's colocated mode, vLLM holds its memory (model weights + KV cache) even during the training phase. If we can enable `ROLLOUT_FREE_CACHE_ENGINE=True` or `ROLLOUT_ENABLE_SLEEP_MODE=True`, vLLM would release its KV cache after rollout, freeing ~28-56 GiB for training.

The script currently disables both due to a **vLLM 0.8.5 bug**, but we run **vLLM 0.12.0** which may have fixed this. If so, the optimal strategy would be:
- Rollout phase: vLLM at 0.85-0.90 (fast generation)
- After rollout: release KV cache
- Training phase: use freed memory for larger token budgets → much higher MFU

**Status**: CONFIRMED SAFE. vLLM 0.12 supports `sleep(level=1)` in colocated mode. KV cache is fully released and recreated from scratch — no numerical differences. Code verified in `verl/workers/rollout/vllm_rollout/vllm_async_server.py:674-687` and `verl/workers/fsdp_workers.py:855-893`.

---

## Iter3: Enable FREE_CACHE_ENGINE (IN PROGRESS)

### Adjustments from Iter1 (skipping iter2 since it had no effect)
| Parameter | Iter1 | Iter3 | Reason |
|-----------|-------|-------|--------|
| `ROLLOUT_FREE_CACHE_ENGINE` | False | **True** | Release KV cache between phases → more vLLM memory during rollout |
| `ROLLOUT_ENABLE_SLEEP_MODE` | False | **True** | Enable sleep/wake mechanism |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.45 | **0.85** | With KV cache released, vLLM can use much more GPU memory during rollout |
| `ROLLOUT_MAX_NUM_SEQS` | 256 | **512** | More KV cache → more concurrent sequences |

### Calculation
```
During rollout:
  vLLM at 0.85: 0.85 × 80 = 68 GiB (model 8 GB + KV cache 60 GB)
  KV per token (Qwen3-4B): ~144 KB
  Max concurrent tokens: 60 GiB / 144 KB ≈ 417K tokens
  At avg 2500 tokens/seq: ~167 concurrent sequences → comfortable

After rollout (sleep):
  vLLM releases KV cache → only model weights ~8 GiB remain
  Available for training: 80 - 8 = 72 GiB → very comfortable (training needs ~31 GiB)

After training (wake_up):
  vLLM KV cache rebuilt from scratch → no numerical differences
```

### Result: Minimal improvement (-1.4%)
| Metric | Iter1 | Iter3 | Delta |
|--------|-------|-------|-------|
| Step time | 149.9s | 147.9s | -1.4% |
| Generation | 55.4s | 55.2s | -0.4% |
| Actor update | 78.2s | 76.6s | -2.2% |

**Analysis**: Rollout is decode-bound (4096 tokens sequential generation), not KV-cache-bound. More KV cache doesn't speed up decode. Free cache engine is safe but doesn't improve throughput.

### Logs
Archived: `recipe/joint_training/archive_logs/iter3-free-cache-engine_*.log`

---

## Iter4: CUDA Graphs (SIGNIFICANT IMPROVEMENT)

### Adjustments from Iter3
| Parameter | Iter3 | Iter4 | Reason |
|-----------|-------|-------|--------|
| `ROLLOUT_ENFORCE_EAGER` | true | **false** | Enable CUDA graphs in vLLM decode → kernel fusion, less launch overhead |

### Result: -6.7% step time, -18.4% generation time
| Metric | Iter1 (baseline) | Iter4 (CUDA graphs) | Delta |
|--------|-----------------|---------------------|-------|
| Step time | 149.9s | **139.8s** | **-6.7%** |
| Generation | 55.4s | **45.2s** | **-18.4%** |
| Actor update | 78.2s | 78.2s | 0.0% |
| Old log prob | 10.2s | 10.0s | -2.0% |
| Throughput | 1149 tok/s | **1233 tok/s** | **+7.3%** |
| GPU mem | 31.17 GiB | 31.17 GiB | 0.0% |
| MFU | 6.59% | 6.59% | 0.0% |

**Analysis**: CUDA graphs (FULL_AND_PIECEWISE mode, 51 graphs captured) eliminate kernel launch overhead in vLLM decode. This is the most effective single optimization so far. No memory cost, no numerical impact. Entire 10s saving comes from generation phase.

### Current time breakdown (iter4)
```
Generation:    45.2s (32.3%)
Actor update:  78.2s (55.9%)
Old log prob:  10.0s (7.1%)
Weight sync:    4.1s (2.9%)
Other:          2.3s (1.6%)
Total:        139.8s
```

### Logs
Archived: `recipe/joint_training/archive_logs/iter4-cuda-graphs_*.log`

---

## Iteration Plan

| Iter | Focus | Status |
|------|-------|--------|
| 0 | Baseline | OOM |
| 1 | Lower vLLM memory (0.45) | Done (149.9s/step, MFU 6.59%) |
| 2 | Larger token budget (24K) + larger log_prob micro_bsz | Done (NO EFFECT) |
| 3 | Enable FREE_CACHE_ENGINE + higher vLLM util (0.85) | Done (-1.4%, minimal) |
| 4 | CUDA graphs (enforce_eager=false) | **Done (-6.7%, significant)** |
| 5 | Optimize training phase (torch_compile, batch sizes) | Planned |
| 6+ | Fine-tune for production | Planned |

---

## Final Summary

### Optimized config written to script
All findings have been applied to `recipe/joint_training/run_baseline_minirl_qwen3_4b_math.sh` as new defaults:

| Parameter | Before | After | Source |
|-----------|--------|-------|--------|
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.70 | **0.85** | iter1/iter3 |
| `ROLLOUT_ENFORCE_EAGER` | true | **false** | iter4 |
| `ROLLOUT_FREE_CACHE_ENGINE` | False | **True** | iter3 |
| `ROLLOUT_ENABLE_SLEEP_MODE` | False | **True** | iter3 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `""` (conditional) | iter3 |
| `MODEL_PATH` | Qwen3-4B-Base | **SFT-Stage-1** | initial setup |

### Net result
```
Step time:    149.9s → 139.8s  (-6.7%)
Generation:    55.4s →  45.2s  (-18.4%)
Throughput:    1149  →  1233 tok/s  (+7.3%)
GPU mem util:  OOM at 0.70  →  85% stable
```

### Key takeaways for future scripts
1. Always enable CUDA graphs (`enforce_eager=false`) for vLLM — it's free performance.
2. Always enable `free_cache_engine=True` with vLLM >= 0.11 — it decouples rollout and training memory budgets.
3. When `free_cache_engine=True`, do NOT set `expandable_segments:True` at startup — let verl manage it.
4. Token budget tuning (`actor_ppo_max_token_len`) has no effect when sequences are much shorter than the limit. Don't over-tune.
5. MFU ~6.6% is a structural limit for small models (4B) on many GPUs (8). The optimization ceiling is in FSDP communication, not parameter tuning.
