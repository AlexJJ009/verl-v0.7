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

---

## A800 Tuning Series (2026-04-04)

> **Hardware**: 8× NVIDIA A800-SXM4-80GB (80 GiB each, 640 GiB total)
> **Key differences from H800**: BF16 ~312 TFLOPS (vs ~495), HBM ~2 TB/s (vs 3.35), NVLink 400 GB/s (vs 900)
> **Duration**: ~3 hours (4 iterations + 2 OOM retries, each 5 training steps)
> **flash_attn**: NOT available in container → USE_REMOVE_PADDING=False (critical limitation)

### A800 Iter 0: Baseline

Same config as H800 optimized (iter4), run on A800.

| Metric | H800 (ref) | A800 (baseline) | Ratio |
|--------|-----------|-----------------|-------|
| Step time | 139.8s | **196.7s** | 1.41× |
| Generation | 45.2s | **45.6s** | 1.01× |
| Actor update | 78.2s | **113.5s** | 1.45× |
| Old log prob | 10.2s | **31.4s** | **3.08×** |
| Weight sync | 4.2s | 4.15s | 0.99× |
| GPU mem allocated | 31.17 GiB | **44.82 GiB** | 1.44× |
| MFU | 6.59% | **4.6%** | 0.70× |

**Key findings**:
- Generation is essentially the same (CUDA graphs + vLLM decode is not compute-bound)
- Actor update is 1.45× slower (expected from lower BF16 TFLOPS + slower NVLink)
- Old log prob is **3.08× slower** — disproportionate! Root cause: `LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4596` allows only ~1.8 sequences per micro-batch → 36 FSDP all-gather rounds per GPU. On A800's slower NVLink, each round costs more.
- GPU memory is 44.82 GiB vs 31.17 GiB because `USE_REMOVE_PADDING=False` → padded sequences consume more activation/logits memory

Logs: `recipe/joint_training/archive_logs/a800-tune-iter0-*.log`

---

### A800 Iter 1: Reduce log-prob FSDP rounds (BEST CONFIG)

| Parameter | Iter 0 | Iter 1 | Reason |
|-----------|--------|--------|--------|
| `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` | 4596 | **18384** | 4× budget → ~9 FSDP rounds instead of ~36 |
| `LOG_PROB_MICRO_BATCH_SIZE` | 4 | **16** | Match higher token budget |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.4 | **0.7** | More KV cache headroom during rollout |

Result: **-3.4% step time, -13.2% old_log_prob**

| Metric | A800 Iter 0 | A800 Iter 1 | Delta |
|--------|-------------|-------------|-------|
| Step time | 196.7s | **190.0s** | **-3.4%** |
| Generation | 45.6s | 44.1s | -3.4% |
| Actor update | 113.5s | 112.3s | -1.0% |
| Old log prob | 31.4s | **27.2s** | **-13.2%** |
| GPU mem | 44.82 GiB | 44.82 GiB | same |

Logs: `recipe/joint_training/archive_logs/a800-tune-iter1-1775297732.log`

---

### A800 Iter 1 (OOM attempts)

**OOM #1**: `ACTOR_PPO_MAX_TOKEN_LEN=36768` + `ppo_mini_batch_size=16`
- Training process: 76.30 GiB, tried to allocate +2.32 GiB → OOM
- Root cause: `ppo_mini_batch_size=16` with `USE_REMOVE_PADDING=False` → 16 sequences × 4596 padded tokens = 73K padded tokens per GPU → 76 GiB for logits + activations

**OOM #2**: `ACTOR_PPO_MAX_TOKEN_LEN=28000` + `ppo_mini_batch_size=16`
- Same OOM (76.30 GiB) — the token budget doesn't matter because padding dominates. The 16 sequences are padded to ~4596 each regardless of the budget.

**Key insight**: Without `USE_REMOVE_PADDING`, increasing `ppo_mini_batch_size` or `actor_ppo_max_token_len` is futile — the memory is dominated by padding to max sequence length. Each additional sequence costs ~4596 × vocab_size × 2 bytes ≈ 1.33 GiB for the logits tensor alone.

Logs: `recipe/joint_training/archive_logs/a800-tune-iter1-1775296{293,809}.log`

---

### A800 Iter 2: HSDP (NO IMPROVEMENT)

| Parameter | Iter 1 | Iter 2 | Reason |
|-----------|--------|--------|--------|
| `fsdp_size` | -1 (8-way) | **4** (HSDP: 2×4) | Reduce all-gather group size |
| `ROLLOUT_MAX_NUM_BATCHED_TOKENS` | 4596 | **16384** | Faster vLLM prefill scheduling |

Result: **No improvement** — actor update 112.5s (vs 112.3s), +5.6 GiB memory

**Analysis**: HSDP reduces all-gather data by ~14% but adds inter-group gradient sync. Net effect is zero. The training bottleneck is fundamentally the padded computation, not communication protocol.

Logs: `recipe/joint_training/archive_logs/a800-tune-iter2-*.log`

---

### A800 Iter 3: torch.compile (NO IMPROVEMENT)

| Parameter | Iter 1 | Iter 3 | Reason |
|-----------|--------|--------|--------|
| `use_torch_compile` | False | **True** | Fuse training kernels |

Result: **No improvement** — actor update 112.2s (vs 112.3s)

**Analysis**: Dynamic shapes from variable-length padded sequences prevent effective graph capture. torch.compile adds compilation overhead without meaningful kernel fusion.

Logs: `recipe/joint_training/archive_logs/a800-tune-iter3-*.log`

---

### A800 Optimized Config (applied to script, pre-flash_attn)

| Parameter | Before | After | Source |
|-----------|--------|-------|--------|
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.4 | **0.7** | A800 iter 1 |
| `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` | 4596 | **18384** | A800 iter 1 |
| `LOG_PROB_MICRO_BATCH_SIZE` | 4 | **16** | A800 iter 1 |

### A800 Net Result (pre-flash_attn)
```
Step time:    196.7s → 190.0s  (-3.4%)
Old log prob:  31.4s →  27.2s  (-13.2%)
Generation:    45.6s →  44.1s  (-3.4%)
GPU mem:       44.82 GiB (56%) — structural limit without remove_padding
```

### A800-Specific Takeaways (pre-flash_attn)
1. **Install flash_attn** — this is the single most impactful unlocked optimization. Without it, `USE_REMOVE_PADDING=False` causes sequences to be padded to max length, wasting ~40-80% of compute and memory. With remove_padding, `ppo_mini_batch_size` can safely increase from 8 to 16+ (halving FSDP rounds), and training memory drops from 44.8 to ~31 GiB. Estimated improvement: **-30 to -40% actor update time**.
2. **LOG_PROB_MAX_TOKEN_LEN_PER_GPU** matters much more on A800 than H800 due to slower NVLink. Increasing from 1× to 4× sequence length reduces old_log_prob by 13%.
3. **HSDP and torch.compile** do not help for this workload (4B model, padded sequences, dynamic shapes).
4. **A800 vs H800 ratio**: generation ~1:1 (decode-bound), training ~1.45× (compute + NVLink), old_log_prob ~3× at default config (NVLink-dominated micro-batching).
5. **GPU memory utilization ceiling** without flash_attn: ~56% during training, ~70% during rollout. The 80-90% target requires remove_padding to eliminate the padding overhead that inflates per-sequence memory cost.

---

## A800 Flash-Attn Tuning Series (2026-04-04)

> **Hardware**: 8× NVIDIA A800-SXM4-80GB (80 GiB each, 640 GiB total)
> **Duration**: ~2 hours (3 iterations, each 5 training steps)
> **flash_attn**: 2.8.1 (installed via docker commit with --network=host)
> **Key change**: Dockerfile fix — move flash_attn install after all other pip deps to prevent uv removal

### A800 Iter 5: flash_attn + USE_REMOVE_PADDING=True + mini_bsz=16 (sdpa)

| Parameter | Iter 1 (old best) | Iter 5 | Reason |
|-----------|-------------------|--------|--------|
| `USE_REMOVE_PADDING` | False | **True** | flash_attn now available |
| `train_prompt_mini_bsz` | 8 | **16** | remove_padding eliminates padding overhead |
| `actor_ppo_max_token_len` | 18384 | **36768** | 2× budget for larger mini-batches |
| `attn_implementation` | sdpa | sdpa | unchanged in this iteration |

Result: **-35.5% step time, -46.3% actor update**

| Metric | A800 Iter 1 | A800 Iter 5 | Delta |
|--------|-------------|-------------|-------|
| Step time | 190.0s | **122.5s** | **-35.5%** |
| Generation | 44.1s | 45.8s | +3.9% |
| Actor update | 112.3s | **60.3s** | **-46.3%** |
| Old log prob | 27.2s | **10.4s** | **-61.9%** |
| GPU mem | 44.82 GiB | **40.90 GiB** | -8.7% |
| MFU | 4.6% | **8.7%** | +89% |

Logs: `recipe/joint_training/archive_logs/a800-iter5-flash-attn-rp_1775304298.log`

---

### A800 Iter 6: mini_bsz=32 (NO IMPROVEMENT)

| Parameter | Iter 5 | Iter 6 | Reason |
|-----------|--------|--------|--------|
| `train_prompt_mini_bsz` | 16 | **32** | Hypothesis: halve FSDP rounds |
| `actor_ppo_max_token_len` | 36768 | **73536** | Match larger mini-batch |

Result: **WORSE** — actor update 81.3s (vs 60.3s), memory 48.6 GiB (vs 40.9 GiB)

**Analysis**: Larger micro-batches increase per-round activation memory and compute without proportionally reducing FSDP communication. On A800's slower NVLink, the communication is already a smaller fraction of round time, so halving rounds doesn't compensate for doubled per-round cost.

Logs: `recipe/joint_training/archive_logs/a800-iter6-minibsz32_1775305339.log`

---

### A800 Iter 7: flash_attention_2 backend (BEST CONFIG — MASSIVE IMPROVEMENT)

| Parameter | Iter 5 | Iter 7 | Reason |
|-----------|--------|--------|--------|
| `attn_implementation` | sdpa | **flash_attention_2** | flash_attn's varlen API is optimized for packed variable-length sequences |

Result: **-63.5% step time, -87.4% actor update** vs Iter 1

| Metric | A800 Iter 1 | A800 Iter 5 (sdpa) | A800 Iter 7 (fa2) | Iter1→7 |
|--------|-------------|--------------------|--------------------|---------|
| Step time | 190.0s | 122.5s | **69.4s** | **-63.5%** |
| Generation | 44.1s | 45.8s | 45.7s | +3.7% |
| Actor update | 112.3s | 60.3s | **14.1s** | **-87.4%** |
| Old log prob | 27.2s | 10.4s | **3.7s** | **-86.6%** |
| GPU mem | 44.82 GiB | 40.90 GiB | **39.1 GiB** | -12.8% |
| MFU | 4.6% | 8.7% | **37.0%** | **8× improvement** |
| Throughput | 294 tok/s | ~470 tok/s | **~850 tok/s** | **2.9×** |

**Analysis**: flash_attention_2's varlen (variable-length) attention API eliminates redundant computation on padding and fuses QKV operations. Combined with remove_padding (which packs sequences tightly), the forward/backward pass becomes dramatically more efficient. The sdpa backend cannot leverage packed sequence layouts as efficiently.

**Per-step data** (iter 7):
| Step | Total | Gen | Actor | LogProb | WSync | Mem | MFU |
|------|-------|-----|-------|---------|-------|-----|-----|
| 1 | 73.1s | 48.9s | 15.1s | 4.9s | 4.1s | 38.83 GiB | 36.2% |
| 2 | 69.1s | 47.5s | 14.0s | 3.7s | 3.8s | 38.83 GiB | 37.1% |
| 3 | 65.8s | 44.5s | 13.9s | 3.6s | 3.8s | 38.83 GiB | 36.9% |
| 4 | 67.3s | 45.0s | 14.6s | 3.7s | 3.9s | 39.35 GiB | 37.2% |
| 5 | 75.3s | 45.9s | 13.9s | 3.6s | 3.8s | 39.35 GiB | 36.9% |

Logs: `recipe/joint_training/archive_logs/a800-iter7-fa2-attn_1775306581.log`

---

### A800 Final Optimized Config (applied to all scripts)

| Parameter | Before (pre-flash_attn) | After | Source |
|-----------|------------------------|-------|--------|
| `USE_REMOVE_PADDING` | False | **True** | iter 5 |
| `train_prompt_mini_bsz` | 8 | **16** | iter 5 |
| `actor_ppo_max_token_len` | 18384 | **36768** | iter 5 |
| `attn_implementation` | sdpa | **flash_attention_2** | iter 7 |

### A800 Final Net Result
```
Step time:    190.0s →  69.4s  (-63.5%)
Actor update: 112.3s →  14.1s  (-87.4%)
Old log prob:  27.2s →   3.7s  (-86.6%)
Generation:    44.1s →  45.7s  (unchanged, decode-bound)
Throughput:     294  →   850 tok/s  (+189%)
MFU:           4.6%  →  37.0%  (8× improvement)
GPU mem:      44.82  →  39.1 GiB  (-12.8%)
```

### A800 Final Takeaways
1. **flash_attention_2 + remove_padding** is by far the most impactful optimization — combined they deliver 87% actor update reduction and 8× MFU improvement.
2. **Attention backend matters enormously**: sdpa with remove_padding gives 60.3s actor update; flash_attention_2 with remove_padding gives 14.1s — a further 77% reduction, purely from the attention kernel.
3. **mini_bsz=16 is the A800 sweet spot**. Going to 32 hurts due to per-micro-batch memory pressure outweighing FSDP round reduction.
4. **Generation is now the bottleneck** (66% of step time). Further optimization requires faster decode (e.g., speculative decoding, longer prefill, or smaller response lengths).
5. **Estimated full-training time**: 700 steps × 69.4s ≈ 13.5 hours (previously ~37 hours at 190s/step).
