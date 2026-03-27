# verl RLHF Training Parameters: Sample & Training Efficiency Tuning Guide

> **Created**: 2026-03-27
> **Last updated**: 2026-03-27
> **Scope**: All verl RLHF training scripts on this project (baseline and joint-training)
> **Validated on**: 8x NVIDIA H800 (80 GB each), Qwen3-4B (bf16), vLLM 0.12.0, PyTorch 2.9.1
>
> **Background**: This guide was produced during a systematic sample-efficiency optimization
> loop (5 iterations, see `docs/joint_training/guides/sample_efficiency_iteration_log.md`).
> Every recommendation below is backed by empirical measurement on real training runs,
> not theoretical estimates alone.
>
> **Motivation**: In colocated RLHF training (verl), rollout (vLLM) and training (FSDP)
> share the same GPUs. Tuning requires balancing memory between phases, maximizing GPU
> compute utilization, and minimizing scheduling overhead. This guide captures the
> parameter interactions and pitfalls discovered during that process.

## Training Pipeline Overview

Each training step has two **sequential** phases sharing the same GPUs:

```
┌─────────────┐  sleep()  ┌──────────────┐  wake_up()  ┌─────────────┐
│  Rollout     │ ────────► │  Training     │ ──────────► │  Next Rollout│
│  (vLLM)      │           │  (FSDP/PPO)   │             │  (vLLM)      │
│  generate    │           │  gradient     │             │              │
│  responses   │           │  updates      │             │              │
└─────────────┘           └──────────────┘             └─────────────┘
     Phase 1                   Phase 2                     Phase 1
  GPU: vLLM 68 GB           GPU: FSDP 31 GB             GPU: vLLM 68 GB
  (model + KV cache)        (model + optim + act)       (KV cache rebuilt)

With free_cache_engine=True, vLLM releases KV cache between phases via sleep()/wake_up().
Without it, vLLM holds KV cache throughout, competing with training for GPU memory.
```

---

## A. Batch & Sample Parameters

### `train_prompt_bsz` (B) — Prompts per step
- **Current**: 64
- **Effect**: Total samples/step = B x G. Larger B = more data per step, more rollout time.
- **Memory**: Determines total rollout workload and total training data volume.

### `n_resp_per_prompt` (G) — Responses per prompt
- **Current**: 8
- **Effect**: Total samples/step = B x G = 512. More G = better advantage estimation (GRPO needs multiple responses).
- **Rollout**: vLLM generates G sequences per prompt.

### `train_prompt_mini_bsz` — PPO mini-batch size (in prompts)
- **Current**: 8
- **Effect**: Number of gradient updates per step = (B x G) / mini_bsz = 512/8 = 64 updates.
- **Memory**: Smaller = less VRAM per update but more updates. Larger = more VRAM, fewer updates.
- **With dynamic_bsz=True**: This is a *target*; actual size varies based on token content.

### `GENERATION_MICRO_BATCH_SIZE` — Micro-batch for vLLM generation dispatch
- **Current**: 16
- **Effect**: How many prompts are sent to vLLM per dispatch batch.

### `LOG_PROB_MICRO_BATCH_SIZE` — Micro-batch for log-prob computation
- **Current**: 4
- **Effect**: Batch size for computing log P(response|prompt) on actor/ref models.
- **Memory**: Higher = faster but more VRAM. Lower = safer, slower.

---

## B. Token Budget Parameters

### `actor_ppo_max_token_len` — Max tokens per GPU for actor PPO training
- **Current**: 18,384 = (500 + 4096) x 4
- **Effect**: Controls how full each GPU gets during training micro-batches.
- **With dynamic_bsz**: Framework packs sequences up to this token limit per micro-batch.
- **Memory math** (H800, 4B model):
  - Model weights: ~8 GB
  - FSDP optimizer (Adam m+v): ~16 GB (sharded across 8 GPUs = ~2 GB/GPU)
  - Gradients: ~1 GB/GPU (sharded)
  - Available for activations/tokens: ~69 GB/GPU
  - Gradient checkpointing enabled: activation memory is modest

### `infer_ppo_max_token_len` — Max tokens per GPU for inference (log-prob)
- **Current**: 27,576 = (500 + 4096) x 6
- **Effect**: Higher budget OK since no backward pass (no grad/optim memory).

### `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` — Max tokens for log-prob per GPU
- **Current**: 4,596 = (500 + 4096)
- **Effect**: Token budget for ref model log-prob. Can be higher than actor since inference-only.

---

## C. Rollout (vLLM) Parameters

### `ROLLOUT_GPU_MEMORY_UTILIZATION` — vLLM GPU memory fraction
- **Optimized default**: **0.85** (was 0.70; requires `free_cache_engine=True`)
- **Effect**: Controls KV cache pool size during rollout phase.
- **Math** (H800 80 GB, 4B model):
  ```
  0.45 → 36 GB total → 28 GB KV cache (conservative, no free_cache_engine)
  0.70 → 56 GB total → 48 GB KV cache
  0.85 → 68 GB total → 60 GB KV cache (recommended with free_cache_engine)
  0.90 → 72 GB total → 64 GB KV cache
  ```
- **KV cache per token** (Qwen3-4B, GQA 8 heads, head_dim=128, 36 layers, bf16):
  ```
  2 x 8 x 128 x 2 bytes x 36 = 147,456 bytes ≈ 144 KB/token
  ```
- **Max concurrent tokens**: 60 GB / 144 KB ≈ 417K tokens (at 0.85)
- **Empirical finding**: Increasing from 0.45 to 0.85 (with free_cache_engine) did NOT speed up generation. Rollout is decode-bound, not KV-cache-bound. The benefit is allowing free_cache_engine to release more memory for training.

### `ROLLOUT_MAX_NUM_SEQS` — Max concurrent sequences
- **Current**: 512
- **Constraint**: Limited by KV cache. At avg 2500 tokens/seq: 333K / 2500 ≈ 133 concurrent seqs.
- **Note**: vLLM manages this dynamically. Setting higher than KV budget just means queuing.

### `ROLLOUT_MAX_NUM_BATCHED_TOKENS` — Max tokens per scheduler iteration
- **Current**: 4,596
- **Effect**: Limits prefill batch size. Higher = more throughput, higher latency per iteration.

### `ROLLOUT_MAX_MODEL_LEN` — Max sequence length for vLLM
- **Current**: 4,596 (= max_prompt + max_response)
- **Do not change** (matches context length constraint).

### `ROLLOUT_TP_SIZE` — Tensor parallel size
- **Current**: 1 (4B fits on 1 GPU)
- **Do not change** for 4B model.

### `ROLLOUT_AGENT_NUM_WORKERS` — Async rollout workers
- **Current**: 4
- **Effect**: Parallelism in dispatching rollout requests.

### `ROLLOUT_ENABLE_CHUNKED_PREFILL` — Chunked prefill
- **Current**: true
- **Effect**: Splits large prompts into chunks, reduces prefill peak memory.

### `ROLLOUT_ENFORCE_EAGER` — Disable CUDA graphs in vLLM
- **Optimized default**: **false** (was true; CUDA graphs enabled)
- **Effect**: When true, forces vLLM into eager mode (no kernel fusion, no CUDA graphs). When false, vLLM uses `cudagraph_mode: FULL_AND_PIECEWISE` — capturing decode kernels into CUDA graphs for replay.
- **Empirical finding (iter4)**: Switching from true→false reduced generation time by **18.4%** (55.4s → 45.2s). This was the single most impactful optimization in the entire iteration series. No memory cost, no numerical impact.
- **Caveat**: First step may be slightly slower due to graph compilation warmup (51 graphs captured).

### `ROLLOUT_FREE_CACHE_ENGINE` / `ROLLOUT_ENABLE_SLEEP_MODE`
- **Optimized default**: **True / True** (was False / False)
- **Effect**: After rollout completes, `sleep(level=1)` releases KV cache, keeping only model weights in GPU memory (~8 GB). Before next rollout, `wake_up()` rebuilds KV cache from scratch. This frees ~60 GB during training.
- **Requirement**: `PYTORCH_CUDA_ALLOC_CONF` must NOT contain `expandable_segments:True` at process startup. The verl framework manages `expandable_segments` dynamically via `set_expandable_segments()` in `verl/utils/device.py`.
- **Numerical safety**: Confirmed safe — KV cache is fully recomputed, `reset_prefix_cache()` clears any stale values. Code path: `verl/workers/rollout/vllm_rollout/vllm_async_server.py:674-687`, `verl/workers/fsdp_workers.py:855-893`.
- **Empirical finding (iter3)**: Enabling free_cache_engine alone improved step time by only 1.4% — rollout is decode-bound, not memory-bound. But it is a prerequisite for using high `gpu_memory_utilization` (0.85) safely.

### `PYTORCH_CUDA_ALLOC_CONF` — PyTorch CUDA memory allocator config
- **Optimized default**: `""` (empty) when `free_cache_engine=True`; `expandable_segments:True` otherwise.
- **Effect**: `expandable_segments:True` reduces CUDA memory fragmentation for FSDP training. However, vLLM's memory pool asserts that this is NOT set (PyTorch issue #147851). When `free_cache_engine=True`, the script leaves this empty and lets verl toggle it dynamically at runtime.
- **Error if wrong**: `AssertionError: Expandable segments are not compatible with memory pool.`

---

## D. FSDP & Memory Parameters

### `use_dynamic_bsz` — Dynamic batch sizing
- **Current**: True
- **Effect**: Packs variable-length sequences into micro-batches by token count, not sample count.
- **How**: Uses `prepare_dynamic_batch()` with workload formula: `workload = 24576*seqlen + seqlen^2`

### `USE_REMOVE_PADDING` — Remove padding before forward pass
- **Current**: True (requires flash-attn)
- **Effect**: Eliminates wasted compute on padding tokens. Critical for variable-length sequences.

### `sp_size` — Sequence parallel size
- **Current**: 1 (disabled). Not needed for 4B model.

### `offload` — FSDP CPU offload
- **Current**: False. Not needed — 80 GB H800 has ample memory for 4B.

### `fsdp_size` — Sharding group size
- **Current**: -1 (full sharding across all 8 GPUs)

### `enable_gradient_checkpointing` — Activation checkpointing
- **Current**: True
- **Effect**: Discards intermediate activations during forward, recomputes them during backward. Trades compute for memory. Essential for large token budgets.
- **Memory impact**: Without it, activation memory can 2-3x. With it, only 1-2 layers' activations are held during backward.

### `use_torch_compile` — PyTorch compiler for training
- **Current**: False
- **Effect**: Fuses training kernels via `torch.compile()`. Could reduce actor update time.
- **Risk**: Compatibility with FSDP + dynamic shapes is uncertain. Not tested in iterations.
- **Status**: Candidate for future optimization of training phase.

### `entropy_from_logits_with_chunking` — Chunked entropy computation
- **Current**: True
- **Effect**: Computes entropy from logits in chunks instead of materializing full logits tensor at once. Reduces peak memory during loss computation.

### `grad_clip` — Gradient clipping threshold
- **Current**: 500.0
- **Note**: Observed gradient norms of 535-654 in early steps, meaning clipping is active. This is expected during warmup with MiniRL loss.

---

## E. Schedule Parameters

### `total_epochs` — Max epochs over training data
- **Current**: 3. **Constraint: cannot exceed 3.**

### `total_training_steps` — Max training steps
- **Current**: 700. For testing: set to 5-10.

### `test_freq` / `save_freq`
- **Current**: 5 / 20. For testing: set test_freq high to avoid validation overhead.

---

## F. Key Relationships

### Samples per step
```
total_samples_per_step = train_prompt_bsz x n_resp_per_prompt
                       = 64 x 8 = 512
```

### Gradient updates per step
```
updates_per_step ≈ total_tokens / (actor_ppo_max_token_len x num_gpus)
                 ≈ (512 x ~2500) / (18384 x 8) ≈ 8.7 micro-batches/GPU
```

### Training throughput
```
tokens_per_step = total_samples x avg_seq_len
                = 512 x (500 + ~2000) ≈ 1.28M tokens
```

### Memory budget per GPU (training phase) — empirical
```
Measured peak (Qwen3-4B, 8x H800, FSDP full shard):
  max_memory_allocated: 31.17 GiB
  max_memory_reserved:  35.66 GiB

Breakdown (estimated from OOM analysis + profiling):
  FSDP base (model shard + optimizer + gradients): ~4 GiB
  Logits tensor (vocab 151936 x tokens x 2B bf16):  ~5 GiB
  Activation recomputation (gradient checkpointing): ~5-8 GiB
  FSDP communication buffers (all-gather/reduce):    ~3-5 GiB
  PyTorch allocator overhead / fragmentation:         ~5-10 GiB
  Total:                                              ~27-31 GiB

Key insight: Most training memory is NOT model weights.
The logits tensor (vocab_size x micro_batch_tokens) dominates.
```

### Memory budget per GPU (rollout phase) — empirical
```
vLLM allocation: 80 GB x gpu_memory_utilization
  At 0.45: 36 GB (model 8 GB + KV cache 28 GB) — conservative, no free_cache
  At 0.85: 68 GB (model 8 GB + KV cache 60 GB) — recommended with free_cache

With free_cache_engine=True:
  During rollout: full allocation (e.g. 68 GB at 0.85)
  After sleep():  only model weights ~8 GB retained
  During training: 80 - 8 = 72 GB available for FSDP (ample for 31 GB peak)

Without free_cache_engine:
  CONSTRAINT: vLLM_mem + training_peak ≤ 80 GB
  At 0.45: 36 + 35.66 = 71.66 GB (OK, ~90% utilization)
  At 0.70: 56 + 28 = 84 GB (OOM! confirmed in iter0)
```

---

## G. Optimization Results & Tuning Priorities

### Achieved performance (8x H800, Qwen3-4B, optimized config)

| Metric | Before (iter0/1) | After (iter4) | Status |
|--------|------------------|---------------|--------|
| GPU memory utilization (peak) | 90% (OOM at 0.70 vLLM) | **85%** (with free_cache) | **Achieved** |
| Step time | 149.9s | **139.8s** | -6.7% |
| Generation time | 55.4s (37%) | **45.2s (32%)** | -18.4% |
| Training time | 78.2s (52%) | 78.2s (56%) | unchanged |
| Throughput | 1,149 tok/s | **1,233 tok/s** | +7.3% |
| MFU (actor) | 6.59% | 6.59% | structural limit |

### What worked (ranked by impact)

1. **CUDA graphs** (`enforce_eager=false`): **-18.4% generation time**. By far the most impactful single change. Eliminates kernel launch overhead in vLLM decode. No memory cost.
2. **`free_cache_engine=True`** + high `gpu_memory_utilization=0.85`: **Prevents OOM** when vLLM and training colocate. Releases KV cache between phases. Direct improvement is small (-1.4%) but enables the safe high-utilization configuration.
3. **Lower vLLM utilization** (0.70→0.45, without free_cache): Only needed if `free_cache_engine` cannot be enabled (e.g. older vLLM < 0.11). Eliminates OOM but wastes rollout capacity.

### What did NOT work

1. **Increasing `actor_ppo_max_token_len`** (18K→24K): No effect. Dynamic batching already packs by actual token count (~2600 avg), well below the 18K limit. Raising the cap changes nothing.
2. **Increasing `LOG_PROB_MICRO_BATCH_SIZE`** (4→8): No effect. The bottleneck in old_log_prob computation is the forward pass itself, not batch dispatch overhead.
3. **More KV cache** (0.45→0.85 utilization, same eager mode): No effect on generation speed. Rollout is decode-bound (sequential token generation), not memory-bound.

### Tuning priorities (corrected, by empirical impact)

1. **Enable CUDA graphs** (`ROLLOUT_ENFORCE_EAGER=false`): Highest impact, try first.
2. **Enable `ROLLOUT_FREE_CACHE_ENGINE=True`**: Safety net for memory, small direct benefit.
3. **Set `ROLLOUT_GPU_MEMORY_UTILIZATION=0.85`**: Requires free_cache_engine. More KV cache is available but effect is minimal for decode-bound workloads.
4. **Don't over-tune token budgets**: `actor_ppo_max_token_len` and `LOG_PROB_MICRO_BATCH_SIZE` are secondary; only tune if actual sequences regularly hit the limit.
5. **Training phase (78s, 56% of time)** is the remaining bottleneck: `use_torch_compile`, reducing GPU count, or hybrid sharding (HSDP) are potential directions, but each carries compatibility risk.

### Structural limits

MFU of ~6.6% is a structural limit for 4B model on 8 GPUs with FSDP full sharding. The model is too small relative to the communication overhead (all-gather + reduce-scatter across 8 GPUs for each layer). This is NOT a tuning problem — it requires architectural changes (fewer GPUs, larger model, hybrid parallelism) to improve significantly.

---

## H. Quick Reference: Recommended Defaults

For 4B model on 8x H800 (80 GB), vLLM 0.12+:

```bash
# Rollout (vLLM) — high utilization, CUDA graphs, cache release
ROLLOUT_GPU_MEMORY_UTILIZATION=0.85
ROLLOUT_ENFORCE_EAGER=false
ROLLOUT_FREE_CACHE_ENGINE=True
ROLLOUT_ENABLE_SLEEP_MODE=True
ROLLOUT_MAX_NUM_SEQS=512
ROLLOUT_ENABLE_CHUNKED_PREFILL=true

# Memory — let verl manage expandable_segments dynamically
PYTORCH_CUDA_ALLOC_CONF=""   # when free_cache_engine=True

# Training (FSDP)
use_dynamic_bsz=True
USE_REMOVE_PADDING=True      # requires flash-attn
enable_gradient_checkpointing=True
offload=False                 # not needed for 4B on 80 GB
fsdp_size=-1                  # full sharding

# Batch sizes
train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=8
```
