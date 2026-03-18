# Training Diagnostic Report: Baseline-MiniRL-Qwen3-1.7B-MATH

**Experiments Compared:**
- **EXP-05:** `Baseline-MiniRL-Qwen3-1.7B-MATH_1773625595` (grad_clip=1.0)
- **EXP-06:** `Baseline-MiniRL-Qwen3-1.7B-MATH-GC500_1773643860` (grad_clip=500.0)

**Date:** 2026-03-16 (updated)
**Model:** Qwen3-1.7B-Base (single model, no joint training)
**Algorithm:** MiniRL loss + Dr.GRPO advantage + token-level IS correction
**Hardware:** 4 GPUs, B=32, G=8, max_response_length=4096

---

## 1. Executive Summary

Two identical training runs were compared — the only difference is gradient clipping (`grad_clip=1.0` in EXP-05 vs `grad_clip=500.0` in EXP-06). Both runs reached **virtually identical peak accuracy**: EXP-05 peaked at 64.0%, EXP-06 at 64.2%.

**The grad_clip fix was mechanically correct** (verified via code audit — `optim.clip_grad=1.0` is dead code for the FSDP path), and EXP-06 gradients flow at full magnitude (97.5% of steps unclipped). **But the plateau at ~60-64% persists**, indicating the original diagnosis was only partially correct.

**Revised root cause analysis:** The ~60% plateau is caused by a combination of factors:

1. **Insufficient training duration**: 200 steps × 32 prompts = 6,400 prompts seen, but the dataset has **7,429 training examples** — the model hasn't even completed 1 epoch (only 0.86 epochs). The script's `total_epochs=3` is overridden by `total_training_steps=200`.
2. **Rapid format learning masks slow RL improvement**: The jump from 42% → 58% in the first 30 steps is primarily the model learning output format (boxed answers, reasoning structure), not mathematical ability. Subsequent RL-driven improvement is much slower.
3. **Model capacity ceiling**: At 1.7B parameters, 60-64% on MATH-500 may be approaching the model's inherent limit for this problem distribution, achievable with minimal RL training.

**Key evidence:**
- EXP-05 (effective lr ≈ 2.8e-9, barely learning) and EXP-06 (full lr = 1e-6, ~350× more learning) achieved the same result
- Offline eval confirms: EVAL-03 (EXP-06) MATH-500 mean@3 = 64.1% vs EVAL-02 (EXP-05) mean@3 = 61.4% — only +2.7% improvement despite dramatically different gradient dynamics
- AIME performance actually **degraded** after training (from 7.7% at step 0 to 0% at step 200)

**Recommendations:**
1. Increase `total_training_steps` to 600-700 to cover 3 full epochs
2. Consider raising lr to 3e-6 or 5e-6 to accelerate RL-phase learning
3. For AIME evaluation, use pass@k with k≥8 — single-sample eval on 26 problems is too noisy

---

## 2. EXP-05 vs EXP-06 Head-to-Head Comparison

### 2.1 Validation Metrics

| Metric | EXP-05 (gc=1.0) | EXP-06 (gc=500) | Delta |
|--------|-----------------|-----------------|-------|
| **MATH-500 peak** | 64.0% (step 145) | 64.2% (step 160) | +0.2% |
| **MATH-500 final** | 64.0% (step 200) | 60.0% (step 200) | -4.0% |
| **MATH-500 step-0** | 42.5% | 41.7% | -0.8% |
| **AIME-25 peak** | 11.5% (step 100) | 7.7% (step 90) | -3.8% |
| **AIME-25 final** | 0.0% (step 200) | 0.0% (step 200) | 0.0% |
| **Steps to reach 60%** | ~40 | ~50 | +10 steps |

### 2.2 Offline Evaluation (n=3 sampling)

| Benchmark | EVAL-02 (EXP-05) | EVAL-03 (EXP-06) | Delta |
|-----------|-------------------|-------------------|-------|
| **MATH-500 mean@3** | 61.4% | 64.1% | +2.7% |
| **MATH-500 pass@3** | 74.4% | 76.2% | +1.8% |
| **MATH-500 maj@3** | 66.8% | 69.2% | +2.4% |
| **AIME-2025 mean@3** | 3.3% | 2.2% | -1.1% |
| **AMC-2023 mean@3** | 37.5% | 35.0% | -2.5% |
| **MinervaMAth mean@3** | 24.6% | 24.8% | +0.2% |
| **OlympiadBench mean@3** | 30.0% | 28.1% | -1.9% |

**Observation:** EXP-06 improved MATH-500 by +2.7% (the training distribution) but showed slight degradation on OOD benchmarks (AMC, AIME, OlympiadBench). This pattern suggests mild **overfitting to the MATH distribution** — the higher effective lr causes the model to specialize more narrowly.

### 2.3 Training Dynamics Comparison

| Metric | EXP-05 (gc=1.0) | EXP-06 (gc=500) |
|--------|-----------------|-----------------|
| **Grad norm range** | 231-622 (pre-clip) | 227-535 (mostly unclipped) |
| **Steps grad-clipped** | 40/40 (100%) | 1/40 (2.5%) |
| **Effective lr** | ~2.8e-9 | 1e-6 (full) |
| **pg_clipfrac mean** | 0.0008 | 0.0008 |
| **pg_loss stabilized at** | -100 to -200 | -30 to -100 |
| **score/mean trend** | +0.09 (first→second half) | +0.06 (first→second half) |
| **Response length trend** | 812→580 | 825→600 |

**Critical observation**: `pg_clipfrac` is virtually identical (~0.08%) in both runs. Since EXP-06 has 350× higher effective learning rate, the fact that the policy ratio clip fraction didn't increase means **the optimization landscape is very flat** near the current policy — there's very little gradient signal to exploit.

---

## 3. Why the Plateau Persists: Detailed Analysis

### 3.1 Insufficient Training Duration

The training dataset has **7,429 prompts** (7,500 total minus 71 filtered for length). At B=32 prompts per step:

| Steps | Prompts Seen | Epochs |
|-------|-------------|--------|
| 100 | 3,200 | 0.43 |
| **200** | **6,400** | **0.86** |
| 232 | 7,429 | 1.00 |
| 464 | 14,858 | 2.00 |
| **696** | **22,287** | **3.00** |

**200 steps covers only 86% of the training set once.** The model has never seen ~1,029 training problems. The script sets `total_epochs=3`, but `total_training_steps=200` terminates training early.

For comparison, the reference script (EXP-00) uses B=128, steps=500, which means 64,000 prompts seen — **10× more training data exposure** than our current setup.

### 3.2 The "Format Learning" Phase vs "RL Learning" Phase

The accuracy curve reveals two distinct phases:

```
Phase 1 (Format Learning, steps 0-30):
  EXP-05: 42.5% → 57.1%  (+14.6 pts in 25 steps)
  EXP-06: 41.7% → 57.2%  (+15.5 pts in 30 steps)
  → Nearly identical despite 350× difference in effective lr!

Phase 2 (RL Improvement, steps 30-200):
  EXP-05: 57.1% → 64.0%  (+6.9 pts in 170 steps)
  EXP-06: 57.2% → 64.2%  (+7.0 pts in 170 steps)
  → Also nearly identical!
```

Phase 1 is dominated by the model learning to format answers in `\boxed{}` and produce step-by-step reasoning chains — this requires minimal gradient signal and happens almost regardless of learning rate. Phase 2 is the actual RL improvement phase, which is very slow and produces only ~7 percentage points in 170 steps.

### 3.3 Why Didn't Higher LR Help?

If `grad_clip=500` restores the full lr, why didn't training improve faster? Several hypotheses:

**H1: The advantage signal is too weak**
- Mean advantage (`adv/mean`) is consistently near zero: -0.03 ± 0.03 across all steps
- With Dr.GRPO normalization (zero-mean advantages), the gradient signal is inherently small
- `score/mean` ≈ 0.25 means ~62% of responses are correct — the model is already performing well on training data, leaving little room for RL improvement

**H2: Token-level IS correction dampens updates**
- The token-level importance sampling with threshold=5.0 can suppress gradients when the rollout policy (vLLM) diverges from the FSDP policy
- This is a known conservative choice — it prevents large updates but also slows learning

**H3: Entropy collapse (unverified)**
- `calculate_entropy` is disabled, so we cannot observe entropy trends
- MiniRL with no KL penalty and `entropy_coeff=0` has no mechanism to maintain exploration
- The model may be collapsing to a narrow solution distribution, reducing the diversity of positive/negative examples and weakening the RL signal

**H4: The model is near its capacity limit**
- 1.7B parameters may genuinely cap around 60-64% on MATH-500 with RL from a base model
- EXP-04 (joint model, same algorithm) peaked at 63.0% — very similar
- The ~60-64% range may represent a "capability frontier" for this model size

### 3.4 AIME Performance Degradation

Detailed analysis of validation generations at step 200:

| Metric | Step 0 (pre-training) | Step 200 (final) |
|--------|----------------------|-----------------|
| AIME accuracy | 7.7% (2/26) | 0.0% (0/26) |
| AIME avg response length | 3,879 chars | 2,652 chars |
| MATH-500 accuracy | 41.7% | 60.0% |
| MATH-500 avg response length | 1,962 chars | 1,643 chars |

**Training actively degrades AIME performance** through:

1. **Response shortening**: RL on MATH (which rewards concise correct answers) reduces average AIME response length by 31.6%. Competition problems require extended reasoning chains that the model now truncates.

2. **Reasoning quality degradation**: Comparing the same AIME problems at step 0 vs step 200:
   - Step-0 responses are more methodical (systematic case-testing, longer derivations)
   - Step-200 responses apply formulas prematurely, skip steps, and commit to wrong approaches early
   - One step-200 response fabricates Python code execution output

3. **No positive AIME signal**: The model never receives positive reward on AIME-like problems during training, so there's no gradient to improve this capability.

4. **3/26 predictions exceed the AIME answer range (0-999)**: Predictions of 18135, 10400, and 4320 show the model doesn't understand the AIME format constraint.

---

## 4. Why Is actor/pg_loss Negative?

**This is expected and correct for MiniRL.**

The MiniRL loss function (`core_algos.py:1782`) is:

```python
pg_losses = -mask * advantages.detach() * log_prob
```

This is the standard REINFORCE-style convention:
- The **RL objective to maximize** is `E[advantage * log_prob]`
- For gradient descent (minimization), we negate it: **loss = -E[advantage * log_prob]**
- `log_prob` values are always negative (log of a probability < 1)
- `loss_agg_mode=seq-mean-token-sum` sums across tokens per sequence without length normalization

The magnitude decrease over training (from -1884 at step 5 to ~-100 at later steps) reflects the model becoming more calibrated.

---

## 5. Why Does critic/score/mean Fluctuate?

`critic/score/mean` is the mean reward across all responses in the batch. With binary rewards ({-1, +1}) and 256 responses (32×8):

```
score/mean = (num_correct - num_incorrect) / 256
```

Volatility comes from prompt difficulty variance (some batches draw harder problems), binary reward amplification, and small batch size (standard error ≈ 0.08).

The trend is positive: EXP-05 first-half avg 0.191 → second-half avg 0.276. EXP-06 shows a similar trend from 0.11 to 0.27.

---

## 6. Gradient Clipping Analysis (Updated)

### 6.1 Code Audit Result

There are two gradient clipping parameters in the config:
- `actor.grad_clip=500.0` — used by `DataParallelPPOActor._optimizer_step()` in `dp_actor.py:400-410`
- `optim.clip_grad=1.0` — **dead code for FSDP training path** (only used by Megatron backend)

**Confirmed: `grad_clip=500.0` is the sole, binding gradient clipping constraint.** The `optim.clip_grad=1.0` parameter has no effect.

### 6.2 EXP-05 vs EXP-06 Gradient Flow

| | EXP-05 (gc=1.0) | EXP-06 (gc=500) |
|---|---|---|
| **Pre-clip grad_norm range** | 231 — 622 | 227 — 535 |
| **Steps clipped** | 40/40 (100%) | 1/40 (2.5%) |
| **Clipping reduction factor** | ~350× average | None (1 outlier at 1.07×) |
| **Effective lr** | ~2.8e-9 | 1e-6 |
| **MATH-500 peak** | 64.0% | 64.2% |

**The 350× increase in effective learning rate produced only +0.2% improvement.** This conclusively shows gradient clipping was not the primary bottleneck.

### 6.3 Why the Previous Diagnosis Was Partially Wrong

The original report hypothesized that `grad_clip=1.0` was the root cause of the plateau. The reasoning was sound (100% of steps clipped, ~300× gradient suppression), but the conclusion was wrong because:

1. The rapid early improvement (42% → 58%) is format learning, not gradient-dependent RL optimization
2. The slow Phase 2 improvement (58% → 64%) is limited by the weakness of the RL signal itself, not by gradient magnitude
3. Both runs have nearly identical `pg_clipfrac` (~0.08%), meaning policy updates are naturally small regardless of lr

---

## 7. Additional Observations

### 7.1 Clip Fraction Remains Near Zero
`actor/pg_clipfrac` averages 0.08% in both experiments. The MiniRL binary mask rarely activates because policy ratios stay close to 1.0 — this is a property of the weak training signal, not of gradient clipping.

### 7.2 No Aborted Responses
`response/aborted_ratio` = 0.0 at every step. All generations complete normally in both runs.

### 7.3 Response Length Decreasing
Mean response length drops from ~825 to ~580 tokens. Correct responses are shorter (1332 chars avg) than incorrect ones (2109 chars avg), suggesting the model becomes more concise as it learns to solve easier problems quickly.

### 7.4 AIME Responses Are Not Truncated
All 26 AIME responses complete with EOS (`has_eos=True`). The issue is wrong answers, not insufficient generation length.

---

## 8. MATH-500 Accuracy Progression (Both Experiments)

```
Step     EXP-05(gc=1)  EXP-06(gc=500)
   0:      —           41.7%  ████████░░░░░░░░░░░░░
   5:    42.5%          40.0%  ████████░░░░░░░░░░░░░
  10:    52.1%          51.5%  ██████████░░░░░░░░░░░
  15:    54.9%          54.1%  ███████████░░░░░░░░░░
  20:    55.9%          54.5%  ███████████░░░░░░░░░░
  25:    58.6%          54.9%  ███████████░░░░░░░░░░
  30:    57.1%          57.1%  ███████████░░░░░░░░░░
  35:    57.3%          59.4%  ████████████░░░░░░░░░
  40:    61.2%          57.8%  ████████████░░░░░░░░░
  50:    58.1%          60.2%  ████████████░░░░░░░░░
  60:    61.0%          60.2%  ████████████░░░░░░░░░
  75:    59.0%          63.0%  █████████████░░░░░░░░
  95:    60.0%          63.2%  █████████████░░░░░░░░
 100:    60.8%          58.6%  ████████████░░░░░░░░░
 120:    57.9%          60.4%  ████████████░░░░░░░░░
 145:    64.0%  ←peak   62.0%  ████████████░░░░░░░░░
 150:    64.0%          62.6%  █████████████░░░░░░░░
 160:    60.4%          64.2%  █████████████░░░░░░░░ ←peak
 185:    —              59.4%  ████████████░░░░░░░░░
 200:    64.0%          60.0%  ████████████░░░░░░░░░
```

Both experiments oscillate in the 58-64% band after step 40, with no sustained upward trend. This band appears to be the current training regime's performance range.

---

## 9. Recommendations

### 9.1 Increase Training Steps (High Priority)

The most impactful change: increase `total_training_steps` to allow at least 3 full epochs:

```bash
# Current: 200 steps = 0.86 epochs (hasn't seen full dataset)
# Recommended: 700 steps = 3.0 epochs
total_training_steps=700

# Or for a quick test: 400 steps = 1.7 epochs
total_training_steps=400
```

**Rationale**: The model has only seen 6,400 of 7,429 training problems. Many harder problems (the ones that would push accuracy beyond 60%) may never have been encountered. The reference script (EXP-00) uses 10× more training data exposure.

### 9.2 Increase Learning Rate (Medium Priority)

With `grad_clip=500.0` confirmed working, lr=1e-6 may be too conservative for RL training:

```bash
# Try 5e-6 or 1e-5 with warmup
actor_rollout_ref.actor.optim.lr=5e-6
actor_rollout_ref.actor.optim.lr_warmup_steps=10
```

**Rationale**: The near-zero advantage magnitudes (~0.03) combined with lr=1e-6 produce very small parameter updates. A higher lr could break through the plateau if the bottleneck is update magnitude rather than signal quality.

### 9.3 Enable Entropy Monitoring (Low Priority, Diagnostic)

```bash
actor_rollout_ref.actor.calculate_entropy=True
actor_rollout_ref.actor.entropy_coeff=0  # keep at 0, just monitor
```

**Rationale**: Without entropy data, we cannot determine if the model has collapsed to a narrow solution distribution. Entropy collapse would explain the weak RL signal and should be diagnosed before other interventions.

### 9.4 AIME Evaluation Improvements

- AIME-25 single-sample eval (n=1, 26 problems) is statistically meaningless. Use offline eval with n≥8.
- Do not use AIME as a training signal — 1.7B model capacity is insufficient.
- AIME degradation after RL training is expected and should not be a concern for MATH-focused training.

### 9.5 Comparison with Joint Model

| | EXP-04 (Joint) | EXP-05 (Baseline gc=1) | EXP-06 (Baseline gc=500) |
|---|---|---|---|
| MATH-500 peak | 63.0% (step 80) | 64.0% (step 145) | 64.2% (step 160) |
| Steps to peak | 80 | 145 | 160 |
| Total steps | 100 | 200 | 200 |
| MATH-500 offline | 64.2% (EVAL-01) | 61.4% (EVAL-02) | 64.1% (EVAL-03) |

The joint model (EXP-04) achieves comparable accuracy in **half the steps**, suggesting the joint architecture provides a stronger learning signal. This advantage would likely compound with longer training.

---

## Appendix A: EXP-06 Per-Step Training Metrics

| Step | pg_loss | grad_norm | score/mean | adv/mean | resp_len | clipfrac | MATH-500 | AIME-25 |
|------|---------|-----------|------------|----------|----------|----------|----------|---------|
| 5 | -1883.8 | 535.0 | -0.367 | -0.096 | 824.5 | 0.0013 | 0.400 | 0.038 |
| 10 | -291.6 | 429.0 | -0.117 | -0.041 | 665.7 | 0.0010 | 0.515 | 0.000 |
| 15 | -241.1 | 437.3 | 0.359 | -0.064 | 614.8 | 0.0009 | 0.541 | 0.038 |
| 20 | -374.2 | 475.8 | 0.117 | -0.115 | 687.4 | 0.0011 | 0.545 | 0.000 |
| 25 | -258.9 | 420.9 | 0.297 | -0.074 | 627.1 | 0.0014 | 0.549 | 0.038 |
| 30 | -152.9 | 357.9 | 0.281 | -0.091 | 605.2 | 0.0007 | 0.571 | 0.000 |
| 35 | -61.7 | 304.2 | 0.281 | -0.016 | 552.6 | 0.0009 | 0.594 | 0.038 |
| 40 | -172.0 | 351.6 | 0.328 | -0.053 | 480.6 | 0.0008 | 0.578 | 0.038 |
| 45 | -188.2 | 409.6 | -0.016 | -0.065 | 666.0 | 0.0010 | 0.584 | 0.038 |
| 50 | -67.0 | 344.4 | 0.227 | -0.039 | 619.2 | 0.0008 | 0.602 | 0.000 |
| 55 | -9.2 | 268.3 | 0.430 | 0.001 | 516.9 | 0.0005 | 0.616 | 0.038 |
| 60 | -115.8 | 315.8 | 0.180 | -0.019 | 735.0 | 0.0006 | 0.602 | 0.000 |
| 65 | -48.2 | 345.6 | 0.078 | -0.040 | 729.2 | 0.0008 | 0.596 | 0.038 |
| 70 | -54.3 | 265.9 | 0.070 | -0.018 | 597.3 | 0.0005 | 0.594 | 0.000 |
| 75 | -82.7 | 416.8 | 0.203 | -0.021 | 615.1 | 0.0010 | 0.630 | 0.038 |
| 80 | -57.6 | 250.1 | 0.266 | -0.009 | 634.6 | 0.0007 | 0.592 | 0.000 |
| 85 | -100.6 | 257.1 | 0.453 | -0.016 | 507.1 | 0.0006 | 0.610 | 0.000 |
| 90 | -94.7 | 255.8 | 0.063 | -0.011 | 640.7 | 0.0005 | 0.588 | 0.077 |
| 95 | -84.6 | 296.4 | 0.281 | -0.002 | 612.9 | 0.0006 | 0.632 | 0.038 |
| 100 | -101.0 | 227.1 | 0.352 | -0.030 | 595.0 | 0.0004 | 0.586 | 0.038 |
| 105 | -89.6 | 317.4 | 0.016 | -0.001 | 688.0 | 0.0009 | 0.598 | 0.038 |
| 110 | -126.3 | 289.8 | 0.188 | -0.019 | 626.7 | 0.0005 | 0.598 | 0.038 |
| 115 | -19.0 | 319.1 | 0.109 | -0.003 | 581.0 | 0.0006 | 0.602 | 0.038 |
| 120 | -108.5 | 357.1 | 0.125 | -0.019 | 680.6 | 0.0008 | 0.604 | 0.038 |
| 125 | -92.8 | 337.7 | 0.422 | 0.001 | 597.6 | 0.0007 | 0.610 | 0.038 |
| 130 | -111.2 | 262.7 | 0.453 | -0.006 | 507.1 | 0.0004 | 0.612 | 0.038 |
| 135 | -95.6 | 295.9 | 0.148 | -0.025 | 582.7 | 0.0007 | 0.588 | 0.000 |
| 140 | -54.2 | 369.9 | 0.406 | -0.049 | 597.1 | 0.0008 | 0.608 | 0.000 |
| 145 | -104.3 | 384.9 | 0.313 | -0.004 | 574.7 | 0.0010 | 0.620 | 0.038 |
| 150 | -54.3 | 278.5 | 0.172 | -0.012 | 536.6 | 0.0009 | 0.626 | 0.000 |
| 155 | -251.1 | 440.2 | 0.344 | 0.002 | 623.7 | 0.0009 | 0.624 | 0.000 |
| 160 | -35.0 | 371.2 | 0.383 | -0.010 | 575.0 | 0.0010 | 0.642 | 0.000 |
| 165 | 9.5 | 272.0 | 0.352 | -0.004 | 551.9 | 0.0008 | 0.622 | 0.000 |
| 170 | -33.9 | 357.7 | 0.117 | -0.007 | 630.2 | 0.0007 | 0.622 | 0.038 |
| 175 | -10.3 | 344.8 | 0.445 | 0.011 | 497.5 | 0.0007 | 0.626 | 0.038 |
| 180 | -29.8 | 404.0 | 0.016 | 0.010 | 605.3 | 0.0010 | 0.620 | 0.000 |
| 185 | -75.3 | 297.2 | 0.023 | -0.019 | 651.6 | 0.0006 | 0.594 | 0.000 |
| 190 | -129.7 | 420.8 | 0.250 | -0.032 | 652.4 | 0.0013 | 0.618 | 0.077 |
| 195 | -172.6 | 299.1 | 0.273 | -0.034 | 565.1 | 0.0007 | 0.636 | 0.000 |
| 200 | -189.0 | 351.4 | 0.227 | -0.022 | 674.0 | 0.0007 | 0.600 | 0.000 |

## Appendix B: EXP-05 Per-Step Training Metrics

| Step | pg_loss | grad_norm | score/mean | resp_len | clipfrac | MATH-500 | AIME-25 |
|------|---------|-----------|------------|----------|----------|----------|---------|
| 5 | -1894.8 | 515.5 | -0.273 | 812.4 | 0.0011 | 0.425 | 0.000 |
| 10 | -408.1 | 464.4 | -0.172 | 698.3 | 0.0011 | 0.521 | 0.000 |
| 15 | -201.6 | 363.6 | 0.383 | 672.1 | 0.0008 | 0.549 | 0.000 |
| 20 | -227.7 | 419.6 | 0.086 | 761.4 | 0.0008 | 0.559 | 0.038 |
| 25 | -627.5 | 443.7 | 0.367 | 613.3 | 0.0010 | 0.586 | 0.038 |
| 30 | -128.6 | 337.4 | 0.352 | 608.6 | 0.0005 | 0.571 | 0.038 |
| 35 | -98.9 | 408.5 | 0.359 | 641.8 | 0.0011 | 0.573 | 0.000 |
| 40 | -98.1 | 324.4 | 0.336 | 600.6 | 0.0006 | 0.612 | 0.000 |
| 45 | -60.6 | 387.9 | 0.055 | 721.0 | 0.0010 | 0.612 | 0.038 |
| 50 | -86.7 | 375.9 | 0.211 | 706.1 | 0.0006 | 0.581 | 0.038 |
| 55 | -83.5 | 293.9 | 0.453 | 547.1 | 0.0006 | 0.590 | 0.077 |
| 60 | -189.8 | 369.3 | 0.266 | 656.9 | 0.0010 | 0.610 | 0.038 |
| 65 | -133.1 | 364.4 | 0.078 | 743.2 | 0.0007 | 0.590 | 0.077 |
| 70 | -181.3 | 362.5 | 0.094 | 645.1 | 0.0006 | 0.596 | 0.038 |
| 75 | -205.4 | 458.7 | 0.164 | 645.8 | 0.0012 | 0.590 | 0.077 |
| 80 | -256.5 | 301.0 | 0.305 | 638.9 | 0.0008 | 0.608 | 0.038 |
| 85 | -112.3 | 312.7 | 0.430 | 517.4 | 0.0008 | 0.604 | 0.038 |
| 90 | -3.0 | 312.8 | 0.148 | 608.0 | 0.0005 | 0.604 | 0.038 |
| 95 | -104.0 | 283.5 | 0.258 | 655.3 | 0.0007 | 0.600 | 0.000 |
| 100 | -790.5 | 402.7 | 0.336 | 654.7 | 0.0008 | 0.608 | 0.115 |
| 105 | -573.3 | 422.0 | 0.008 | 710.7 | 0.0009 | 0.620 | 0.038 |
| 110 | -184.0 | 329.8 | 0.148 | 641.7 | 0.0008 | 0.586 | 0.077 |
| 115 | -203.1 | 365.5 | 0.016 | 590.6 | 0.0009 | 0.594 | 0.077 |
| 120 | -154.5 | 390.5 | 0.219 | 696.2 | 0.0008 | 0.579 | 0.038 |
| 125 | -281.5 | 390.2 | 0.422 | 582.1 | 0.0011 | 0.612 | 0.000 |
| 130 | -59.6 | 231.3 | 0.477 | 534.1 | 0.0004 | 0.616 | 0.000 |
| 135 | -342.1 | 335.0 | 0.234 | 629.1 | 0.0005 | 0.610 | 0.038 |
| 140 | +2.7 | 435.8 | 0.414 | 662.4 | 0.0009 | 0.628 | 0.115 |
| 145 | -455.4 | 398.7 | 0.305 | 645.7 | 0.0007 | 0.640 | 0.038 |
| 150 | -208.3 | 337.4 | 0.219 | 562.1 | 0.0008 | 0.640 | 0.000 |
| 155 | -369.0 | 424.7 | 0.359 | 641.6 | 0.0012 | 0.614 | 0.077 |
| 160 | -210.0 | 330.6 | 0.391 | 579.2 | 0.0007 | 0.604 | 0.038 |
| 165 | -142.7 | 267.3 | 0.313 | 621.4 | 0.0003 | 0.602 | 0.000 |

## Appendix C: Validation Generation Quality (EXP-06 Step 200)

### MATH-500
- 298/497 correct (60.0%)
- All responses complete with EOS (no truncation)
- Correct answers: avg 1,332 chars | Incorrect: avg 2,109 chars
- 37 near-misses (integer answers within ±5 of ground truth)
- Verification: all via `string_match` or `verl_math_verify`

### AIME-25
- 0/26 correct at step 200
- All responses complete with EOS (no truncation)
- Average response length: 2,652 chars (range 1,044–8,285)
- Median absolute error: 153, mean: 1,443
- 3 predictions exceed AIME range (0-999): 18135, 10400, 4320
- 1 non-integer prediction: `\dfrac{15}{32767}` (expected integer 237)
- 1 fabricated code execution output
- 1 near-miss: pred=59 vs gts=60
