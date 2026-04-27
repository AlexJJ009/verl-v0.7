---
name: Reverse SFT — v2 online stable; offline model1 eval pending
description: EXP-17 (1B, β=0.1 under v2 loss) completed 2026-04-21; online tracks 1A within 0.5 pp; v1 "必崩" claim refuted at training level, offline model1 eval still needed to close the case
type: project
originSessionId: dab68424-681d-451a-941d-802e7e92a9f9
---
Reverse SFT (L- term, β>0) was earlier declared permanently abandoned based on EXP-12 (M5, lr=1e-6) and EXP-14 (M5.6, lr=5e-7). **Under v2 loss that claim is refuted at the training/online level** — EXP-17 (1B, β=0.1 under `wdl_sft_is` at lr=5e-7) completed 300 steps cleanly 2026-04-21 with online MATH-500 model2-only peak 70.97% (steps 225 & 275), step-300 final 70.36%. Tracks EXP-16 (1A, same lr, β=0) within 0.5 pp across all 13 val points.

**Why:** v1's failure mode was never "training diverges" (M5.6 also completed 300 steps). It was **EVAL-15's offline model1 format-compliance collapse**: MATH-500 −21.6%, extraction_fail uniform 24–28% across all 7 benchmarks. This is invisible in online metrics. v2's lower-bound binary mask on negatives (`ratio < 1 − clip_ratio_low` → zero gradient) is the hypothesized countermeasure, but **has not yet been verified offline on 1B model1**.

**How to apply:**
- Do NOT say "reverse SFT is abandoned" or "β>0 is safe under v2" — both are overclaims.
- Correct summary: "Under v2, β=0.1 is online-stable and model2-competitive; decisive offline eval on 1B model1 (extraction_fail vs EVAL-15's 24–28%) is pending."
- The decisive next step is offline eval on `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/global_step_{225,275,300}/` — extract model1, measure MATH-500 mean@3 + extraction_fail. If extraction_fail ≈ M5.5 m1's 4.8–8.3%, v2 genuinely fixed it. If ≈ M5.6 m1's 24–28%, v2 did not help on model1 and the earlier judgment was right for the wrong reason.
