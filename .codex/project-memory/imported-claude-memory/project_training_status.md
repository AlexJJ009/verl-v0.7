---
name: Training status as of 2026-04-21 (1A/1B complete, 1C running)
description: v2 loss: EXP-16 (1A) and EXP-17 (1B) complete 300 steps, online peaks ~71%; EXP-18 (1C) running; offline eval pending for 1A full trio + 1B full trio + 1C
type: project
originSessionId: dab68424-681d-451a-941d-802e7e92a9f9
---
Current project state (2026-04-21):

**Direction**: v2 loss (wdl_sft_is) as of 2026-04-19. Adds binary-mask ratio clipping + token-level `rollout_is_weights`. Online val path goes through `_is_joint_training` → `update_weights(eval_only=True)` → `extract_sub_model_weights(sub_model_index=1)`, so online metrics are model2-only (not fused) — verified in 1A/1B logs.

**v1 evals (loss_mode=wdl_sft, EXP-12~15)** — frozen reference:
- EXP-13 M5.5 (lr=5e-7, β=0, step 300): model2 offline MATH-500 mean@3 = 78.6%; model1 = 70.5%
- EXP-14 M5.6 (lr=5e-7, β=0.1, step 300): model2 = 79.1%; model1 collapsed to 48.9% (−21.6%, extraction_fail 24–28%) — the v1 reverse-SFT failure mode
- EXP-15 LR3 (lr=1e-6, β=0, step 125 best): model2 = 79.6%; peaked then drifted
- EXP-12 M5 (lr=1e-6, β=0.1): diverged

**v2 runs (loss_mode=wdl_sft_is)**:
- **EXP-16 (1A, lr=5e-7, β=0)** — complete 2026-04-20, 300 steps, one resume (disk-full at ckpt save). Online MATH-500 model2-only peak 71.37% @ step 225, final 70.36% @ step 300. Breaks v1 online ceiling (+2.4 pp vs M5.5). Run ID `WDL-SFT-Qwen3-4B-MATH-1A_1776594597`. Step 225 model2 preliminary offline MATH-500 mean@3 = 83.07% (EVAL-ID pending). Step 225 model2 already extracted to `/data-1/model_weights/WDL-SFT-4B-MATH-1A/step_225_model2/`. Checkpoints 125/150/175/200/250/275/300 deleted 2026-04-21 to free space; step_225 FSDP retained.
- **EXP-17 (1B, lr=5e-7, β=0.1)** — complete 2026-04-21, 300 steps, zero resumes. Online MATH-500 model2-only peak 70.97% @ steps 225 & 275, final 70.36% @ step 300. Tracks 1A within 0.5 pp throughout. Run ID `WDL-SFT-Qwen3-4B-MATH-1B_1776695220`. All 12 checkpoints (25–300) at `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1B_1776695220/`. Note: first-launch attempt `_1776683653` was killed at step 26 (my mistake re: model2-only val status) and cleaned; this is the clean re-run.
- **EXP-18 (1C, lr=1e-6, β=0)** — **running** (launched 2026-04-21 18:53, tmux `wdl_sft_is_1c`, Run ID `WDL-SFT-Qwen3-4B-MATH-1C_1776768784`). ETA ~20h. Key milestone step 125 (v1 LR3 peak-and-crash point).

**Pending offline eval priority** (decisive for v2 conclusions):
1. **1B model1** — decisive for "reverse SFT under v2" claim. v1 EVAL-15 showed β>0 destroyed model1 format (extraction_fail 24–28%). v2 lower-bound clip is the hypothesized fix but untested.
2. 1B model2 — for comparison against 1A's 83.07%.
3. 1A step 225 model1 + full EVAL-XX ID to record existing 83.07% model2 result.
4. 1C step 125 (peak checkpoint) + step 300 (final) — after training completes.

**Decision status**: 1A proves v2 breaks v1 ceiling; 1B proves v2 β=0.1 is online-stable (refutes v1-era "reverse SFT 必崩" at training level). Final β=0.1 rehabilitation verdict waits on 1B model1 offline extraction_fail number.

**WandB sync note**: v2 scripts set WANDB_DIR to /data-1/wandb_runs/{RUN_PREFIX}, survives docker --rm. Run `wandb sync <offline-run-dir>` in container to push. 1A synced 2026-04-20; 1B and 1C pending sync.
