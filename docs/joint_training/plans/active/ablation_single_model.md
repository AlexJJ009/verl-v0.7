# Ablation Plan: Single-Model WDL-SFT-IS (Series 2X)

**Status**: Active (created 2026-04-22)
**Owner**: Alex
**Parent branch**: `feature/on-policy-wdl-sft`
**Scripts**: `recipe/on_policy_wdl_sft/ablation_single_model/`

## 1. Research Question

The main-line experiments (1A/1B/1C) combine three novel ingredients:

1. **Joint model** with weights for model1 (weak) + model2 (strong)
2. **Fused-logit rollout**: responses sampled from $P_\text{mix} = \text{Softmax}((1-\lambda) z_\text{weak} + \lambda z_\text{strong})$
3. **WDL-SFT-IS loss**: forward SFT on correct set $C$, optional reverse SFT on incorrect set $I$, with IS correction + binary-mask ratio clip

**Question**: is ingredient #3 (the loss) doing the work on its own, or does it genuinely require #1 + #2 to produce the observed +2.4pp online MATH-500 lift at step 300?

## 2. Hypotheses

- **H1 (loss-sufficient)**: `wdl_sft_is` applied to a standard single model reaches most of the joint's lift.
  *Implication if true*: drop the joint infrastructure for the next iteration — simpler pipeline with comparable results.

- **H2 (fusion-critical)**: single-model wdl_sft_is is significantly worse than joint 1A.
  *Implication if true*: the gradient-routing-through-fused-logits mechanism is real; keep joint.

- **H3 (init-dominant)**: the gap is mostly due to SFT init (model2 starts from an already-finetuned checkpoint), not the loss or fusion. 2X-BASE ≪ 2X-SFT ≈ 1X.

## 3. Design

### 3.1 Variables

| Variable | Joint (1A/B/C) | Ablation (2A/B/C) | Baseline (2Z) |
|---|---|---|---|
| Model architecture | JointModel (2 backbones) | Qwen3 (1 backbone) | Qwen3 (1 backbone) |
| Rollout source | Fused logits | Single-model logits | Single-model logits |
| Loss | `wdl_sft_is` | `wdl_sft_is` | `minirl` |
| Init | Base (m1) + SFT (m2) | Base **or** SFT | Base **or** SFT |

### 3.2 Constants (identical across all 8 runs and aligned with 1A/B/C)

- Dataset: EnsembleLLM train_rl_format.parquet
- Eval sets: MATH-500, AIME-2025 (with system prompt)
- Reward function: `custom_reward_function_latex_verify.py` (3-tier LaTeX semantic verify)
- Adv estimator: GRPO (no std normalization), `norm_adv_by_std_in_grpo=False`
- Data budget: 300 steps × 64 prompts/step × 8 rollouts/prompt = 153,600 responses
- Mini-batch: 8 prompts (→ 8 grad updates per rollout batch — the scenario v2 IS is designed for)
- Optimizer: AdamW, weight_decay=0.1, warmup=5, grad_clip=500.0
- Sampling: T=1.0, top_p=1.0, top_k=-1; val T=1.0, val top_p=0.95
- Sequence: max_prompt=500, max_response=4096
- IS: token-level, threshold=5.0, batch_normalize=false
- Clip: ratio_low=0.2, ratio_high=0.27 (binary mask under v2; clip under minirl)
- Eval cadence: every 25 steps, val_before_train=True
- No KL term anywhere (`use_kl_in_reward=False`, `use_kl_loss=False`)

### 3.3 Run matrix

| Run | Init | Loss | β | lr | Paired with |
|---|---|---|---|---|---|
| **2A-base** | Qwen3-4B-Base | wdl_sft_is | 0 | 5e-7 | 1A |
| **2A-sft** | Qwen3-4B-Base-SFT-stage-1 | wdl_sft_is | 0 | 5e-7 | 1A |
| **2B-base** | Qwen3-4B-Base | wdl_sft_is | 0.1 | 5e-7 | 1B |
| **2B-sft** | Qwen3-4B-Base-SFT-stage-1 | wdl_sft_is | 0.1 | 5e-7 | 1B |
| **2C-base** | Qwen3-4B-Base | wdl_sft_is | 0 | 1e-6 | 1C |
| **2C-sft** | Qwen3-4B-Base-SFT-stage-1 | wdl_sft_is | 0 | 1e-6 | 1C |
| **2Z-base** | Qwen3-4B-Base | minirl | — | 5e-7 | reference floor |
| **2Z-sft** | Qwen3-4B-Base-SFT-stage-1 | minirl | — | 5e-7 | reference floor |

## 4. Control Rigor — What We Control, What We Don't

| Dimension | Controlled? | Reason |
|---|---|---|
| Data budget (steps × batch × N) | **Yes** | This is the correct "sample efficiency" definition. |
| lr / β / optimizer / warmup / grad_clip | **Yes (paired)** | 2X↔1X match exactly; apples-to-apples on loss ingredients. |
| Rollout N, temperature, max lengths | **Yes** | Same sample distribution hyperparameters. |
| IS config + clip thresholds | **Yes** | Stability machinery held constant. |
| Seed, dataset, eval cadence, val set | **Yes** | Standard rigor. |
| **GPU-hour / wall-clock** | **No** | Joint has 2× params → forward/backward is ~2× slower. Controlling GPU-hour would force single-model to run ~2× the steps, polluting the comparison. |
| **Parameter count trained** | **No (IV)** | Joint updates both sub-models; single updates one. By design — this is part of what we're ablating. Evaluation is on model2 only, whose capacity matches in both. |
| Rollout source distribution | **No (IV)** | Fused vs single-policy is exactly what we're testing. |

**Key principle**: we control the *data budget* (how much the model sees), not the *compute budget*. Single-model runs will finish in ~half the wall-clock of joint runs; that is the natural consequence of the design, not a confound.

## 5. Decomposing the Result

Define:
- $G_\text{joint} = \text{score}(1X) - \text{score}(2Z_\text{sft})$ — total joint lift over pure RL from SFT init
- $L_\text{loss} = \text{score}(2X_\text{sft}) - \text{score}(2Z_\text{sft})$ — loss effect alone
- $L_\text{fusion} = \text{score}(1X) - \text{score}(2X_\text{sft})$ — fusion effect alone
- $L_\text{init} = \text{score}(2X_\text{sft}) - \text{score}(2X_\text{base})$ — SFT-init contribution

Identity: $G_\text{joint} \approx L_\text{loss} + L_\text{fusion}$ (assuming small interaction).

Reading the result:

| Observation | Interpretation |
|---|---|
| $L_\text{loss}$ large, $L_\text{fusion}$ small | H1: drop joint, keep loss. |
| $L_\text{loss}$ small, $L_\text{fusion}$ large | H2: joint mechanism is essential. |
| $L_\text{loss}$ moderate, $L_\text{fusion}$ moderate | Both matter; keep joint as a useful multiplier. |
| $L_\text{init}$ dominates everything | H3: the SFT pre-training is doing most of the work, not the RL loop. Reconsider whether the RL phase is worth its compute. |

## 6. Expected Timeline

- Writing scripts: **done** (2026-04-22)
- Verification dry-runs: next (ensure Hydra args diff from 1C is minimal and expected)
- Local kickoff (priority: 2A-sft first, since it's the cleanest apples-to-apples with 1A): after 1C completes (frees GPUs ~06:00 2026-04-22)
- Meituan platform batch: all 8 runs in parallel once scripts are validated on a local 2A-sft run

## 7. Decision Criteria

After 300 steps on each run:

1. If **2A-sft online MATH-500 ≥ 1A − 1pp**: H1 supported for β=0 regime.
2. If **2B-sft does not diverge AND 2B-sft online MATH-500 ≥ 1B − 2pp**: H1 supported for β>0 regime.
3. If both above fail: H2 — joint is real, drop ablation thread.
4. Any configuration that crashes (NaN, grad explode): flag the v2 stability claim and revisit the clip/IS thresholds.

Offline eval (model2 mean@3 on MATH-500) on each run's best online step is the definitive scorecard — online model2-only numbers can diverge from offline (v1 EVAL-15 showed a +3pp online but −21pp offline for model1). Do not declare H1/H2 solely from online curves.

## 8. Open Questions

- Should 2Z use `minirl` or `vanilla` as the baseline loss? Currently `minirl` — shares the IS + binary-mask machinery with v2, so the loss-delta is cleaner ("forward/reverse SFT vs clipped PG" rather than "with vs without IS stability"). **Resolved 2026-04-23**: both are now run side-by-side as ABL-MINIRL-03 (2Z-BASE, minirl) and ABL-MINIRL-04 (2G-BASE, vanilla/GRPO). Keeping both, since the online reading already shows they separate (2Z drifts, 2G is stable-but-lower — see §9).
- If 2A-base completely stalls (no C-set signal for many steps), consider a warmup phase with minirl loss before switching to wdl_sft_is. **Resolved 2026-04-23**: 2A-BASE did NOT stall — it ramped from 30.65% (step 0) to 72.58% (step 25) in the first validation window, without a minirl warmup. The C-set signal bootstraps fine on Base init. No warmup needed.

## 9. Results — 2X-BASE batch (Meituan AFO, online only, 2026-04-23)

The SFT-init side ran locally (ABL-MINIRL-01/02, already written up in `EXPERIMENT_INDEX.md`). The Base-init side ran on Meituan AFO on 2026-04-23 — five runs, all 300 steps, all stable enough to finish. Full per-run entries live in `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md` (ABL-MINIRL-03 through 07). Offline eval (n=3, 7 benchmarks) is **pending** on all five — checkpoints still on dolphinfs, wandb not yet synced. Everything below is online MATH-500 mean@1 only.

### 9.1 Matrix — Base vs SFT init, online MATH-500 (mean@1)

| Run | Loss | β | lr | Init | Peak MATH (step) | Step-300 | Drift? |
|---|---|---|---|---|---|---|---|
| ABL-MINIRL-01 (2Z-SFT) | minirl | — | 5e-7 | SFT | 70.56% (275) | 70.16% | no |
| ABL-MINIRL-02 (2A-SFT) | wdl_sft_is | 0 | 5e-7 | SFT | 70.4% (300) | 70.4% | no |
| ABL-MINIRL-03 (2Z-BASE) | minirl | — | 5e-7 | Base | **76.21%** (100) | 69.35% | **yes, −11.7 pp to step 225, recovers to −6.9 pp end** |
| ABL-MINIRL-04 (2G-BASE) | vanilla/GRPO | — | 5e-7 | Base | 71.17% (275) | 70.16% | no |
| ABL-MINIRL-05 (2A-BASE) | wdl_sft_is | 0 | 5e-7 | Base | **76.21%** (225) | **74.80%** | no |
| ABL-MINIRL-06 (2C-BASE) | wdl_sft_is | 0 | 1e-6 | Base | 74.19% (100) | 72.18% | mild |
| ABL-MINIRL-07 (2B-BASE) | wdl_sft_is | 0.1 | 5e-7 | Base | 73.19% (75) | 67.74% | **yes, early peak then sustained** |

### 9.2 Online-only L-term reading (pending offline confirmation)

- **L_loss on Base init ≠ L_loss on SFT init**. At matched step 300 online: 2A-BASE − 2Z-BASE = **+5.45 pp** (74.80 − 69.35); 2A-SFT − 2Z-SFT ≈ **+0.24 pp** online (70.4 − 70.16) and **−0.6 pp** on offline mean@3 (ABL-MINIRL-02 already confirmed L_loss ≈ 0 offline on SFT init). Pattern: `wdl_sft_is` adds clear online value over MiniRL on Base init, ~zero over MiniRL on SFT init. Hypothesis: SFT init already encodes the format/anchor compliance that `wdl_sft_is` adds via its positive-set forward SFT; Base init has the headroom.
- **L_init (online direction is reversed from §2's H3)**. H3 predicted 2X-BASE ≪ 2X-SFT. Online readings say the opposite on mean@1 at matched step: 2A-BASE peak 76.21% > 2A-SFT peak 70.4%; 2Z-BASE peak 76.21% > 2Z-SFT peak 70.56%. **Caveat**: online mean@1 has previously diverged sharply from offline mean@3 (1B: online 70.97% m2 vs offline 82.9%; EXP-15: online 68.2% vs offline 79.6%). Do NOT redraw §2/§5 conclusions until offline n=3 on 2X-BASE ckpts is done.
- **β>0 damage is visible online on Base but not on joint**. 2B-BASE step 300 − 2A-BASE step 300 = **−7.06 pp online** (67.74 − 74.80). On joint (1B − 1A), the same β term was invisible online (within 0.5 pp at every val point) and only surfaced at offline m1 (ext_fail 37–49%). Reading: the joint's fused-logit rollout absorbs the reverse-SFT gradient through model1 while model2 benefits from the correct-set fusion — remove the joint scaffold and the damage lands on the val-visible trajectory. Reinforces the "β=0 is the default" recommendation.
- **lr=5e-7 > lr=1e-6 on Base single-model too** (matches 1C vs 1A). 2A-BASE − 2C-BASE at step 300 = +2.62 pp online (74.80 − 72.18). Same sign as 1A vs 1C. Higher lr reaches the plateau slightly earlier (2C peak at step 100 vs 2A peak at step 225) at a lower steady-state.
- **GRPO vs MiniRL on Base**: 2G-BASE is the most stable Base-init run (no drift) but has the lowest Base-init online ceiling. 2Z-BASE peaks 5 pp higher but drifts. IS + asymmetric-clip machinery gives more online headroom at the cost of late-training stability when starting from Base.

### 9.3 Decision-criterion update (§7)

Criteria §7.1–§7.2 were written against the SFT-init arm and have been resolved by ABL-MINIRL-01/02 offline (see `EXPERIMENT_INDEX.md` cross-experiment block): 2A-SFT online ≥ 1A − 1 pp ✓ but offline mean@3 says L_loss ≈ 0 and L_fusion ≈ +3 pp — H1 only partially supported, and the joint MATH-500 lift is now attributed to architecture, not loss.

**Added for the Base-init arm (2026-04-23, pending offline)**:

5. If **2A-BASE offline mean@3 ≥ 2A-SFT offline mean@3**: confirms the online L_init-reversal finding — Base init is a viable starting point for `wdl_sft_is` single-model RL, possibly preferable to SFT init if compute/curriculum allows.
6. If **2A-BASE offline mean@3 < 2A-SFT offline mean@3 despite 2A-BASE online > 2A-SFT online**: another case of online/offline divergence (add to the list that includes 1B, EXP-15). Treat online mean@1 on Base init with heavier skepticism going forward.
7. If **2B-BASE offline m1-style format collapse is comparable to 1B m1 (ext_fail 30–50%)**: β>0 anchor damage is an invariant of the reverse-SFT mechanism across arch/init, not a joint-specific artifact. Shut the β>0 thread definitively.

### 9.4 Artifacts (Meituan-resident, pending transfer)

- Training logs + metrics jsonl + validation jsonls: `recipe/on_policy_wdl_sft/ablation_single_model/{*.log, metrics/, validation/}` — **local as of 2026-04-23** (unzipped from `/data-1/verl-exp/logs.zip`, five runs).
- Checkpoints: **all on Meituan AFO dolphinfs** under `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02/lgx/verl-exp/checkpoints/<run_id>/global_step_{25,50,…,300}/`. Not yet transferred to local `/data-1`. Candidate promotion steps per run listed in `EXPERIMENT_INDEX.md` Extracted/Checkpoint Inventory block.
- wandb: **offline runs on Meituan**, not yet synced to the public project.
