# On-Policy Weak-Driven SFT (WDL-SFT) — Agent Entry Point

This file is the table of contents for coding agents working on the **On-Policy WDL-SFT** experiment on branch `feature/on-policy-wdl-sft`. This experiment extends standard Weak-Driven SFT with on-policy rollout and **forward-only** SFT training on correct rollouts.

## Experiment Overview

**On-Policy WDL-SFT** combines logit fusion with on-policy rollout. The algorithm is now versioned:

- **v1 (naive, `loss_mode=wdl_sft`)** — EXP-12 through EXP-15. No stability mechanisms in the loss.
- **v2 (IS-corrected, `loss_mode=wdl_sft_is`)** — EXP-16+. Adds binary-mask ratio clipping and `rollout_is_weights` correction.

### Common to both versions

1. **Fused Rollout**: Sample N responses from the fused distribution `P_mix = Softmax((1-λ)·z_weak + λ·z_strong)`.
2. **Reward Judgment**: Score each response via reward function → correct set C, incorrect set I.
3. **Both sub-models trained**: `freeze_model1=False` (default). Gradients flow into both model1 and model2. model1 is intentionally "sacrificed" to amplify model2's gradient through the fused logits — **evaluation targets model2 only**, not the joint fused model.

### v1 loss (naive)

$$L = L^+ + \beta \cdot L^-$$

where $L^+ = -\tfrac{1}{|C|}\sum_{i \in C}\sum_t \log P_\theta(y^i_t)$ and $L^- = \tfrac{1}{|I|}\sum_{j \in I}\sum_t \log P_\theta(y^j_t)$.
No `old_log_prob`, no IS correction, no clip.

### v2 loss (IS-corrected)

Same $L^+, L^-$ structure, plus:
- Binary mask on ratio $r = \pi_\theta / \pi_\text{old}$: upper-bound clip on C, lower-bound clip on I (MiniRL-style).
- `rollout_is_weights` (token-level, from `rollout_correction`) multiplied into token loss.
- Addresses the two sources of π_old ≠ π_new relevant to this setup: (A) multi-mini-batch updates per rollout, (B) vLLM/FSDP numerical mismatch. The fused-vs-submodel mismatch (source C) is intentional and left unmodified.

See `docs/joint_training/specs/wdl_sft_is.md` for exact formulas and `docs/joint_training/plans/active/wdl_sft_is.md` for the current experiment plan.

### Status of "reverse SFT abandoned"

Previous documents declared reverse SFT (β>0) permanently abandoned based on EXP-12 (M5, lr=1e-6) and EXP-14 (M5.6, lr=5e-7) instability. **Under v2 loss (EXP-17, completed 2026-04-21) this conclusion is no longer supported at the training/online level**: 1B (β=0.1, lr=5e-7, v2) ran 300 steps without drift and tracked 1A (β=0) to within 0.5 pp throughout — online MATH-500 model2-only peaked at 70.97% (steps 225/275) and landed at 70.36% (step 300), indistinguishable from 1A. **Final decision still pending offline eval on 1B model1**: v1's actual failure mode was EVAL-15's model1 format-compliance collapse (MATH-500 −21.6%, extraction_fail 24–28%), which was invisible online. v2's lower-bound binary mask on negatives is the hypothesized countermeasure, but it has not yet been offline-verified on 1B's model1.

## Environment

The project runs inside Docker image `verl-harness` (no conda). All paths assume identical `/data-1` layout across servers.

```bash
# Build or load the image (first time on a new server)
bash /data-1/verl07/build.sh            # loads tar or builds from Dockerfile
bash /data-1/verl07/build.sh --save     # + export to /data-1/docker_images/verl-harness.tar

# Interactive shell
bash /data-1/verl07/run_train.sh
```

- Docker image: `verl-harness` (tar: `/data-1/docker_images/verl-harness.tar`)
- Dockerfile: `docker/Dockerfile.joint-training.cu126`
- Scripts: `/data-1/verl07/build.sh`, `/data-1/verl07/run_train.sh`
- Python 3.12 managed by uv
- PyTorch 2.9.1+cu126, vLLM 0.12.0, flash-attn 2.8.1
- Apex, TransformerEngine, Megatron-LM compiled from source

### Attention Backend Constraints
- **vLLM (rollout)**: Must use FlashInfer backend (`export VLLM_ATTENTION_BACKEND=FLASHINFER`). Do NOT use Flash Attention for vLLM.
- **FSDP (training)**: Must use `attn_implementation=flash_attention_2` for performance.

## Repository

- Repo: `https://github.com/AlexJJ009/verl-v0.7`
- Upstream: `https://github.com/verl-project/verl`
- Branch: `feature/on-policy-wdl-sft`
- Parent branch: `feature/joint-training` (Stage 1 & 2 complete — provided the joint model and fused rollout infrastructure)
- `recipe/` is a submodule → `https://github.com/AlexJJ009/verl-recipe.git`, branch `feature/on-policy-wdl-sft`

## Models

| Role | Model | Path |
|------|-------|------|
| Weak model (model1, sub_models.0) | Qwen3-4B-Base (pretrained init; trained jointly, not frozen) | `/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base` |
| Strong model (model2, sub_models.1) | Qwen3-4B-Base-SFT-stage-1 (SFT-finetuned init; trained jointly) | `/data-1/.cache/Qwen3-4B-Base-SFT-stage-1` |

## Datasets

| Dataset | Purpose | Path |
|---------|---------|------|
| EnsembleLLM (RL format) | Training data (parquet, ~111K entries) | `/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet` |
| MATH-500 | Validation | `/data-1/dataset/MATH-500/` |
| AIME-2025 | Validation | `/data-1/dataset/AIME-2025/` |

## Key Code Paths

```
verl/trainer/ppo/core_algos.py        # Loss functions (WDL-SFT loss to be added here)
verl/trainer/ppo/ray_trainer.py       # Training main loop
verl/workers/fsdp_workers.py          # FSDP backend
verl/workers/actor/dp_actor.py        # Actor worker
verl/workers/rollout/vllm_rollout/    # vLLM rollout (fused sampling via logit mixing)
verl/models/joint_model/              # Joint model implementation (from parent branch)
recipe/on_policy_wdl_sft/             # On-Policy WDL-SFT recipe (scripts, configs)
recipe/joint_training/                # Joint-training recipe (ARCHIVAL — from parent branch)
```

## Reward Function

- Path: `recipe/joint_training/custom_reward_function_latex_verify.py`
- 3-tier verification: LaTeX semantic → math_verify → string matching
- Returns binary reward: +1.0 (correct) / -1.0 (incorrect), -1.0 for truncated (no EOS)

## Current Hyperparameters (v2 defaults — for experiments 1a/1b/1c)

| Parameter | Value | Notes |
|-----------|-------|-------|
| loss_mode | `wdl_sft_is` | v2. Pin to `wdl_sft` to reproduce v1 runs. |
| λ (logit mixing weight) | 0.5 | |
| β (reverse SFT weight) | **0.0** (1a/1c), 0.1 (1b) | Re-testing reverse SFT under v2. |
| N (rollouts per prompt) | 8 | |
| Learning rate | **5e-7** (1a/1b), 1e-6 (1c) | Full re-test with v2 stability |
| Batch size (prompts/step) | 64 | |
| Max prompt length | 500 | |
| Max response length | 4096 | |
| grad_clip | 500.0 | |
| weight_decay | 0.1 | |
| lr_warmup_steps | 5 | |
| clip_ratio_low / high | 0.2 / 0.27 | v2 only (binary mask thresholds) |
| rollout_is | `token` | v2 only (token-level IS weights) |
| rollout_is_threshold | 5.0 | v2 only (MiniRL default) |

## Training History

**v1 runs (loss_mode=wdl_sft)**:

| Run | Config | Steps | MATH-500 mean@3 (model2 offline) | Notes |
|-----|--------|-------|------|-------|
| EXP-12 M5 | lr=1e-6, β=0.1 | ~1000 | — | Diverged, checkpoints discarded |
| EXP-13 M5.5 | lr=5e-7, β=0 | 300 (complete) | 78.6% | Stable baseline |
| EXP-14 M5.6 | lr=5e-7, β=0.1 | ~458 | 79.1% | Stable at lr=5e-7; model1 collapsed −21.6% |
| EXP-15 LR3 | lr=1e-6, β=0 | 125 (best) → 274 (killed) | 79.6% | Peaked at step 125, then drifted |

Key finding from v1: model2 ceiling ≈ 79-80% MATH-500 mean@3 regardless of lr/β — the loss itself caps performance. Baseline MiniRL (same init) hits ~74% mean@1 at step 100 while v1 reaches ~68% mean@1 at step 300. Motivates v2.

**v2 runs (loss_mode=wdl_sft_is)**:

| Run | Config | Status | Peak MATH-500 (online model2, step) | Step-300 final | Run ID |
|-----|--------|--------|------|------|--------|
| **EXP-16 (1a)** | lr=5e-7, β=0, v2 | Complete (2026-04-20) | **71.37%** (step 225) | 70.36% | `WDL-SFT-Qwen3-4B-MATH-1A_1776594597` |
| **EXP-17 (1b)** | lr=5e-7, β=0.1, v2 | Complete (2026-04-21) | **70.97%** (step 225 & 275) | 70.36% | `WDL-SFT-Qwen3-4B-MATH-1B_1776695220` |
| EXP-18 (1c) | lr=1e-6, β=0, v2 | **Running** (launched 2026-04-21 18:53) | — | — | `WDL-SFT-Qwen3-4B-MATH-1C_1776768784` |

Key finding so far: **v2 breaks the v1 online ceiling** (+2.4 pp at step 300 vs M5.5). 1B matches 1A online despite β=0.1 — training-level evidence that v2 contains the reverse SFT instability. Preliminary offline eval on 1A step 225 model2: MATH-500 mean@3 = 83.07% (vs v1 EVAL-10 = 79.6%).

**Current focus**: Monitor 1C — key milestone step 125 (v1's LR3 peak-and-crash). Run offline eval on 1A full trio (step 225 model1 + model2) and 1B full trio (model1 is the critical test — see "Status of reverse SFT abandoned" above). See `docs/joint_training/plans/active/wdl_sft_is.md` §4 for decision criteria.

## Documentation (Archival)

Documentation in `docs/joint_training/` was created during the parent branch's joint-training development (Stages 1 & 2). It remains useful for understanding the underlying infrastructure but does **not** describe the On-Policy WDL-SFT algorithm or training loop.

| Directory | Purpose | Status |
|---|---|---|
| `specs/` | Technical specs for joint model / logit fusion | ARCHIVAL — infrastructure reference |
| `constraints/` | Development rules and boundaries | Still applicable |
| `plans/active/README.md` | Active plan index | ACTIVE |
| `plans/active/wdl_sft_is.md` | **WDL-SFT v2 (IS-corrected)** — post-fix rerun matrix remains open; historical 1A/1B/1C are pre-fix | ACTIVE |
| `plans/active/ablation_single_model.md` | **Single-model ablation (2A/B/C + 2Z baseline)** — partially complete; post-fix rows remain open | ACTIVE |
| `plans/completed/dual_submodel_rollout_wdl_sft.md` | Dual-submodel rollout WDL-SFT — implemented and smoke-passed, but 3A real run failed as method-level negative result | ARCHIVAL |
| `plans/completed/on_policy_wdl_sft_v1.md` | v1 plan (original) | ARCHIVAL — superseded by v2 |
| `specs/wdl_sft_is.md` | v1 vs v2 loss comparison | ACTIVE |
| `plans/completed/lr_search.md` | LR search — abandoned, see wdl_sft_is.md | ARCHIVAL |
| `plans/completed/` | Archived plans from Stage 1 & 2 | ARCHIVAL |
| `courses/` | Educational docs on joint-training theory | ARCHIVAL — background reference |
| `guides/` | Testing, tuning, migration guides | Partially applicable |
| `guides/meituan_platform.md` | **Meituan AFO layered launch + cross-host portability playbook** — MUST follow when adding any experiment that will run on Meituan | ACTIVE |
| `references/` | External articles and papers | ARCHIVAL — background reference |

## Operational Best Practices (MANDATORY)

Before launching any training, monitoring, checkpoint transfer, or large file operation, **always read and follow** these rules:

1. **Use tmux for all long-running work**: Training scripts, monitoring scripts, checkpoint transfers (`rsync`/`cp` to secondary mounts), and large file downloads must all run inside tmux sessions. This prevents job loss due to SSH disconnection or terminal closure. Example:
   ```bash
   tmux new-session -s train
   bash recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_m5_5.sh
   # Ctrl-B D to detach; tmux attach -t train to re-attach
   ```

2. **Checkpoint transfer**: When transferring checkpoints to a secondary mount, always use tmux. Large model checkpoints (~8GB+ per checkpoint) take significant time.

3. **Monitor in tmux**: If running a monitoring/tail script alongside training, put it in its own tmux pane or window.

4. **Meituan-bound experiments follow the layered playbook**: When an experiment is confirmed to run on the Meituan AFO platform, it MUST be authored according to `docs/joint_training/guides/meituan_platform.md` — four-layer launch path, default-local-overridable-everything paths in `run_*.sh`, dolphinfs overrides isolated to `recipe/.../meituan/env.sh`. Local-only experiments don't need layers 1–3, but must still write `run_*.sh` by the same portability rules so migration later is a one-file change. Every experiment must run on BOTH the local box and Meituan without per-host branches in the experiment script itself.

## Agent Guidelines

- **Subagents**: Use subagents (Agent tool) for exploratory/independent work to save main context. Subagents should use the **Haiku** model (`model: "haiku"`) for cost efficiency — do NOT use Opus for subagent work unless the task specifically requires strong reasoning.
- **Plans**: Active plans live in `docs/joint_training/plans/active/`. When creating or updating experiment plans, always update the index in this CLAUDE.md file.

## Quick Links

- Active plan index: `docs/joint_training/plans/active/README.md`
- **Current focus**: `docs/joint_training/plans/active/wdl_sft_is.md` (v2 plan; post-fix rerun matrix still open)
- Archived dual-submodel rollout negative result: `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft.md`
- Dual-submodel 3A failure analysis: `docs/joint_training/plans/completed/dual_submodel_rollout_wdl_sft_3a_failure_analysis.md`
- Single-model ablation plan: `docs/joint_training/plans/active/ablation_single_model.md`
- Single-model ablation scripts: `recipe/on_policy_wdl_sft/ablation_single_model/`
- **Meituan platform playbook** (how to add experiments that run on both local + AFO): `docs/joint_training/guides/meituan_platform.md`
- v1 vs v2 loss spec: `docs/joint_training/specs/wdl_sft_is.md`
- v1 loss code: `verl/trainer/ppo/core_algos.py:1861` (wdl_sft)
- v2 loss code (pending): same file, registered as `wdl_sft_is`
- MiniRL reference (clip/IS implementation): `verl/trainer/ppo/core_algos.py:1782`
- Rollout correction helper: `verl/trainer/ppo/rollout_corr_helper.py`
- On-Policy WDL-SFT original plan (v1): `docs/joint_training/plans/completed/on_policy_wdl_sft_v1.md`
- Archived LR search plan: `docs/joint_training/plans/completed/lr_search.md`
- Joint model code: `verl/models/joint_model/modeling_joint_qwen3.py`
- Joint config: `verl/models/joint_model/configuration_joint_qwen3.py`
- Weight utils: `verl/models/joint_model/weight_utils.py`
- Enable joint training: `+actor_rollout_ref.model.joint_training=True`

<!-- BEGIN CODEX PROJECT MEMORY -->
## Codex Project Memory

Codex: this project keeps project-scoped memory in `.codex/project-memory/MEMORY.md`.
For tasks that depend on prior project history, recurring workflows,
experiments, or user/project preferences, read that index first, then open only
the referenced topic files that match the task. Treat dated status as possibly
stale and verify live state before acting.

Do not store secrets in project memory. Prefer durable workflow lessons and
known pitfalls over transient status.
<!-- END CODEX PROJECT MEMORY -->
