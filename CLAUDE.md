# On-Policy Weak-Driven SFT (WDL-SFT) — Agent Entry Point

This file is the table of contents for coding agents working on the **On-Policy WDL-SFT** experiment on branch `feature/on-policy-wdl-sft`. This experiment extends standard Weak-Driven SFT with on-policy rollout and **forward-only** SFT training on correct rollouts.

## Experiment Overview

**On-Policy WDL-SFT** combines logit fusion with on-policy rollout:

1. **Fused Rollout**: Sample N responses from the fused distribution `P_mix = Softmax((1-λ)·z_weak + λ·z_strong)`
2. **Reward Judgment**: Score each response via reward function → correct set C, incorrect set I
3. **Forward SFT (L+)**: SFT on correct rollouts only (β=0, reverse SFT disabled)
4. **Loss**: `L = L+` (cross-entropy on correct rollouts)

> **Note**: Reverse SFT (L-, β>0) was experimentally tested in M5 (lr=1e-6) and M5.6 (lr=5e-7) — both runs crashed due to training instability. Reverse SFT is **abandoned** as it is inherently unstable for this setup. All future work uses forward-only mode (β=0).

This is distinct from MiniRL/GRPO — it preserves SFT simplicity while gaining on-policy self-adaptive training signals.

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
| Weak model | Qwen3-4B-Base (pretrained) | `/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base` |
| Strong model | Qwen3-4B-Base-SFT-stage-1 (SFT-finetuned) | `/data-1/.cache/Qwen3-4B-Base-SFT-stage-1` |

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

## Current Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| λ (logit mixing weight) | 0.5 | |
| β (reverse SFT weight) | **0.0** | Reverse SFT disabled — unstable |
| N (rollouts per prompt) | 8 | |
| Learning rate | **5e-7** | Baseline from M5.5; LR search planned |
| Batch size (prompts/step) | 64 | |
| Max prompt length | 500 | |
| Max response length | 4096 | |
| grad_clip | 500.0 | |
| weight_decay | 0.1 | |
| lr_warmup_steps | 5 | |

## Training History

| Run | Config | Steps | Result |
|-----|--------|-------|--------|
| M5 | lr=1e-6, β=0.1 (bidirectional) | ~1000 | Unstable, training diverged |
| M5.5 | lr=5e-7, β=0.0 (forward-only) | **300 (complete)** | Stable. 12 checkpoints (every 25 steps). Baseline run. |
| M5.6 | lr=5e-7, β=0.1 (reverse re-test) | ~236 | Crashed — confirms reverse SFT instability |

Checkpoints from M5.5 have been transferred to a secondary mount. These serve as the basis for upcoming LR hyperparameter search.

**Next step**: Learning rate search — see `docs/joint_training/plans/active/lr_search.md`

## Documentation (Archival)

Documentation in `docs/joint_training/` was created during the parent branch's joint-training development (Stages 1 & 2). It remains useful for understanding the underlying infrastructure but does **not** describe the On-Policy WDL-SFT algorithm or training loop.

| Directory | Purpose | Status |
|---|---|---|
| `specs/` | Technical specs for joint model / logit fusion | ARCHIVAL — infrastructure reference |
| `constraints/` | Development rules and boundaries | Still applicable |
| `plans/active/on_policy_wdl_sft.md` | On-Policy WDL-SFT plan | ACTIVE |
| `plans/completed/` | Archived plans from Stage 1 & 2 | ARCHIVAL |
| `courses/` | Educational docs on joint-training theory | ARCHIVAL — background reference |
| `guides/` | Testing, tuning, migration guides | Partially applicable |
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

## Agent Guidelines

- **Subagents**: Use subagents (Agent tool) for exploratory/independent work to save main context. Subagents should use the **Haiku** model (`model: "haiku"`) for cost efficiency — do NOT use Opus for subagent work unless the task specifically requires strong reasoning.
- **Plans**: Active plans live in `docs/joint_training/plans/active/`. When creating or updating experiment plans, always update the index in this CLAUDE.md file.

## Quick Links

- On-Policy WDL-SFT plan: `docs/joint_training/plans/active/on_policy_wdl_sft.md`
- LR search plan: `docs/joint_training/plans/active/lr_search.md`
- Joint model code: `verl/models/joint_model/modeling_joint_qwen3.py`
- Joint config: `verl/models/joint_model/configuration_joint_qwen3.py`
- Weight utils: `verl/models/joint_model/weight_utils.py`
- Enable joint training: `+actor_rollout_ref.model.joint_training=True`
