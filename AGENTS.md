# AGENTS.md — Quick Reference for Coding Agents

> See `CLAUDE.md` for full project context. This file is a concise reference for any agent working in this codebase.

## Environment

```bash
conda activate verl07          # REQUIRED before any Python/pytest command
```

## Repo Layout (key paths only)

```
verl/trainer/ppo/ray_trainer.py     # Main training loop (fit_and_evaluate)
verl/trainer/ppo/core_algos.py      # GRPO/PPO loss, advantage computation
verl/workers/fsdp_workers.py        # FSDP hybrid engine (init_model, rollout_mode, update_weights)
verl/workers/actor/dp_actor.py      # Actor forward/backward (_forward_micro_batch, compute_log_prob, update_policy)
verl/checkpoint_engine/base.py      # Weight sync coordinator (CheckpointEngineManager)
verl/utils/torch_functional.py      # logprobs_from_logits, log_probs_from_logits_all_rmpad
verl/models/joint_model/            # ★ Joint training model (QwenJointForCausalLM)
recipe/                             # Git SUBMODULE — commit separately: cd recipe && git add/commit
tests/joint_training/               # Joint training tests (feat/ and regression/)
docs/joint_training/                # Investigation report, target spec, progress tracker
```

## Joint Training Architecture

```
logits_fused = (1 - λ) * logits_model1 + λ * logits_model2
```

- Model class: `verl/models/joint_model/modeling_joint_qwen3.py` — `QwenJointForCausalLM`
- Two `Qwen3ForCausalLM` sub-models in `nn.ModuleList`, fused in `forward()`
- `forward(eval_only=True)` returns model2-only logits (for evaluation)
- Fusion is transparent to upper layers — `core_algos.py` and `dp_actor.py` are **unchanged**
- Joint mode activated via Hydra config: `+actor_rollout_ref.model.joint_training=True`
- Weight extraction: `verl/models/joint_model/weight_utils.py` (extract_sub_model_weights)
- Eval flow: `ray_trainer._validate()` → `update_weights(eval_only=True)` → `rollout_mode(eval_only=True)` → extracts model2 weights for vLLM

## Modified Files (from upstream)

| File | What changed |
|------|-------------|
| `verl/workers/fsdp_workers.py` | `rollout_mode(eval_only=False)` param, joint weight extraction block |
| `verl/trainer/ppo/ray_trainer.py` | `_is_joint_training` flag, `_validate()` eval-only weight sync |
| `verl/checkpoint_engine/base.py` | `update_weights(eval_only=False)` param passthrough |

## Running Tests

```bash
pytest tests/joint_training/ -v          # All joint training tests (46 pass, 2 skip)
pytest tests/joint_training/feat/ -v     # Feature tests only
pytest tests/joint_training/regression/ -v  # Regression tests only
```

## Preparing Joint Model Weights

```bash
python -m verl.models.joint_model.prepare_joint_weights \
    --base_model_path Qwen/Qwen3-1.7B-Base \
    --output_path .cache/huggingface/QwenJoint-1.7B \
    --fusion_lambda 0.5
```

## Training

```bash
bash recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh
```

## Data

- **Dataset**: `/data-1/dataset/gsm8k` — `train.parquet` (7473 rows), `test.parquet`
- **Columns**: `data_source`, `prompt` (chat format), `ability`, `reward_model` (has `ground_truth`), `extra_info`

## Known Issues

- HuggingFace model download stalls behind SOCKS proxy — prepare weights manually when network is available
- `AutoModelForVision2Seq` import error in transformers — pre-existing env issue, causes 2 test skips
- `recipe/` is a git submodule — always commit inside it separately

## Pending Work (Phase 2)

1. Download and prepare Qwen3-1.7B-Base joint weights
2. vLLM integration for joint model rollout
3. End-to-end GPU training validation
