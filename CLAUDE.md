# verl: Volcano Engine Reinforcement Learning for LLMs

verl is a flexible, efficient, and production-ready RL training framework for large language models, supporting diverse algorithms (PPO, GRPO, DAPO, etc.) with seamless integration of FSDP, Megatron-LM, vLLM, and SGLang.

## Repository Information

- **Your Fork**: https://github.com/AlexJJ009/verl-v0.7 (stable v0.7.0 version)
- **Old Fork**: https://github.com/AlexJJ009/verl (older version, for reference)
- **Upstream**: https://github.com/verl-project/verl (original project)
- **Branch**: `feature/joint-training` (active development)

### Git Remote Configuration
```bash
origin   -> https://github.com/AlexJJ009/verl-v0.7.git (your fork)
upstream -> https://github.com/verl-project/verl.git (original)
```

### Multi-Node Setup Purpose
This repository is configured for multi-node distributed training/inference across multiple devices with identical hardware configurations. Each compute node will clone from your fork to ensure consistent codebase versions.

## Directory Structure

```
verl/
├── verl/                              # Core package
│   ├── single_controller/             # Hybrid controller for RL dataflows
│   │   ├── base/                      # Base controller implementations
│   │   └── ray/                       # Ray-based distributed controller
│   ├── trainer/                       # Training orchestration
│   │   ├── ppo/                       # PPO/GRPO trainer (ray_trainer.py, core_algos.py)
│   │   └── config/                    # Training configuration templates
│   ├── workers/                       # Distributed worker implementations
│   │   ├── actor/                     # Actor workers (dp_actor.py)
│   │   ├── critic/                    # Critic workers (value function)
│   │   ├── rollout/                   # Rollout generation workers
│   │   ├── reward_model/              # Reward model workers
│   │   ├── reward_manager/            # Reward computation (naive, dapo, prime, batch)
│   │   ├── engine/                    # Backend engine abstractions
│   │   ├── sharding_manager/          # Model sharding utilities
│   │   ├── fsdp_workers.py            # ★ Main FSDP hybrid engine worker
│   │   └── config/                    # Worker configuration
│   ├── models/                        # Model implementations
│   │   ├── joint_model/               # ★ Joint training model (our addition)
│   │   │   ├── modeling_joint_qwen3.py      # QwenJointForCausalLM
│   │   │   ├── configuration_joint_qwen3.py # QwenJointConfig
│   │   │   ├── weight_utils.py              # Weight extraction utilities
│   │   │   └── prepare_joint_weights.py     # Script to create joint weights
│   │   ├── llama/                     # LLaMA model support
│   │   ├── qwen2/                     # Qwen2 model support
│   │   ├── transformers/              # HuggingFace transformers integration
│   │   └── mcore/                     # Megatron-Core integration
│   ├── utils/                         # Utilities
│   │   ├── checkpoint/                # Checkpoint management
│   │   ├── dataset/                   # Dataset processing
│   │   ├── torch_functional.py        # logprobs_from_logits, etc.
│   │   └── ...                        # logger, metric, reward_score, vllm, sglang, etc.
│   ├── checkpoint_engine/             # Weight sync between trainer and rollout
│   │   └── base.py                    # ★ CheckpointEngineManager (modified)
│   └── ...                            # interactions, third_party, tools, experimental
│
├── tests/                             # Test suite
│   ├── joint_training/                # ★ Joint training tests (our addition)
│   │   ├── feat/                      # Feature tests
│   │   │   ├── test_joint_model.py          # 21 tests: model class, fusion, gradients
│   │   │   ├── test_weight_utils.py         # 8 tests: weight extraction
│   │   │   ├── test_grpo_integration.py     # 6 tests: full GRPO step simulation
│   │   │   └── test_auto_model_loading.py   # 4 tests: AutoModel trust_remote_code
│   │   └── regression/               # Regression tests
│   │       └── test_existing_functionality.py # 9 tests: standard model, utils unchanged
│   └── ...                            # single_controller, trainer, workers, models, utils
│
├── recipe/                            # Algorithm recipes (git submodule)
│   ├── joint_training/                # ★ Joint training recipe (our addition)
│   │   └── run_joint_grpo_qwen3_1.7b.sh
│   ├── dapo/                          # DAPO
│   └── ...                            # prime, r1, sppo, entropy, etc.
│
├── docs/                              # Documentation
│   └── joint_training/                # ★ Joint training docs
│       ├── progress.md                      # Implementation progress tracker
│       ├── verl_joint_training_investigation_report.md  # Architecture analysis
│       └── GRPO_Joint_Training_Target_v1.md             # Target specification
│
├── examples/                          # Training examples
└── scripts/                           # Utility scripts
```

## Environment Setup

**CRITICAL**: All code and scripts MUST be run within the Miniconda environment:

```bash
conda activate verl07
```

## Key Development Principles

### 1. Test-Driven Development (TDD)
- Write tests BEFORE implementing new features
- Two test categories: `feat/` (new functionality) and `regression/` (existing behavior)
- Run joint training tests: `pytest tests/joint_training/ -v`
- Run all tests: `pytest tests/`

### 2. Codebase Hygiene
- Follow existing code patterns and conventions
- `recipe/` is a **git submodule** — see workflow below
- Use pre-commit hooks: `.pre-commit-config.yaml`

### 3. Git Submodule Workflow (`recipe/`)

**Submodule remote configuration:**
- `.gitmodules` points to personal fork: `https://github.com/AlexJJ009/verl-recipe.git`
- Upstream (`verl-project/verl-recipe.git`) does not have our custom commits — always push to the fork
- Inside `recipe/`, both `origin` and `myfork` remotes point to `AlexJJ009/verl-recipe.git`

**Correct update workflow (order matters):**
```bash
# Step 1: commit & push inside the submodule FIRST
cd recipe
git add <files>
git commit -m "feat: ..."
git push origin feature/joint-training   # submodule content must reach remote BEFORE step 2

# Step 2: update parent repo pointer
cd ..
git add recipe
git commit -m "chore(recipe): update submodule pointer"
git push origin feature/joint-training
```

**On a new server (first clone):**
```bash
git clone https://github.com/AlexJJ009/verl-v0.7.git
cd verl-v0.7
git submodule update --init --recursive
```

**On an existing server (after pulling parent repo):**
```bash
git pull origin feature/joint-training
git submodule update --recursive
```

**Why recipe content doesn't update on other servers (pitfall):**
The parent repo only stores a pointer (commit SHA) to the submodule. If submodule commits are not pushed to their own remote, other servers cannot fetch them even though the parent repo's pointer is updated. Always push submodule changes to `AlexJJ009/verl-recipe.git` before committing the pointer in the parent repo.

### 3. Documentation Discipline
- Write documentation ONLY when explicitly requested by the user
- Code should be self-documenting through clear naming

## Current Development Goal: Joint Training with GRPO

### Algorithm
```
logits_fused = (1 - λ) × logits_model1 + λ × logits_model2
```

| Phase | Logits Used | Notes |
|-------|-------------|-------|
| Rollout (training) | Fused logits | Both models contribute |
| compute_log_prob | Fused logits | old_log_probs for PPO ratio |
| update_policy | Fused logits | new_log_probs, standard GRPO loss |
| Evaluation | model2 only | Assess standalone capability |

### Implementation Status: Phase 1 Complete

**Created files:**
| File | Purpose |
|------|---------|
| `verl/models/joint_model/modeling_joint_qwen3.py` | QwenJointForCausalLM model class |
| `verl/models/joint_model/configuration_joint_qwen3.py` | QwenJointConfig |
| `verl/models/joint_model/weight_utils.py` | Weight extraction (is_joint, extract_sub_model) |
| `verl/models/joint_model/prepare_joint_weights.py` | Create joint weights from base model |
| `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh` | Training launch script |

**Modified files:**
| File | Change |
|------|--------|
| `verl/workers/fsdp_workers.py` | `rollout_mode(eval_only=True)` — extracts model2 weights for eval |
| `verl/trainer/ppo/ray_trainer.py` | `_validate()` switches to model2-only, restores after |
| `verl/checkpoint_engine/base.py` | `update_weights(eval_only=True)` passthrough |

**Key design:** Logit fusion is encapsulated in the model class — `core_algos.py`, `dp_actor.py` are **unchanged**. Joint mode activates via config: `+actor_rollout_ref.model.joint_training=True`.

**Tests: 46 passing, 2 skipped** (skips are pre-existing env issue with AutoModelForVision2Seq import)

### Pending Work
- Download Qwen/Qwen3-1.7B-Base (network issue — run `prepare_joint_weights.py` manually)
- vLLM integration for joint model rollout (Phase 2)
- End-to-end GPU training test

### Models and Data
- **Model**: `Qwen/Qwen3-1.7B-Base` (both sub-models use this)
- **Joint weights output**: `.cache/huggingface/QwenJoint-1.7B`
- **Dataset**: `/data-1/dataset/gsm8k` (train.parquet, test.parquet)

### Reference Documents
- Investigation report: `docs/joint_training/verl_joint_training_investigation_report.md`
- Target specification: `docs/joint_training/GRPO_Joint_Training_Target_v1.md`
- Progress tracker: `docs/joint_training/progress.md`

## Important Notes

- This is a research and production framework used by ByteDance Seed Team
- The codebase scales to 671B parameter models and hundreds of GPUs
- Performance is critical for on-policy RL algorithms
- Follow the [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) when optimizing
- **Known env issue**: `AutoModelForVision2Seq` import fails — affects `verl.trainer.ppo.core_algos` import chain. Pre-existing, not caused by joint training changes.
