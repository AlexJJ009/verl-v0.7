# verl: Volcano Engine Reinforcement Learning for LLMs

verl is a flexible, efficient, and production-ready RL training framework for large language models, supporting diverse algorithms (PPO, GRPO, DAPO, etc.) with seamless integration of FSDP, Megatron-LM, vLLM, and SGLang.

## Repository Information

- **Your Fork**: https://github.com/AlexJJ009/verl-v0.7 (stable v0.7.0 version)
- **Old Fork**: https://github.com/AlexJJ009/verl (older version, for reference)
- **Upstream**: https://github.com/verl-project/verl (original project)

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
├── verl/                          # Core package
│   ├── single_controller/         # Hybrid controller for RL dataflows
│   │   ├── base/                  # Base controller implementations
│   │   └── ray/                   # Ray-based distributed controller
│   ├── trainer/                   # Training orchestration
│   │   ├── ppo/                   # PPO trainer implementation
│   │   └── config/                # Training configuration templates
│   ├── workers/                   # Distributed worker implementations
│   │   ├── actor/                 # Actor model workers (policy)
│   │   ├── critic/                # Critic model workers (value function)
│   │   ├── rollout/               # Rollout generation workers
│   │   ├── reward_model/          # Reward model workers
│   │   ├── reward_manager/        # Reward computation management
│   │   ├── engine/                # Backend engine abstractions
│   │   ├── sharding_manager/      # Model sharding utilities
│   │   └── config/                # Worker configuration
│   ├── models/                    # Model implementations
│   │   ├── llama/                 # LLaMA model support
│   │   ├── qwen2/                 # Qwen2 model support
│   │   ├── transformers/          # HuggingFace transformers integration
│   │   └── mcore/                 # Megatron-Core integration
│   ├── utils/                     # Utilities
│   │   ├── checkpoint/            # Checkpoint management
│   │   ├── dataset/               # Dataset processing
│   │   ├── logger/                # Logging utilities
│   │   ├── metric/                # Metrics computation
│   │   ├── reward_score/          # Reward scoring utilities
│   │   ├── vllm/                  # vLLM integration utilities
│   │   ├── sglang/                # SGLang integration utilities
│   │   ├── megatron/              # Megatron utilities
│   │   ├── kernel/                # Custom CUDA kernels
│   │   └── profiler/              # Performance profiling
│   ├── interactions/              # RL interaction protocols
│   ├── third_party/               # Third-party integrations
│   │   ├── vllm/                  # vLLM patches and extensions
│   │   └── torch/                 # PyTorch extensions
│   ├── tools/                     # CLI tools and utilities
│   ├── model_merger/              # Model merging utilities
│   └── experimental/              # Experimental features
│
├── examples/                      # Training examples
│   ├── ppo_trainer/               # PPO training examples
│   ├── grpo_trainer/              # GRPO training examples
│   ├── remax_trainer/             # ReMax training examples
│   ├── rloo_trainer/              # RLOO training examples
│   ├── sft/                       # Supervised fine-tuning examples
│   ├── sglang_multiturn/          # Multi-turn dialogue examples
│   ├── split_placement/           # GPU placement examples
│   └── tuning/                    # Hyperparameter tuning examples
│
├── recipe/                        # Algorithm recipes and research implementations
│   ├── dapo/                      # DAPO (AIME 50 points)
│   ├── prime/                     # PRIME algorithm
│   ├── r1/                        # R1-style reasoning
│   ├── sppo/                      # Self-play preference optimization
│   ├── entropy/                   # Entropy mechanism experiments
│   └── open_math_reasoning/       # Math reasoning recipes
│
├── tests/                         # Test suite
│   ├── single_controller/         # Controller tests
│   ├── trainer/                   # Trainer tests
│   ├── workers/                   # Worker tests
│   ├── models/                    # Model tests
│   ├── utils/                     # Utility tests
│   ├── special_e2e/               # End-to-end tests
│   └── special_distributed/       # Distributed training tests
│
├── docs/                          # Documentation
├── docker/                        # Docker configurations
└── scripts/                       # Utility scripts
```

## Key Development Principles

### 1. Test-Driven Development (TDD)
- Write tests BEFORE implementing new features
- Maintain two test categories:
  - **Feature tests**: Validate new functionality works as intended
  - **Regression tests**: Ensure existing functionality remains intact
- Test cases co-evolve with the development lifecycle
- Run tests frequently during development: `pytest tests/`
- For distributed tests: `pytest tests/special_distributed/`
- For end-to-end tests: `pytest tests/special_e2e/`

### 2. Documentation Discipline
- Write documentation ONLY when explicitly requested by the user
- Code should be self-documenting through clear naming and structure
- Focus on code quality over documentation quantity

### 3. Codebase Hygiene
- Keep the repository clean and organized
- Remove unused code and dependencies
- Follow existing code patterns and conventions
- Use pre-commit hooks: `.pre-commit-config.yaml`
- Maintain consistent formatting and style

## User Objectives

The user is working with this repository to:

1. **Develop custom RL/SFT algorithms**: Build new training pipelines and algorithms on top of verl's infrastructure
2. **Reuse existing code**: Leverage verl's modular architecture, configurations, and implementations
3. **Learn software engineering**: Study verl's design patterns to advance from junior to expert developer level

## Key Technical Concepts

### Hybrid Controller Architecture
- Decouples computation and data dependencies
- Enables flexible RL dataflow representation (PPO, GRPO, etc.)
- Supports various device mappings for efficient resource utilization

### Backend Integrations
- **Training**: FSDP, FSDP2, Megatron-LM
- **Inference**: vLLM (≥0.8.2), SGLang (0.5.6), HuggingFace Transformers
- **Models**: Qwen-3, Llama3.1, Gemma2, DeepSeek, etc.

### Supported Algorithms
PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, PF-PPO, and more

## Environment Setup

**CRITICAL**: All code and scripts MUST be run within the Miniconda environment:

```bash
conda activate verl07
```

Always activate this environment before:
- Running any Python scripts or training jobs
- Executing tests
- Installing dependencies
- Running examples or recipes

## Development Workflow

1. **Before coding**: Write tests that define expected behavior
2. **During coding**: Run tests iteratively to validate implementation
3. **After coding**: Ensure all tests pass and no regressions introduced
4. **Code review**: Check for cleanliness, organization, and adherence to patterns

## Important Notes

- This is a research and production framework used by ByteDance Seed Team
- The codebase scales to 671B parameter models and hundreds of GPUs
- Performance is critical for on-policy RL algorithms
- Follow the [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) when optimizing
