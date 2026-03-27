# Joint Training — Agent Entry Point

This file is the table of contents for coding agents working on the joint-training branch. It is intentionally short. Deeper context lives in `docs/joint_training/`.

## Environment

The project runs inside a Docker container (no conda).

```bash
# Build the image (once)
bash /data-1/verl07/build.sh          # → verl-train:cu126

# Launch a training container
bash /data-1/verl07/run_train.sh
```

- Dockerfile: `docker/Dockerfile.joint-training.cu126`
- Python 3.12 managed by uv
- PyTorch 2.9.1+cu126, vLLM 0.12.0, flash-attn 2.8.1
- Apex, TransformerEngine, Megatron-LM compiled from source

## Repository

- Repo: `https://github.com/AlexJJ009/verl-v0.7`
- Upstream: `https://github.com/verl-project/verl`
- Branch: `feature/joint-training`
- `recipe/` is a submodule → `https://github.com/AlexJJ009/verl-recipe.git`

## Current Status

Stage 1 (runtime bring-up) is complete. Stage 2 (algorithm correctness) is active.

- Active plan: `docs/joint_training/plans/active/stage2.md`
- Completed plan: `docs/joint_training/plans/completed/stage1.md`

## Design Anchor

```
logits_fused = (1 - lambda) * logits_model1 + lambda * logits_model2
```

- Model class: `verl/models/joint_model/modeling_joint_qwen3.py`
- Joint config: `verl/models/joint_model/configuration_joint_qwen3.py`
- Weight utils: `verl/models/joint_model/weight_utils.py`
- Enable: `+actor_rollout_ref.model.joint_training=True`

## Key Code Paths

```
verl/trainer/ppo/ray_trainer.py       # Training main loop
verl/trainer/ppo/core_algos.py        # Core algorithms
verl/workers/fsdp_workers.py          # FSDP backend
verl/workers/actor/dp_actor.py        # Actor worker
verl/workers/rollout/vllm_rollout/    # vLLM rollout
verl/models/joint_model/              # Joint model implementation
tests/joint_training/                 # Joint-training tests
recipe/joint_training/                # Recipes and launchers
```

## Documentation Map

All deep documentation lives in `docs/joint_training/`:

| Directory | Purpose | When to read |
|---|---|---|
| `specs/` | Technical specs (agent-facing) | Before implementing any feature |
| `constraints/` | Rules, boundaries, commit conventions | Before writing or committing code |
| `plans/` | Active and completed development plans | Before starting a task |
| `codereview/` | Review results (active and completed) | When reviewing or fixing issues |
| `courses/` | Educational docs (human-facing) | When learning the codebase |
| `guides/` | Practical how-tos (testing, migration) | When running tests or setting up |
| `references/` | External articles and media | For background context |
| `constraints/experiment_tracking/` | Experiment index maintenance rules | Before registering or cleaning up experiments |

## Quick Links

- Development principles: `docs/joint_training/constraints/principles/development_principles.md`
- Modification boundaries: `docs/joint_training/constraints/boundaries/modification_boundaries.md`
- Commit rules: `docs/joint_training/constraints/commit_rules/commit_conventions.md`
- Testing guide: `docs/joint_training/guides/testing.md`
- Sample efficiency tuning: `docs/joint_training/guides/sample_efficiency_tuning.md`
- Tuning iteration log: `docs/joint_training/guides/sample_efficiency_iteration_log.md`
- Algorithm spec: `docs/joint_training/specs/joint_training_algorithm_v1.md`
- Experiment index: `recipe/joint_training/EXPERIMENT_INDEX.md`
- Inference results: `recipe/joint_training/INFERENCE_RESULTS.md`
- Experiment tracking spec: `docs/joint_training/constraints/experiment_tracking/experiment_index_spec.md`
