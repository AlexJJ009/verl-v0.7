# Modification Boundaries

This document defines which files and paths are safe to modify and which must be treated with extreme care.

## Joint-Training Owned Code (Safe to Modify)

These paths are owned by the joint-training effort and can be modified freely:

```
verl/models/joint_model/           # Joint model class, config, weight utils
tests/joint_training/              # All joint-training tests
docs/joint_training/               # All joint-training documentation
recipe/joint_training/             # Joint-training recipes and scripts
```

## Framework Code (Modify With Care)

These paths are shared framework code. Modifications must:
- Preserve existing single-model behavior
- Include regression tests
- Be justified as reusable (not joint-training-only hacks)

```
verl/trainer/ppo/ray_trainer.py    # Training main loop
verl/trainer/ppo/core_algos.py     # Core algorithm implementations
verl/workers/fsdp_workers.py       # FSDP training backend
verl/workers/actor/dp_actor.py     # Actor worker
verl/checkpoint_engine/base.py     # Checkpoint engine
verl/utils/checkpoint/             # Checkpoint utilities
verl/utils/torch_functional.py     # Torch utilities
verl/workers/rollout/vllm_rollout/ # vLLM rollout engine
```

## Do Not Modify

These paths should not be touched by joint-training work:

```
verl/trainer/config/               # Upstream Hydra config schemas
verl/single_controller/            # Single controller mode (unrelated)
verl/protocol.py                   # Core data protocol
```

## Persistent Runtime Constraints

1. H800 memory is **not pooled** across 8 GPUs. OOM reasoning must be per GPU.
2. Colocated vLLM rollout and actor/ref workloads share the same device budget.
3. Root-mounted paths (`/tmp`, `/root/.config`) can break runs if `/` fills up. Use `/data-1` for all large or high-churn runtime paths.
4. Optional acceleration dependencies (`flash_attn`) must fail safe.
5. A training run that completes operationally may still be algorithmically wrong.
