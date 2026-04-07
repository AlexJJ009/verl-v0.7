# On-Policy Weak-Driven SFT — Agent Entry Point

This file is the table of contents for coding agents working on the on-policy-wdl-sft branch. It is intentionally short. Deeper context lives in `docs/joint_training/`.

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

## Repository

- Repo: `https://github.com/AlexJJ009/verl-v0.7`
- Upstream: `https://github.com/verl-project/verl`
- Branch: `feature/on-policy-wdl-sft`
- Parent branch: `feature/joint-training` (Stage 1 & 2 complete)
- `recipe/` is a submodule → `https://github.com/AlexJJ009/verl-recipe.git`

## Current Status

Joint-training runtime (Stage 1) and algorithm correctness (Stage 2) are complete on the parent branch. This branch implements **On-Policy Weak-Driven SFT** — a new training algorithm combining logit fusion with on-policy rollout and forward/reverse SFT.

- Active plan: `docs/joint_training/plans/active/on_policy_wdl_sft.md`
- Completed plans (from parent branch): `docs/joint_training/plans/completed/`

## Design Anchor

```
# Fused rollout (on-policy)
z_mix = (1 - lambda) * z_weak + lambda * z_strong
responses = autoregressive_sample(softmax(z_mix), N)

# Forward WD-SFT on correct rollouts
L+ = -mean[ log P_mix(y_correct) ]

# Reverse WD-SFT on incorrect rollouts
L- = +mean[ log P_mix(y_incorrect) ]

# Combined loss
L = L+ + beta * L-
```

- Model class: `verl/models/joint_model/modeling_joint_qwen3.py`
- Joint config: `verl/models/joint_model/configuration_joint_qwen3.py`
- Weight utils: `verl/models/joint_model/weight_utils.py`
- Enable: `+actor_rollout_ref.model.joint_training=True`

## Key Code Paths

```
verl/trainer/ppo/ray_trainer.py       # Training main loop
verl/trainer/ppo/core_algos.py        # Core algorithms (loss functions)
verl/workers/fsdp_workers.py          # FSDP backend
verl/workers/actor/dp_actor.py        # Actor worker
verl/workers/rollout/vllm_rollout/    # vLLM rollout (fused sampling)
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
| `plans/active/` | Current development plan | Before starting a task |
| `plans/completed/` | Archived plans from parent branch | For historical context |
| `codereview/` | Review results (active and completed) | When reviewing or fixing issues |
| `courses/` | Educational docs (human-facing) | When learning the codebase |
| `guides/` | Practical how-tos (testing, migration) | When running tests or setting up |
| `references/` | External articles and media | For background context |
| `constraints/experiment_tracking/` | Experiment index maintenance rules | Before registering or cleaning up experiments |

## Quick Links

- Active plan: `docs/joint_training/plans/active/on_policy_wdl_sft.md`
- Development principles: `docs/joint_training/constraints/principles/development_principles.md`
- Modification boundaries: `docs/joint_training/constraints/boundaries/modification_boundaries.md`
- Commit rules: `docs/joint_training/constraints/commit_rules/commit_conventions.md`
- Testing guide: `docs/joint_training/guides/testing.md`
- Sample efficiency tuning: `docs/joint_training/guides/sample_efficiency_tuning.md`
- Algorithm spec (logit fusion): `docs/joint_training/specs/joint_training_algorithm_v1.md`
- SFT analysis (theory): `docs/joint_training/references/external/sft_analysis.tex`
- Experiment index: `recipe/joint_training/EXPERIMENT_INDEX.md`
- Inference results: `recipe/joint_training/INFERENCE_RESULTS.md`
- Experiment tracking spec: `docs/joint_training/constraints/experiment_tracking/experiment_index_spec.md`
