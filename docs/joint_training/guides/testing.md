# Testing Guide

## Environment Setup

All tests run inside the Docker container (`verl-harness`). Launch it with:

```bash
bash /data-1/verl07/run_train.sh
```

## Running Joint-Training Tests

Run all joint-training tests:

```bash
cd /data-1/verl07/verl
pytest tests/joint_training/ -v
```

Run a specific test file:

```bash
pytest tests/joint_training/regression/test_validation_generation_logging.py -v
```

Run with specific test function:

```bash
pytest tests/joint_training/reward/test_reward_function_latex_verify.py::test_specific_function -v
```

## Test Categories

### Joint Training Tests (`tests/joint_training/`)

- `regression/` — Regression tests for stabilized behavior
- `feat/` — Feature tests
  - Model: `test_joint_model.py`, `test_auto_model_loading.py`, `test_weight_utils.py`
  - Rollout: `test_vllm_joint_rollout.py`, `test_hf_rollout_joint.py`, `test_hf_sync_rollout_manager.py`
  - Training: `test_grpo_integration.py`, `test_gpu_e2e.py`
  - P1 metrics: `test_p1_grad_metrics.py`, `test_p1_logit_disagreement.py`, `test_p1_reward_aggregation_metrics.py`
  - P2 audit: `test_p2_semantic_audit.py`
- `reward/` — Reward function tests (e.g., LaTeX verification)

### Related Framework Tests

- `tests/workers/actor/test_special_dp_actor.py` — DP actor tests
- `tests/workers/critic/test_dynamic_dp_critic.py` — Dynamic DP critic
- `tests/workers/test_fsdp_workers.py` — FSDP workers
- `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py` — Checkpoint cleanup
- `tests/utils/test_attention_utils_on_cpu.py` — Attention utils
- `tests/utils/test_torch_functional.py` — Torch functional utils

## Writing New Tests

1. Every real fix should include a regression test.
2. Prefer CPU-only tests when possible for faster iteration.
3. Place joint-training tests in the appropriate subdirectory under `tests/joint_training/`.
4. Tests that verify framework behavior (not joint-training-specific) go in the standard `tests/` hierarchy.

## End-to-End Validation

Unit tests are necessary but not sufficient. Real E2E reruns still find lifecycle bugs that unit tests miss. After significant changes:

1. Run the full test suite: `pytest tests/joint_training/ -v`
2. If the change affects runtime behavior, consider a short E2E run with the recipe launcher.
