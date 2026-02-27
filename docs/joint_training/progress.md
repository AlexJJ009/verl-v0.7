# Joint Training GRPO - Implementation Progress

## Status: In Progress

### Completed Tasks

1. **QwenJointForCausalLM Model Class** (21 tests passing)
   - `verl/models/joint_model/modeling_joint_qwen3.py`
   - `verl/models/joint_model/configuration_joint_qwen3.py`
   - Supports logit fusion, gradient flow, parameter freezing, eval_only mode

2. **Weight Extraction Utilities** (8 tests passing)
   - `verl/models/joint_model/weight_utils.py`
   - Functions: is_joint_model_state_dict, extract_sub_model_weights, etc.

3. **FSDP Worker Modifications**
   - `verl/workers/fsdp_workers.py`: rollout_mode(eval_only=True) support
   - `verl/checkpoint_engine/base.py`: update_weights(eval_only=True) passthrough

4. **Ray Trainer Modifications**
   - `verl/trainer/ppo/ray_trainer.py`: _validate() switches to model2-only weights

5. **Regression Tests** (7 passed, 2 skipped due to env)
   - `tests/joint_training/regression/test_existing_functionality.py`

6. **Training Recipe**
   - `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh`

### In Progress

- **Model Weight Preparation**: Downloading Qwen/Qwen3-1.7B-Base and creating joint weights
  - Script: `verl/models/joint_model/prepare_joint_weights.py`
  - Output: `.cache/huggingface/QwenJoint-1.7B`

### Remaining Work

- [ ] Complete model download and weight preparation
- [ ] Test end-to-end training pipeline
- [ ] vLLM integration for joint model rollout (Phase 2)
- [ ] Add joint-specific metrics monitoring
- [ ] Performance optimization and memory profiling

### Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| feat/test_joint_model.py | 21 | All passing |
| feat/test_weight_utils.py | 8 | All passing |
| regression/test_existing_functionality.py | 7 passed, 2 skipped | OK |
| **Total** | **36 passing** | |

### Architecture Decisions

1. **Logit fusion in model class**: Encapsulated inside QwenJointForCausalLM, transparent to upper layers
2. **Eval-only via parameter**: `eval_only=True` on forward() returns model2-only logits
3. **Dual-mode weight sync**: rollout_mode(eval_only=True) extracts model2 weights for evaluation
4. **Minimal core changes**: core_algos.py, dp_actor.py, metric_utils.py unchanged
5. **Configuration-driven**: `+actor_rollout_ref.model.joint_training=True` in Hydra config
