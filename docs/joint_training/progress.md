# Joint Training GRPO - Implementation Progress

## Status: Phase 1 Complete (Core Training Pipeline)

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

5. **GRPO Integration Tests** (6 tests passing)
   - `tests/joint_training/feat/test_grpo_integration.py`
   - Full simulation: old_log_probs → advantages → policy loss → backward

6. **AutoModel Loading Tests** (4 tests passing)
   - `tests/joint_training/feat/test_auto_model_loading.py`
   - Verified trust_remote_code loading path used by fsdp_workers.py

7. **Regression Tests** (7 passed, 2 skipped due to env)
   - `tests/joint_training/regression/test_existing_functionality.py`

8. **Training Recipe**
   - `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh`

9. **Model Weight Preparation Script**
   - `verl/models/joint_model/prepare_joint_weights.py`
   - Downloads base model and creates joint model checkpoint

### Pending

- [ ] Download Qwen/Qwen3-1.7B-Base (network issue, run manually)
- [ ] End-to-end training test with actual GPU
- [ ] vLLM integration for joint model rollout (Phase 2)
- [ ] Joint-specific metrics monitoring (optional)

### Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| feat/test_joint_model.py | 21 | All passing |
| feat/test_weight_utils.py | 8 | All passing |
| feat/test_grpo_integration.py | 6 | All passing |
| feat/test_auto_model_loading.py | 4 | All passing |
| regression/test_existing_functionality.py | 7 passed, 2 skipped | OK |
| **Total** | **46 passing, 2 skipped** | |

### Git Commits

| Hash | Description |
|------|-------------|
| 245908cc | feat: add QwenJointForCausalLM model class with logit fusion |
| 7a9e8691 | feat: add FSDP dual-mode weight sync and eval-mode switching |
| d54ae321 | test: add regression tests for joint training changes |
| 2f70926d | feat: add GRPO integration tests, recipe, and progress tracking |
| 961892c3 | test: add AutoModel loading tests for joint model |

### Architecture Decisions

1. **Logit fusion in model class**: Encapsulated inside QwenJointForCausalLM, transparent to upper layers
2. **Eval-only via parameter**: `eval_only=True` on forward() returns model2-only logits
3. **Dual-mode weight sync**: rollout_mode(eval_only=True) extracts model2 weights for evaluation
4. **Minimal core changes**: core_algos.py, dp_actor.py, metric_utils.py unchanged
5. **Configuration-driven**: `+actor_rollout_ref.model.joint_training=True` in Hydra config

### How to Prepare Model Weights (when network is available)

```bash
conda activate verl07
python -m verl.models.joint_model.prepare_joint_weights \
    --base_model_path Qwen/Qwen3-1.7B-Base \
    --output_path .cache/huggingface/QwenJoint-1.7B \
    --fusion_lambda 0.5
```

### How to Run Training

```bash
conda activate verl07
bash recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh
```
