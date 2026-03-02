# Joint Training GRPO - Implementation Progress

## Status: Phase 2 Debugging - Environment Fixes Complete, Rollout Architecture Refactoring Needed

### Phase 1: Core Training Pipeline (Complete - 64 tests passing)

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

6. **AutoModel Loading Tests** (4 tests passing)
   - `tests/joint_training/feat/test_auto_model_loading.py`

7. **Regression Tests** (9 tests passing)
   - `tests/joint_training/regression/test_existing_functionality.py`
   - Fixed API incompatibilities with updated core_algos

8. **Model Weight Preparation Script**
   - `verl/models/joint_model/prepare_joint_weights.py`

### Phase 2A: HuggingFace Rollout Integration (Partial)

9. **GenerationMixin Support** ✅
   - `verl/models/joint_model/modeling_joint_qwen3.py`: Added `GenerationMixin` inheritance
   - `model.generate()` works with fused logits automatically
   - `_eval_only_mode` attribute for switching between fused and model2-only during generation

10. **HFRollout Integration in FSDP Workers** ✅
    - `verl/workers/fsdp_workers.py`:
      - `_build_rollout()`: Special case for HF rollout — shares FSDP model instance
      - `rollout_mode()`: Early return for HFRollout — sets `_eval_only_mode`
      - `trainer_mode()`: New method — resets `_eval_only_mode`
      - `generate_sequences_hf()`: New remote-callable method for agent loop

11. **HFRollout Bug Fix** ✅
    - `verl/workers/rollout/hf_rollout.py`: Fixed `super().__init__()` call

12. **HF Rollout + Joint Model Tests** (9 tests passing) ✅
    - `tests/joint_training/feat/test_hf_rollout_joint.py`

13. **GPU End-to-End Tests** (7 tests passing) ✅
    - `tests/joint_training/feat/test_gpu_e2e.py`

14. **HFRolloutReplica Registration** ⚠️ Partial
    - `verl/workers/rollout/replica.py`: Registered HFRolloutReplica for 'hf' mode
    - Implemented HYBRID mode support
    - **Issue**: Architectural mismatch between HFRollout (in-process FSDP) and AgentLoopManager (async remote servers)

### Phase 2B: Environment Fixes (Complete)

**Flash Attention & Dependencies:**
- ✅ Installed flash_attn v2.7.4 via conda-forge
- ✅ Fixed torch/torchvision version mismatch (torch 2.6.0 → 2.5.1, torchvision 0.23.0 → 0.21.0)
- ✅ Patched torchvision NMS operator registration for torch 2.5.1

**Import & Configuration Fixes:**
- ✅ Fixed AutoModelForVision2Seq import (conditional try/except wrapper)
- ✅ Disabled torch.compile (incompatible with torch 2.5.1)
- ✅ Added `joint_training` field to HFModelConfig

### Current Blocker: AgentLoopManager Architecture Mismatch

**Issue**: Training script hits architectural incompatibility when trying to initialize rollout:
- **Root Cause**: HFRollout designed for in-process FSDP HYBRID mode (shared process with trainer)
- **Expected Architecture**: AgentLoopManager expects remote async servers (vLLM, SGLang, TRT-LLM)
- **Error**: AgentLoopManager calls `server.generate.remote()` but FSDPWorker has `generate_sequences_hf()`
- **Impact**: Training script loads model weights successfully but fails at validation/rollout generation

**Resolution Options**:
1. **Option A (Recommended)**: Implement vLLM backend with joint logit fusion support
   - Pros: Maintains async architecture, can scale to multi-node, proven vLLM infrastructure
   - Cons: Requires vLLM custom kernel development for fusion

2. **Option B**: Refactor trainer to support HYBRID in-process rollout for HFRollout
   - Pros: Uses existing HFRollout without new kernels
   - Cons: Architectural change, breaks async pattern, single-node only

3. **Option C**: Create vLLM rollout wrapper that delegates to HFRollout
   - Pros: Minimal changes to trainer
   - Cons: Redundant code, still doesn't solve kernel issues

### Pending

- [ ] Resolve AgentLoopManager/HFRollout architectural incompatibility
- [ ] End-to-end distributed training test
- [ ] Joint-specific metrics monitoring (optional)

### Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| feat/test_joint_model.py | 21 | ✅ All passing |
| feat/test_weight_utils.py | 8 | ✅ All passing |
| feat/test_grpo_integration.py | 6 | ✅ All passing |
| feat/test_auto_model_loading.py | 4 | ✅ All passing |
| feat/test_hf_rollout_joint.py | 9 | ✅ All passing |
| feat/test_gpu_e2e.py | 7 | ✅ All passing |
| regression/test_existing_functionality.py | 9 | ✅ All passing |
| **Total** | **64 passing** | |

### Git Commits

| Hash | Description |
|------|-------------|
| 245908cc | feat: add QwenJointForCausalLM model class with logit fusion |
| 7a9e8691 | feat: add FSDP dual-mode weight sync and eval-mode switching |
| d54ae321 | test: add regression tests for joint training changes |
| 2f70926d | feat: add GRPO integration tests, recipe, and progress tracking |
| 961892c3 | test: add AutoModel loading tests for joint model |
| f207e1ab | feat: add HF rollout integration for joint training with fused logits |
| 1cbdd086 | fix(joint_training): Phase 2 debugging - environment and training setup |

### Architecture Notes

1. **Logit fusion**: `logits = (1-λ) * forward(model1) + λ * forward(model2)`
2. **HFRollout design**: Shares FSDP model instance, calls `model.generate()` → `model.forward()`
3. **Eval-only mode**: `_eval_only_mode` attribute checked in `forward()` for eval-only switching
4. **FSDP integration**: rollout_mode/trainer_mode context switching in FSDPWorker
5. **Configuration**: `+actor_rollout_ref.model.joint_training=True` enables feature

### How to Run Training (when resolved)

```bash
conda activate verl07
bash recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh
```

