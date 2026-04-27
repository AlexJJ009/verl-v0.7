# Dual-Submodel Rollout WDL-SFT — Plan and Open Decisions

- Status: **ACTIVE PLAN — design discussion before implementation**
- Created: 2026-04-27
- Base branch: `feature/on-policy-wdl-sft`
- Planned working branch: `feature/on-policy-wdl-sft-dual-rollout`
- Recipe target: `recipe/on_policy_wdl_sft/dual_submodel_rollout/`
- Parent plans:
  - `docs/joint_training/plans/active/wdl_sft_is.md`
  - `docs/joint_training/plans/active/ablation_single_model.md`

## 1. Motivation

Current On-Policy WDL-SFT samples rollout trajectories from the fused distribution:

$$
P_{\text{mix}}=\text{Softmax}((1-\lambda)z_1+\lambda z_2)
$$

This makes rollout quality depend on fused logits. Recent results and advisor discussion suggest this is likely the wrong place to use fusion: the fused rollout distribution can be weaker than model2-only rollout, while model2 alone gives higher-quality trajectories.

The proposed change is to decouple **data generation** from **gradient amplification**:

1. During rollout, generate trajectories from model1 and model2 separately.
2. By default, select only model2 trajectories as labeled training data.
3. During training, keep the joint fused-logit forward/backward path unchanged.
4. Update both submodels from the fused-logit training loss.

In this design, fusion is no longer the rollout policy. Fusion becomes a training-time gradient amplifier.

## 2. Algorithm Boundary

### Kept From v2 WDL-SFT-IS

- Joint model architecture: `QwenJointForCausalLM` with `sub_models.0` and `sub_models.1`.
- Both model1 and model2 train together; `freeze_model1=False`.
- Training forward pass uses fused logits:

$$
z_{\text{train}}=(1-\lambda)z_1+\lambda z_2
$$

- Evaluation target remains model2, not fused joint output.
- Default core loss remains `wdl_sft_is` unless explicitly changed after discussion.
- Default experiment scale remains the 1A schedule: 300 steps, save/test every 25 steps, `lr=5e-7`, default `β=0`, `rollout.n=8`, prompt batch size 64.
- Reverse-SFT weight `β` must remain configurable through `actor_rollout_ref.actor.policy_loss.wdl_sft_beta` / `WDL_SFT_BETA`, supporting both `0.0` and nonzero values such as `0.1`.

### Changed

- Training rollout source changes from fused logits to per-submodel logits.
- Default selected rollout source changes to model2:

$$
D_{\text{train}} = D_{\text{rollout},2}
$$

- Model1 rollout is still generated initially, but used for diagnostics and future selection policies, not default training.

### Not In Scope For First Implementation

- Do not train on both model1 and model2 trajectories by default.
- Do not select the best trajectory across model1/model2 by reward yet.
- Do not introduce a new reward function.
- Do not change checkpoint format.
- Do not change offline eval scripts except where needed to point at the new run family.
- Do not change Meituan platform layers unless this experiment is confirmed to run there; local scripts should still follow the default-local-overridable-everything rules.

## 3. Current Code Facts

The current implementation has the required pieces, but their default behavior is fused rollout:

- HF/FSDP training model: `verl/models/joint_model/modeling_joint_qwen3.py`
  - Default forward runs both submodels and fuses logits.
  - `eval_only=True` switches to model2-only.
- vLLM rollout model: `verl/models/joint_model/vllm_modeling_joint_qwen3.py`
  - Default forward packs both submodel hidden states.
  - `compute_logits()` fuses model1/model2 logits.
  - `_use_model2_only=True` exists, but currently only used when loading non-joint model2-only weights.
- Trainer loop: `verl/trainer/ppo/ray_trainer.py`
  - Currently generates one rollout batch per training step.
  - It then rewards that batch, recomputes `old_log_probs`, computes rollout correction, computes advantages, and updates the actor.
- Weight sync: `verl/workers/fsdp_workers.py`
  - `rollout_mode(eval_only=True)` can sync only model2 weights for validation.
  - Normal training sync sends full joint weights to rollout.

## 4. Proposed Implementation

### 4.1 Branch and Commit Hygiene

1. Commit the current worktree state before implementation if the user confirms the existing dirty changes should be preserved as a checkpoint.
2. Create branch:

```bash
git switch -c feature/on-policy-wdl-sft-dual-rollout
```

Do not revert existing dirty files unless the user explicitly asks.

### 4.2 Config Surface

Add config under `actor_rollout_ref.rollout`, preferably under `custom` to avoid broad config-schema churn if possible:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      joint_rollout_sources: ["sub_model_0", "sub_model_1"]
      joint_rollout_select: "sub_model_1"
      joint_rollout_train_on_selected_only: true
```

Semantics:

- `fused`: current behavior.
- `sub_model_0`: rollout from model1 logits only.
- `sub_model_1`: rollout from model2 logits only.
- `joint_rollout_sources`: list of sources to generate each step.
- `joint_rollout_select`: source whose batch enters training.
- `joint_rollout_train_on_selected_only=true`: preserve first implementation boundary.

Default behavior in the base config should remain current fused rollout for backwards compatibility. The new recipe explicitly opts into dual-submodel rollout.

### 4.3 Joint Model Rollout Source Switching

Add an explicit rollout source mode to both joint model implementations:

- `verl/models/joint_model/modeling_joint_qwen3.py`
- `verl/models/joint_model/vllm_modeling_joint_qwen3.py`

Candidate attribute:

```python
self._rollout_source = "fused"  # fused | sub_model_0 | sub_model_1
```

Expected behavior:

- Training forward path stays fused unless the rollout manager explicitly switches source.
- HF rollout can set this on the shared model object.
- vLLM rollout needs a worker-extension RPC to set this on every vLLM worker model.
- Validation can continue using the existing model2-only eval sync path.

### 4.4 Trainer Rollout Flow

In `verl/trainer/ppo/ray_trainer.py`, wrap generation with source switching:

1. Build the normal repeated generation batch.
2. For each source in `joint_rollout_sources`:
   - set rollout source on the rollout engine,
   - generate sequences,
   - compute reward or preserve reward output if generated by agent loop,
   - record source-specific metrics.
3. Select `joint_rollout_select` for training.
4. Continue the existing pipeline using the selected batch only:
   - union with original repeated prompt batch,
   - compute `response_mask`,
   - compute reward tensor,
   - recompute `old_log_probs`,
   - compute rollout correction if enabled,
   - compute advantages / reward labels,
   - update actor.

Metrics to log:

- `dual_rollout/model1_correct_ratio`
- `dual_rollout/model2_correct_ratio`
- `dual_rollout/model1_response_len_mean`
- `dual_rollout/model2_response_len_mean`
- `dual_rollout/selected_source`
- Optional: per-source extraction failure and truncation rate from reward metadata.

### 4.5 Reward Labels for `wdl_sft_is`

Status: **confirmed implementation/spec mismatch; must be fixed before dual-rollout launch.** See the full handoff report in `docs/joint_training/plans/active/wdl_sft_is.md` §9.

The trainer currently overrides `advantages` with raw reward labels only when:

```python
loss_mode == "wdl_sft"
```

But `compute_policy_loss_wdl_sft_is()` also reads:

```python
reward_labels = advantages[:, 0]
```

So the trainer should treat both `wdl_sft` and `wdl_sft_is` as WDL-style losses:

```python
if loss_mode in {"wdl_sft", "wdl_sft_is"}:
    reward_labels = batch.batch["token_level_scores"].sum(dim=-1)
    batch.batch["advantages"] = reward_labels.unsqueeze(-1).expand_as(response_mask).clone()
```

This is a correctness cleanup for label semantics. It should be fixed before or together with the dual-rollout implementation.

Historical `wdl_sft_is` results should be labeled as pre-fix/current-implementation results until rerun or audited. Do not silently treat EXP-16/17/18 as spec-correct `wdl_sft_is` runs.

## 5. Clip / Mask Decisions

Status: **confirmed design boundary for first implementation**.

Principle: keep masks that protect valid token accounting or training-time stability after the selected data batch is fixed. Disable mechanisms that would implicitly turn model2 rollout data back into fused-policy rollout data.

### 5.1 Keep: Training-Time Ratio Binary Mask

Decision: keep `wdl_sft_is` ratio masking between `log_prob` and `old_log_prob`.

Why:

- It controls drift across multiple mini-batches from the same selected rollout batch.
- It is about training stability after the batch is selected, not about whether rollout came from fused or model2.
- It stays meaningful if `old_log_probs` are recomputed under the same fused training policy used by the actor update.

Risk:

- If `old_log_probs` are computed under fused logits but the data came from model2, the ratio is not correcting sampling-policy mismatch. It is only a local trust-region proxy for fused training updates.

Default for first implementation: **keep it**.

### 5.2 Disable By Default: `rollout_is_weights` Loss Multiplication

Current v2 uses `rollout_is_weights` computed from:

$$
\rho = \exp(\log \pi_{\text{old/FSDP}} - \log \pi_{\text{rollout/vLLM}})
$$

In fused-rollout v2, this mostly corrects vLLM-vs-FSDP numerical mismatch for the same intended fused policy.

In dual-submodel rollout, selected data comes from model2. If the training policy is fused, then:

- rollout log-prob is model2 policy,
- old log-prob is fused policy,
- the weight becomes a real cross-policy correction from model2 to fused.

That may conflict with the new philosophy: model2 is deliberately the data policy, while fused logits are deliberately the training-time gradient amplifier.

Decision for first implementation: **disable loss multiplication by `rollout_is_weights`, but keep rollout/FSDP log-prob diagnostics if cheap**.

Why:

- This preserves the new algorithm idea cleanly: model2 is deliberately the data policy.
- It avoids suppressing model2's higher-quality data distribution toward the fused policy.
- The remaining ratio binary mask still provides a local training-time trust-region proxy.

Tradeoff:

- This gives up the original v2 interpretation of `rollout_is_weights` as vLLM/FSDP mismatch correction.
- If later evidence suggests this correction is needed, add an explicit opt-in ablation. Do not make it the default for 3A.

### 5.3 Disable: Rejection Sampling Masks

Current recipe sets `rollout_rs=null`, so no rejection-sampling mask is active.

Decision: keep it disabled in the first dual-submodel run.

Why:

- The first question is whether model2-only rollout plus fused training improves the algorithm.
- Rejection sampling would add another data-selection mechanism and make attribution harder.

### 5.4 Keep: Response Mask

Always keep `response_mask`.

Why:

- It is padding / valid-token masking, not an algorithmic trust-region choice.
- Removing it would corrupt loss aggregation.

### 5.5 Default First-Run Mask Stack

Recommended initial default:

| Mechanism | Default | Reason |
|---|---:|---|
| `response_mask` | keep | required correctness mask |
| WDL-SFT-IS ratio binary mask | keep | controls mini-batch drift in fused training |
| `rollout_is_weights` loss multiplication | disable | avoid correcting away model2 rollout policy |
| rollout correction metrics | keep if available | diagnostic only |
| rejection sampling `rollout_rs` | disable | avoid confounding data selection |
| reverse SFT `β` | default `0.0`, configurable | match 1A default while allowing `β=0.1` ablation |

### 5.6 Reverse-SFT `β` Is A Configurable Loss Weight

`β` is not a clip or mask. It controls whether incorrect responses contribute the reverse-SFT term:

$$
L = L^+ + \beta L^-
$$

Required behavior:

- `β=0.0` must be supported and remains the 3A default.
- `β=0.1` and other nonnegative values must be supported through the same config path used by existing scripts:

```bash
WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
+actor_rollout_ref.actor.policy_loss.wdl_sft_beta=${WDL_SFT_BETA}
```

Interpretation:

- `β=0.0`: forward-only learning from correct model2 rollouts.
- `β>0`: bidirectional WDL-SFT, with reverse-SFT pressure on incorrect selected model2 rollouts.
- After the `wdl_sft_is` reward-label fix, all-incorrect groups only produce reverse-SFT signal when `β>0`; this should be covered by tests.

## 6. Recipe Plan

Create:

```text
recipe/on_policy_wdl_sft/dual_submodel_rollout/
├── README.md
├── _common_dual_rollout.sh
├── run_3a_model2_rollout_beta0.sh
└── run_3b_model2_rollout_beta01.sh
```

Initial experiments:

| Run | Rollout sources | Selected data | Training forward | Loss | β | lr | Compare |
|---|---|---|---|---|---:|---:|---|
| 3A | model1 + model2 | model2 only | fused joint logits | `wdl_sft_is` | 0 | 5e-7 | 1A, 2A-SFT, 2Z-SFT |
| 3B | model1 + model2 | model2 only | fused joint logits | `wdl_sft_is` | 0.1 | 5e-7 | 3A, 1B, 2B-SFT |

Script rules:

- Use Docker via `/data-1/verl07/run_train.sh` for launch.
- Use tmux for long-running training.
- Use default-local-overridable-everything path style.
- Keep `VLLM_ATTENTION_BACKEND=FLASHINFER`.
- Keep FSDP `attn_implementation=flash_attention_2`.
- Put shared defaults in `_common_dual_rollout.sh`, with `WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}` as an overridable environment variable.
- `run_3a_model2_rollout_beta0.sh` should export `WDL_SFT_BETA=0.0`.
- `run_3b_model2_rollout_beta01.sh` should export `WDL_SFT_BETA=0.1`.

## 7. Tests and Validation Gates

### Unit / CPU Tests

Add or update tests for:

- HF joint model source switching:
  - fused output equals weighted logits,
  - `sub_model_0` output ignores model2,
  - `sub_model_1` output ignores model1.
- vLLM joint model source mode, with mocked submodels if full vLLM construction is too heavy.
- trainer label override:
  - `wdl_sft` and `wdl_sft_is` both receive raw reward labels in `advantages`.
  - all-correct groups produce forward-SFT signal.
  - all-incorrect groups produce reverse-SFT signal when `WDL_SFT_BETA=0.1`.
- selector behavior:
  - when `joint_rollout_select=sub_model_1`, only model2 batch reaches actor update.

### Smoke Test

Run inside Docker with tiny settings:

```bash
TOTAL_TRAINING_STEPS=1 \
TRAIN_PROMPT_BSZ=2 \
TRAIN_PROMPT_MINI_BSZ=1 \
ROLLOUT_AGENT_NUM_WORKERS=1 \
bash recipe/on_policy_wdl_sft/dual_submodel_rollout/run_3a_model2_rollout_beta0.sh
```

Validation criteria:

- both model1 and model2 rollout paths execute,
- selected source is model2,
- reward metrics exist for both sources,
- actor update executes once,
- no shape mismatch in `old_log_probs`, `rollout_log_probs`, `advantages`, or `response_mask`,
- checkpoint/update-weight cycle still works.

## 8. Progress Checklist

- [x] Plan drafted into docs.
- [ ] Commit or otherwise checkpoint current dirty worktree.
- [ ] Create implementation branch.
- [ ] Add config surface.
- [ ] Add HF joint source switching.
- [ ] Add vLLM joint source switching and RPC.
- [ ] Add trainer dual-rollout generation and selected-batch training.
- [ ] Fix `wdl_sft_is` reward-label override.
- [ ] Add recipe folder and wrapper.
- [ ] Add tests.
- [ ] Run targeted CPU tests.
- [ ] Run Docker smoke test.
- [ ] Update this plan with implementation notes and any changed decisions.

## 9. Discussion Queue

Before implementation, confirm:

1. Should `rollout.n=8` mean 8 per source, or total 8 split across model1/model2?
   - Current recommendation: 8 per source, selected model2 gives 8 training samples per prompt, matching 1A data budget while adding model1 diagnostics cost.
2. Should model1 rollout be generated in the first run if it is not selected?
   - Current recommendation: yes, because it gives direct evidence for the advisor question and future selector policies.
3. How should historical `wdl_sft_is` results be labeled after the reward-label fix?
   - Current recommendation: mark EXP-16/17/18 and related 2X runs as pre-fix/current-implementation results; use new experiment IDs for post-fix spec-correct reruns.
4. Should the new algorithm get a new public name beyond "dual-submodel rollout WDL-SFT"?
   - Current recommendation: keep the descriptive name until the boundary is stable.
