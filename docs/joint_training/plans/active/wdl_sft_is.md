# WDL-SFT-IS：给 On-Policy WDL-SFT 补齐稳定性机制

- 状态：**ACTIVE — 1a/1b 训练完成，1c 运行中，offline eval 待跑**
- 分支：`feature/on-policy-wdl-sft`（不新开分支，本算法仍是 WDL-SFT 家族）
- 创建日期：2026-04-19
- 前置：
  - `on_policy_wdl_sft.md`（原方案，本文档称为 **WDL-SFT v1**）
  - `lr_search.md`（已归档到 `completed/`，结论：LR 搜索不是解药）

---

## 1. 动机（数据驱动）

跑完 EXP-12 ~ EXP-15 四个实验 + 多次 offline eval，**看到三件事**：

1. **WDL-SFT v1 的 model2 在 MATH-500 mean@3 的上限固定在 79–80%**，和 lr、β、训练步数都弱相关：
   - EXP-13 M5.5 (lr=5e-7, β=0, step 300): **78.6%**
   - EXP-14 M5.6 (lr=5e-7, β=0.1, step 300): **79.1%**
   - EXP-15 LR3 (lr=1e-6, β=0, step 125): **79.6%**
2. **同机同初始化（Qwen3-4B-Base）下的 baseline MiniRL** 在 step 100 就能到 MATH-500 mean@1 ≈ **74%**。WDL-SFT v1 训练完 300 步 mean@1 ≈ **68%**。
3. **训练中期（step 125 左右）**开始出现 response 长度、entropy、extraction failure 同步漂移的现象；不是 LR 问题，是 loss 层没有任何稳定性机制。

**结论**：上限由 loss 决定，不是 LR 决定。v1 缺的是 RL 文献里的常规装备：importance sampling 修正、ratio clipping、train-infer 一致性修正。

## 2. π_old 与 π_new 的三个来源（为什么一定需要 IS）

在 on-policy WDL-SFT 里，rollout 得到的数据虽然是"新采的"，但**训练时使用这份数据的 log-prob 是在不同的 θ 下算的**，有三个来源：

**A. 一次 rollout → 多次梯度更新**（主要来源）
- 配置：`train_batch_size=64 × rollout.n=8 = 512` samples，`ppo_mini_batch_size=8`，`ppo_epochs=1`
- `verl/workers/actor/dp_actor.py:613` 明确定义了 `on_policy = len(mini_batches) == 1 and ppo_epochs == 1`。本配置下 `len(mini_batches) > 1`，按 verl 自己的定义就是 off-policy。
- 效果：mini-batch 2..N 用的是 θ_{i-1} 产生的 logit，但 label 是 θ_0 采的。Drift 每个 mini-batch 累积。

**B. vLLM ≠ FSDP 数值差异**
- Rollout：vLLM + FlashInfer + BF16
- Training：FSDP + FlashAttention2 + mixed precision
- 即使 θ 完全相同，两侧 log-prob 也不一致。标准补丁：token 级 `rollout_is_weights`（MiniRL 用的 threshold=5.0）。

**C. Fused rollout vs per-submodel 训练**（WDL-SFT 特有）
- Rollout 从 $P_{\text{mix}} = \text{softmax}((1{-}\lambda)z_1 + \lambda z_2)$ 采样
- Training loss 用 $\log P_{\theta_2}(a|s)$（单个 submodel）
- 这是算法设计意图：**用 fused 分布放大 model2 的梯度**，model1 是牺牲品。这一项不需要修正，是 feature 不是 bug。

**修复范围**：A + B。C 不动。

## 3. 算法变更：v1 → v2

### v1（保留，不动）

`verl/trainer/ppo/core_algos.py:1861-1917` 的 `compute_policy_loss_wdl_sft` + `compute_wdl_sft_loss`：

```python
L+ = -(1/k) * sum_{i in C} sum_t log P_θ(y^i_t | ...)
L- = (1/(N-k)) * sum_{j in I} sum_t log P_θ(y^j_t | ...)
L  = L+ + β * L-
```

- 不使用 `old_log_prob`
- 不使用 `rollout_is_weights`
- 无 clip

### v2（新增，`loss_mode="wdl_sft_is"`）

在正负两支上都加 binary-mask clipping（MiniRL 风格）+ 接通 `rollout_is_weights`：

```python
ratio = exp(clamp(log_prob - old_log_prob, -20, 20))

# 正样本（i ∈ C）：上界 mask
mask_pos[i, t] = 1 if (reward_label_i == +1) else 0
mask_pos[ratio > 1 + clip_ratio_high] = 0   # ratio 过大 → token 被丢

# 负样本（j ∈ I）：下界 mask
mask_neg[j, t] = 1 if (reward_label_j == -1) else 0
mask_neg[ratio < 1 - clip_ratio_low]  = 0   # ratio 过小 → token 被丢

# Forward SFT token loss (正号)
token_loss_pos = -mask_pos * log_prob * response_mask
# Reverse SFT token loss (负号)
token_loss_neg = +mask_neg * log_prob * response_mask

# Rollout IS weights (vLLM↔FSDP 修正)
if rollout_is_weights is not None:
    token_loss_pos = token_loss_pos * rollout_is_weights.detach()
    token_loss_neg = token_loss_neg * rollout_is_weights.detach()

# Per-group 归一化（保留 v1 的 1/k、1/(N-k) 语义）
k       = correct_count
n_neg   = incorrect_count
L_pos   = token_loss_pos.sum() / max(k, 1)
L_neg   = token_loss_neg.sum() / max(n_neg, 1)
L       = L_pos + β * L_neg
```

### 为什么用 binary mask 而不是 PPO min-clip

- **MiniRL / CISPO / Qwen 系** 最近都用 binary mask
- 对反向 SFT 友好：mask 的"超出 1-ε 就丢弃"语义比 PPO min-clip 在负样本上更直观
- 被 clip 的 token **完全不回传梯度**，清爽；PPO min-clip 会保留 clip 边界处的梯度，信号混杂

### 归一化策略的权衡

v1 用 `seq-mean-token-sum` + `1/k` 归一化：长序列贡献的 token 数更多，grad 能量更大。v2 **暂时保留这个聚合**，保证 v1/v2 仅在 IS/clip 上有差异，可以做干净的 A/B。**长度归一化是独立的问题**，放到后续再测（如果 v2 仍然长度漂移，再单独改成 `token-mean` 做第二轮 A/B）。

## 4. 实验计划（v2 的三个关键点）

所有实验共享的配置：
- 初始化：Qwen3-4B-Base（model1）+ Qwen3-4B-Base-SFT-stage-1（model2）
- λ=0.5，n_resp=8，batch=64 prompts，max_resp_len=4096
- `clip_ratio_low=0.2`, `clip_ratio_high=0.27`, `clip_ratio_c=10.0`（复用现有 config）
- `algorithm.rollout_correction.rollout_is="token"`, `threshold=5.0`
- 300 steps，val_freq=25，save_freq=25

| 实验 | loss_mode | lr | β | 状态 | 在线 MATH-500 peak / step-300 final (model2-only) | Run ID |
|---|---|---|---|---|---|---|
| **1a** (EXP-16) | wdl_sft_is | 5e-7 | 0 | **完成** 2026-04-20 | 71.37% (step 225) / 70.36% | `WDL-SFT-Qwen3-4B-MATH-1A_1776594597` |
| **1b** (EXP-17) | wdl_sft_is | 5e-7 | 0.1 | **完成** 2026-04-21 | 70.97% (step 225 & 275) / 70.36% | `WDL-SFT-Qwen3-4B-MATH-1B_1776695220` |
| **1c** (EXP-18) | wdl_sft_is | 1e-6 | 0 | **运行中** 2026-04-21 18:53 | — | `WDL-SFT-Qwen3-4B-MATH-1C_1776768784` |

**决策流**（更新）：
- ✅ 1a mean@1 = 71.37% > 70% → IS/clip 必要，v2 突破 v1 在线 ceiling（+2.4 pp @ step 300 vs M5.5）；方向有效。
- ✅ 1b model2 在线 ≈ 1a（差距 < 0.5 pp 贯穿 13 个 val 点）→ **β=0.1 在 v2 下训练侧稳定**，v1 时期"reverse SFT 必崩"在训练层被推翻。
- ⚠️ 1b 决策**仍未最终**：v1 的真正失败模式是 EVAL-15 model1 **offline** 格式崩溃（MATH-500 −21.6%，extraction_fail 24–28%），**online 看不到**。v2 的 lower-bound clip 是假设的反制，但尚未在 1b model1 offline 上验证。
- 1c 正在跑，关注 step 125（v1 LR3 崩溃点）。

**离线 eval 下一步优先级**：
1. 1b model1 offline（**决定性**，验证 β>0 在 v2 下是否保持 format compliance）
2. 1b model2 offline（对标 1a 83.07% preliminary）
3. 1a step 225 model2 offline 已跑得 MATH-500 mean@3 = 83.07%，补齐 EVAL-XX ID + model1 eval
4. 1c 完成后同上

## 5. Validation 策略变更：model2-only

**v1 的 online val 用 fused P_mix 做 inference**，但算法设计意图是 model1 牺牲、model2 才是交付物。
- 证据：EVAL-10（fused online: 68.15% vs offline model2-only: 79.6%，gap +11.5%）
- 证据：EVAL-11（model1 只有 63.7%，比 model2 低 15.9%）

**从 v2 开始**：
- **Offline eval** 仍然只评 model2（单独提取权重、单独跑 vLLM），与 v1 一致
- **Online val during training** 考虑增加 model2-only 路径（不再只报 fused）。实现方式有两种：
  - (a) 加一个 config `trainer.val_submodel_only=2`，val 时走 single-submodel 路径
  - (b) 保留 fused val 不变，额外记录 model2 log-probs 和某种代理指标
- 先跑 1a 用 v1 的 fused val 保持可比性，实现 (a) 放到 1a 结果出来之后

## 6. 代码变更清单

| 文件 | 改动 |
|---|---|
| `verl/trainer/ppo/core_algos.py` | ✅ 新增 `compute_policy_loss_wdl_sft_is(...)`，registered as `wdl_sft_is`。不改动 `wdl_sft` 原函数。 |
| `tests/on_policy_wdl_sft/test_wdl_sft_is_loss.py` | ✅ 新建单元测试，12/12 通过（详见 §7）。 |
| `recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_1a.sh` | ✅ 新建，基于 m5_5 脚本仅改 `loss_mode=wdl_sft_is` + `rollout_is=token` + `WANDB_DIR` 持久化。 |
| `recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_1b.sh` | ✅ 基于 1a 改 `WDL_SFT_BETA=0.1`。 |
| `recipe/on_policy_wdl_sft/run_on_policy_wdl_sft_qwen3_4b_math_1c.sh` | ✅ 基于 1a 改 `LR=1e-6`。 |

GPU/sample-efficiency config（`ACTOR_PPO_MAX_TOKEN_LEN`, `TRAIN_PROMPT_BSZ`, `TRAIN_PROMPT_MINI_BSZ`, `ROLLOUT_GPU_MEMORY_UTILIZATION`, `GENERATION_MICRO_BATCH_SIZE`, `LOG_PROB_MICRO_BATCH_SIZE`, `ROLLOUT_MAX_NUM_SEQS` 等）在三个脚本里保持完全一致，保证实验公平对照。

## 7. Tests（Claude 执行前必须先跑过）

必需的单元测试（`tests/joint_training/test_wdl_sft_is_loss.py`）：

1. **Backward compat**: 当 `old_log_prob == log_prob` 且 `rollout_is_weights=None` 且 β=0 时，`compute_policy_loss_wdl_sft_is` 应与 `compute_policy_loss_wdl_sft`（v1 forward-only）在数值上一致（误差 < 1e-6）。
2. **Upper mask**: 构造一个 token 使 ratio=2.0（> 1+clip_ratio_high=1.27），验证该 token 的梯度为 0。
3. **Lower mask on negatives**: 构造一个负样本 token 使 ratio=0.5（< 1-clip_ratio_low=0.8），验证该 token 的梯度为 0。
4. **β>0 path**: 同时有正负样本时，`L_pos` 和 `L_neg` 分别按 k 和 (N-k) 归一化，total = L_pos + β * L_neg。
5. **rollout_is_weights 效果**: 传入非 None 的 weights 时，loss 被 weights 乘进去。
6. **Boundary: k=0**: 全部 incorrect 时 L_pos=0，L=β·L_neg。**重要修改**：v1 在 k=0 时跳过整个 prompt（return zero），v2 应该仍保留 L_neg 部分（因为 β>0 时负样本应产生梯度）。
7. **Boundary: N-k=0**: 全部 correct 时 L_neg=0，L=L_pos。

所有测试用 `torch.manual_seed(42)` 保证可重现。

## 8. Timeline

1. ✅ 文档与 spec：2026-04-19 完成
2. ✅ 代码 + 单元测试：2026-04-19 完成（12/12 通过 + MiniRL 10/10 回归通过）
3. ✅ Git commit + push：2026-04-19 完成（主 repo `a8aaedc7`，recipe submodule `b37a0e3`）
4. ✅ 1a 训练：2026-04-19 16:57 → 2026-04-20 完成（300 steps，peak 71.37% @ step 225）
5. ✅ 1a step 225 model2 初步 offline：MATH-500 mean@3 = 83.07%（EVAL-ID 待补）
6. ✅ 1b 训练：2026-04-20 14:30 → 2026-04-21 完成（300 steps，peak 70.97%；在线稳定，β=0.1 v2 训练侧不崩）
7. ✅ 1c 启动：2026-04-21 18:53（tmux `wdl_sft_is_1c`，ETA ~20h）
8. 🔄 1b offline eval（model1 决定性 + model2 对标）：下一步
9. 🔄 1c 训练完成 + offline eval：预计 2026-04-22 下午
10. 最迟 2026-04-25 给出 v2 三联对照的完整结论

## 9. 相关代码引用

- v1 loss 实现：`verl/trainer/ppo/core_algos.py:1861` (wrapper), `:1920` (core)
- MiniRL 参考实现：`verl/trainer/ppo/core_algos.py:1782`
- Vanilla PPO clip 参考：`verl/trainer/ppo/core_algos.py:1159`
- Rollout correction helper：`verl/trainer/ppo/rollout_corr_helper.py`
- Off-policy 讨论参考：https://richardli.xyz/rl-collapse、https://fengyao.notion.site/off-policy-rl
