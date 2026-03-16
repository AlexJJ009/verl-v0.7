# MiniRL vs 当前联合训练算法：对比分析报告

> 日期：2026-03-15
> 参考论文：*Stabilizing Reinforcement Learning with LLMs: Formulation and Practices* (Qwen Team, COLM 2024)
> 目标：评估将联合训练从当前 GRPO 配置迁移到 MiniRL 的可行性和难度

---

## 1. 当前联合训练算法配置

当前联合训练使用以下算法栈：

| 组件 | 配置 | 说明 |
|------|------|------|
| 优势估计 | GRPO（z-score 归一化） | `A_i = (r_i - μ_g) / (σ_g + ε)` |
| 策略损失 | 双重裁剪 PPO（vanilla） | `ε_low=0.2, ε_high=0.28, c=10.0` |
| 重要性采样 | 序列级 IS 修正 | `threshold=2.0`，修正 vLLM/FSDP 数值差异 |
| KL 正则 | 完全关闭 | `kl_coef=0, use_kl_loss=False` |
| 聚合方式 | token-mean | 对所有 token 取均值 |
| 批量大小 | 16 prompts × 8 responses = 128 | 已从 8×4=32 扩大 |

**当前面临的核心问题**：

- `rollout_n=8` + 二元奖励 → 组内奖励全相同的概率仍然不低
- 组内 reward 方差为 0 时 → advantage 全为 0 → 梯度为零，浪费 step
- Adam 优化器动量因零梯度 step 被稀释，影响收敛

---

## 2. MiniRL 算法核心设计

### 2.1 损失函数

```
J_MiniRL(θ) = E[ Σ_t  M_t · sg[π_θ(y_t|·) / μ_{θ_old}(y_t|·)] · Â(x,y) · log π_θ(y_t|·) ]
```

其中：

- `sg[·]` = stop-gradient，IS 权重不参与梯度计算
- `μ_{θ_old}` = 推理引擎的策略分布（vLLM BF16）
- `π_{θ_old}` = 训练引擎的策略分布（FSDP FP32），用于裁剪判断
- `π_θ` = 当前正在优化的策略，梯度仅通过 `log π_θ` 流动

### 2.2 裁剪掩码

```
r_t = π_θ(y_t|·) / π_{θ_old}(y_t|·)    ← 策略陈旧度比率

M_t = 0   若 Â > 0 且 r_t > 1 + ε_high (0.27)
M_t = 0   若 Â < 0 且 r_t < 1 - ε_low  (0.2)
M_t = 1   其他情况
```

关键区别：使用**二值掩码**（直接屏蔽 token）而非 PPO 的 min-clip。

### 2.3 优势估计

```
Â(x, y) = R(x, y) - E_{y'}[R(x, y')]  =  R(x, y) - μ_g
```

**仅做均值中心化，不除以标准差**。这与 Dr.GRPO（`norm_adv_by_std_in_grpo=False`）等价。

### 2.4 截断重要性采样 (TIS)

```
w_t = clip(π_θ(y_t|·) / μ_{θ_old}(y_t|·),  max=5)
```

IS 权重在 token 级别截断，阈值为 5。

---

## 3. 逐项差异对比

| 维度 | 当前联合训练 | MiniRL | 影响 |
|------|-------------|--------|------|
| **优势归一化** | z-score: `(r - μ) / (σ + ε)` | 仅均值中心化: `r - μ` | MiniRL 避免 σ=0 时的除零问题，但不解决零优势问题 |
| **IS 修正粒度** | **序列级** IS: `prod_t(π/μ)` | **token 级** IS: 每个 token 独立修正 | token 级更精确，论文证明序列级会造成方差爆炸 |
| **IS 权重梯度** | IS 权重参与梯度（乘在 pg_loss 上） | IS 权重在 `sg[·]` 内，**不参与梯度** | MiniRL 更稳定：梯度仅通过 `log π` |
| **裁剪方式** | PPO 双重裁剪: `max(rA, clip(r)A)` | 二值掩码: 超范围 token 直接屏蔽 | MiniRL 更简洁，不需要 dual-clip |
| **IS 截断阈值** | 序列级 threshold=2.0 | token 级 threshold=5.0 | token 级天然更紧凑，可用更大阈值 |
| **长度归一化** | token-mean（隐式长度归一化） | **无长度归一化** | 论文证明长度归一化破坏一阶近似的理论保证 |
| **KL 正则** | 关闭 | 关闭 | 一致 |
| **梯度流向** | 通过 ratio (`π/π_old`) | 仅通过 `log π` | MiniRL 是 REINFORCE 风格，ratio 只作为权重 |

### 3.1 最关键的三个差异

**差异 1：IS 修正粒度（序列级 vs token 级）**

当前实现计算序列级 IS 权重 `w = exp(Σ_t log(π/μ))`，这在长序列上方差极大。MiniRL 在 token 级别做 IS 修正，每个 token 有独立的修正因子。论文实验显示，**去掉 token 级 IS 修正会导致训练快速崩溃和熵急剧下降**。

**差异 2：梯度流（ratio 梯度 vs stop-gradient）**

当前 PPO 损失让梯度通过 ratio `π_θ/π_{θ_old}` 流动。MiniRL 将 IS 权重放在 `sg[·]` 内，梯度仅通过 `log π_θ`。这使得 MiniRL 本质上是一个**加权 REINFORCE**，更简单且梯度更稳定。

**差异 3：长度归一化**

当前使用 `token-mean` 聚合（等效于除以序列长度）。MiniRL 论文明确指出这会**破坏一阶近似的理论有效性**，导致长序列被系统性低估。

---

## 4. verl 代码库中与 MiniRL 最接近的已有算法

### 4.1 CISPO（最接近）

`core_algos.py` 中注册为 `"cispo"` 的损失函数：

```python
loss = -stop_grad(clip(ratio)) * A * log_prob
```

**相似点**：
- 梯度仅通过 `log_prob` 流动（stop-gradient on ratio）✅
- 使用裁剪比率作为权重 ✅

**差异**：
- 使用 PPO 式裁剪而非二值掩码
- 没有分离 IS 修正权重和策略陈旧度裁剪
- 没有 token 级 IS 修正

### 4.2 Dr.GRPO（优势估计最接近）

GRPO 的 `norm_adv_by_std_in_grpo=False` 模式：

```python
advantage = score - group_mean  # 不除以 std
```

与 MiniRL 的优势估计完全一致。

### 4.3 GPG（结构最接近但缺少 IS）

```python
loss = -log_prob * A  # 纯 REINFORCE，无 IS ratio
```

MiniRL 实质上是 GPG + token 级 IS 权重 + 二值裁剪掩码。

### 4.4 综合评估

| verl 已有算法 | 与 MiniRL 的距离 | 缺少什么 |
|--------------|-----------------|---------|
| CISPO | 最近 | 需改为二值掩码、加 token 级 IS |
| Dr.GRPO (优势) | 优势部分完全匹配 | — |
| GPG (损失) | 结构匹配 | 需加 IS 权重和裁剪掩码 |

---

## 5. MiniRL 是否解决零优势问题？

**答案：不直接解决。**

MiniRL 的优势 `Â = r - μ_g` 在组内奖励完全相同时，同样为 0。这是所有基于组内比较的方法（GRPO、RLOO、MiniRL）的固有问题。

但 MiniRL 有间接缓解：
- **不除以 std**：避免 0/0 的数值问题
- **不使用长度归一化**：保留更完整的训练信号
- 论文使用 `B=64, G=16`（每步 1024 responses），规模远大于我们的 `B=16, G=8`

**真正解决零优势问题的方向**：
1. 使用全局 baseline（如 EMA moving average）而非组内 baseline → verl 中无现成实现
2. 使用 RLOO（leave-one-out baseline）→ verl 中有 `rloo` / `rloo_vectorized`
3. 增大 `rollout_n` 以降低全同概率 → 受 GPU 显存限制
4. 跳过零优势 batch 的 optimizer step → 需修改训练循环

---

## 6. 迁移可行性评估

### 6.1 实现路径

将 MiniRL 实现为 `core_algos.py` 中的新策略损失函数 `"minirl"`：

```python
# 伪代码
def minirl_loss(log_prob, old_log_prob, advantages, response_mask,
                rollout_log_prob,   # μ_{θ_old} 的 log prob
                eps_high=0.27, eps_low=0.2, is_threshold=5.0):

    # 1. Token 级 IS 权重（训练-推理修正）
    is_weight = torch.exp(log_prob - rollout_log_prob)  # π_θ / μ_{θ_old}
    is_weight = torch.clamp(is_weight, max=is_threshold)  # TIS
    is_weight = is_weight.detach()  # stop-gradient

    # 2. 策略陈旧度比率（裁剪判断）
    ratio = torch.exp(log_prob - old_log_prob)  # π_θ / π_{θ_old}

    # 3. 二值裁剪掩码
    mask = torch.ones_like(ratio)
    mask[(advantages > 0) & (ratio > 1 + eps_high)] = 0
    mask[(advantages < 0) & (ratio < 1 - eps_low)] = 0

    # 4. MiniRL 损失
    loss = -mask * is_weight * advantages * log_prob
    loss = (loss * response_mask).sum() / response_mask.sum()

    return loss
```

### 6.2 需要的修改

| 修改点 | 文件 | 难度 | 说明 |
|--------|------|------|------|
| 新增 MiniRL 损失函数 | `core_algos.py` | ⭐低 | 注册新的 policy loss，~50 行 |
| 优势改为 Dr.GRPO | 配置修改 | ⭐低 | `norm_adv_by_std_in_grpo=False` |
| IS 修正改为 token 级 | `rollout_corr_helper.py` | ⭐⭐中 | 当前仅支持序列级，需扩展 |
| 传递 rollout log_prob | `dp_actor.py` + batch 数据流 | ⭐⭐中 | 需确保 `rollout_log_prob` 独立于 `old_log_prob` |
| 去掉长度归一化 | 配置修改 | ⭐低 | 改 `loss_agg_mode` 或在新损失函数中处理 |

### 6.3 关键前置条件

1. **`rollout_log_prob` 和 `old_log_prob` 必须分离**：
   - `rollout_log_prob` = vLLM 推理引擎产出的 log prob（μ_{θ_old}）
   - `old_log_prob` = FSDP 训练引擎前向传播的 log prob（π_{θ_old}）
   - 当前代码已经有这两个值（rollout batch 里的 `old_log_probs` 来自 vLLM，actor 前向重计算的是 `log_prob`）
   - 需确认数据流中这两者的命名和传递是否正确

2. **聚合方式**：MiniRL 不做长度归一化，需要 `seq-mean-token-sum` 或自定义聚合

### 6.4 风险评估

| 风险 | 等级 | 说明 |
|------|------|------|
| 训练稳定性 | 中 | MiniRL 在 Qwen 系列上验证过，但联合模型的 logit 融合可能引入额外方差 |
| 零优势问题 | 中 | MiniRL 不直接解决此问题，需结合其他手段 |
| 调参成本 | 低 | 论文提供了详细超参（ε_high=0.27, ε_low=0.2, TIS=5） |
| 代码侵入性 | 低 | 新增损失函数不影响现有代码 |

### 6.5 总体评估

| 维度 | 评分 |
|------|------|
| 实现难度 | **低-中**（2-3 天工作量） |
| 理论收益 | **中-高**（更稳定的 IS 修正、更干净的梯度） |
| 对零优势问题的帮助 | **低**（不直接解决，需结合 RLOO 或更大 rollout_n） |
| 推荐优先级 | **建议先试 Dr.GRPO**（一行配置改动），再考虑完整 MiniRL |

---

## 7. 建议的实验路径

### 第一步（零成本，立即可试）

```bash
# 改为 Dr.GRPO：去掉 std 归一化
+algorithm.norm_adv_by_std_in_grpo=False
```

这消除了 σ=0 时的数值问题，并与 MiniRL 的优势估计对齐。

### 第二步（低成本）

尝试 RLOO 优势估计（leave-one-out baseline），verl 已有实现：

```bash
algorithm.adv_estimator=rloo
```

RLOO 的 baseline 是组内**其他**样本的均值，对全同奖励的情况同样产生零优势，但在部分分化时比 GRPO 有更低方差的估计。

### 第三步（中等成本）

实现完整 MiniRL 损失函数：
1. 在 `core_algos.py` 注册 `"minirl"` 损失
2. 将 IS 修正从序列级改为 token 级
3. 去掉长度归一化
4. 使用论文推荐的超参

### 第四步（探索性）

如果零优势问题仍然严重，考虑：
- 实现全局 EMA baseline（跨 batch 的移动平均）
- 在零优势 batch 上跳过 optimizer step
- 探索 `optimal_token_baseline`（verl 已有实现，per-token baseline）

---

## 附录 A：论文关键发现摘要

1. **Token 级 IS 修正是必须的**：去掉它会导致训练崩溃和熵急剧下降
2. **长度归一化有害**：破坏一阶近似的理论保证
3. **二值裁剪掩码优于 PPO min-clip**：更简洁，效果一致
4. **IS 权重应在 stop-gradient 内**：防止不稳定梯度项
5. **On-policy 训练最稳定**：Off-policy 程度越大，裁剪越关键
6. **裁剪阈值可以不对称**：正方向 ε_high=0.27，负方向 ε_low=0.2

## 附录 B：MiniRL 与 PPO/GRPO/REINFORCE 的谱系关系

```
REINFORCE (无 baseline, 无 IS)
  └─ + group baseline → GRPO
  └─ + IS weight      → GPG + IS (≈ MiniRL 无裁剪)
  └─ + IS + clip mask → MiniRL

PPO (ratio-based, gradient through ratio)
  └─ + stop-grad ratio → CISPO (verl 已有)
  └─ + IS decomposition → MiniRL 方向
```

MiniRL 本质是 **REINFORCE + token 级 IS 修正 + 二值裁剪掩码**，比 PPO 更简单，但通过 IS 修正获得了 off-policy 能力。
