# 零 Advantage 批次问题

**状态**：调研中（2026-03-15 记录）
**来源**：4B MATH 训练日志分析（`Joint-GRPO-Qwen3-4B-RolloutCorr-MATH_1773505000.log`）

## 问题描述

在 GRPO + 二值奖励的联合训练中，频繁出现整个 batch 的 advantage 全部为零的情况，导致该训练步的梯度为零，更新完全浪费。

## 观测数据

### 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | QwenJoint-4B |
| 数据集 | MATH (7500 samples) |
| train_prompt_bsz | 8 |
| rollout_n | 4 |
| 每步总响应数 | 32 (8 × 4) |
| norm_adv_by_std_in_grpo | True |
| 总训练步数 | 200 |

### 零梯度步的完整指标

| Step | score/mean | advantages max/mean/min | pg_loss | actor/grad_norm |
|------|-----------|------------------------|---------|-----------------|
| 80   | 0.75      | 0.0 / 0.0 / 0.0       | 0.0     | 0.0             |
| 95   | 0.50      | 0.0 / 0.0 / 0.0       | 0.0     | 0.0             |
| 120  | 0.75      | 0.0 / 0.0 / 0.0       | 0.0     | 0.0             |
| 125  | 0.25      | 0.0 / 0.0 / 0.0       | 0.0     | 0.0             |
| 155  | 0.75      | 0.0 / 0.0 / 0.0       | 0.0     | 0.0             |
| 195  | 0.50      | 0.0 / 0.0 / 0.0       | 0.0     | 0.0             |

pg_loss 和 grad_norm 为**精确的 0.0**，非近似值。

### 对比：相同 batch 均分但梯度非零的步

步 165 和 180 的 `score/mean = 0.75`（与步 80/120/155 相同），但 advantages 非零（max=1.5, min=-1.5），梯度也非零。区别在于 8 个 prompt 组的**组内分布不同**——至少有一个组存在对/错混合。

### 发生频率

6 / 40 个训练步 = **15%** 的训练步完全浪费。

## 根因分析

### 机制

`core_algos.py:314-327` 中 GRPO 的 advantage 计算：

```python
advantage = (score - group_mean) / (group_std + 1e-6)
```

当一个 prompt 的 `rollout_n=4` 条响应全对或全错时：
- `group_std = 0` → `score - group_mean = 0` → `advantage = 0`

如果 batch 中所有 8 个 prompt 组都恰好全对或全错，则整个 batch 的 advantage 全为零。

### 为什么 score/mean 非 0 或 1 也会出现零梯度

以 `score/mean = 0.75` 为例（24/32 正确）：
- 若 6 个组全对（4/4）+ 2 个组全错（0/4） → 每个组内无方差 → 所有 advantage = 0
- 对比：5 个组全对 + 1 个组 3/4 + 2 个组全错 → 混合组有方差 → 部分 advantage ≠ 0

### 加剧因素

1. **rollout_n 太小（4）**：每组仅 4 次采样，全对/全错的概率高
2. **二值奖励**：正确 +1 / 错误 -1，无中间值，无法产生组内方差
3. **数据集难度极端**：MATH 中极难/极简的题目占比高，4 次采样容易全同

### 副作用

零梯度步不仅浪费算力（forward + backward + optimizer step），还会**污染 Adam 优化器状态**：
- 零梯度使一阶动量 `m` 衰减向零
- 二阶动量 `v` 同样衰减
- 后续非零梯度步的有效学习率因此被扭曲

## 已采取措施

### 2026-03-15：增大 batch size 和 rollout_n

脚本 `run_joint_grpo_qwen3_4b_rollout_corr.sh` 已修改：

| 参数 | 原值 | 新值 | 理由 |
|------|------|------|------|
| `train_prompt_bsz` | 8 | **16** | 更多 prompt 组，降低全组均匀的概率 |
| `rollout_n` | 4 | **8** | 每组 8 次采样，全对/全错概率大幅下降 |
| `ppo_mini_batch_size` | 2 | **4** | 匹配更大的 batch |
| `RUN_PREFIX` | `...-MATH` | `...-MATH-bsz16` | 新 checkpoint 目录，从头训练 |

每步总响应数从 32 (8×4) → 128 (16×8)。

显存可行性依据（8×H800 80GB）：
- 原配置峰值 25.2 GB / 80 GB（含 CPU offload），余量充足
- `rollout_n` 增加不影响 vLLM 峰值显存（micro_batch=1 逐个生成）
- `ppo_mini_batch_size` 不变时训练侧峰值也不变

**状态**：⏳ 等待训练运行结果验证

### 待验证

1. 新配置下零 advantage 步的频率是否显著下降（预期从 15% 降至接近 0%）
2. 是否出现 OOM
3. 训练效果（acc/mean@1 趋势）是否改善

## 待调研（如当前措施不足）

1. 是否应跳过零 advantage batch 的优化器更新
2. 是否存在替代的 advantage 归一化策略
3. 该问题是否为联合训练特有，还是标准 GRPO 训练同样存在
