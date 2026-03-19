# 联合训练进展汇报

**日期**：2026-03-16
**阶段**：Stage 2（算法正确性验证）

---

## 1. 框架选型

调研了两个可行的 RL 框架：

| 框架 | 训练后端 | Rollout 引擎 | 状态 |
|------|----------|-------------|------|
| **verl** | FSDP (ZeRO-3) | vLLM | **已选用** |
| slime | Megatron | SGLang | 暂不使用 |

选择 verl 的原因：

<!-- TODO: 补充选型理由 -->

---

## 2. 已完成的工作

### 2.1 Infra：联合训练的 logit 融合

核心设计是在输出层进行 logit 线性组合：

$$
\text{logits}_{\text{fused}} = (1 - \lambda) \cdot \text{logits}_{\text{model}_1} + \lambda \cdot \text{logits}_{\text{model}_2}
$$

其中 $\text{model}_1$ 为锚定模型（frozen），$\text{model}_2$ 为可训练模型。两个子模型独立前向传播，仅在 logit 层融合，没有 hidden states 的交互。融合后的 logits 用于计算 RL 损失并反向传播——因为 $\text{model}_1$ 的参数被冻结，梯度只更新 $\text{model}_2$。

实现要点：
- 模型类 `QwenJointForCausalLM`：支持 HuggingFace `from_pretrained` 加载，兼容 FSDP 分片
- vLLM rollout 集成：训练时用融合 logits 采样，评估时提取 model2 独立推理
- 已完成 25 项语义测试（融合计算、old-log-prob 重算、权重提取等）

### 2.2 损失函数：从 GRPO 切换到 MiniRL

最初用标准 GRPO（vanilla 模式）跑联合训练，在 GSM8K 上 0% 准确率（EXP-01），没有学到任何东西。切换到 **MiniRL**（[arXiv:2512.01374](https://arxiv.org/abs/2512.01374)）后，训练开始收敛。

#### MiniRL 算法简介

MiniRL 的核心改动是用**二值裁剪掩码**替代 PPO/GRPO 的 min-clip 机制。标准 GRPO 的 policy loss：

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(r_t \hat{A}_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比率。这个公式通过 `min` 操作同时对 ratio 和 advantage 做限制。

MiniRL 将其简化为：

$$
\mathcal{L}_{\text{MiniRL}} = -\mathbb{E}\left[m_t \cdot \hat{A}_t \cdot \log \pi_\theta(a_t|s_t)\right]
$$

其中 $m_t$ 是二值掩码：

$$
m_t = \begin{cases}
0, & \text{if } \hat{A}_t > 0 \text{ and } r_t > 1 + \epsilon_{\text{high}} \\
0, & \text{if } \hat{A}_t < 0 \text{ and } r_t < 1 - \epsilon_{\text{low}} \\
1, & \text{otherwise}
\end{cases}
$$

**与 GRPO 的关键区别**：

| 特性 | GRPO (vanilla) | MiniRL |
|------|---------------|--------|
| 裁剪方式 | `min(ratio * adv, clip(ratio) * adv)` | 二值掩码 $m_t \in \{0, 1\}$ |
| 梯度通路 | 通过 ratio 和 log\_prob | 仅通过 $\log \pi_\theta$（REINFORCE 风格） |
| 损失聚合 | token-mean（除以 token 数） | seq-mean-token-sum（不做 token 级归一化） |
| Advantage | 可以 attach 梯度 | 必须 `.detach()`，作为常数 |

MiniRL 本质上回到了 REINFORCE 的梯度形式，但加了一个基于 policy staleness 的二值门控。它的理论优势是在 `seq-mean-token-sum` 聚合下保持了梯度估计的无偏性。

#### Advantage 计算：Dr.GRPO

我们使用 Dr.GRPO 变体来计算 advantage。标准 GRPO 对每个 prompt 的 $G$ 条响应做归一化：

$$
\hat{A}_i^{\text{GRPO}} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}
$$

Dr.GRPO 去掉了标准差归一化，直接用均值中心化：

$$
\hat{A}_i^{\text{Dr.GRPO}} = r_i - \text{mean}(\mathbf{r})
$$

其中 $r_i \in \{-1, +1\}$ 是二值奖励。去掉标准差的动机是：当一组响应全对或全错时，$\text{std}(\mathbf{r}) = 0$，标准 GRPO 的归一化会产生数值不稳定；而且对于二值奖励，除以标准差不提供额外信息，反而引入不必要的缩放。

### 2.3 实验结果

在 H800 服务器（8 GPU）上用 Qwen3-1.7B-Base 训练，速度很快（100 步约 1.5 小时）。共跑了 6 组实验：

#### 实验总览

| 实验 | 模型 | 算法 | 联合训练 | MATH-500 最佳 | 状态 |
|------|------|------|---------|--------------|------|
| EXP-01 | 1.7B | GRPO | 是 ($\lambda$=0.5) | 0% | 失败 |
| EXP-02 | 4B | GRPO+IS | 是 ($\lambda$=0.5) | N/A（AIME 23.1%） | 完成 |
| EXP-03 | 4B | GRPO+IS | 是 ($\lambda$=0.5) | N/A（AIME 11.5%） | 中断 |
| EXP-04 | 1.7B | MiniRL | 是 ($\lambda$=0.55) | **63.0%** | 完成 |
| EXP-05 | 1.7B | MiniRL | **否** | **64.0%** | 完成 |
| EXP-06 | 1.7B | MiniRL | **否** | 待运行 | 计划中 |

#### 离线评估对比（EVAL-01 vs EVAL-02）

用 vLLM 离线推理，对 EXP-04（联合训练）和 EXP-05（单模型 baseline）在 5 个数学基准上评测（$n=3$，$T=1.0$）：

| Benchmark | 样本数 | 联合训练 mean@3 | Baseline mean@3 | 差值 |
|-----------|-------|----------------|-----------------|------|
| **MATH-500** | 500 | **64.2%** | 61.4% | **+2.8%** |
| AIME-2025 | 30 | 4.4% | 5.6% | -1.2% |
| AMC-2023 | 40 | 36.7% | 41.7% | -5.0% |
| MinervaMAth | 272 | 24.4% | 27.6% | -3.2% |
| OlympiadBench | 674 | 28.3% | 28.1% | +0.2% |

**观察**：联合训练在训练域（MATH-500）上优于 baseline（+2.8%），但在 OOD 基准上整体弱于 baseline。不过这个对比**不公平**——baseline 存在严重的梯度裁剪问题（见下文），实际上没有充分训练。

### 2.4 发现的关键问题：梯度裁剪瓶颈

在分析 EXP-05（baseline）的训练日志时，发现了一个严重的超参数不匹配：

**`grad_clip=1.0` 对 MiniRL 来说太激进了。**

MiniRL 使用 `seq-mean-token-sum` 聚合（token 级损失求和，不除以序列长度），导致梯度范数天然比 GRPO 的 `token-mean` 大 100-500 倍：

| 聚合模式 | 典型 grad\_norm | `grad_clip=1.0` 下的裁剪率 | 有效学习率 |
|---------|----------------|---------------------------|-----------|
| token-mean (GRPO) | 0.5 - 5 | ~20-50%（只裁极端值） | $\approx 10^{-6}$ |
| seq-mean-token-sum (MiniRL) | 231 - 515 | **100%（每步都裁）** | $\approx 2.8 \times 10^{-9}$ |

这意味着 MiniRL 的有效学习率被压缩了约 300 倍：

$$
\text{lr}_{\text{effective}} = \text{lr} \times \frac{\text{grad\_clip}}{\|\nabla\|} = 10^{-6} \times \frac{1.0}{368} \approx 2.8 \times 10^{-9}
$$

这解释了 MATH-500 准确率在 step 40 之后就进入平台期（~60%），后续 125 步只涨了约 4 个百分点。

**修复方案**：将 `grad_clip` 从 1.0 提高到 500.0。在已观测到的 33 个 step 中，只有 1 个 step 的 grad\_norm 超过 500（step 5 的 515.5），其余都在 500 以内。EXP-06 就是用 `grad_clip=500.0` 重跑的实验，目前计划中。

---

## 3. 后续计划

### 3.1 验证联合训练 vs 单模型

当前的对比（EXP-04 vs EXP-05）因为 baseline 的梯度裁剪问题，结论不可靠。需要等 EXP-06（`grad_clip=500.0` 的 baseline）跑完后，才能做公平对比。

核心问题：**logit 融合在 RL 中是否比单模型更有效？**

### 3.2 梯度分析

在跑实验的间隙，准备推导联合训练的梯度公式，验证其正确性。对于融合后的 policy $\pi_{\text{fused}}$，其 log-probability 为：

$$
\log \pi_{\text{fused}}(a_t | s_t) = \log \text{softmax}\left((1-\lambda) \cdot z_1 + \lambda \cdot z_2\right)_{a_t}
$$

梯度关于 $\text{model}_2$ 参数 $\theta_2$ 的表达式：

$$
\frac{\partial \log \pi_{\text{fused}}}{\partial \theta_2} = \lambda \left(\mathbf{e}_{a_t} - \pi_{\text{fused}}(\cdot|s_t)\right) \frac{\partial z_2}{\partial \theta_2}
$$

需要确认：这个梯度方向是否合理？$\lambda$ 的取值如何影响训练动态？

### 3.3 探索更多 weak-driven 方法

联合训练本质上是一种 **weak-driven learning**：让弱模型（$\text{model}_1$）在 logit 层提供参考信号，放大强模型（$\text{model}_2$）的学习效果。

这个思路可以推广到其他层次：

| 层次 | 方法 | 弱模型的角色 |
|------|------|-------------|
| **Logit 层** | 当前方案（logit 融合） | 直接混合输出分布 |
| **Reward / Sequence 层** | 弱模型作为 reward signal 或 baseline | 提供有指向性的优势估计 |
| **Tool 层** | 弱模型参与工具调用 / 验证 | 在推理管道中提供辅助判断 |

核心原理是：弱模型提供的是**有结构的、有指向性的**梯度放大信号，而不是随机噪声。这种信号之所以有效，是因为即使弱模型本身不够强，它的知识分布与强模型是相关的——它们来自同一个预训练家族，共享底层的语言和推理模式。

---

## 4. 经验总结

1. **GRPO 在联合训练中不 work**：EXP-01 跑了 100 步，GSM8K 0% 准确率。原因可能是 GRPO 的 ratio-based 裁剪在融合 logits 下表现不稳定。
2. **MiniRL 比 GRPO 更适合联合训练**：切换到 MiniRL 后立即收敛。MiniRL 的 REINFORCE 风格梯度（只通过 $\log \pi$）可能在 logit 融合场景下更稳定。
3. **超参数不能直接迁移**：MiniRL 的 `seq-mean-token-sum` 聚合使梯度范数比 GRPO 大 100-500 倍，必须相应调整 `grad_clip`。这是一个容易被忽略但影响巨大的问题。
4. **离线评估很重要**：训练中的 validation（$n=1$）噪声大，尤其是在 AIME-2025 这样只有 30 个样本的基准上。搭建了离线 vLLM 推理 + pass@k/maj@k 评估管道后，结果更可靠。
5. **checkpoint 管理是刚需**：6 组实验消耗约 665 GB checkpoint 空间，必须有系统化的实验索引和清理策略。
