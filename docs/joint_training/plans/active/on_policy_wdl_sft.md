# On-Policy Weak-Driven SFT（仅正向）

- 状态：**进行中 — LR 搜参阶段**
- 分支：`feature/on-policy-wdl-sft`
- 创建日期：2026-04-06
- 更新日期：2026-04-16
- 前置分支：`feature/joint-training`（Stage 1 & 2 完成）

> **重要变更 (2026-04-16)**: 反向 SFT（β>0）已永久放弃。M5 (lr=1e-6) 和 M5.6 (lr=5e-7) 两次实验均因训练崩溃证实反向 SFT 本身是不稳定因素。后续所有工作均使用 β=0（仅正向 SFT）。当前基线为 M5.5 (lr=5e-7, β=0, 300 steps 完成)。下一步：学习率搜参，见 `lr_search.md`。

---

## 前置：标准 Weak-Driven SFT

### 设定

两个预训练语言模型：Weak model（参数 $\theta_w$）和 Strong model（参数 $\theta_s$），其中 Strong 是在 Weak 基础上进一步训练得到的，能力更强。给定带标签的 SFT 数据集 $\{(x, y^*)\}$，其中 $x$ 为 prompt，$y^* = (y^*_1, \ldots, y^*_T)$ 为 ground truth response。

### Logit Fusion

两个模型对同一输入独立做前向传播，各自输出 logits $z_w, z_s \in \mathbb{R}^{|V|}$（$|V|$ 为词表大小）。在 softmax 之前做线性融合：

$$z_{\text{mix}} = (1-\lambda)\, z_w + \lambda\, z_s, \quad \lambda \in [0, 1]$$

融合后的预测概率分布为 $P_{\text{mix}} = \text{Softmax}(z_{\text{mix}})$。

两个模型**不共享隐藏状态**，仅在 logit 层交互。$\lambda$ 控制 Strong model 的贡献权重。

### 损失函数

标准交叉熵，label 为 ground truth 序列 $y^*$（teacher-forcing）：

$$\mathcal{L}_{\text{WD-SFT}} = -\sum_{t=1}^{T} \log P_{\text{mix}}(y^*_t \mid x, y^*_{<t})$$

### 梯度流

损失对融合 logits 的梯度为标准残差：

$$g = \frac{\partial \mathcal{L}}{\partial z_{\text{mix}}} = P_{\text{mix}} - e_{y^*_t}$$

由融合公式的线性性，梯度按比例分配到两个子模型：

$$\nabla_{\theta_w} \mathcal{L} = (1-\lambda)\, J_w^T g, \quad \nabla_{\theta_s} \mathcal{L} = \lambda\, J_s^T g$$

其中 $J_i = \partial z_i / \partial \theta_i$ 为各模型的 Jacobian。两个模型接收**同方向、不同幅度**的梯度——Strong model 能力更强（NTK 更大），在相同方向梯度下更新更快，但 $\lambda$ 较大时其梯度幅度也更大，这两个因素的交互决定了训练动力学（详见 `sft_analysis.tex` 的三阶段分析）。

### 核心特点

- **Off-policy**：训练数据（ground truth）与模型当前分布无关，是固定数据集
- **Teacher-forcing**：训练时 context 是 ground truth 前缀 $y^*_{<t}$，非模型自回归生成
- **双模型联合更新**：一次 loss 同时驱动两个模型的参数更新

---

## 概述

在上述标准 Weak-Driven SFT 的基础上，引入 on-policy rollout：用 fused model 自回归生成 response，正确的做正向 WD-SFT（用 rollout 作为 label），错误的做反向 WD-SFT（最小化其概率）。数据由模型自己实时生成，训练信号来自 rollout 结果而非固定数据集的 ground truth label。

## 流程

```
对每个训练 prompt x：

1. Fused Rollout
   ┌─────────────────────────────────────────────┐
   │  Weak model, Strong model 分别输出 logits    │
   │  z_mix = (1-λ) z_weak + λ z_strong          │
   │  从 P_mix = Softmax(z_mix) 自回归采样 N 条    │
   │  response: {y¹, y², ..., yᴺ}                │
   └─────────────────────────────────────────────┘
                        ↓
2. Reward 判定
   ┌─────────────────────────────────────────────┐
   │  对每条 yⁱ，用 reward function 判断正误        │
   │  R(x, yⁱ) ∈ {0, 1}（如：答案与 gt 匹配）      │
   │  正确集 C = {i : R=1}，错误集 I = {i : R=0}   │
   └─────────────────────────────────────────────┘
                        ↓
3. Weak-Driven SFT 训练（teacher-forcing）
   ┌─────────────────────────────────────────────┐
   │  对 yⁱ ∈ C：正向 WD-SFT（用 yⁱ 作为 label）   │
   │  对 yʲ ∈ I：反向 WD-SFT（最小化 yʲ 的概率）    │
   │  两个模型通过 fused logits 接收梯度并更新参数     │
   └─────────────────────────────────────────────┘
```

注意：步骤 3 中的 teacher-forcing 仍需 Weak 和 Strong 两个模型做一次 forward pass 来计算 fused logits 下的交叉熵，这与 rollout 阶段是**两次独立的前向过程**。

## 损失函数

给定 prompt $x$，fused rollout 得到 $N$ 条 response $\{y^1, \ldots, y^N\}$，其中 $k = |\mathcal{C}|$ 条正确。

$$\mathcal{L}(x) = \underbrace{-\frac{1}{k}\sum_{i \in \mathcal{C}} \sum_{t=1}^{T_i} \log P_{\text{mix}}(y^i_t \mid x, y^i_{<t})}_{\mathcal{L}^+:\text{ 正向 WD-SFT，label = 正确 rollout}} + \; \beta \cdot \underbrace{\frac{1}{N-k}\sum_{j \in \mathcal{I}} \sum_{t=1}^{T_j} \log P_{\text{mix}}(y^j_t \mid x, y^j_{<t})}_{\mathcal{L}^-:\text{ 反向 WD-SFT，推离错误 rollout}}$$

- $P_{\text{mix}}(y_t \mid \cdot) = \text{Softmax}\big((1-\lambda)z_{\text{weak}} + \lambda z_{\text{strong}}\big)_{y_t}$
- $\beta \geq 0$：反向 SFT 的权重系数（需要调参或按 $k/(N-k)$ 动态设定）
- 当 $\mathcal{C} = \emptyset$ 时跳过该 prompt（或 fallback 到 ground truth 做标准 WD-SFT）
- 当 $\mathcal{I} = \emptyset$ 时 $\mathcal{L}^- = 0$，仅保留正向项

梯度传播到子模型的方式与标准 WD-SFT 一致：

$$\nabla_{\theta_{\text{weak}}} \mathcal{L} = (1-\lambda) \cdot g, \quad \nabla_{\theta_{\text{strong}}} \mathcal{L} = \lambda \cdot g$$

## On-Policy 提供的优势

| 优势                | 说明                                                                       |
| ----------------- | ------------------------------------------------------------------------ |
| **自适应训练信号**       | Label 来自模型当前能力下的成功路径，推理风格与模型匹配，不受 ground truth 解题风格限制                    |
| **隐式 curriculum** | 难题（$k$ 小）自然产生更多负样本梯度，简单题（$k$ 大）产生更多正样本梯度，训练精力自动分配                        |
| **全正确时不浪费**       | $k = N$ 时仍有正向 SFT 梯度（MiniRL/GRPO 在此情况下 advantage = 0，梯度为零）               |
| **与 WD-SFT 理论兼容** | 梯度流结构与 `sft_analysis.tex` 的分析一致，logit fusion 的 gradient shielding 等性质仍成立 |

## 代价与风险

| 代价 / 风险       | 说明                                                                                                                                           |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **计算开销大**     | 每个 prompt 需要 (1) fused auto-regressive rollout $N$ 条 + (2) teacher-forcing forward-backward 1 次。Rollout 是主要瓶颈，粗略估计总开销为标准 WD-SFT 的 **5-10 倍** |
| **负 SFT 不稳定** | $\mathcal{L}^-$ 将概率推离观测 token 但不指定去向，可能导致分布坍塌或熵爆炸。需要 $\beta$ 调参或引入 mask 机制（如屏蔽低概率 token 的梯度）                                                 |
| **正负不平衡**     | 当正确率低（$k=1, N=8$）时，负样本数量是正样本的 7 倍。$\beta$ 的设定需要平衡这一点                                                                                         |
| **强化错误推理的风险** | 正确 rollout 可能答案对但中间推导有误（碰巧正确），正向 SFT 会强化这种错误推理                                                                                               |
| **架构要求**      | 需要推理（rollout）和训练（SFT）交替进行，最好支持推训分离的框架（如 vLLM 推理 + FSDP 训练）                                                                                   |

## 与 MiniRL 的关系

本方案在结构上接近**去掉 IS 修正和 binary mask 的 MiniRL**（见第 3.6 节分析，这里没有）。差异在于：
(1) 不做 advantage 加权（正负样本等权，由 $\beta$ 控制比例）；
(2) 通过 logit fusion 联合训练两个模型。如果逐步添加 advantage 加权 → IS 修正 → binary mask，本方案会逐步退化为标准 MiniRL + logit fusion。因此本方案的独特价值在于它**保持 SFT 的简洁性**，同时通过 on-policy rollout 获得自适应数据。

---

## 实现计划

### Phase 0: 基础设施准备 — DONE

- [x] 在 `core_algos.py` 中注册新的 loss function `wdl_sft`
- [x] 实现正向 WD-SFT loss（fused logits + teacher-forcing on correct rollouts）
- [x] 实现反向 WD-SFT loss（minimize probability of incorrect rollouts）
- [x] 添加 $\beta$ 系数配置
- [x] 单元测试覆盖

### Phase 1: 训练循环集成 — DONE

- [x] 修改 `ray_trainer.py` 中的训练循环以支持 on-policy SFT 模式
- [x] 复用现有 vLLM rollout 基础设施进行 fused rollout
- [x] 复用现有 reward computation 判定正误
- [x] 将 rollout 结果按正确/错误分组后送入 SFT loss
- [x] 处理边界情况：$\mathcal{C} = \emptyset$（全错）和 $\mathcal{I} = \emptyset$（全对）

### Phase 2: 训练验证 — DONE（结论：仅正向可行）

- [x] M5: 双向训练（β=0.1, lr=1e-6）— 训练不稳定，~1000 步后崩溃
- [x] M5.5: 仅正向训练（β=0, lr=5e-7）— **稳定完成 300 步**，12 个 checkpoint
- [x] M5.6: 重新测试双向（β=0.1, lr=5e-7）— ~236 步崩溃，确认反向 SFT 不稳定
- [x] **结论**：反向 SFT 永久放弃，后续仅使用 β=0

### Phase 3: 学习率搜参 — IN PROGRESS

详见 `lr_search.md`。

- [ ] LR grid search: 1e-7, 2e-7, (5e-7 已完成), 1e-6, 2e-6, 5e-6
- [ ] 对比各 LR 在 MATH-500 和 AIME-2025 上的验证精度
- [ ] 选定最佳 LR

### Phase 4: 最终训练与评估

- [ ] 使用最佳 LR 进行训练（~300 步，足够收敛）
- [ ] 综合评估（/vllm-eval）
- [ ] 与 baseline（weak model、strong model、static WD-SFT）对比
