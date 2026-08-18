# On-Policy Weak-Driven Learning 为什么有效？

## 从实验异象、梯度放大到 Product-of-Experts 耦合与可证伪机制实验

- 文档类型：机制调研与实验设计沉淀
- 创建日期：2026-08-16
- 适用范围：Qwen3-1.7B、Math-first、`beta=0` 的 On-Policy WDL-SFT 主实验
- 当前状态：理论框架与第一批实验合同草案完成；fixed-Model1 代码与配置已准备但尚无正式运行凭据；Dynamic Permutation 及后续机制控制仍未实现或启动
- 相关方法论文：[Weak-Driven Learning: How Weak Agents Make Strong Agents Stronger](https://arxiv.org/abs/2602.08222)
- 原始同熵实验草案：[动态同熵 Weak-Structure Ablation](https://ocnwds5io8yp.feishu.cn/docx/NEIvdnwU0o0vszxi2wycfcTHnjd)
- 审稿记录：[OpenReview](https://openreview.net/forum?id=WAqz1qihuI)

---

## 0. 这份文档试图回答什么

我们已经看到一批初步有效性信号：在当前 Qwen3-1.7B 数学任务合同、单一训练 seed、online `n=3` 条件下，continuous On-Policy WDL-SFT 相比 matched-scale no-weak control 和若干现有路线显示出更早增长与更高点估计。但这些结果仍是用于触发机制研究的 local diagnostic，尚未构成可直接发表的方法有效性结论。

但是，“方法有效”和“我们知道它为什么有效”是两件不同的事情。

如果只停留在现有效果上，我们最多能够说：加入 Weak Model 1（下文简称 WM1）后，整个训练 procedure 的结果变好了。我们还不能说：

1. WM1 是不是必须持续训练；
2. WM1 提供的是不是 token-specific 的语义结构；
3. 效果是不是只来自 entropy、target confidence 或梯度尺度；
4. WM1 能不能被一个更简单、无需第二个在线模型的机制替代；
5. Model1 和 Model2 同时变好是不是某种隐式蒸馏；
6. 同样的解释能不能统一我们在 On-Policy SFT/SFD 和 OPD 上看到的结果。

因此，这份文档不是一份“WDL 已经被解释完毕”的总结，而是一份发现过程的完整记录。它要说明：我们先看到了什么，为什么现有解释不够，新的理论语言是怎样出现的，每一项实验为什么必须在前一项实验之后进行，以及不同结果将如何改变论文叙事。

这条发现路径可以概括为四步：

1. 先看到 C 相对标准 On-Policy SFT（A）的完整方法增益，再用 D0 拆出 matched-scale weak-logit 增量，同时观察到 Model1 与 Model2 都在提升；
2. 再发现“梯度放大”和“隐式蒸馏”都只能解释一部分现象；
3. 于是回到 fused logits 本身，推导出 PoE/Chernoff identity；
4. 最后把理论变量转化为 fixed-M1、Dynamic Permutation、geometry controls 和 no-online-WM1 controller 等可证伪实验。

### 0.1 当前证据与主张层级

| 层级 | 当前可以说 | 当前不能说 |
| --- | --- | --- |
| 已观察 | C-A 与 C-D0 在单 seed、online `n=3` 下都出现初步正向差距；C 中两模型都提升 | 已经得到 publication-level efficacy 或 causality |
| 已推导 | fused logits 严格对应 geometric PoE；成功轨迹监督项具有 verifier-gated hard self-training 解释 | PoE 保证 endpoint 更好；Model1 是标准 teacher |
| 工作假设 | WM1 通过 fused distribution 改变局部梯度几何，并与 Model2 共同适应 | scalar、rank、token semantics 和 trajectory 各自贡献已经确定 |
| 待验证 | fixed-M1、DynPerm、geometry matching、controller、reverse 和 same-rollout 2x2 | 机制已经被证实；online WM1 已可去掉 |

### 0.2 本文使用的实验缩写

| 缩写 | 含义 |
| --- | --- |
| CS0 | cold-start Model2 起点 |
| S1 | Stage1 训练后的 Model2 起点 |
| A | 普通 correctness-filtered On-Policy SFT continuation |
| B | WDL20 后转 Model2-only Stage3 40 steps 的 staged baseline |
| C | continuous WDL，训练 logits 为 `0.2 z1 + 0.8 z2` |
| D0 | matched-scale no-weak control，训练 logits 为 `0.8 z2` |
| P60 | post-Stage1 训练到第 60 step；不是 pass@60 |
| Math-7 | 当前七个数学 validation datasets 的汇总合同 |
| online `n=3` | 训练过程中每题采样 3 个回答的诊断性验证 |
| frozen `n=8/n=256` | 固定 checkpoint、decoder 与 evaluator 的离线确认评测 |

全文的核心工作假设是：

> On-Policy WDL 在数据层进行 verifier-gated、reward-projected 的 hard self-training；在 logit 层使用一个可训练的几何乘积耦合。WM1 不是一个需要 Model2 模仿的显式 teacher，而是通过 fused objective 改变当前监督目标的局部梯度几何；当两条分支都可训练时，它们又会在同一批 verifier-selected trajectories 上共同适应。

这只是当前最有解释力的工作假设，不是已经被实验证实的最终结论。后文的所有实验都是为了让它能够被证伪，而不是为了事后配合已有结果。

---

## 1. 故事的起点：我们先确认了“能学”，却没有确认“为什么学”

### 1.1 早期 Stage123 实验解决的是链路问题

早期的 Stage123 矩阵完成了 16 条 authoritative runs。它确认了 CoT-v3 数据、严格格式合同、Stage1/Stage2/Stage3 训练链路和双模型基础设施确实能够工作。多个 Stage3 Model2 分支在后半程都有明显增长，说明模型不是因为 reward、格式、loss mask 或训练脚本错误而“假学习”。

但这一批结果不能作为干净的 WDL 机制证据。原因是 matched Stage1 continuation control 在相近 effective steps 也出现了明显跃升；而 Stage2、Stage3 与 control 同时改变了训练目标、模型形态、optimizer state 和数据窗口。它们证明了训练链路具有学习能力，却没有把 WDL treatment 从“继续训练本身”中隔离出来。

这迫使我们把问题重新收窄：不再问“Stage123 是否整体有效”，而是问：在同一个 Model2 起点、同一批 post-Stage1 prompts、同样的 60-step 预算和同样的 Model2-only rollout 机制下，仅仅让 weak logits 进入 fused training objective，是否仍然带来增益？这里的“机制相同”不表示两条 run 共享实际 trajectories；首个不同 update 之后，两个 Model2 proposer 及其后续采样就会分叉。

### 1.2 A/C 建立方法增益，C/D0 再缩小机制入口

strict scorer 修复后，我们重新构造了 post-Stage1 的 A/B/C/D0 对比：

| Arm | 训练方式 | P60 Math-7 mean@3 | Exact pass@3 | 在故事中的角色 |
| --- | --- | ---: | ---: | --- |
| A | Standard On-Policy SFT continuation | 66.352% | 74.621% | 最直接的 practical selected-SFT baseline |
| B | WDL20 后转为 Stage3 Model2-only 40 steps | 68.530% | 76.239% | legacy staged/allocation baseline |
| D0 | matched-scale no-weak，`0.8 z2` | 67.394% | 75.323% | 隔离 scale 与 weak-logit contribution 的内部 control |
| C | continuous WDL，`0.2 z1 + 0.8 z2` | **70.802%** | **77.407%** | 当前主方法候选 |

因此，在当前单一训练 seed、online `n=3` 条件下：

- `C - A = +4.450 pp mean@3`；
- `C - A = +2.786 pp pass@3`；
- `C - D0 = +3.408 pp mean@3`；
- `C - D0 = +2.084 pp pass@3`；
- 七个数学数据集的方向均为正；
- C 比 D0 更早进入明显增长区间，并在后续区间保持领先。

这两组差值不能互相替代。A 是单模型 correctness-filtered On-Policy SFT continuation，也是用户真正会采用的标准算法。`C-A` 回答论文首先必须回答的 practical question：**在相同 Model2 起点、rollout source、verifier、数据顺序和 post-Stage1 budget 下，加入 WM1 与 fused training 后，完整 WDL 方法是否优于标准 On-Policy SFT？** 因而，A 应当是外部结果表和方法叙事中的主 baseline。

D0 不是一个需要与外部工作并列的标准训练算法，而是本文内部构造的 matched-scale mechanism control。A 使用 $z_2$；D0 使用 $0.8z_2$；C 使用 $0.2z_1+0.8z_2$。因此，`A-D0` 主要暴露 strong-logit scale/temperature 的影响，`D0-C` 再问固定 Model2 coefficient 后加入 weak logits 会发生什么。D0 的作用是拆掉 `C-A` 中的一个混杂项，而不是把 A 从主比较中替换掉。

D0 与 C 共用 Stage1 Model2 起点、Model2-only rollout 机制、相同 prompt/data order 和 60-step budget，关键区别是 fused objective 中是否存在 `0.2z1`。但两者从首个不同 update 后仍会产生不同的 on-policy trajectories，因此 C-D0 隔离的是 weak-logit treatment package，不是固定 successful-rollout manifest 上的纯 objective effect。

但这仍然只是机制研究的起点，而不是机制证明。当前结果仍缺少 common frozen evaluation、paired bootstrap、第二个 training seed，以及 A/C/D0/GRPO 的共同 `n=256` pass@k 与 diversity 评测。因此，下文会把这些数值称为“触发机制猜想的事实”，而不是 publication-level efficacy conclusion。

### 1.3 一个比 C-D0 更意外的事实：两个模型都变好了

C 的双模型在线验证给出了一个无法被“牺牲弱模型”叙事轻易解释的结果：

| View | P0 Math-7 mean@3 | P60 Math-7 mean@3 | 增量 |
| --- | ---: | ---: | ---: |
| C Model1 | 39.02% | **71.04%** | +32.02 pp |
| C Model2 | 42.61% | 70.80% | +28.20 pp |
| D0 Model1（冻结） | 38.80% | 38.77% | -0.03 pp |
| D0 Model2 | 42.50% | 67.39% | +24.90 pp |

更关键的是，C 的 rollout source 是 Model2-only。Model1 并没有生成自己的成功轨迹，却从 39.02% 提升到了 71.04%。同时，Model1 format success 只增加约 5.38 pp，而正确率增加 32.02 pp，因此这一增长不能主要归结为格式修复。

这带来了第一个真正的机制问题：如果 Model1 不是 rollout proposer，也不是一个固定 teacher，它为什么会跟 Model2 一起变好？

---

## 2. 我们最初的解释：梯度放大

### 2.1 旧理论为什么自然产生

对 one-hot target token `y`，cross-entropy 对任意 non-target logit `k` 的梯度为：

$$
\frac{\partial \ell}{\partial z[k]}=p(k),\qquad k\neq y.
$$

也就是说，一个错误 token 在当前分布中概率越高，它受到的抑制梯度就越大。

早期 WDL 理论据此提出：如果 weak model 在某些 non-target tokens 上更加困惑，那么把 weak logits 混入 strong logits 后，这些 weak-revealed hard negatives 的概率会上升；cross-entropy 会把它们当成更严重的错误，从而对 strong model 施加更强的纠正。其直觉是：weak model 把 strong model 已经“看轻”的错误重新放大，使训练信号不至于在局部饱和区域过早消失。

设

$$
z_m=(1-\lambda)z_1+\lambda z_2,
$$

其中 `z1` 来自 WM1，`z2` 来自 Model2。在 weak margin 对所有 non-target tokens 都不大于 strong margin 的充分条件下，旧理论能够证明 fused target probability 不高于 strong target probability，因而总 non-target probability mass 与相应 logit-coordinate gradient 会增大。

这套解释有两个优点：它直观，而且确实指出了 logit mixing 与普通 SFT 的一个局部差异。

### 2.2 为什么它仍然不够

当我们把它用于当前 On-Policy WDL-SFT 时，出现了五个缺口。

第一，旧理论比较 fused model 与未缩放的 strong distribution，这恰好对应完整方法层面的 `C-A` 问题；但仅靠 C-A 无法区分 weak logits 与 strong-logit scale。为了进一步做机制归因，才需要加入 matched-scale D0。D0 的分布是：

$$
p_{D0}=\operatorname{softmax}(\lambda z_2),
$$

而 A 的分布是 $p_2=\operatorname{softmax}(z_2)$。所以，A 不能省略，因为它代表标准 On-Policy SFT；D0 也不能省略，因为没有它，温度/scale 与 weak contribution 会被混在一起。二者回答的是不同层级的问题。

第二，旧理论是 same-state、single-token、logit-coordinate 层面的充分条件。真正的参数梯度还需要乘模型 Jacobian，之后还会经过 gradient clipping、Adam moments 和 optimizer update。某个 logit coordinate 被放大，并不自动等于 parameter-gradient norm 被放大，更不等于多步训练后准确率更高。

第三，旧理论没有解释 Model1 为什么会变好。当前 C 中两个模型都可训练，并共同收到 fused residual 的梯度；把 WM1 简单描述成“被牺牲的噪声发生器”，与观测到的 Model1 大幅增长不一致。

第四，旧理论没有处理 on-policy 数据分布。Model2 的 rollout、verifier 的正确性筛选以及下一轮参数更新会共同改变后续成功轨迹分布；这不是固定离线 SFT 分析能够直接覆盖的。

第五，梯度放大理论容易落入事后解释：看到某个结果更好，就说它得到了“有益的放大”；如果结果不好，又可以说放大方向不对。除非理论能提前规定要测哪些量、如何反向操纵它们，以及哪些结果会否定解释，否则它还不能指导实验。

因此，我们并不否定梯度放大，而是把它降级为一个需要进一步解释的局部现象：它可能是机制的一部分，但不能独自承担全部叙事。

---

## 3. 第二个猜想：这里是否存在隐式蒸馏

### 3.1 为什么会产生这个猜想

Model2 生成 rollout，verifier 选择成功轨迹，之后 Model1 和 Model2 都在这些轨迹上接受 teacher-forced likelihood 训练。Model1 虽然没有负责采样，却在学习由 Model2 proposer 发现、并被 verifier 接受的解法。从数据流上看，这很像一种“强模型发现训练数据，弱模型也从中学习”的蒸馏过程。因此，最自然的问题是：WDL loss 能不能改写成 self-distillation？Model2 是否在隐式地教 Model1？

这个直觉需要拆成两个问题。第一个问题是，Model2 生成并经 verifier 筛选的轨迹，是否构成一种 self-training data。第二个问题是，Model1 与 Model2 之间是否存在标准蒸馏所要求的 teacher-distribution matching。前者可以严格成立，后者则不能直接从当前 loss 推出。

### 3.2 从成功轨迹筛选到 hard self-training

先固定一个问题 $x$。令 $\tau$ 表示一整条生成轨迹，$\pi_{\mathrm{roll}}(\tau\mid x)$ 表示 rollout policy 生成该轨迹的概率，$R(\tau)\in\{0,1\}$ 表示该轨迹是否通过 binary verifier。这里所谓“成功轨迹的条件分布”，不是所有数学上正确的推理过程所服从的真实分布，而是：从当前 rollout policy 反复采样，并且只观察 verifier 接受的轨迹时，这些被接受轨迹所服从的分布。形式化地，

$$
q^+(\tau\mid x)
\triangleq
\Pr_{\tau\sim\pi_{\mathrm{roll}}}
\bigl(\tau\mid x,R(\tau)=1\bigr).
$$

根据条件概率定义，

$$
\Pr(\tau\mid x,R=1)
=
\frac{\Pr(\tau,R=1\mid x)}{\Pr(R=1\mid x)}.
$$

当 verifier 是确定性的 binary function 时，联合概率可以写成

$$
\Pr(\tau,R=1\mid x)
=
\pi_{\mathrm{roll}}(\tau\mid x)\mathbf 1[R(\tau)=1],
$$

而分母

$$
Z^+(x)
=
\sum_{\tau'}
\pi_{\mathrm{roll}}(\tau'\mid x)\mathbf 1[R(\tau')=1]
=
\Pr(R(\tau)=1\mid x)
$$

是当前 rollout policy 在问题 $x$ 上产生成功轨迹的总概率。因此，

$$
q^+(\tau\mid x)
=
\frac{\pi_{\mathrm{roll}}(\tau\mid x)\mathbf 1[R(\tau)=1]}
{Z^+(x)}.
$$

这个式子只做了两件事：首先用 reward indicator 将失败轨迹的概率置零，然后用 $Z^+(x)$ 对剩余概率重新归一化。它等价于对 rollout policy 做 rejection sampling 后得到的总体分布。如果 $Z^+(x)=0$，当前 policy 不可能产生成功轨迹，$q^+$ 便没有定义。还应注意，若 verifier 只检查最终答案，$R(\tau)=1$ 只代表轨迹被 verifier 接受，并不保证中间推理的每一步都正确。

实际训练并不显式枚举所有轨迹，也不显式计算 $Z^+(x)$。给定本轮采样的 $N$ 条轨迹

$$
\tau_1,\ldots,\tau_N
\overset{\mathrm{i.i.d.}}{\sim}
\pi_{\mathrm{roll}}(\cdot\mid x),
$$

代码先形成成功集合

$$
\mathcal C(x)=\{\tau_i:R(\tau_i)=1\}.
$$

当 $|\mathcal C(x)|>0$ 时，这些有限样本定义了一个经验分布：

$$
\widehat q^+(\tau\mid x)
=
\frac{1}{|\mathcal C(x)|}
\sum_{\tau_i\in\mathcal C(x)}
\mathbf 1[\tau=\tau_i].
$$

也可以把上式中的 $\mathbf 1[\tau=\tau_i]$ 写成 point mass $\delta_{\tau_i}(\tau)$。在离散轨迹空间里，它只是一个非常朴素的记号：当待查询轨迹 $\tau$ 与样本 $\tau_i$ 完全相同时取 $1$，否则取 $0$。因此，这个公式表达的只是“统计每条轨迹在成功样本中出现了多少次，再除以成功样本总数”。它不是在假设轨迹空间连续，也不需要读者调用连续空间中 Dirac delta 的积分性质；严格说，离散情形称为 point mass 或 Kronecker delta 更不易引起误解。

文中后续不再使用容易与模型分支混淆的 `positive branch`，而统一称为**成功轨迹监督项**，记作 $L^+$；代码中的 `loss_positive` 对应这个量。当前 v1 实现先用 `reward_labels > 0` 生成 `correct_mask`，再计算

$$
L^+(x)
=
-\frac{1}{|\mathcal C(x)|}
\sum_{\tau_i\in\mathcal C(x)}
\log\pi_{\theta,m}(\tau_i\mid x).
$$

其中每条轨迹的 log-probability 是有效 response tokens 的 log-probability 之和，因此当前 v1 聚合方式是“成功 response 等权、每条 response 内 token log-probability 求和”，并不是对所有成功 token 直接做全局平均。v2 在此基础上加入 ratio clip 与可选 rollout IS weight；被保留的 token mask 和 IS weight 都停止梯度，所以 v2 对应的是经过 detached token weighting 的经验目标，而不是最简形式的 $\widehat q^+$。

#### $q^+$ 到底有没有梯度

如果 $\pi_{\mathrm{roll}}$ 来自当前参数，那么从纯数学依赖关系看，$q^+$ 确实随 rollout policy 参数变化。若把它写成 $q^+_{\theta}$，对其求全导数会同时涉及 $\pi_{\mathrm{roll},\theta}$ 与归一化项 $Z^+_{\theta}(x)$。但是，当前算法没有沿这条依赖反向传播。采样出的 token IDs、binary reward、成功集合 $\mathcal C(x)$ 以及由它们定义的 $\widehat q^+$ 都被当作本轮固定训练数据；真正保留梯度的是当前 fused policy 对这些 token 给出的 $\log\pi_{\theta,m}$。

因此，最准确的说法不是“$q^+$ 本身在数学上永远没有梯度”，而是：

> $q^+$ 会随着各轮 rollout policy 的变化而变化，但在一次参数更新的计算图中，它作为 detached target 停止梯度。

可以把第 $k$ 轮训练写成一个交替过程：

$$
q_k^+
=
\operatorname{Condition}
\bigl(\pi_{\mathrm{roll},\theta_k},R=1\bigr),
$$

$$
\theta_{k+1}
\leftarrow
\arg\min_{\theta}
\operatorname{CE}
\bigl(\operatorname{sg}[q_k^+],\pi_{\theta,m}\bigr),
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。下一轮会用更新后的模型重新采样，因此 $q_{k+1}^+$ 可能不同于 $q_k^+$；但当前更新不会穿过离散采样与 verifier 选择去优化“怎样产生这批样本”。如果取消 stop-gradient，就需要处理离散采样和条件归一化带来的 score-function 项，目标会转向 policy-gradient 或 reward-weighted policy optimization，而不再是当前的 verifier-filtered SFT。

在上述 stop-gradient 约定下，成功轨迹监督项就是经验分布上的 cross-entropy：

$$
L^+(x)
=
\operatorname{CE}
\bigl(\widehat q^+,\pi_{\theta,m}\bigr).
$$

又因为

$$
\operatorname{CE}(q^+,\pi_{\theta,m})
=
H(q^+)
+
D_{\mathrm{KL}}(q^+\Vert\pi_{\theta,m}),
$$

而 $H(q^+)$ 对本轮被优化的 $\theta$ 是常数，所以本轮最小化成功轨迹上的 CE，等价于最小化从 detached $q^+$ 到 fused policy 的 forward KL。这个等价关系只在逐轮、stop-gradient 的意义下成立；整个 on-policy 训练过程不是在最小化一个固定不变的 KL。

这里的 forward KL 具体做的是：把本轮 verifier 接受的成功轨迹视为目标分布，并提高 fused policy 对这些轨迹的 likelihood。若本轮只有一条成功轨迹，目标便退化为这条轨迹上的 point mass，优化项就是它的 negative log-likelihood；若有多条成功轨迹，则同时提高经验分布所覆盖的这些成功模式的概率。它不会告诉模型失败轨迹为什么失败，也不会覆盖本轮没有采样到的其他正确解，更没有匹配 proposer 在完整 vocabulary 上的 soft distribution。

因此，必须区分“可以写成 forward KL”与“属于知识蒸馏”这两个命题。任何经验数据上的 maximum likelihood 或 SFT，都可以写成从经验分布到模型分布的 forward KL；这个恒等式本身不提供 teacher，也不构成蒸馏证据。对当前 WDL，更准确的结论是：$L^+$ 的数值与 $D_{\mathrm{KL}}(q^+\Vert\pi_{\theta,m})$ 相差一个本轮参数无关的 $H(q^+)$，二者具有相同梯度和最优解；但它们描述的是成功轨迹上的 selected MLE，而不是标准 teacher-student distillation。

### 3.3 这与标准 self-distillation 有什么关系

传统 knowledge distillation 会构造一个停止梯度的 teacher soft distribution，并让 student 显式匹配它。以温度 $T$ 为例，teacher 与 student 的分布为

$$
p_T^{(T)}(v\mid s)
=
\operatorname{softmax}\!\left(\frac{z_T(s)}{T}\right)_v,
\qquad
p_S^{(T)}(v\mid s)
=
\operatorname{softmax}\!\left(\frac{z_S(s)}{T}\right)_v,
$$

典型的 soft-target loss 为

$$
L_{\mathrm{KD}}
=
T^2D_{\mathrm{KL}}
\left(
\operatorname{sg}[p_T^{(T)}]
\Vert
p_S^{(T)}
\right).
$$

当真实 hard label 也可用时，通常再与 hard-label CE 加权组合。对 autoregressive LM，GKD 将它写成沿一条轨迹的 token-level teacher/student divergence；on-policy GKD 的变化主要是让 student 生成被访问的 prefixes，然后在这些 prefixes 上查询 teacher distribution，并且同样不对 student 的采样分布反向传播。2026 年的 OPSD 虽然让同一个模型在不同 context 下同时扮演 teacher 与 student，但仍然保留一个明确的 privileged teacher distribution、一个 student distribution，以及二者之间的 full-vocabulary token divergence；梯度只通过 student 一侧传播。

当前 WDL 不具备这套标准结构。它没有让 Model1 去匹配 Model2 的完整 next-token distribution，也没有让 Model2 去匹配 Model1；Model1 与 Model2 都通过同一个 fused CE 更新。Model2 proposer 与 verifier 提供的是被选中的 hard trajectories，而不是一个 detached teacher soft distribution。因此，更稳妥的分类是：成功轨迹监督项严格属于 **verifier-gated on-policy hard self-training**；只有在“当前模型系统生成 hard pseudo-label，再由同一系统学习”的宽泛数据流意义下，才可以附带称为 hard sequence-level self-distillation。

这意味着，forward-KL 改写可以帮助我们理解 fused policy 正在向哪里移动，却不能把当前 loss 纳入标准蒸馏语境。若使用“隐式蒸馏”这一说法，必须明确它只是一种 hard pseudo-label 数据流类比，而不是 loss-level 等价：当前不存在独立 teacher soft target、full-vocabulary teacher/student divergence，也不存在只从 teacher 一侧停止梯度并只更新 student 的非对称结构。因而，当前理论主线不应继续寻找一个并不存在的标准 KD loss，而应解释同一 hard target 经 fused logits 作用后，为什么会产生不同于 single-model selected SFT 的优化动力学。

#### STaR 提供了更直接的历史源头

[STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) 在 2022 年已经给出了与这里非常接近的 reasoning self-training 骨架：模型先生成 rationale 与答案，只保留最终答案正确的生成结果，再在这些成功 rationales 上 fine-tune，并重复这一过程。对于第一次没有生成正确答案的问题，STaR 还会把正确答案作为 hint，引导模型重新生成能够解释该答案的 rationale，再把通过验证的结果加入训练集。

2026 年的 [On-Policy Supervised Fine-Tuning for Efficient Reasoning](https://arxiv.org/abs/2602.13407) 把这个骨架改造成了更细粒度的在线训练过程：每个 minibatch 先由当前或极近期 policy 生成多条 responses，再按正确性与长度限制筛选，随后立即在被选中的 responses 上做 SFT 更新。该工作从简化 GRPO 的角度重新推导这一训练形式，并把研究目标集中在准确率与推理长度的 Pareto frontier，而不是首次提出“模型生成—结果筛选—重新监督学习”这一思想。

从方法谱系看，可以把 2026 On-Policy SFT 理解为 **STaR-style reasoning self-training 的在线、逐 minibatch 版本**。反方向上，STaR 更适合称为它的**批式或外循环祖先**，而不宜直接称为“off-policy On-Policy SFT”。严格地说，STaR 每一轮的成功轨迹同样由该轮当前模型生成，并非来自一个固定外部行为策略；它的“非即时性”来自先生成较大的训练集、在同一批数据上做多步 fine-tuning、经典算法每轮从原始 base model 重新训练，以及随后才进入下一轮数据再生。也就是说，二者的主要差异是 policy lag、数据复用与更新粒度，而不是一个简单的 on-policy/off-policy 二分。

| 维度 | STaR | 2026 On-Policy SFT | 当前实验 A |
| --- | --- | --- | --- |
| 基本骨架 | generate-filter-finetune 外循环 | rollout-filter-SFT 逐 minibatch 更新 | Model2 rollout-filter-SFT continuation |
| 成功标准 | 最终答案正确 | 正确且满足长度限制 | correctness、format、EOS 与非截断合同 |
| 失败样本 | 可用正确答案作为 hint 做 rationalization | 不进入正样本 loss | `beta=0` 时不贡献正向梯度 |
| 数据更新 | 先构造较大数据集，再进行多步训练 | 每步使用当前或极近期 policy 的新 rollout | 每步使用 Model2 的新 rollout |
| 模型更新 | 经典算法每轮从原始模型训练 | 持续更新同一 policy | 从 Stage1 Model2 持续更新 |
| 主要目标 | 自举 rationale/reasoning 能力 | 准确率—长度效率 | WDL 的同预算 practical baseline |

因此，WDL 的“Model2 生成候选轨迹—verifier 筛选—在成功轨迹上做 SFT”并不是 2026 年才出现的新范式。它可以追溯到 STaR、rejection-sampling fine-tuning 和 sequence-level self-training。WDL 真正新增的问题是：在已经采用 verifier-filtered On-Policy SFT 的前提下，额外加入 weak Model1，并让两个模型通过同一个 fused objective 更新，为什么能够优于标准单模型 On-Policy SFT；D0 只在下一层帮助判断这项收益中有多少来自 weak logits，而不是 strong-logit scale。

#### 实验 A 是什么层级的 On-Policy SFT 复现

现有 A（Standard On-Policy SFT continuation）已经复现了 2026 On-Policy SFT 最核心的训练形式：当前 Model2 生成多条 response，verifier 选出成功 response，然后对这些 hard trajectories 做 teacher-forced CE，并在更新后重新 rollout。从这个意义上说，A 可以称为 **correctness-filtered On-Policy SFT 的基础复现**，也可以在内部实验叙事中称为 `OPSFT-core reproduction`。

但 A 不是该论文实验配置的 exact reproduction。二者在 length-efficiency 目标、rollout 数、batch size、response 长度、loss aggregation、format/EOS 规则与训练数据上仍有差异。论文写作中推荐使用 **Correctness-filtered On-Policy SFT continuation (our implementation)**，并说明它覆盖 Zhao et al. 的核心 selected-CE objective；除非进一步对齐论文的 correct-and-concise filter、聚合方式和主要超参数，否则不应直接写成“复现了 Zhao et al. 的完整方法”。

这个定位也使 A 成为最自然的 practical baseline：A 与 C 可以匹配 Model2 起点、rollout source、verifier、数据顺序和 post-Stage1 budget，而 C 的核心新增 treatment 是 weak-model fused training。因而，`C-A` 回答的是完整 WDL 方法是否优于同预算 correctness-filtered On-Policy SFT；`C-D0` 则进一步隔离 matched-scale weak-logit treatment。若未来需要进入外部 baseline 主表，只需先审计 A 与论文 objective 在同一 frozen batch 上是否仅相差全局 scalar 或 aggregation，再决定是否有必要新增一个 paper-aligned `A-OPSFT-loss` arm，而不是默认重跑整套训练。

### 3.4 为什么不能把 Model1 称为 teacher

上述 KL 与 self-training 解释有严格边界：

1. 它只覆盖成功集合 $\mathcal C(x)$ 非空时的成功轨迹监督项；
2. 在当前 `beta=0` 主实验中，全错 group 没有训练信号；
3. finite group 只给出经验近似，并受到“是否发现成功轨迹”的概率加权；
4. 如果 `beta>0` 加入 reverse SFT，signed objective 不再是对合法概率目标的 KL；
5. 采样、verifier、成功集合与 $q^+$ 的归一化都不参与本轮反向传播；
6. 当前没有一个正 KL 让 Model2 拟合 Model1 的 soft distribution，也没有一个正 KL 让 Model1 拟合 Model2 的 soft distribution。

所以，更准确的说法不是“Model1 是隐藏 teacher”，而是：

> Model2 proposer 和 verifier 共同构造 hard training data；Model1 与 Model2 通过 fused objective 共同拟合这些数据。hard self-distillation 的类比只存在于数据生成层，WM1 的独特作用仍然需要在 logit coupling 层解释。

这一解释说明了 Model1 为什么能够收到来自成功轨迹的直接学习信号，也说明 Model1 与 Model2 同时提升并不矛盾；但它不能单独保证 Model1 最终一定变好，更不能解释为什么加入 `0.2 z1` 会让 C 超过 D0。后一个问题仍然需要由 fused-logit geometry、相应的替代实验和反向操纵实验回答。

既然数据层的 hard self-training 解释不了 C-A 的完整方法差距，也不能进一步解释 C-D0 的 weak-logit 增量，我们就必须回到 fused logits 本身：相对标准单模型 loss，它增加了什么；在 fixed-scale control 下，又是哪一部分来自 WM1。

### 3.5 本节结论：隐式蒸馏是一条被排除的解释

第三节的价值不在于建立了一个新的“隐式蒸馏”理论，而在于完成了一次必要的排除。我们之所以产生这个猜想，是因为 Model2 负责 rollout，verifier 从中选择成功轨迹，而 Model1 也会在这些轨迹上收到梯度并明显变好；从数据流上看，这很像“较强分支发现数据，较弱分支跟随学习”。进一步地，成功轨迹上的 CE 又能写成从 detached 经验目标 $q^+$ 到 fused policy 的 forward KL，这使蒸馏解释显得更加诱人。

但逐项检查后，这条解释不能成立为严格机制。forward-KL 改写是任何经验分布 maximum likelihood 都具有的恒等式，不会自动产生 teacher；当前 loss 中也不存在独立的 teacher soft distribution、Model1/Model2 之间的 full-vocabulary divergence，或只更新 student 的非对称梯度路径。Model1 与 Model2 实际上是在共同拟合同一个 verifier-filtered hard target，而不是一个模型把自己的完整知识分布蒸馏给另一个模型。

因此，本节得到的结论是：**当前 WDL 不存在一个可以严格识别为标准知识蒸馏或标准自蒸馏的隐式 loss。**“hard sequence-level self-distillation”最多只能保留为描述自生成 pseudo-label 数据流的宽泛类比，不能作为 loss-level 等价关系，更不能作为解释 C 优于 A/D0 的主要机制主张。这个结论也不声称系统内部绝不存在任何广义的信息传递；它只排除了“现有 objective 本质上就是一个隐藏的标准蒸馏目标”这一具体猜想。

这次排除使问题重新变得清晰：成功轨迹来自哪里，可以由 STaR/On-Policy SFT 式 hard self-training 解释；但为什么额外加入 weak Model1 会改变学习速度和最终效果，仍必须由 fused-logit coupling、Product-of-Experts geometry 以及后续可证伪实验来回答。下一节因此不再继续寻找隐藏 teacher，而转向分析 fused logits 本身。

---

## 4. 理论上的关键转折：fused logits 是几何乘积，而不是概率平均

### 4.1 从 logits 混合到 Product-of-Experts

令

$$
p_i(v)=\operatorname{softmax}(z_i)(v),
\qquad
z_m=(1-\lambda)z_1+\lambda z_2.
$$

把 softmax 展开，可以得到一个不依赖额外假设的严格恒等式：

$$
p_m(v)
=
\frac{p_1(v)^{1-\lambda}p_2(v)^\lambda}
{C_\lambda(p_1,p_2)},
$$

其中

$$
C_\lambda(p_1,p_2)
=
\sum_v p_1(v)^{1-\lambda}p_2(v)^\lambda.
$$

这说明 logits 的线性混合，在概率空间里不是 arithmetic average，而是一个加权几何乘积，即 geometric product-of-experts。

这里的 Product-of-Experts（PoE）不是本文新造的名称。Hinton 在经典 PoE 工作中用“将多个 expert 的概率相乘后重新归一化”来组合模型；与 mixture 平均不同，product 只会保留各 expert 共同支持的区域。我们的式子是带权版本：Model1 贡献指数 $1-\lambda$，Model2 贡献指数 $\lambda$。需要强调的是，本文只借用 PoE 的代数结构，并没有照搬 Hinton 当时用于生成模型的 contrastive-divergence 训练算法（见附录 D.18）。

### 4.2 Chernoff affinity 到底是什么

原文直接把 $C_\lambda$ 称为 Chernoff affinity，确实省略了必要背景。`affinity` 在这里不是“吸引力”，而是“相似度”或“重叠程度”：数值越大，表示两个分布在相同 outcome 上分配高概率的程度越高。它不是某一种唯一指标的专名，而是一类 similarity coefficient 的通称。

更标准而不易歧义的名称是 fixed-$\lambda$ **Chernoff coefficient**，也常写作 skew Bhattacharyya coefficient：

$$
C_\lambda(p_1,p_2)
=
\sum_v p_1(v)^{1-\lambda}p_2(v)^\lambda.
$$

本文后续第一次出现时写作“fixed-$\lambda$ Chernoff coefficient（affinity）”，之后简称 $C_\lambda$。`affinity` 只是为了提醒读者它的方向是“越大越相似”，不是本文另行发明了一个 divergence。

对于 $0\leq\lambda\leq1$，逐坐标使用 weighted AM-GM inequality，有：

$$
p_1(v)^{1-\lambda}p_2(v)^\lambda
\leq
(1-\lambda)p_1(v)+\lambda p_2(v).
$$

对词表求和后得到 $C_\lambda\leq1$。当 $0<\lambda<1$ 且两个分布相同时取等号；分布在高概率 token 上越缺少重叠，$C_\lambda$ 通常越小。因此 $C_\lambda$ 可以直观理解为两个 token distribution 的加权几何重叠。

对 coefficient 取负对数，定义：

$$
D_\lambda(p_1,p_2)=-\log C_\lambda(p_1,p_2)\geq0.
$$

这个固定 $\lambda$ 的量更稳妥的名称是 **skew Bhattacharyya distance**；部分文献也称它为 fixed-order Chernoff divergence。这里的“distance”并不保证满足 metric 的全部公理，尤其在 $\lambda\neq1/2$ 时一般不对称。为了沿用本文已有符号，后文仍记作 $D_\lambda$，但不再把它说成一个名称唯一确定的“Chernoff divergence component”。

三组容易混淆的术语可以这样区分：

| 术语 | 定义 | 在本文中的作用 |
|---|---|---|
| fixed-$\lambda$ Chernoff coefficient / affinity | $C_\lambda=\sum_v p_1(v)^{1-\lambda}p_2(v)^\lambda$ | 恰好是 fused PoE 的 normalization constant |
| skew Bhattacharyya distance | $D_\lambda=-\log C_\lambda$ | 固定当前 fusion weight 后得到的分布分离量 |
| Chernoff information / Chernoff distance | $C^*(p_1,p_2)=\max_{\lambda\in[0,1]}D_\lambda=-\log\min_\lambda C_\lambda$ | 对 $\lambda$ 再优化后的经典假设检验量；不是本文训练时实际优化的对象 |

当 $\lambda=1/2$ 时，$C_{1/2}=\sum_v\sqrt{p_1(v)p_2(v)}$ 是经典 Bhattacharyya coefficient，$D_{1/2}$ 是 Bhattacharyya distance。Chernoff 在 1952 年引入相关最优化量，原始语境是二元假设检验：它刻画区分两个分布时 Bayes error 随样本数下降的最优指数速率。它并不是为神经网络或 LLM 训练提出的；它在本文出现，是因为 weighted geometric fusion 的归一化常数恰好具有同一形式（见附录 D.15–D.17）。

### 4.3 从 PoE 到 CE 分解：中间步骤不能省略

对任意监督分布 $q$，无论它是离线 SFT 的 one-hot label、on-policy 成功轨迹形成的 hard empirical target，还是 OPD 的 soft teacher distribution，只要 loss 是 fused distribution 上的 cross-entropy，就有：

$$
\operatorname{CE}(q,p_m)
=
(1-\lambda)\operatorname{CE}(q,p_1)
+\lambda\operatorname{CE}(q,p_2)
-D_\lambda(p_1,p_2).
$$

这条式子不是额外的训练假设，而是一个严格恒等式。先对 fused probability 取对数：

$$
\log p_m(v)
=(1-\lambda)\log p_1(v)
+\lambda\log p_2(v)
-\log C_\lambda.
$$

再代入 cross-entropy：

$$
\begin{aligned}
\operatorname{CE}(q,p_m)
&=-\sum_v q(v)\log p_m(v)\\
&=(1-\lambda)\operatorname{CE}(q,p_1)
+\lambda\operatorname{CE}(q,p_2)
+\log C_\lambda\sum_v q(v)\\
&=(1-\lambda)\operatorname{CE}(q,p_1)
+\lambda\operatorname{CE}(q,p_2)
-D_\lambda(p_1,p_2),
\end{aligned}
$$

其中最后一步只用了 $\sum_vq(v)=1$ 和 $D_\lambda=-\log C_\lambda$。因此，fused CE 可以精确分成两个模型对同一 target 的 weighted CE，再加上一个由几何归一化产生的 coupling term。

这个推导给出了第一个重要结论：只要 Model1 与 Model2 都可训练，两个模型同时变好并不反常。它们不是各自面对不同 target，而是通过同一个 fused residual 共同拟合 $q$。

### 4.4 coupling term 做了什么，又没有做什么

第二个结论需要更谨慎地推出来。$-D_\lambda$ 不是一个让 Model1 模仿 Model2 的 positive KL。若 objective 中存在显式 distillation，梯度里通常会出现类似 $p_1-p_2$ 或 $p_2-p_1$ 的“模型间残差”；这里直接对 fused CE 求导，得到的却是两个模型共享的 target residual：

$$
\frac{\partial L}{\partial z_1}
=(1-\lambda)(p_m-q),
\qquad
\frac{\partial L}{\partial z_2}
=\lambda(p_m-q).
$$

两条分支都被拉向同一个监督目标，但 objective 没有要求 $p_1$ 与 $p_2$ 在整个 non-target distribution 上逐 token 一致。这正是它与 mutual distillation 的关键差别。

那么，为什么分解里会出现一个带负号的 $-D_\lambda$？原因是 PoE 先逐 token 相乘，再用 $C_\lambda$ 重新归一化。两个模型怎样把概率质量排列到相同 token 上，会改变这个 normalizer；normalizer 又会同时缩放所有 fused probabilities。所谓 coupling，具体指的就是：一个模型对某个 token 的赋值，必须和另一个模型在同一个 token 上的赋值共同决定 fused probability，任何一条分支都不能独立解释最终分布。

`-D_lambda` 不能脱离前面两个 target CE 项单独解释。三个项依赖同一组参数；改变某个 logit 时，两个 CE 项与 $D_\lambda$ 会一起变化。因此，式子里有负号不等于训练代码显式加入了“鼓励两个模型分歧”的 regularizer，也不等于 $D_\lambda$ 越大，训练或 generalization 就一定越好。

到这里，真正需要回答的问题是：**什么因素决定 $p_m(y)$ 下降还是上升？** 只看“两个模型不一致”仍然太粗，因为

$$
p_m(y)
=
\frac{p_1(y)^{1-\lambda}p_2(y)^\lambda}
{\sum_v p_1(v)^{1-\lambda}p_2(v)^\lambda}.
$$

分子只看 target token，分母却汇总整个词表。weak model 既可能改变 target 分子，也可能改变各个 non-target token 对分母的贡献。为了把这两种作用分开，我们先构造一个只有三个 token 的例子，而不再用“重叠”“错开”这类难以想象的文字。

设词表只有 $[y,a,b]$，其中 $y$ 是当前正确 target，$a,b$ 是两个 non-target token，并取 $\lambda=1/2$。固定 Model2：

$$
p_2=(0.6,0.3,0.1).
$$

这表示 Model2 给 target $y$ 的概率为 $0.6$；在两个错误 token 中，它更偏向 $a$，因为 $p_2(a)=0.3>p_2(b)=0.1$。matched-scale D0 为：

$$
p_{D0}
=
\frac{(\sqrt{0.6},\sqrt{0.3},\sqrt{0.1})}
{\sqrt{0.6}+\sqrt{0.3}+\sqrt{0.1}}
\approx
(0.473,0.334,0.193).
$$

因此，D0 的 target residual 是 $1-p_{D0}(y)\approx0.527$；对 Model2 raw logit $z_2(y)$ 的实际梯度还要乘外层 $\lambda=1/2$，即约为 $0.264$。下面只改变 Model1：

| Model1 分布 $p_1(y,a,b)$ | 具体含义 | 融合后的 $p_m(y)$ | 去掉共同 $\lambda$ 后、相对 D0 的 target residual |
|---|---|---:|---:|
| $(0.8,0.1,0.1)$ | Model1 也强烈支持正确 token $y$ | $0.717$ | $1-0.717=0.283$，明显衰减 |
| $(0.3,0.6,0.1)$ | Model1 在 non-target 中也把 $a$ 排在 $b$ 前面；两个模型都把较高的错误概率放在同一个 token $a$ 上 | $0.447$ | $1-0.447=0.553$，相对 D0 放大 |
| $(0.3,0.1,0.6)$ | Model2 在错误 token 中偏向 $a$，Model1 却偏向 $b$；二者的 non-target 排序相反 | $0.504$ | $1-0.504=0.496$，相对 D0 衰减 |

后两行是最关键的受控比较。它们具有相同的 Model1 target probability $p_1(y)=0.3$，也具有完全相同的 probability values multiset ${0.3,0.6,0.1}$，所以 Model1 entropy 也相同。唯一差别是：$0.6$ 被分配给了 token $a$，还是 token $b$。仅仅改变这个对应关系，就把 Model2 的 target gradient 从放大翻转成了衰减。

这里的“放大/衰减”严格限定为相对 matched-scale D0，因为 C 与 D0 对 $z_2$ 都带有相同的外层 $\lambda=1/2$。若改为和标准 On-Policy SFT 的 A 比较，A 的 target gradient 是 $1-p_2(y)=0.4$；上述 Align case 的实际 C gradient 是 $0.5\times(1-0.447)\approx0.276$。因此，同一个例子可以“相对 D0 放大、相对 A 仍然衰减”。这不是矛盾，而是说明 gradient amplification 必须明确写出参照系。

为了不让这一区别停留在一句定性判断上，下面把同一组三维向量完整地放回实验 A 的参照系中计算。实验 A 直接用 Model2 policy $p_2$ 做标准 On-Policy SFT；D0 使用缩放后的 strong-only policy；实验 C 使用 Model1 与 Model2 的 PoE fused policy。对于 hard target $y$，表中的 loss 是 $-\log p(y)$；“Model2 target 梯度”则统一指对 **raw Model2 target logit** $z_2(y)$ 的梯度绝对值。因此，C 和 D0 必须保留外层系数 $\lambda=1/2$，不能只比较 $1-p(y)$。

| 实验或局部状态 | loss 实际监督的 target probability | hard-target loss $-\log p(y)$ | $\left|\partial L/\partial z_2(y)\right|$ | 相对实验 A 的梯度倍率 | 在 A 参照系下的解释 |
|---|---:|---:|---:|---:|---|
| A：标准 On-Policy SFT，$p_2=(0.6,0.3,0.1)$ | $0.600$ | $0.511$ | $1-0.600=0.400$ | $1.000$ | 基础方法与主参照 |
| D0：matched-scale strong-only | $0.473$ | $0.749$ | $0.5\times(1-0.473)=0.264$ | $0.659$ | 相对 A 衰减 |
| C：Model1 强烈支持 target，$p_1=(0.8,0.1,0.1)$ | $0.717$ | $0.332$ | $0.5\times(1-0.717)=0.141$ | $0.354$ | 相对 A 明显衰减 |
| C-Align：$p_1=(0.3,0.6,0.1)$ | $0.447$ | $0.805$ | $0.5\times(1-0.447)=0.276$ | $0.691$ | 相对 D0 放大，但相对 A 仍衰减 |
| C-Anti-align：$p_1=(0.3,0.1,0.6)$ | $0.504$ | $0.686$ | $0.5\times(1-0.504)=0.248$ | $0.620$ | 相对 A 与 D0 都衰减 |

这个表揭示了两层不同的问题。第一层是论文中最重要的方法比较：$C-A$ 回答“加入 weak model 后，完整 WDL 相对标准 On-Policy SFT 到底改变了什么”。在这个构造例子中，所有 C 情形对 Model2 raw target logit 的瞬时梯度都小于 A。作为反例，它已经足以否定“WDL 在任意训练状态、每一步都会把 Model2 梯度放大到超过标准 On-Policy SFT”这一无条件命题；但它并不能反过来证明真实训练中的 C 梯度始终小于 A。

第二层才是内部机制消融：$C-D0$ 在相同 $\lambda$ 下消去 strong-logit scale 的影响，用来判断 weak logits 的加入能否把梯度从 matched-scale 状态进一步放大或衰减。Align 行说明，两种结论可以同时成立：weak logits 相对 D0 产生了局部放大，但完整 WDL 相对 A 仍是局部衰减。因此，“梯度放大”不能继续作为一个相对实验 A、无条件成立的总体机制；更准确的定位是：它是 PoE coupling 在特定 target probability 与 non-target assignment 下、相对某个明确参照系可能出现的**条件性局部现象**。

这会改变后续机制叙事。WDL 相对 A 的总体收益，不能仅由单步 Model2 gradient magnitude 解释；还必须考虑 fused policy 改变了哪些 rollout 被生成和筛选、Model1 自身的更新、target 与 non-target 几何，以及这些因素跨 step 形成的反馈动力学。换言之，旧版“梯度放大机制”的强表述站不住脚，但 PoE 几何会条件性地改变梯度这一较弱结论仍然成立，而且可以被后续干预实验检验。

这里还要避免把静态 loss 数值误读成最终效果。例如，第一种 C 状态的 loss 最小，只是因为它在当前 token 上已经给出了更高的 target probability，并不等价于“这组训练一定更好”。这张表承担的是局部动力学诊断：它精确比较同一时刻的监督分布与 raw Model2 gradient；长期 accuracy、收敛速度和 rollout 分布如何变化，仍要由 A/C 主比较和后续受控实验回答。上述 $\lambda=1/2$ 仅用于保持三维例子的计算直观；正式实验使用其他 $\lambda$ 时，应代入同一组公式重新计算，而不是沿用表中的数值。

这里所谓“non-target 排序相反”，不是指 token 在 embedding 空间中距离很远，也不是泛泛地说两个向量不相似。它有一个非常具体的含义：

- Model2 对 non-target 的排序是 $a>b$；
- 第二行的 Model1 排序也是 $a>b$；
- 第三行的 Model1 排序则是 $b>a$。

因此，这个例子得出的结论不是“对齐一定更好”，而是：**即使 target probability、entropy 和 value multiset 全部固定，non-target probability 与 token identity 的对应关系仍然可以改变 fused target gradient 的方向。** 这也说明，gradient amplification 并不是 PoE 的无条件结果；同一个 coupling 既可能放大，也可能衰减。

#### 为什么接下来要研究 cross-model rank alignment

三维例子暴露出了一个此前没有被单独控制的变量：两个模型是否把较高的 non-target probability 分配给相同 token。我们把这种相对排序关系称为 **cross-model rank alignment**。

这里的“rank alignment”不表示两个 token 的名次只差一位或两位，也不涉及 embedding 位置。它关注的是两个模型的排序方向是否一致：Model2 排名靠前的 non-target tokens，在 Model1 中是否也倾向于排名靠前。最简单的情形就是上面的 $a>b$ 对 $a>b$；反方向则是 $a>b$ 对 $b>a$。

为什么这一变量会从 coupling 公式中自然出现？固定 target coordinate，并固定 weak non-target probability values 的 multiset，只改变这些 values 被分配给哪些 token。令

$$
a_v=p_1(v)^{1-\lambda},
\qquad
b_v=p_2(v)^\lambda,
$$

则 non-target 对 normalizer 的贡献是 $\sum_{v\neq y}a_vb_v$。根据 rearrangement inequality：

- 将最大的 weak value 配给最大的 strong value（Align），这部分和最大；
- 随机配对（Random），其值位于最大值和最小值之间；
- 将最大的 weak value 配给最小的 strong value（Anti-align），这部分和最小。

在 target numerator 固定时，因此有：

$$
C_\lambda^{\mathrm{Align}}
\geq
C_\lambda^{\mathrm{Random}}
\geq
C_\lambda^{\mathrm{Anti}}.
$$

$C_\lambda$ 越大，作为同一分母的 $p_m(y)$ 越小，target residual $1-p_m(y)$ 越大。这就是为什么 rank assignment 可能成为控制局部 amplification/attenuation 的一个候选旋钮。它不是从外部文献突然引入的额外理论，而是我们在固定 target 与分布数值后，对 coupling denominator 继续做变量分解得到的。

#### 描述性统计不能单独建立因果关系

仅仅计算 top-$K$ overlap、Spearman/Kendall correlation 或 top-$K$ probability mass，只能回答“训练过程中是否伴随某种 rank alignment”。即使这些统计量与 accuracy 同时上升，也只能说明相关性，不能证明 rank alignment 导致了性能变化。它们在实验中的作用应限于 telemetry 和 manipulation check，而不能单独承担机制结论。

要获得因果证据，必须主动改变这一变量。具体做法是：在同一个 frozen state 上保持 target coordinate、weak value multiset 及其他输入不变，只重新分配 non-target values，构造 `Align / Random / Anti-align`。首先验证 $C_\lambda$、$p_m(y)$ 和 target-gradient magnitude 是否按理论方向变化；只有 manipulation check 成功后，才值得进入短训练实验。

即便短训练 endpoint 随之变化，也仍需检查 intervention 是否同时改变了 rollout data。若要把差异归因于训练 objective 本身，最终还需要 same-rollout 或对应的 $2\times2$ control。也就是说，rank statistics 本身不是因果证据；受控置换提供局部因果干预，而完整训练机制还要进一步排除 trajectory feedback。

#### Top-K 可以限制干预范围，但不应停止其余坐标的训练

在完整词表上做 `Align / Random / Anti-align` 虽然理论上最干净，工程上却可能过强：大量概率极小的长尾 token 会参与重排，既增加实现与统计负担，也可能制造与真实训练无关的扰动。第一版实验可以只在 high-mass non-target Top-K 内改变 assignment，其余 token 保持原样。

但这里必须区分“只改 Top-K”和“只优化 Top-K”。推荐的是前者。对每个 frozen state，用停止梯度的 Model2 ranking 定义：

$$
S_K
=
\operatorname{TopK}_{v\neq y}
\bigl(\operatorname{sg}[p_2(v)]\bigr).
$$

随后只在 $S_K$ 内置换 Model1 logits，target coordinate 和集合外坐标保持不变：

$$
\widetilde z_1(y)=z_1(y),
\qquad
\widetilde z_1(v)=
\begin{cases}
z_1(\pi(v)), & v\in S_K,\\
z_1(v), & v\notin S_K.
\end{cases}
$$

`Align / Random / Anti-align` 只对应不同的 permutation $\pi$。因为这只是集合内的 permutation，它精确保留 Model1 的完整 logit multiset、softmax normalizer、entropy、target probability、Top-K 总质量以及集合外所有 token；改变的只有 Top-K values 与 token identity 的对应关系。

Top-K 之外的 logits 仍然进入完整 softmax、PoE normalization 和 cross-entropy，因此仍会获得标准训练梯度。我们没有把它们 detach，也没有把它们从 loss 中删除。“只允许 Top-K 获得梯度、其他坐标完全不优化”会变成一个新的 sparse-gradient objective，可能引入概率质量泄漏和额外优化偏差，不适合作为第一轮机制控制。

这种 Top-K restricted permutation 不是一个可以直接引用为标准 baseline 的固定算法，更合适的定位是机制干预。它能够检验的命题也必须收窄为：**高概率 non-target tokens 内的 assignment 是否对局部 geometry 和训练结果具有因果作用。** 它不能排除 Top-K 之外的长尾结构仍然重要。

固定 $K$ 仍可能在不同 state 上覆盖截然不同的 probability mass。因此，实验除了报告 $K$，还必须报告 $S_K$ 覆盖的 non-target mass 分布。主实验可以先用已有 checkpoint 选择一个能够覆盖大部分 non-target mass 的 $K$，再预注册一个较小或较大的 sensitivity arm；不能在看到 endpoint 后再挑选最有利的 $K$。如果不同 token position 的分布尖锐度差异很大，也可以用累计质量阈值定义动态集合，但那应作为第二版设计，避免第一轮同时改变集合大小和 assignment。

因此，读到这一节为止，我们只知道：在构造的 frozen state 上，rank assignment 可以成为一个可控的局部 gradient knob。我们仍不知道自然训练中主要处于哪种状态、不知道放大方向是否跨 token 和 step 稳定，也不知道这种局部控制能否解释最终 endpoint。后续实验的任务正是回答这些尚未解决的问题。

### 4.5 两层比较：先对标准 On-Policy SFT，再用 D0 拆 scale 混杂

PoE identity 本身不依赖 D0，也不要求先构造任何 baseline。但若研究问题是“在标准 On-Policy SFT 上加入 WM1 后发生了什么”，第一参照系必须是单模型 On-Policy SFT，也就是实验 A。

在同一个 frozen state 和同一个 hard target $y$ 上，A 使用：

$$
p_2=\operatorname{softmax}(z_2),
\qquad
L_A=-\log p_2(y),
$$

其 Model2 logit gradient 为：

$$
\frac{\partial L_A}{\partial z_2}=p_2-e_y.
$$

C 使用 fused policy：

$$
p_m=\operatorname{softmax}\bigl((1-\lambda)z_1+\lambda z_2\bigr),
\qquad
\frac{\partial L_C}{\partial z_2}=\lambda(p_m-e_y).
$$

因此，对任意监督分布 $q$，WDL 与标准单模型 CE 的 exact loss difference 是：

$$
\operatorname{CE}(q,p_m)-\operatorname{CE}(q,p_2)
=
(1-\lambda)
\bigl[
\operatorname{CE}(q,p_1)-\operatorname{CE}(q,p_2)
\bigr]
-D_\lambda(p_1,p_2).
$$

对 one-hot target $y$，它化为：

$$
L_C-L_A
=
(1-\lambda)
\log\frac{p_2(y)}{p_1(y)}
-D_\lambda(p_1,p_2).
$$

这就是 PoE 理论和标准 On-Policy SFT loss 的直接比较。它表明，加入 WM1 后，loss 同时多出了 branch target-fit difference 与 geometric normalizer coupling；并不需要借助 D0 才能写出理论差异。

对应的 Model2 raw-logit target-coordinate 梯度幅度比为：

$$
A_y^{C/A}
=
\lambda
\frac{1-p_m(y)}{1-p_2(y)}.
$$

对 non-target coordinate $k\neq y$：

$$
A_k^{C/A}
=
\lambda
\frac{p_m(k)}{p_2(k)}.
$$

这里的外层 $\lambda$ 不能抵消。对 target coordinate，C 相对 A 放大的充要条件是：

$$
A_y^{C/A}>1
\quad\Longleftrightarrow\quad
p_m(y)
<
1-\frac{1-p_2(y)}{\lambda}.
$$

这个条件不是恒成立的。特别地，当 $p_2(y)\leq 1-\lambda$ 时，右端不大于零，而 probability 不可能小于零，所以 target-coordinate amplification 相对 A 根本不可能发生。即使 $p_2(y)>1-\lambda$，也只有当 fused policy 把 target probability 压低到上述阈值以下时，增大的 residual 才足以抵消 Model2 coefficient 从 $1$ 变为 $\lambda$。因此，“相对 D0 放大”不自动等于“相对标准 On-Policy SFT 也放大”，更不支持无条件的逐步放大命题。

D0 在第二层才出现。它使用：

$$
p_{D0}=\operatorname{softmax}(\lambda z_2),
\qquad
\frac{\partial L_{D0}}{\partial z_2}
=
\lambda(p_{D0}-e_y).
$$

因为 C 与 D0 对 $z_2$ 都带有相同外层 $\lambda$，这个因子在 C/D0 ratio 中抵消：

$$
A_y^{C/D0}
=
\frac{1-p_m(y)}{1-p_{D0}(y)},
\qquad
A_k^{C/D0}
=
\frac{p_m(k)}{p_{D0}(k)}.
$$

所以三组实验应当形成两层解释：

| 比较 | 回答的问题 | 论文中的地位 |
|---|---|---|
| `C-A` | 完整 WDL 是否优于同预算标准 On-Policy SFT | 方法有效性的主比较、外部 baseline |
| `D0-A` | 单独把 Model2 logits 从 $z_2$ 缩放为 $\lambda z_2$ 会带来什么 | scale/temperature control |
| `C-D0` | 固定 strong coefficient 后，加入 weak logits 与双分支训练会带来什么 | 内部机制消融，不是标准外部 baseline |

三者仍然都是 same-state、single-token、logit-coordinate 层面的比较。参数梯度还要乘 Model2 Jacobian：

$$
\nabla_{\theta_2}L_A
=J_2^\top(p_2-e_y),
$$

$$
\nabla_{\theta_2}L_C
=J_2^\top\lambda(p_m-e_y),
\qquad
\nabla_{\theta_2}L_{D0}
=J_2^\top\lambda(p_{D0}-e_y).
$$

之后的 gradient clipping、optimizer moments、trajectory distribution 以及多步参数分叉都会改变最终 update。因此，C-A 负责证明完整方法是否值得使用；C-D0 负责缩小机制候选；任何 local amplification ratio 都不是 endpoint accuracy 的充分解释。

### 4.6 本节结论：从恒等式得到可证伪假设，而不是直接得到机制证明

第四节现在形成了一条完整链路。我们先从 fused logits 推导出 weighted geometric PoE；PoE 的 normalizer 恰好是 fixed-$\lambda$ Chernoff coefficient；把它代回 CE，得到两个 target CE 与一个负的 skew Bhattacharyya distance term 组成的严格分解。这个 coupling 不要求两个模型彼此模仿，而是让它们通过同一个 fused residual 学习，并让两条分支在相同 token 上的概率排列共同决定 fused distribution。

在此基础上，gradient amplification 获得了一个有条件且依赖参照系的局部解释。相对 A，需要 $\lambda(1-p_m(y))$ 超过 $1-p_2(y)$；相对 D0，只需比较 $1-p_m(y)$ 与 $1-p_{D0}(y)$。三维例子与 rearrangement inequality 只说明，在固定 state、target coordinate 和 value multiset 的条件下，rank assignment 可以定向改变 normalizer、target probability 与 gradient magnitude。它还没有说明自然训练中的 WM1 是否主要通过这条路径起作用，也没有说明这种方向能否在训练过程中稳定控制。

因此，本节严格证明的是 PoE/CE identity、WDL 相对标准 On-Policy SFT 的 exact loss difference，以及两种参照系下的单步梯度关系。Top-K permutation 只是由该推导得到的候选控制手段；尚未证明的是“这些局部几何变化解释了 C 相对 A 的完整方法收益，以及 C 相对 D0 的内部 weak-logit 增量”。后文的 Dynamic Permutation、Align/Random/Anti、TargetGradMatch 与 no-online-WM1 controller，正是为了逐层检验这条尚未完成的因果链。

---

## 5. 新理论与旧理论到底有什么不同

新的 PoE/Chernoff 解释并不是简单地把旧理论换一个术语。它改变了分析的起点、适用范围和实验责任。

| 比较维度 | 旧梯度放大/局部动力学 | 当前 PoE/Chernoff 分析 |
| --- | --- | --- |
| 起点 | weak hard-negative margin 与局部梯度 | fused distribution 的 exact probability identity |
| 是否需要条件 | 需要 weak/strong margin、局部 kernel 或 sensitivity 假设 | PoE 与 CE decomposition 本身无额外 margin 假设 |
| 主要比较 | fused vs raw strong SFT | 第一层 C vs 标准 On-Policy SFT A；第二层 C vs matched-scale D0 |
| 能解释什么 | 某些 non-target gradient 在局部被放大 | 两分支共同 target fitting、分布 affinity、局部放大与全 objective coupling |
| 对 Model1 的处理 | 容易把 WM1 写成噪声源、历史残差或被牺牲分支 | 明确两分支都收到同一个 fused residual，并可能 co-adapt |
| 对 on-policy data 的处理 | 基本是固定监督样本分析 | 与 reward-projected hard self-training 分层组合 |
| 与 distillation 的关系 | 不明确，容易类比成 teacher/student | 明确区分 data-layer hard self-training 与 logit-layer non-KL coupling |
| 对实验的要求 | 容易停留在 post hoc gradient explanation | 指定 affinity、target probability、rank geometry、reverse controls 与 surrogate |

旧理论中的 margin analysis、branch sensitivity、gradient shielding 和 null-space drift 仍可以作为局部诊断视角。它们并没有被推翻。但新的理论把这些局部现象放回一个严格的 objective identity 中，并要求每个解释都对应一个能够失败的实验。

这也是我们避免“理论永远追着结果跑”的关键变化：理论不再只说“某种放大可能发生”，而是提前规定哪些变量应该有序变化、怎样构造相反方向的干预、哪些结果将迫使我们放弃某种叙事。

---

## 6. 如何统一 Offline SFT、On-Policy SFT/SFD 与 OPD

PoE identity 对任意监督目标分布 $q$ 都成立，因此 WDL 可以被理解为一个 supervision-agnostic coupling operator；不同训练范式的差别，主要在于 $q$ 从哪里来。

| 训练范式 | 监督目标 $q$ 的来源 | 可以统一的部分 | 不能混为一谈的部分 |
| --- | --- | --- | --- |
| Offline SFT | 固定数据集中的 hard labels | fused CE 仍具有同一个 PoE/Chernoff decomposition | 没有 on-policy rollout、verifier projection 或 self-distillation claim |
| On-Policy selected SFT/SFD | 当前 rollout 经 verifier 过滤后的成功 hard trajectories | 数据层是 hard self-training，logit 层是同一个 PoE coupling | finite sampling、all-wrong groups、clip/IS、reverse branch 会破坏纯 KL 表述 |
| OPD | teacher 在 student states 上给出的 soft distribution | 若对 full soft target 做 fused CE，可写成两个 weighted forward-KL terms 减 Chernoff coupling | 当前 WDL 实现没有显式 teacher KL，数据生成与监督路径不同 |

因此，如果 WDL 在 hard on-policy SFT 和 soft OPD 中都有效，真正共同的候选机制应当是 fused-logit geometry，而不是笼统地说“它们都有蒸馏”。

这也解释了为什么 On-Policy Distillation 不是 2026 年才出现的新概念。GKD / On-Policy Distillation 和 MiniLLM 都已发表于 ICLR 2024；它们在 student-generated states 上使用显式 teacher distribution。2026 年的 OPSD、SDPO 等工作进一步使用 privileged context 或 feedback 构造 self-teacher，但仍包含显式 token-distribution supervision。当前 WDL 的关键差别在于：WM1 不是需要被模仿的 teacher distribution，而是 fused objective 的一个可训练分支。

同时，Rethinking OPSD 和 Anti-Self-Distillation 的负面或反向结果提醒我们：在长推理模型中，过强的 teacher agreement 可能抑制高 entropy 的分叉、自检和回溯行为。因此，WDL 中允许 non-target tails 保持差异，可能是一个值得检验而不是应当消除的特征。但在实验完成前，它只能作为研究动机，不能写成已验证优势。

---

## 7. 从一个漂亮公式走向一组可证伪假设

PoE 恒等式提供了一个更准确的分析语言，但一个恒等式本身不是机制证明。它只告诉我们 fused objective 可以怎样分解，没有告诉我们 C 为什么在当前训练合同下超过 D0。为了避免再次用理论追赶结果，我们必须先列出所有仍然可能成立的解释，再为每个解释指定一个能够让它失败的实验。

### 7.1 假设 H0：当前 C-A 与 C-D0 差距并不稳定

这是最先需要保留的可能性。当前 C-A 与 C-D0 都来自单一训练 seed 和 online `n=3`，即使七个数据集同向，也不能替代 common frozen evaluation、paired uncertainty 和第二个 seed。

如果 common evaluator、固定 decoder 和第二个 seed 下 C-A 消失，那么完整 WDL 的 practical advantage 没有站稳；如果 C-A 存在而 C-D0 消失，则增益更可能来自 scale、训练结构或其他 treatment，而不是稳定的 weak-logit contribution。只有两层差值分别报告，才能知道机制研究是否还有依据。

因此，任何机制实验都不能绕开效力确认。机制实验可以与 common evaluation 的准备并行，但论文中的机制结论必须以主效应可复现为前提。

### 7.2 假设 H1：C 只是更长或更复杂的 selected SFT

A、B、C、D0 的意义之一，就是排除“多训练了一段时间”这一最平凡解释。C 与 D0 共享相同的 Stage1 Model2 起点、相同 post-Stage1 prompts、相同 Model2-only rollout 机制和相同 P60 horizon；D0 还保留了 `lambda=0.8` 的 logit scale。后续实际采样轨迹并不共享，因此这仍是完整 procedure 的对比，而不是 fixed-data objective ablation。

如果 C 稳定超过 A，H1 就不足以用“只是继续做普通 selected SFT”解释完整方法收益；如果 C 同时稳定超过 D0，weak-logit treatment 才获得进一步证据。因此，论文中必须把 C-A 放在方法主比较位置，同时把 C-D0 明确标成内部机制消融。

### 7.3 假设 H2：WM1 只提供一个 scalar 级别的置信度或 entropy 变换

PoE 融合会改变 fused target probability、fused entropy、non-target mass 和 gradient magnitude。即使 weak tail 与真实 token identity 完全没有关系，仅仅降低目标概率、提高某些梯度坐标，也可能改变学习速度。

这个假设的关键预测是：只要用 Model2-only controller 匹配 C 的 target probability、target-gradient magnitude 或 entropy，就能复现大部分 C-D0 增益。

如果成立，WDL 的核心机制将从“weak model 提供结构化知识”收缩为“weak branch 实现了一个自适应的局部几何控制器”。这仍然是有价值的结果，但论文需要把重点放在可计算的 scalar geometry 和训练效率上，而不是 dark knowledge。

### 7.4 假设 H3：WM1 提供的分布形状有用，但真实 token assignment 不重要

entropy 和 target confidence 之外，完整 weak value spectrum 也可能重要。例如，同样的 entropy 可以由截然不同的 tail 形状产生；PoE 会把这个 tail 与 Model2 的 tail 相乘，从而改变整条 non-target gradient vector。

如果对 weak non-target values 做逐步随机置换后，效果仍接近真实 C，而又明显超过 D0，那么证据将偏向：weak value multiset、spectrum 或由融合引起的 optimization perturbation 足以产生效果，真实 token identity 的证据变弱。

注意，这一结果仍不能推出“WM1 可以删除”。置换后的 values 仍由 WM1 online forward 产生，Model1 仍然参与反向传播和更新。它只是说明 WM1 输出中的哪一部分信息可能不是必要条件。

### 7.5 假设 H4：WM1 与 Model2 的 cross-model rank geometry 才是关键

DynPerm 会改变 weak tail 与 strong tail 的相对排列，因此同时改变 Chernoff affinity、fused target probability 和 gradient strength。若真实 C 优于随机置换，差距可能来自 token-specific semantics，也可能只是来自 cross-model rank alignment。

这个假设要求更细的方向性干预：在保留 weak target coordinate、weak entropy 和 value multiset 的前提下，主动构造 `AlignSort`、`RandomPerm` 与 `AntiAlignSort`，让 weak/strong non-target ranks 的对齐程度按预注册方向变化。

如果 measured affinity、target probability 和梯度量按预测排序，endpoint 也出现同向排序，那么 cross-model rank geometry 就获得方向性证据。反之，如果 telemetry 没有按预期移动，endpoint 无论高低都不能用于支持这套机制，因为干预本身没有成功。

### 7.6 假设 H5：真实 token-specific weak assignment 具有额外作用

这是最接近“WM1 提供结构化语义”的强假设。它要求在尽量匹配 scalar geometry 和 rank geometry 后，只破坏精确 token identity，然后真实 C 仍然更好。

`RankBinPerm` 是为此设计的近似控制：先按 Model2 non-target rank 划分窄 bin，只在 bin 内置换 weak values。这样既打乱精确 token 对应，又尽量不改变 rank alignment、Chernoff affinity、fused target probability 和 gradient ratio。

只有当以下两个条件同时成立时，real C 与 `RankBinPerm` 的差距才能作为 token-specific residual 的候选证据：

1. permutation coverage 足够高、fixed-point rate 足够低，token identity 确实被破坏；
2. affinity、fused target probability 和 gradient-ratio 分布在预注册容差内匹配，几何因素确实被控制。

如果窄 bin 保留了太多 identity，或宽 bin 破坏了 geometry，这个实验只能作为 diagnostic，不能承担因果结论。

### 7.7 假设 H6：优势主要来自 on-policy trajectory，而不是 fused objective

即使当前 rollout source 是 Model2-only，C 与 D0 一旦发生一次不同 update，后续 Model2 proposer 的 trajectory distribution 就会分叉。最终 C-D0 差距可能主要来自 C 早期发现了更好的成功轨迹，而不是同一批数据上的 fused loss 更优。

这个假设要求把 data source 与 training objective 分开：冻结相同的 successful rollout manifest，然后交叉使用 C fused loss 与 D0/model2-only loss。只有在共享轨迹上 C 仍占优，我们才能说 fused objective 具有独立贡献；如果 D0 使用 C 的成功轨迹后追上，reward-projected data transfer 就可能是主导因素。

### 7.8 为什么这些假设必须按顺序排除

这组假设不是并列的实验菜单，而是一条从低成本、粗粒度问题走向高成本、细粒度问题的判别链：

| 顺序 | 先回答的问题 | 如果跳过，会留下什么歧义 |
| ---: | --- | --- |
| 1 | 主效应是否可复现 | 后续机制实验可能在解释 noise |
| 2 | WM1 是否需要更新 | 无法区分 static guidance 与 adaptive co-training |
| 3 | 真实 assignment 或与其绑定的 fusion geometry 是否有粗粒度贡献 | 无法判断 weak values 只是 shape，还是与真实 token correspondence 绑定 |
| 4 | scalar / rank geometry 是否足够 | real-vs-permutation 差距会被误写成 semantics |
| 5 | 匹配 scalar/rank telemetry 后是否仍有 token-specific residual | 不能判断 residual 是否属于 semantic dark knowledge |
| 6 | online WM1 是否可被低维 controller 替代 | fixed/permute arms 都仍保留 WM1 forward |
| 7 | rollout data 与 loss geometry 各占多少 | endpoint 差距仍混合了 on-policy distribution drift |
| 8 | 反向干预是否产生反向曲线 | 只能说相关或 close-the-gap，不能说方向性因果 |

这条顺序把“为什么有效”改写为一组可以失败的问题。它也明确告诉我们，某个负结果不是整个理论失败，而是帮助我们把解释空间收窄到下一层。

这里还要提前区分两个容易混淆的 control：fixed-M1 保留 WM1 forward 与 `0.2 z1`，只移除 WM1 update；D0 则从训练 logits 中完全移除 weak contribution，仅保留 `0.8 z2`。

---

## 8. 审稿人的“同熵”问题为什么成为突破口

### 8.1 审稿人真正质疑的不是一个超参数

审稿人提出 uniform 或 temperature-matched、same-entropy control，实质是在追问：如果 weak model 的贡献只是改变了分布的不确定性或形状，那么为什么需要一个具有真实 token semantics 的 WM1？

这个问题直指旧梯度放大理论的薄弱处。旧理论说 weak distribution 会暴露 hard negatives，但没有区分两种可能：

1. 哪些 token 被放大很重要，即 token-specific assignment 携带结构；
2. 只要某些梯度被放大即可，具体落在哪些 token 上并不重要。

Dynamic Permutation 的价值，就在于它可以在不摘掉 WM1、也不预知 WM1 未来会学到什么的情况下，在线破坏第一类信息，同时保留大量第二类统计。

### 8.2 “同熵”不能等同于 uniform

literal uniform distribution 只有最大 entropy。对于任意非最大 entropy 的 weak distribution，不存在一个同时 uniform 且与它同熵的分布。

更重要的是，把 weak logits 替换为常数向量 `c`，有：

$$
\operatorname{softmax}((1-\lambda)c+\lambda z_2)
=
\operatorname{softmax}(\lambda z_2).
$$

所以 uniform weak logits 在当前设置下等价于 matched-scale D0。它是一个必要的 no-weak endpoint，却不是一个保留当前 weak entropy 的同熵干预。

temperature matching 也只匹配一个 scalar entropy，不保留 target probability、value multiset 或 token assignment。它应当作为 `EntropyMatch` 独立控制，而不是与 Dynamic Permutation 混为一谈。

### 8.3 canonical Dynamic Permutation 的定义

对每个样本、每个 response token、每个 optimizer step，设 weak logits 为 $z_1$，当前 hard target token 为 $y$。构造一个 permutation matrix $P_{y,t}$：

- 固定 target coordinate `y`；
- 只在所有 non-target coordinates 上置换；
- permutation seed 由 training seed、global step、sample index 和 token position 确定；
- 每一步重新采样，不给模型学习固定逆置换的机会。

干预后的 weak logits 为：

$$
\widetilde z_1=P_{y,t}z_1,
$$

训练仍使用：

$$
\widetilde z_m=(1-\lambda)\widetilde z_1+\lambda z_2.
$$

如果需要 partial dose，可以只对比例为 $\rho$ 的 non-target coordinates 置换。第一批只做两个端点：

- `rho=0`：严格 no-op，应与真实 C 完全等价；
- `rho=1`：所有 non-target coordinates 参与动态置换。

只有端点产生稳定、可解释的 separation 后，`rho=0.25/0.5` 才有必要用于描绘 dose-response。

### 8.4 该干预精确保留什么

在同一个 forward 内，target-preserving permutation 精确保留：

1. weak target logit 与 weak target probability；
2. weak non-target values 的完整 multiset；
3. weak 分布的 entropy；
4. weak centered-logit norm 以及任何只依赖 value multiset 的统计；
5. Model1 与 Model2 两条反向传播路径；
6. 双模型仍然 trainable 的 procedure。

它破坏的是 weak non-target values 与 vocabulary token identity 的一一对应。因此，它能够对真实 assignment 及其绑定的 fusion geometry 是否影响完整 adaptive training procedure 做粗粒度测试。

这类“保留预测值、置换类别对应”的思路并非凭空出现。Born-Again Networks 在 ICML 2018 的 Dark Knowledge with Permuted Predictions（DKPP）实验，已经用相似干预区分 teacher prediction spectrum 与 class identity。我们的版本把这一思想扩展到 autoregressive token、target-preserving、step-resampled 和双分支共同训练的场景。

### 8.5 它没有保留什么

Dynamic Permutation 不保留：

- mixed/fused entropy；
- $C_\lambda(P_{y,t}p_1,p_2)$ 或 fixed-$\lambda$ skew Bhattacharyya distance；
- fused target probability；
- Model2 的 target/non-target logit-gradient magnitude；
- parameter-gradient direction、clip frequency 和 optimizer update norm；
- 干预发生后未来 rollout trajectory 的 entropy 与成功率。

原因很直接：PoE 依赖 weak 与 strong 在相同 token 上的乘积。虽然 weak 自身的 values 没变，但它们与 strong tail 的对应关系变了，fusion geometry 就会随之改变。

所以，`real C > DynPerm` 最多说明“真实 token assignment 或与之绑定的 cross-model geometry 有贡献”。它不能单独证明 semantic dark knowledge。要把 semantics 与 geometry 分开，还必须补 `Align/Random/Anti` 和 `RankBinPerm`。

### 8.6 为什么必须动态重采样，而不能用固定 permutation

固定 permutation 会在整个训练过程中保持同一个 token 映射。由于两条分支都可训练，Model1 可能逐渐学会把需要作用于 token `k` 的值输出到固定映射后的 coordinate 上，相当于适应甚至反演干预。最终“置换没有影响”可能只是模型学会了绕过控制。

逐 optimizer step 重采样让映射不再稳定，使训练无法建立一个固定逆变换。这样，实验估计的是“持续保留真实 token assignment 是否重要”，而不是“模型能否学会适应一个新的固定 vocabulary 编码”。

### 8.7 为什么这个实验仍然不能去掉 WM1

DynPerm 仍然执行 WM1 forward，仍然从 WM1 产生 values，也仍然把梯度传回 WM1。它只改变中间的 value-to-token assignment。

因此，无论结果如何，都不能写：

- “Dynamic Perturbation 去掉了 WM1”；
- “WM1 只是一段随机噪声”；
- “我们已经找到无需 weak model 的等价方法”。

真正的 WM1 replacement 必须满足更严格的 operational definition：正式训练时不运行 online WM1 forward，不更新 WM1 optimizer，也不依赖当前 batch 的 WM1 logits。若 replacement 的参数来自历史 WM1 traces，还必须把结论限制为“online WM1 不必要”，而不是“weak-model information 从未被使用”。

### 8.8 Dynamic Permutation 的结果应怎样写

| 观测 | 可以支持的解释 | 仍然不能支持的解释 |
| --- | --- | --- |
| real C 明显优于 `DynPerm-100` | 真实 assignment 或与其绑定的 fusion geometry 对 adaptive procedure 有贡献 | 每个 hard negative 都具有语义；WM1 不可替代 |
| real C 与 `DynPerm-100` 近似等价，且二者都优于 D0 | token identity 证据变弱，weak shape/spectrum 或 generic perturbation 可能足够 | 已证明 no-WM1；语义完全无用 |
| `DynPerm-100` 比 D0 还差或训练不稳 | 置换破坏了必要结构，或引入了不利的 geometry/optimization stochasticity | 可以直接归因于语义被破坏 |
| partial permutation 单调退化 | assignment-preservation ratio 与效果相关 | 关系一定是语义因果；仍需 geometry telemetry |
| partial permutation 非单调 | 可能存在最佳扰动强度或多种机制竞争 | 理论被简单否定；先检查干预有效性与 optimizer 行为 |

这里必须使用等价性界限，而不是把“不显著”当作“相同”。如果要说 `real≈perm`，需要预先规定允许的 effect-size margin、paired interval 和 seed-level consistency。

---

## 9. 从“WM1 有用”到“WM1 起什么作用”：完整机制实验链

本节按发现顺序说明每一个实验为什么出现。每一步都只解决前一步留下的一个核心歧义。

这不是把所有可能实验平铺成菜单：M0 先确认理论变量是否真实移动；M1 问 WM1 是否必须在线更新；M2 对真实 assignment 及其绑定的 geometry 做粗粒度消融；M3 再把 scalar、rank geometry 与 token-specific residual 分开；M4 才尝试 no-online-WM1 replacement；M5 用反向操纵和 same-rollout 设计补上因果方向；M6 最后再与显式 distillation 对齐。顺序本身就是识别设计，目的是避免看到结果后临时选择最顺手的解释。

### 9.1 M0：先用已有 checkpoint 看清发生了什么

第一步不需要新训练。使用已有 C 与 D0 的 P0/P25/P40/P55/P60 checkpoint 和固定 validation samples，逐 token、逐 confidence bin 记录：

1. weak、strong、fused、D0 的 target probability 与 target rank；
2. weak/strong entropy 与 centered-logit norm；
3. `log C_lambda` 与 fixed-$\lambda$ skew Bhattacharyya distance；
4. C/D0 target 和 non-target amplification ratio；
5. weak/strong top-k overlap、non-target rank correlation；
6. Model1-only、Model2-only、fused 的 correctness overlap；
7. positive rate、all-wrong/all-correct group rate；
8. pre-clip grad norm、post-clip norm、clip frequency、optimizer update norm；
9. format、EOS、truncation、response length 与 answer diversity。

这一步的目的不是用 correlation 证明机制，而是决定后续 intervention 的方向。例如，反向 target-margin 实验的符号不能凭直觉指定；必须先看 C 相对 D0 在真正贡献增益的 confidence bins 中，是降低还是提高了 target probability。

M0 还承担一个现实责任：如果 C 与 D0 的 supposed mechanism variables 根本没有可测差异，后续 controller 和 directional permutation 就失去了依据。

### 9.2 M1：fixed-M1 先区分静态引导与共同适应

fixed-M1 是当前最高优先级的新增机制实验。它保留 WM1 forward 和：

$$
z_m=0.2z_1+0.8z_2,
$$

但冻结 WM1 参数，只训练 Model2。

计划包含两个 P60 arms：

- `Cold Start -> fixed M1`：从 cold-start Model2 source 开始；
- `Stage1 -> fixed M1`：从与 C/D0 对齐的 Stage1 Model2 source 开始。

每个 arm 所使用的 frozen WM1 checkpoint identity 必须与对应 source contract 绑定并预注册，不能在 CS0 与 Stage1 arms 之间临时复用来源不同的 WM1。否则 fixed-CS0、fixed-S1 与 A/C/D0 的比较都会混入 weak initialization 差异。

两组都必须保留 Model1 与 Model2 的 online validation；即使 Model1 冻结，也要验证它的指标确实近似不动，并记录 Model2、必要时 fused view 的曲线。否则我们无法区分“冻结成功”与“日志遗漏”。

核心比较是：

| 比较 | 精确回答的问题 |
| --- | --- |
| fixed-S1 vs joint C | adaptive WM1 update 是否提供净收益 |
| fixed-S1 vs D0 | 静态 weak logits 是否已经足以超过 no-weak |
| fixed-CS0 vs A | 从 cold start 出发，static weak guidance 是否优于普通 selected SFT |
| fixed-CS0 vs fixed-S1 | strong initialization 是否改变 fixed guidance 的价值 |

判读如下：

- `fixed≈joint>D0`，且 grad/clip/update telemetry 无解释性偏差：静态 weak guidance 基本足够，co-adaptation 不是必要条件；
- `joint>fixed>D0`，且 optimization telemetry 合格：静态 guidance 与 adaptive co-training 都有贡献；
- `joint>fixed≈D0`，且不能由可训练参数量或 clipping 解释：WM1 的在线共同适应是主候选；
- `fixed` 或 `joint≤D0`：当前 source 下没有稳定的正 weak contribution。

fixed-M1 仍然不能回答 token assignment 是否重要，也不能宣称去掉 WM1。它只节省 WM1 backward、optimizer update 和部分通信，不节省 WM1 forward。

它还引入了一个新的 optimization confounder：冻结分支会改变 global grad norm、clipping 和 optimizer behavior。因此必须同时记录 per-branch pre/post-clip norm、clip frequency 与 Model2 update norm，不能只看 endpoint accuracy。

### 9.3 M2：Dynamic Permutation 先做 assignment/geometry 的粗粒度消融

完成 `rho=0` exact no-op test 后，先做一个 `DynPerm-100` P20/P30 pilot，而不是直接铺开 P60 全矩阵。pilot 的 admission 条件包括：

- target coordinate 完全不变；
- weak entropy、target probability 与 sorted-value checksum 在数值容差内不变；
- permutation coverage 合格；
- Model1/Model2 gradient connectivity 都存在；
- RNG 可复现；
- `rho=0` 在 forward、backward、RNG 和 config 上与 C 完全等价；
- runtime、格式与 reward contract 正常。

如果 P20/P30 已出现可解释 separation，再把端点推进到 P60。若没有 separation，则先做等价性分析和第二 seed，不应立刻追加多个 partial ratios。只有端点对比真实且需要定位转折，才补 `rho=0.25/0.5`。

fixed-M1 与 DynPerm 的先后顺序有逻辑含义。fixed-M1 先回答 WM1 是否必须共同适应；DynPerm 再在保持 WM1 forward/trainability 的条件下，对真实 assignment 及其绑定的 fusion geometry 做粗粒度消融。这个结果不能直接写成 semantic assignment，必须由 M3a/M3b/M3c 继续拆分 scalar、rank 与 identity。若先做 DynPerm，负结果还会和 co-adaptation behavior 混在一起，解释更困难。

### 9.4 M3a：EntropyMatch 与 TargetGradMatch 拆掉 scalar geometry

在同一批 frozen tokens 上，先构造两个不训练的 Model2-only counterfactual：

- `EntropyMatch`：调节 Model2-only temperature，使变换后分布的 entropy 匹配 C fused entropy；
- `TargetGradMatch`：只给 target logit 加 detached bias，使 target probability 匹配 C，从而精确匹配 one-hot target-coordinate gradient magnitude 与 aggregate non-target mass。

它们回答的是：C-D0 的差距是否可以由一个或两个 scalar 几何量解释。

为什么 DynPerm 不够？因为 DynPerm 虽然保留 weak entropy，却改变 fused entropy、fused target probability 和 gradient magnitude。如果 `real>perm`，我们仍不知道差异来自 token assignment，还是来自这些 scalar 量被意外改变。`EntropyMatch/TargetGradMatch` 专门拆这个混杂。

如果 frozen diagnostic 显示这些 scalar 量能解释 C gains 集中的 bins，再进入短训练 controller。若它们与增益无关，就不应耗费 GPU 跑完整 controller。

### 9.5 M3b：Align / Random / Anti 拆 cross-model rank geometry

在固定 target coordinate 和 weak non-target value multiset 的前提下，把 weak tail 重新分配到 strong tail ranks：

1. `AlignSort`：最大的 weak tail values 配对最大的 strong tail values；
2. `RandomPerm`：随机配对；
3. `AntiAlignSort`：最大的 weak tail values 配对最小的 strong tail values。

根据 rearrangement inequality，这三组会按方向改变 non-target 对 Chernoff affinity 的贡献。在同一个 frozen forward 中，target coordinate 固定且只重排 non-target tail，因此 target numerator $p_1(y)^{1-\lambda}p_2(y)^\lambda$ 不变；此时：

$$
C_\lambda\uparrow
\Longrightarrow
p_m(y)\downarrow
\Longrightarrow
1-p_m(y)\uparrow.
$$

这个推导只预注册 telemetry 的方向，不直接预言 endpoint accuracy。真正支持 geometry mechanism 的结果必须同时满足：

1. measured affinity、fused target probability 与 gradient strength 按设计排序；
2. endpoint 或 learning curve 也出现稳定、可复现的相应排序。

若 telemetry 排序失败，说明干预没有实现理论目标；若 telemetry 成功而 endpoint 不排序，则说明 geometry 虽被操纵，但长期效果还受 trajectory、Jacobian、optimizer 或其他因素支配。这不是“公式错误”，而是说明 local geometry 不是充分机制。

### 9.6 M3c：RankBinPerm 才真正接近 semantic control

`Align/Random/Anti` 故意改变 rank geometry，所以真实 C 优于这些 arms 仍不能证明 token semantics。`RankBinPerm` 尝试在相近 rank 范围内打乱精确 identity，以近似保持 affinity 和 target-gradient distribution。

这一步只有在 M3b 表明 geometry 本身重要后才值得做。否则我们甚至不知道需要匹配哪些量。它的 admission 也最严格：不仅要报告 global mean，还要按 Model2 confidence、position、target rank 等 bins 检查 fused target probability 和 gradient ratio 的完整分布。

若 real C 在 geometry matching 合格时仍优于 `RankBinPerm`，我们才获得 token-specific residual 的额外证据。更稳妥的结论是：真实 value-to-token correspondence 中存在超出当前已匹配 scalar/rank telemetry 的 residual information。它只是 semantic-assignment 的候选证据，还需要按 token family、frequency、format token、position 和 dataset type 做排除性分析；不能声称每个 high weak logit 都是“有意义 hard negative”。

### 9.7 M4：只有到这里，才开始尝试 no-online-WM1 replacement

前述所有实验都仍然使用 WM1 logits。要回答“WM1 能否被替代”，必须根据 M0-M3 找到的 sufficient statistics，构造一个不运行 online WM1 的低维 deterministic controller。

候选形式可以是按 token position、Model2 target confidence 和 entropy 分 bin 的查表函数：

$$
\widehat z_w
=
G_{b(t),b(p_2(y)),b(H(p_2))}
\big(\operatorname{rank}(z_2),y\big).
$$

它不应是另一个完整 neural network；否则只是重新发明一个 WM1。第一版最小候选应当是下一章推导的 target-margin controller，只有当 target probability 不足以解释结果时，才增加 entropy、tail spectrum 或 rank-alignment statistics。

真实 replacement arm 必须满足：

- calibration、training、evaluation prompts 分离；
- evaluated run 中没有 online WM1 forward/backward；
- controller 参数量、存储与 calibration cost 披露；
- 与 C、fixed-M1、D0 使用相同 Model2 source、数据、horizon 和 evaluator；
- 若 calibration 使用过 WM1 traces，论文只写“无需 online WM1”。

### 9.8 M5a：reverse control 把 close-the-gap 变成方向性证据

一个 controller 接近 C，只说明某种变换能够复现部分现象；它仍可能是一个与真实机制无关的优化技巧。更强的检验是反向移动同一个机制变量，并预期 learning curve 也反向。

两类 reverse controls 分别针对不同解释：

- geometry reversal：`AlignSort` 与 `AntiAlignSort` 反转 affinity、target probability 与 gradient-strength telemetry；
- scalar reversal：在 M0 确定 C 相对 D0 的 target-probability 位移方向后，做大小匹配、方向相反的 target-margin arm。

反向 arm 不能简单使用相同绝对值的 `|delta|`。logit bias 与 probability/gradient displacement 是非线性的，必须匹配可行的 target-probability 或 target-gradient 位移，并审计 pre/post-clip norm、clip frequency 和 update norm。

如果正向 controller close the gap，而反向 controller 在 validity metrics 合格时使曲线向相反方向移动，scalar gradient mechanism 才获得方向性因果证据。若反向只引起爆梯度、过度 clipping 或长度崩溃，它只是 optimization stress test。

### 9.9 M5b：same-rollout 2x2 拆 data source 与 objective

on-policy 训练最大的识别难题是：objective 一旦不同，未来数据也会不同。为此冻结 rollout manifest，构造局部 2x2：

它放在 controller 与主效应确认之后，是因为 frozen-manifest replay 的工程成本更高，而且只有 full online C-D0 gap 稳定时，拆 data/objective 才有实际识别价值；这不是说 data layer 在理论上更次要。

| Frozen rollout manifest generated by | Replay objective from the same common base state |
| --- | --- |
| C checkpoint 的 Model2 proposer | C fused loss |
| 同一份 C-proposer manifest | D0/model2-only loss |
| D0 checkpoint 的 Model2 proposer | C fused loss |
| 同一份 D0-proposer manifest | D0/model2-only loss |

四个 replay-training cells 必须从同一个预注册 base training checkpoint，以及相同 optimizer/scheduler state、prompt shard 和短 horizon 启动。C-proposer manifest 与 D0-proposer manifest 只作为 detached frozen data source 被重放；selected-token mask 随 manifest 固定。若直接使用 C checkpoint 或 D0 checkpoint 本身作为训练起点，就会把 model state 与 data source 混在一起，必须另开 `state × data × objective` 设计，不能称为纯 same-rollout 2x2。所有 cells 都要记录 positive rate、all-wrong/all-correct composition、length、format、EOS 与 token accounting。

判读是：

- D0 用 C manifest 后追上 C：trajectory quality / reward-projected self-training 主导局部差异；
- C 在同一 manifest 上仍领先：fused objective 有独立贡献；
- 只有 C manifest × C loss 的组合有效：data 与 geometry 存在 interaction；
- 短 horizon 无差异但完整 online 有差异：机制主要通过长期 distribution drift 累积。

这项实验估计的是固定数据下的局部 interaction，不替代 full online endpoint。它的价值在于把“更好的数据”和“更好的 objective”第一次放到同一张因果表里。

### 9.10 M6：显式 distillation bridge 是最后的解释桥，而不是第一步

如果最终要在论文中比较 WDL 与标准 distillation，可以在相同 successful prefixes 上增加：

- hard selected-SFT only；
- explicit Model2-to-Model1 KL；
- symmetric JS 或 deep-mutual-learning-style KL；
- WDL fused objective；
- 在可用时，WDL 加显式 OPD target。

这一步用来回答 fused-logit mechanism 与显式 teacher/student matching 的关系。它不是把成功轨迹监督项解释为 verifier-gated hard self-training 所必需的，也不应在 fixed-M1、DynPerm 和 geometry controls 之前占用预算。

如果 explicit KL 复现 WDL，论文可以把 WDL 写成一种计算或优化上不同的 distillation coupling；如果 explicit KL 明显不如 WDL，说明“共同拟合 hard target + 保留 tail disagreement”的结构可能更重要。两种结果都不能仅凭 loss 形式下结论，仍要比较 trajectory、entropy、forking 和成本。

---

## 10. 如何构造一个真正可解释的 no-online-WM1 controller

### 10.1 为什么先从 target margin 开始

如果 M0 显示 C 相比 D0 最稳定的差异是 fused target probability 与 target-gradient magnitude，那么最小替代物不应是另一个网络，而应是一个只改变 target margin 的确定性变换。

这样设计有三个原因：

1. 它直接对应旧梯度放大理论最清楚的可测量对象；
2. 它不改变 Model2 non-target tokens 之间的相对排序，因此不会暗中引入一套新的 token semantics；
3. 它的作用可以精确推导，不需要用“神经网络学到了某种形状”来解释。

### 10.2 精确推导

从 matched-scale D0 分布出发：

$$
q_0=\operatorname{softmax}(\lambda z_2).
$$

只给 hard target token `y` 的 logit 加一个 bias $\delta$，其他 logits 不变。变换后的 target probability 为：

$$
q_\delta(y)
=
\frac{e^\delta q_0(y)}
{e^\delta q_0(y)+1-q_0(y)}.
$$

若希望它等于预定目标概率 $r$，解得：

$$
\delta
=
\operatorname{logit}(r)
-
\operatorname{logit}(q_0(y)).
$$

对任意 non-target token $k\neq y$：

$$
q_\delta(k)
=
q_0(k)\frac{1-r}{1-q_0(y)}.
$$

因此，这个 controller 精确做到两件事：

- 匹配 hard-target CE 的 target-coordinate gradient magnitude $1-r$；
- 匹配 aggregate non-target gradient mass $1-r$。

它同时保留 Model2 non-target distribution 的相对排序，因为所有 non-target probabilities 只乘同一个比例。

### 10.3 它故意不匹配什么

target-margin controller 不匹配：

- C 的完整 non-target gradient vector；
- WM1 tail 的 token-specific assignment；
- weak/strong cross-model rank geometry；
- Model1 与 Model2 的共同适应；
- Model2 Jacobian 映射后的 parameter-gradient direction；
- clipping、Adam moments 作用后的真实 update；
- soft-target OPD 中 full distribution 的梯度。

这不是设计缺陷，而是识别策略。一个好的机制控制不应同时复制所有可能因素；它要只复制一个候选机制，然后观察能够 close 多少 gap。

### 10.4 stop-gradient 边界必须写进实现合同

$q_0(y)$、目标 $r$ 和由此得到的 $\delta$ 都必须从 detached values 计算。反向传播只能通过“加过 bias 的 Model2 logits”进入 Model2 参数。

如果允许 $q_0\rightarrow\delta$ 的梯度路径，controller 可能在微分意义上把 target probability 锁成常数，抵消或重定义原本想测试的训练信号。这样得到的结果无法再解释为“匹配 C 的 target-gradient magnitude”。

实现时还必须把 $q_0(y)$ 和 $r$ clamp 到 $[\epsilon,1-\epsilon]$，并记录 clamp rate。若大量 token 被 clamp，说明当前 controller 在这些状态上不可用，不能把数值饱和后的结果解释成机制证据。

### 10.5 三个阶段：diagnostic、calibrated、replacement

同一个公式可以对应三种证据强度，必须分开命名。

**阶段一：oracle diagnostic（不是可部署训练方法）。**

直接读取当前 C forward 的 fused target probability 作为 $r$，在同一批 frozen tokens 上重放 D0 counterfactual。这能回答“若只复制 target probability，局部梯度有多接近”，但它依赖当前 WM1 信息，不能作为训练方法。

**阶段二：WM1-trace-calibrated controller。**

在 disjoint calibration split 上，用 detached Model2 confidence、position、step 和 entropy bins 拟合：

$$
\widehat r=f\big(p_2(y),H(p_2),t,\text{step}\big).
$$

正式训练不运行 online WM1，只查表或调用低维函数得到 $\widehat r$。如果这一版本接近 C，可以说“online WM1 可以被预先校准的低维统计替代”。由于 calibration targets 来自 WM1 traces，仍不能说整个方法从未使用 weak-model information。

**阶段三：WM1-free rule。**

若进一步从 Model2-only statistics、理论预设或跨任务共享规则得到 $r$，且整个 calibration 与训练过程都不使用 WM1 信息，才接近真正的 no-WM1 method。它的难度更高，不应作为第一轮机制实验的成功标准。

### 10.6 两个 scalar 不够时，怎样扩展而不重新发明 WM1

如果 target-only controller 只能解释部分差距，可以增加一个 detached temperature $T$，同时匹配 target probability 与 fused entropy：

$$
q_{T,\delta}
=
\operatorname{softmax}\left(\frac{\lambda z_2+\delta e_y}{T}\right).
$$

这是一个 two-scalar root-finding 问题。每个 token 的目标对 $(r,H)$ 未必可行，尤其在 vocab 很大、目标概率接近边界或 entropy 目标与 non-target ordering 不兼容时。实现必须：

- 检查解是否存在；
- 记录 root-finding failure rate；
- 拒绝不稳定解，而不是静默 clamp 到任意数值；
- 报告匹配误差的完整分布；
- 保持 $T$ 与 $\delta$ detached。

若 two-scalar controller 仍不足，再根据 M3 证据添加 tail spectrum 或 rank-alignment table。每次只增加一个由前一实验明确指出的统计，不允许直接引入一个高容量网络把所有差距拟合掉。

### 10.7 controller 的正向与反向预注册

先用 M0 确认在贡献增益的 bins 中，C 相对 D0 的 $p(y)$ 位移。如果 C 主要降低 $p(y)$、增大正样本梯度，则：

- 正向 arm：向更低的 $r$ 移动，复现 C 的 target-gradient displacement；
- 反向 arm：向更高的 $r$ 移动，并在 probability/gradient space 中匹配位移大小。

如果真实 M0 方向相反，则预注册符号也必须相反。不能因为“梯度放大”叙事听起来合理，就在看过结果后挑选有利方向。

最终只有在以下链条同时成立时，才能说 scalar target-gradient strength 是一个因果组件：

1. C 与 D0 的 telemetry 存在预注册方向差异；
2. forward controller 匹配该差异并 close 一部分 endpoint gap；
3. reverse controller 产生相反 telemetry；
4. learning curve 也按相反方向移动；
5. clipping、update norm、长度和格式没有提供更简单解释。

---

## 11. 预先写好结果解释：避免实验结束后重新发明故事

机制研究最容易犯的错误，是每个结果出现后都临时找到一个听起来合理的解释。为避免这种情况，我们先把主要结果组合与允许的叙事写出来。

### 11.1 第一层：fixed-M1 的三种关键结果

以下三种判读都以冻结生效、Model1/Model2 validation 完整，并且 grad/clip/update telemetry 没有提供更简单的优化解释为前提。

**情形 A：`fixed≈joint>D0`。**

这说明 static weak guidance 已经能解释大部分增益。Model1 的在线共同适应不是 Model2 提升的必要条件，尽管 joint run 中 Model1 仍可能因为共同 hard targets 而变好。

下一步重点应转向 DynPerm 和 low-dimensional controller：固定 weak distribution 的哪些统计在起作用？能否离线缓存或低维复制？

不能写“无需 WM1”，因为 fixed arm 仍执行 WM1 forward。

**情形 B：`joint>fixed>D0`。**

静态 guidance 与 adaptive co-training 都有贡献。这是最符合双层机制的结果：初始 weak geometry 提供局部扰动，双方又在 verifier-selected trajectories 上共同适应。

下一步必须比较 joint 与 fixed 下的 affinity、target probability、gradient ratio、correctness overlap 和 update norms，寻找 co-adaptation 如何改变有效统计。

**情形 C：`joint>fixed≈D0`。**

在线共同适应是主候选，静态 weak shape 几乎不足。此时 no-online-WM1 controller 更难，但更有理论价值；它必须模拟的是 trajectory-dependent statistics，而非一个固定 prior。

DynPerm 仍应运行，因为它能判断共同适应需要真实 token assignment，还是只需要一个随训练变化的 distributional shape。

### 11.2 第二层：DynPerm 与 geometry controls 的联合判读

| real C | RandomPerm | RankBinPerm | 主要解释 |
| --- | --- | --- | --- |
| 高 | 接近 C | 未做或接近 C | token identity 证据弱；shape/rank/optimization perturbation 可能足够 |
| 高 | 低 | 接近 C | real-random 差距主要由 rank geometry 解释，精确 identity 不是必要条件 |
| 高 | 低 | 也低，且 geometry matching 合格 | token-specific assignment 有额外证据 |
| 高 | 低 | 低，但 geometry matching 失败 | 无法区分 semantics 与几何残差，实验不具结论性 |
| 低或不稳 | 任意 | 任意 | 先回到主效应与训练合同，不继续扩张机制叙事 |

若 endpoint 沿 M0 预注册的有利方向排序，且 affinity/gradient telemetry 同序，cross-model rank geometry 获得方向性支持；在“lower target probability 有利”的设定下，这可能表现为 `Align>Random>Anti`。若 endpoint 排序存在、telemetry 不存在，应先判定 intervention invalid；若 telemetry 存在、endpoint 不排序，则 local geometry 不是 endpoint 的充分条件。

### 11.3 第三层：controller 是否真正替代 online WM1

**controller 接近 C。**

如果 disjoint-calibrated controller 在无 online WM1 forward 的 run 中接近 C，同时优于 D0 和 fixed-M1 的相应对照，那么可以写：在该模型、任务、horizon 与 calibration contract 下，online WM1 的主要作用可由低维统计近似。

若 calibration 使用 WM1 traces，claim 必须保留这一依赖；若跨 seed、跨 source 或代码任务失效，也不能写 universal equivalence。

**controller 位于 D0 与 C 之间。**

这是非常有信息量的部分成功。它说明 scalar target confidence 或 entropy 解释了 WDL 增益的一部分，剩余部分可能来自 tail vector、rank alignment、co-adaptation 或 trajectory interaction。

此时不应把 controller 判定为“失败”；应根据 residual gap 与 M3 telemetry 决定下一项最小扩展。

**controller 接近 D0。**

低维统计不足。若 fixed-M1 仍接近 C，则更丰富的 cached weak tail 或 rank structure 可能重要；若只有 joint C 有效，则 online co-adaptation 与 trajectory-dependent coupling 更可能是核心。

这个结果只能否定当前 controller，不能证明所有可想象的 surrogate 都不可能，也不能证明 WM1 在所有 domain 中不可替代。

### 11.4 第四层：same-rollout 2x2 如何重写 on-policy 故事

| 结果 | 叙事重心 |
| --- | --- |
| D0 在 C manifest 上追上 C | C 主要改善 successful trajectory discovery；hard self-training/data quality 是主因 |
| C 在共享 manifest 上仍领先 | fused geometry 在固定数据上有独立作用 |
| 只有 C manifest × C loss 强 | data source 与 objective 存在互相增强的 interaction |
| frozen 2x2 差异小，full online 差异大 | 长期 policy-induced distribution drift 是主要机制 |

这一层决定“隐式蒸馏”应该占多大比重。如果 data-source effect 主导，reward-projected hard self-training 是主线；如果 shared-rollout 下 fused loss 仍明显更好，PoE geometry 才能承担更强的独立解释。

### 11.5 哪些结果会真正否定当前主假设

当前主假设不是不可推翻的。以下结果会迫使我们明显降级或放弃它：

1. common frozen evaluation 与第二 seed 下 C-A 完整方法效应消失，或 C-D0 weak-logit 增量消失；两种失败对应不同结论，不能合并解释；
2. M0 中 PoE/gradient variables 与 gains 没有稳定关系，且 direction controls 无法操纵它们；
3. TargetGradMatch、rank geometry 与 reverse controls 都不能产生预注册方向的 telemetry；
4. 同一 rollout manifest 下 C 与 D0 无差异，full online 的差异又完全由数据 composition 解释；
5. fixed/joint/permute/controller 的差异都可由 clipping、length、format 或 token accounting 解释；
6. Model1 的提升在 common frozen evaluator 中消失，说明在线双模型增长是 evaluator artifact。

若出现这些结果，更合适的结论会变成：WDL 是一个在当前 procedure 中有效的训练配方，但现有 PoE/gradient mechanism 没有获得因果支持。承认这一点比保留一个无法失败的理论更有研究价值。

### 11.6 一个理想但并非预设必然的结果组合

最理想的证据链不是“每个 WDL arm 都赢”，而是各项操纵按理论分层：

1. C-A 在 common eval 与第二 seed 下稳定，证明完整方法收益；C-D0 也稳定，证明 weak-logit treatment 值得继续拆解；
2. fixed 介于 joint 与 D0，显示 static guidance 与 co-adaptation 都存在；
3. real C 优于 RandomPerm；
4. endpoint 沿 M0 预注册的有利方向排序；例如，若 lower target probability 被预注册为有利方向，可能表现为 `Align>Random>Anti`，且 affinity/gradient telemetry 同序；
5. real C 仍优于 geometry-matched RankBinPerm；
6. target-margin controller close 一部分 gap，richer low-dimensional controller close 更多；
7. reverse arm 使 telemetry 与曲线反向；
8. same-rollout 2x2 显示 data layer 与 fused objective 都有独立贡献。

如果这一组合成立，论文可以给出一个层次化结论：WDL 的收益同时来自 reward-projected data、PoE 引起的 scalar/rank geometry、以及真实 weak assignment 中的剩余结构；online WM1 的部分作用可被控制器复制，但完整共同适应仍有价值。

这只是一个预先声明的理想图景。实验的目的不是让所有结果都落入它，而是判断实际证据在哪一层停止。

---

## 12. Math-first 实施合同：怎样让结果具有可解释性

### 12.1 第一阶段为什么只做 Qwen3-1.7B 数学任务

当前工作的目标不是立刻证明跨模型普适性，而是先在一个模型规模上把机制做扎实。Qwen3-1.7B Math 具有三个现实优势：

- 现有 A/B/C/D0、GRPO 和双模型曲线最完整；
- 训练和 validation 周期较短，适合短 pilot 与快速否定；
- 数学 verifier 的 correctness 信号比代码任务更直接，代码当前提升还混入格式、长度和执行依赖。

因此，第一阶段所有机制实验固定在 Qwen3-1.7B、Math-first、`beta=0`、`lambda=0.8`、Model2-only rollout 和现有 Math-7 evaluator 下。只有机制变量与主效应在这里稳定后，才进入代码任务、其他 model family 或 model size。

### 12.2 先写停止规则，再决定是否扩实验

这条机制链不是默认全部执行。预注册的主要停止规则是：

- 若 common frozen evaluation 与第二 seed 不支持 C-A，停止把 WDL 写成具有稳定 practical advantage；若 C-A 成立但 C-D0 不成立，停止以 weak-logit geometry 为主线的高成本机制扩展；
- 若 M0 中目标 telemetry 没有稳定差异，停止对应 controller/reverse 分支；
- 若 fixed-M1 本身不具备可解释结果，先排查冻结、clipping 与 source contract，再决定 DynPerm 如何解释；
- 若 DynPerm 端点没有 separation，先做等价性界限，不扩展 partial-ratio grid；
- 若 Align/Anti 没有实现预注册的 affinity/gradient 排序，不推进 RankBin endpoint；
- 若 controller 的匹配误差或 clamp/failure rate 不合格，不把训练结果用于机制结论；
- 若 same-rollout 的 common base state 无法固定，不把设计称为纯 2x2。

停止并不等于机制研究失败。它意味着当前证据不足以支持更细的识别问题，应该保留预算而不是用更多 arms 稀释结论。

### 12.3 immediate wave：最小新增 GPU 预算

| 顺序 | 新训练 | Horizon | 进入下一步的门槛 |
| ---: | --- | ---: | --- |
| 0 | 无：M0 frozen diagnostics | 无 | telemetry 能区分 C/D0 与关键 confidence bins |
| 1 | fixed-M1 Cold Start | P60 | 冻结、双模型 validation、grad/clip/update telemetry 完整 |
| 2 | fixed-M1 Stage1 | P60 | 与 joint C、D0 的 source/fairness contract 对齐 |
| 3 | `DynPerm-100` pilot | P20/P30 | `rho=0` no-op proof 与所有 intervention validity checks 通过 |
| 4 | DynPerm endpoint confirmation | 最多 P60 | pilot separation 真实且可解释；common revision、matched seed |

这就是当前最小 immediate GPU cost：两个 fixed-M1 P60 arms 加一个 P20/P30 DynPerm pilot。不能在 pilot 前一次性启动 partial permutation、RankBin、synthetic surrogate 和 reverse grid。

fixed-M1 的代码、admission 与 diagnostic run 可以和 common evaluation 准备并行；但 DynPerm 的正式 P60 endpoint、controller、reverse arms 以及任何论文机制结论，都必须先确认 C-A 的完整方法收益，并进一步确认 C-D0 weak-logit 增量通过 common frozen evaluation 和第二 seed。

### 12.4 conditional waves

| Wave | 条件 | 新实验 |
| --- | --- | --- |
| M3 scalar | M0 显示 entropy/target probability 与 gains 有稳定联系 | frozen EntropyMatch/TargetGradMatch；必要时短训练 |
| M3 rank | DynPerm 有差异或 geometry telemetry 明显 | AlignSort、AntiAlignSort P20/P30；Random 作为中点 |
| M3 semantics | rank geometry 已被验证且仍有 residual question | RankBinPerm，先 validity 再 endpoint |
| M4 replacement | M1-M3 指向少量 sufficient statistics | 一个 calibrated target-margin/entropy controller |
| M5 reverse | 正向 controller 或 geometry arm 已产生稳定方向 | magnitude-matched reverse arm |
| M5 data/objective | C-A practical gap 与 C-D0 internal gap 均稳定 | same-rollout 2x2 短 horizon |
| M6 bridge | 论文确实需要 distillation positioning | explicit KL / mutual-learning controls |

每一 wave 都有停止条件。负结果或 validity failure 会阻止相关分支扩张，而不是自动产生更多超参数搜索。

### 12.5 每个 model view 都必须在线验证

joint 与 fixed arms 的训练日志必须分别记录：

- Model1-only Math-7 `mean@n` / `pass@n`；
- Model2-only Math-7 `mean@n` / `pass@n`；
- fused view 的对应指标（若 evaluator 合同支持）；
- reward-positive rate、all-wrong/all-correct group rate；
- 每个模型的 format success、EOS、truncation、response length；
- per-branch grad norm、update norm、clip frequency。

fixed-M1 中 Model1 validation 不是多余指标。它用于证明冻结确实生效、初始 weak source 没有被意外替换，也提供“固定但持续参与 forward 的 reference curve”。

Model1 与 Model2 的 validation 必须采用相同 prompt、decoder 与 scorer，不能把两个不同 evaluation contracts 的绝对数直接比较。

### 12.6 evaluation hierarchy：不同采样规模回答不同问题

| 层级 | 用途 | 允许的结论 |
| --- | --- | --- |
| online `n=3` | 监控训练、定位转折和异常 | 只作 trajectory diagnostic，不作最终方法排序 |
| common frozen `n=8` | 用统一 decoder/evaluator 筛选机制 arms | 支持 paired uncertainty 下的中等成本对比 |
| selected checkpoints `n=256` | pass@k、maj@k、diversity、truncation 与 publication confirmation | 用于最终性能与 coverage 叙事，仍需多 seed |

在线与离线指标不能混用绝对数，也不能用在线最佳 step 的点估计替代预注册 checkpoint selection。

### 12.7 efficacy 与机制 telemetry 必须分账

最终 accuracy 只回答实验有没有价值；它不证明 intervention 按理论工作。每个干预都必须同时满足两类证据：

**Efficacy metrics。**

- common frozen `n=8`，再对入选 checkpoint 做 `n=256`；
- mean@k、pass@k、maj@k；
- paired prompt-level correctness 与 bootstrap interval；
- peak 与 terminal checkpoint；
- 第二个 training seed；
- response length、format、EOS、truncation、answer/full-response diversity。

**Mechanism and validity metrics。**

- weak/strong/fused/D0 entropy 与 target probability；
- `log C_lambda` / fixed-$\lambda$ skew Bhattacharyya distance；
- target/non-target gradient ratios；
- cross-model rank overlap/correlation；
- pre/post-clip grad norm、clip frequency、optimizer update norm；
- permutation coverage、fixed-point rate、multiset checksum；
- positive-trajectory composition；
- Model1/Model2 correctness overlap。

若 endpoint 改变但干预的目标统计没有改变，不能按预注册机制解释；若目标统计改变但 endpoint 不变，则说明它不是 endpoint 的充分因素。

### 12.8 budget 必须作为结果的一部分披露

WDL 每一步更新两个 submodels，不能用相同步数暗示与 single-model SFT 或 GRPO 成本相同。报告至少包含：

- rollout generated tokens；
- selected/training tokens；
- wall-clock GPU-hours；
- 近似 FLOPs；
- peak 与平均显存；
- checkpoint/storage；
- validation compute；
- fixed-M1 节省的 backward/optimizer/communication 与仍保留的 WM1 forward；
- surrogate calibration cost。

机制实验的主张是识别原因，不是利用更多计算获得更高分。因此，只要 compute 不匹配，就必须在表格中单列，而不能把性能差异全归因于机制。

### 12.9 publication validity gates

下列任一项失败，都只能把 run 视为 local diagnostic：

1. code revision、submodule revision、model/data identity 不完整；
2. scorer、prompt template、decoder 或 validation set 不一致；
3. `rho=0` no-op equivalence 未证明；
4. entropy/multiset/target-coordinate preservation 未通过；
5. Model1 或 Model2 gradient connectivity 与 arm 定义不符；
6. scheduler/admission receipt 缺失，run 无法证明真正启动；
7. release gate 未通过；
8. 只有 online summary、没有 raw prompt-level outputs；
9. 只报告最佳 checkpoint、不披露 terminal 与选择规则；
10. null result 没有等价性 margin 与 uncertainty。

截至本文创建时，fixed-M1 的底层代码、配置字段、权重准备工具、单测和两个 Math wrapper 已经存在，但没有可核验的正式 Slurm 运行、独立 joint cache 与 admission receipt。因此，本文把它们称为“prepared arms”，不称为“running”或“completed”。Dynamic Permutation 及后续 controls 也尚未实现或启动。

---

## 13. 最终叙事：现在可以说什么，未来可能说到哪里

### 13.1 当前最稳妥的机制表述

基于现有公式、源码与实验事实，当前可以使用的完整表述是：

> On-Policy WDL 在数据层使用 verifier-gated、reward-projected 的 hard self-training：Model2 proposer 产生 trajectories，verifier 选出成功样本，两条可训练分支共同拟合这些 hard targets。在 logit 层，线性融合对应一个 geometric product-of-experts；WM1 不是显式 teacher，而是通过 fused distribution 改变 target probability、non-target tail 与局部梯度几何。两条分支在相同 hard targets 上共同适应，因此 Model1 与 Model2 同时提升并不矛盾。

这段话同时解释了我们看到的三个事实：

1. C 相对标准 On-Policy SFT A 的初步增益说明完整 WDL 方法值得研究；C 相对 matched-scale D0 的进一步增益说明 weak-logit package 值得单独做机制消融；
2. Model1 没有负责 rollout 却显著提升，因为它也拟合 verifier-selected trajectories；
3. 两个模型不必通过显式 KL 互相模仿，PoE objective 已经让它们共享 target residual，同时允许 non-target tails 保留差异。

### 13.2 当前可以写入论文的结论

- strict scorer 修复后的 Math-first 训练链路显示 continuous C 是当前主要候选；
- 单 seed、online `n=3` 下，C 相对标准 On-Policy SFT A 出现初步完整方法增益，且相对 matched-scale D0 出现初步 weak-logit contribution signal；
- 成功轨迹监督项可以解释为 verifier-gated hard self-training；只有在“数据层 hard-target、而非 teacher-distribution distillation”的限定意义下，才可附带称为 hard sequence-level self-distillation；
- Model1 不是标准 teacher distribution；
- fused logits 严格对应 geometric PoE，并具有 Chernoff affinity decomposition；
- local gradient amplification 是可测量的几何诊断，不是 endpoint accuracy 的充分解释；
- fixed-M1、DynPerm、geometry matching、controller 与 same-rollout 2x2 构成一条逐层排除替代解释的实验链。

### 13.3 当前不能写入论文的结论

- WDL 的方法级胜利已经确认；
- WM1 是隐藏 teacher；
- 两个模型同时提升证明发生了普通 knowledge distillation；
- Dynamic Permutation 能去掉 WM1；
- same weak entropy 等于 same fused entropy；
- `real>random permutation` 自动证明 semantic dark knowledge；
- fixed-M1 与 D0 相同；
- frozen WM1 不需要 forward；
- PoE 分解中的 negative skew Bhattacharyya distance term 保证分布越不一致效果越好；
- logit-gradient amplification 保证 parameter update 或 endpoint accuracy 提升；
- 当前 single-seed online 数值足以支持 publication-level causality；
- hard-target SFT/SFD 的 controller 结论自动适用于 soft-target OPD。

### 13.4 如果实验链成功，最终理论会比旧理论多出什么

旧理论的核心是一条局部充分条件：weak hard negatives 可以放大 strong branch 的某些梯度。新的理论与实验链试图多完成四件事：

1. 用 exact PoE/Chernoff identity 说明两分支 objective 的结构，而不依赖特定 margin 条件；
2. 把 on-policy data generation 与 logit coupling 分开，使 hard self-training 和 geometry 各有清楚边界；
3. 把“结构”拆为 scalar confidence、entropy、value spectrum、cross-model rank geometry 与 token identity，逐层操纵；
4. 构造正向、反向和 replacement experiments，使理论能够预测失败，而不只是解释成功。

如果最终 low-dimensional controller 能复现 C，理论将反过来指导方法简化：WM1 可能只是一个昂贵的自适应几何生成器。如果只有真实、可训练 WM1 有效，理论也会更清楚：有效因素不是一般梯度放大，而是 trajectory-dependent co-adaptation 或 token-specific structure。两种答案都比“多一个模型所以更好”更有价值。

### 13.5 这次探索真正得到的不是一个答案，而是一条判别路径

我们最初从“WDL 比 baseline 好”出发，首先想到的是梯度放大；当 Model1 与 Model2 同时提升时，又自然怀疑存在隐式蒸馏。进一步推导后，我们发现这两个直觉都只说对了一层：

- self-distillation 解释的是成功轨迹怎样变成两条分支共同学习的数据；
- gradient amplification 解释的是 PoE fused objective 在局部怎样改变 hard-target 梯度；
- 两者都不能单独解释完整的 on-policy dynamics。

PoE/Chernoff identity 把这两个层次接在了一起，也暴露出真正缺失的证据：我们不知道 WM1 的 static shape、online adaptation、cross-model rank geometry 和 token semantics 各占多少；也不知道最终差距来自训练 objective，还是它诱导出的后续 trajectory distribution。

因此，这份文档的最终结论不是“WDL 为什么一定有效”，而是把这个问题变成了一组明确、有限、能够失败的判别题：

1. 主效应能否复现？
2. WM1 是否必须更新？
3. 真实 token assignment 是否重要？
4. scalar 或 rank geometry 能解释多少？
5. 在 geometry 匹配后是否仍有 semantic residual？
6. online WM1 能否被低维 controller 替代？
7. rollout data 与 fused objective 各自贡献多少？
8. 反向操纵是否产生反向结果？

当这些问题被依次回答后，我们才能从“一个有效的双模型训练配方”走向“一个有清楚作用对象、可被替代、也可被反驳的机制解释”。

---

## 附录 A：核心公式与符号

| 符号 | 含义 |
| --- | --- |
| $z_1,z_2$ | Model1/weak 与 Model2/strong logits |
| $\lambda$ | Model2 权重；当前 C 中为 `0.8`，Model1 权重为 `0.2` |
| $p_1,p_2$ | 两分支 softmax distribution |
| $p_m$ | fused distribution |
| $p_{D0}$ | matched-scale no-weak distribution `softmax(lambda z2)` |
| $q$ | 任意监督 target distribution |
| $C_\lambda$ | fixed-$\lambda$ Chernoff coefficient（affinity），即 PoE normalization constant |
| $D_\lambda$ | $-\log C_\lambda$，非负 skew Bhattacharyya distance；本文沿用 $D_\lambda$ 作为简写 |
| $q^+$ | verifier 选择后的成功轨迹条件分布 |
| $r$ | target-margin controller 希望匹配的 target probability |
| $\rho$ | Dynamic Permutation 的 non-target 置换比例 |

核心等式：

$$
p_m(v)
=
\frac{p_1(v)^{1-\lambda}p_2(v)^\lambda}
{\sum_u p_1(u)^{1-\lambda}p_2(u)^\lambda}.
$$

$$
\operatorname{CE}(q,p_m)
=
(1-\lambda)\operatorname{CE}(q,p_1)
+\lambda\operatorname{CE}(q,p_2)
-D_\lambda(p_1,p_2).
$$

$$
A_y^{C/D0}
=
\frac{1-p_m(y)}{1-p_{D0}(y)},
\qquad
A_k^{C/D0}
=
\frac{p_m(k)}{p_{D0}(k)}.
$$

$$
q^+(\tau\mid x)
=
\frac{\pi_{\mathrm{roll}}(\tau\mid x)\mathbf{1}[R(\tau)=1]}
{\sum_{\tau'}\pi_{\mathrm{roll}}(\tau'\mid x)\mathbf{1}[R(\tau')=1]}.
$$

$$
\delta
=
\operatorname{logit}(r)
-\operatorname{logit}(q_0(y)).
$$

---

## 附录 B：实验—问题—结论边界总表

**C vs D0。** 精确控制是否加入真实 weak logits，并保持 strong scale；没有控制 single seed 与长期 trajectory 分叉。它能回答 weak-logit treatment package 是否有初步作用，不能回答 WM1 的哪一部分起作用。

**fixed-M1。** 精确控制 WM1 是否更新，并保留 WM1 forward/fusion；没有控制 global clipping、optimizer 与通信变化。它能回答 static guidance 与 co-adaptation 的相对贡献，不能回答 no-WM1 或 token semantics。

**Dynamic Permutation。** 精确保留 weak entropy、target probability、value multiset 与双梯度路径；不保留 fused entropy、affinity 和完整 gradient。它对真实 assignment 及其绑定的 geometry 做粗粒度消融，不能单独区分 semantics 与 rank geometry，也不能回答 no-WM1。

**EntropyMatch。** 精确匹配 fused entropy 这个 scalar；不匹配 target probability、full tail 或 co-adaptation。它能测试 entropy 是否足以解释差距，不能回答 token/rank structure。

**TargetGradMatch。** 精确匹配 target probability、target-gradient magnitude 与 aggregate tail mass；不匹配 full tail vector 或 parameter-gradient direction。它能测试 scalar target-gradient 是否足够，不能自动解释 OPD 的 full-distribution mechanism。

**Align/Random/Anti。** 方向性操纵 tail rank affinity；不保留 token identity，也不直接控制长期 endpoint。它能检验 cross-model rank geometry 是否是因果候选，不能证明 semantic necessity。

**RankBinPerm。** 近似匹配 rank/gradient geometry，同时破坏精确 identity；完美几何等价很难保证。它能检验已匹配 telemetry 之外是否还有 token-specific residual，不能直接把 residual 定义为语义。

**no-online controller。** evaluated run 不使用 online WM1；如果 calibration 使用 traces，仍保留历史 weak information。它能检验 online WM1 是否可替代，不能支持 universal no-WM1。

**same-rollout 2x2。** 固定 detached data source 并交叉 training objective；不覆盖完整 online dynamics。它能估计 trajectory 与 loss 的局部贡献，不能穷尽长期闭环机制。

**reverse control。** 反转一个已确认的候选变量方向；不保证跨 loss family 或 domain 泛化。它能提供方向性因果证据，不能自动统一 hard SFT/SFD 与 soft OPD。

---

## 附录 C：本地证据与实现锚点

- 当前机制总方案：`docs/joint_training/plans/active/qwen3_1p7b_wdl_mechanism_program.md`
- On-Policy SFT baseline 调研：`docs/joint_training/plans/active/qwen3_1p7b_on_policy_sft_baseline_extension.md`
- Math Stage123 / A-B-C-D0 结果：`docs/joint_training/reports/qwen3_1p7b_math_stage123_matrix_results_20260723.md`
- Math 主实验计划：`docs/joint_training/plans/active/qwen3_1p7b_math_stage123.md`
- fused logits 实现：`verl/models/joint_model/modeling_joint_qwen3.py`
- WDL loss 实现：`verl/trainer/ppo/core_algos.py`
- fixed-M1 测试：`tests/on_policy_wdl_sft/test_math_wdl_fixed_m1_p60.py`（代码已准备；无正式 run receipt）
- Dynamic Entropy-Matched Weak-Structure Ablation 飞书原方案：`NEIvdnwU0o0vszxi2wycfcTHnjd`

这些路径记录的是本文创建时的证据位置。实验状态会继续变化；任何“已运行”“已完成”或“已发布”的判断，都应以当时的 scheduler、container/GPU、日志、checkpoint、raw evaluation 和 release-gate 凭据重新核验。

---

## 附录 D：相关工作与思想来源

1. [Weak-Driven Learning: How Weak Agents Make Strong Agents Stronger](https://arxiv.org/abs/2602.08222)：旧 SFT 梯度放大、branch sensitivity 与 weak-driven learning 背景。
2. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)，ICLR 2024：student-generated states 上的显式 teacher feedback；说明 OPD 至少在 2023/2024 已形成正式工作。
3. [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)，ICLR 2024：reverse-KL on-policy distillation，关注 autoregressive sequence distribution mismatch。
4. [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)，2026 preprint：同一个模型以 privileged 与 ordinary context 扮演 teacher/student，并使用显式 token divergence。
5. [Reinforcement Learning via Self-Distillation / SDPO](https://arxiv.org/abs/2601.20802)，2026 preprint：用 feedback-conditioned self-teacher 把 rich feedback 转化为 dense token supervision。
6. [Rethinking On-Policy Self-Distillation for Thinking Models](https://arxiv.org/abs/2607.05184)，2026 preprint：指出 privileged teacher 可能压制 thinking model 的 high-entropy forks、自检与回溯。
7. [Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information](https://arxiv.org/abs/2605.11609)，2026 preprint：通过反向 divergence 检验“更强 teacher agreement 不一定更好”。
8. [On-Policy Self-Distillation without Any Supervision](https://arxiv.org/abs/2608.06296)，2026 preprint：通过 self-consistency 构造 pseudo-solution 与 privileged self-teacher；仍属于显式 distillation。
9. [Born-Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html)，ICML 2018：DKPP 提供“保留 prediction values、置换 class identity”的直接实验先例。
10. [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)，CVPR 2018：两个 peer models 共同训练并使用显式 distribution matching；可作为 co-adaptation 先例，但不是 WDL 的等价 objective。
11. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)，NIPS 2014 Deep Learning Workshop：temperature-softened teacher targets 与 hard-label CE 的经典 knowledge distillation 形式。
12. [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/)，EMNLP 2016：把 teacher-generated sequences 作为 hard sequence targets，提供从 token-distribution KD 到 sequence-level hard distillation 的直接来源。
13. [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)，NeurIPS 2022：生成 rationale、按最终答案筛选、在成功轨迹上 fine-tune，并用 answer-conditioned rationalization 扩大可训练集合；是本文 verifier-filtered reasoning self-training 解释的直接历史源头。
14. [On-Policy Supervised Fine-Tuning for Efficient Reasoning](https://arxiv.org/abs/2602.13407)，2026 preprint：把 correctness-and-length filtered self-generated responses 上的 CE 写成逐 minibatch on-policy SFT，并从简化 GRPO 与推理效率的角度系统化这一训练形式；本文实验 A 覆盖其核心 selected-CE objective，但不是配置完全一致的 exact reproduction。
15. [A Measure of Asymptotic Efficiency for Tests of a Hypothesis Based on the Sum of Observations](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-23/issue-4/A-Measure-of-Asymptotic-Efficiency-for-Tests-of-a-Hypothesis/10.1214/aoms/1177729330.full)，Herman Chernoff，Annals of Mathematical Statistics 1952：Chernoff information 的原始假设检验语境与 error-exponent 来源。
16. A. Bhattacharyya, “On a Measure of Divergence between Two Statistical Populations Defined by Their Probability Distributions,” Bulletin of the Calcutta Mathematical Society 35, 99–109, 1943：Bhattacharyya coefficient/distance 的历史来源。
17. [Revisiting Chernoff Information with Likelihood Ratio Exponential Families](https://arxiv.org/abs/2207.03745)，Entropy 2022：系统区分 skew Bhattacharyya distance、Chernoff information、geometric mixture 与 log-normalizer，本文术语收紧主要参考该文。
18. [Training Products of Experts by Minimizing Contrastive Divergence](https://www.cs.toronto.edu/~fritz/absps/tr00-004.html)，Geoffrey E. Hinton，Neural Computation 2002：将多个 expert distribution 相乘并重新归一化的经典 PoE 工作；本文只借用其 probability geometry，不沿用其训练算法。

2026 年的相关工作在本文创建时按 primary venue record 仍视为 preprint。它们用于提出假设和界定术语，不代表学界已经形成定论。

---

## 附录 E：一句话判读清单

- `C>D0`：说明 weak-logit treatment 值得研究，不说明 WM1 的具体机制。
- `fixed≈joint`：说明 WM1 更新可能不必要，不说明 WM1 forward 不必要。
- `real≈DynPerm`：说明 token identity 证据变弱，不说明 WM1 可删除。
- `real>DynPerm`：说明 assignment 或 cross-model rank geometry 有贡献，不自动说明语义。
- endpoint 沿 M0 预注册的有利方向排序，且 telemetry 同序：说明 rank geometry 是候选因果组件，不说明 endpoint 只由 geometry 决定。
- `real>RankBin` 且 matching 合格：说明已匹配 telemetry 之外仍有 token-specific residual；是否属于语义信息，还需排除 token family、frequency、format 和 position 等因素。
- controller≈C：说明被复制的低维统计可能足够；若用 WM1 traces，只能说无需 online WM1。
- reverse telemetry 与曲线同步反转：机制方向得到较强支持；若只发生梯度爆炸，则只是 stress test。
- D0 on C rollouts 追上：说明 trajectory/data layer 主导局部差异，不说明 fused geometry 永远无效。
- C on shared rollouts 仍领先：说明 objective geometry 有独立贡献，不说明它单独解释长期表现。
