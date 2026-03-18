# Joint MiniRL 中 Logit Mixing 的 Reinforce 理论分析

更新时间: 2026-03-18

## 1. 这份报告回答什么问题

你现在关心的不是“联合训练能不能跑”，而是三个更本质的问题：

1. 当前 `verl` 代码里，强化学习版 joint training 到底在优化什么目标？
2. 如果从最简单的 REINFORCE 出发，logit mixing 能不能像 SFT 一样产生“梯度放大”？
3. 这种机制为什么可能一边带来额外训练信号，一边又带来 `model2-only` 评测退化、重复输出、`extraction_fail` 飙升？

这份文档的核心结论先写在最前面：

- 当前实现里，训练时的 rollout policy 确实是融合策略，不是 `model2-only`；奖励也是对这个融合策略生成出来的 response 打分。
- 因此，训练真正优化的是融合策略的期望回报
  $$
  J_{\text{mix}}(\theta_1,\theta_2)=\mathbb{E}_{y\sim \pi_{\text{mix}}}[R(x,y)],
  $$
  而不是部署时真正关心的
  $$
  J_2(\theta_2)=\mathbb{E}_{y\sim \pi_2}[R(x,y)].
  $$
- 在 REINFORCE / RLOO / GRPO / MiniRL 这一整类“`detach` 的 advantage 乘 `log_prob`”目标下，logit mixing 的梯度结构完全一样：它们只是在 REINFORCE 核心项前面多了一个标量权重，方向仍然是 `e_a - pi_mix`。
- 因此，RL 版确实可以得到 SFT 风格的“去饱和 / 增残差”机制，但这是**条件性的梯度放大**，不是无条件的。
- 更具体地说，mixing 会把强模型已经很尖锐的分布重新拉平，从而增大 `log π` 的梯度残差；但由于 `model2` 的链式系数前面还要乘一个 `λ`，所以是否真的比 `model2` 单独训练时更大，要看“去饱和收益”能不能盖过 `λ` 的缩放。
- RL 和 SFT 最大的差别是：SFT 的目标 token 永远是对的；RL 里被放大的，是“采样到的 token 在当前 advantage 符号下的更新”。正 advantage 时它放大强化，负 advantage 时它放大抑制，同时也会放大方差和坏模式。

如果只用一句话概括：

> 在 joint MiniRL 里，logit mixing 既改变了梯度大小，也改变了 rollout 轨迹分布；所以它不是“在同一批数据上给 model2 更大梯度”这么简单，而是在优化一个和 `model2-only` 部署目标并不相同的融合策略。

---

## 2. 先把代码中的“事实”钉死

这一节不讲理论，先回答“代码到底在干嘛”。

### 2.1 当前 recipe 的具体实验设定

从 `recipe/joint_training/run_joint_minirl_qwen3_1.7b_math.sh` 可以直接读出当前训练脚本的关键设定：

- `MODEL2_PATH` 指向 `/data-1/model_weights/EXP-06_Baseline-MiniRL-1.7B-MATH-GC500/step_680`，也就是已经跑过约 680 step 的强模型。
- `BASE_MODEL_PATH` 是 `Qwen3-1.7B-Base`，作为弱模型。
- joint weights 是通过 `verl.models.joint_model.prepare_joint_weights` 生成的，默认 `fusion_lambda=0.5`。
- 训练打开了 `+actor_rollout_ref.model.joint_training=True`。
- policy loss 用的是 `loss_mode=minirl`。
- advantage estimator 用的是 `adv_estimator=grpo`，并且 `algorithm.norm_adv_by_std_in_grpo=False`。
- reward 端没有 KL reward，也没有 KL loss。
- rollout correction 用的是 token-level truncated IS，阈值 5.0。
- `loss_agg_mode="seq-mean-token-sum"`，也就是**不做 per-token 长度归一化**。

这些都在脚本里有明确配置：

- `recipe/joint_training/run_joint_minirl_qwen3_1.7b_math.sh:72`
- `recipe/joint_training/run_joint_minirl_qwen3_1.7b_math.sh:188-209`
- `recipe/joint_training/run_joint_minirl_qwen3_1.7b_math.sh:302-350`

这意味着当前实验不是“两个完全一样的模型平均一下”，而是“一个 base 弱模型 + 一个已经强化学习过的强模型”的双模型 joint training。你之前直觉里“强弱模型已经存在差异”这一点，在当前 recipe 下是成立的。

### 2.2 joint model 在 HF/FSDP 侧到底怎么做 mixing

HF 侧的 joint model 是 `verl/models/joint_model/modeling_joint_qwen3.py` 里的 `QwenJointForCausalLM`。

它的前向非常直接：

1. 两个 `Qwen3ForCausalLM` 子模型独立 forward。
2. 得到 `logits_0` 和 `logits_1`。
3. 线性融合：
   $$
   z_{\text{mix}}=(1-\lambda)z_1+\lambda z_2.
   $$
4. 后续 loss / log_probs 都基于这个 `z_mix`。

对应代码：

- 初始化两个子模型: `verl/models/joint_model/modeling_joint_qwen3.py:30-35`
- 融合 logits: `verl/models/joint_model/modeling_joint_qwen3.py:91-99`
- `eval_only=True` 时只走 `sub_models[1]`: `verl/models/joint_model/modeling_joint_qwen3.py:55-73`

这一步非常关键：**训练侧 actor 看到的 policy，不是 model1，也不是 model2，而是 softmax 之后的 `pi_mix`。**

### 2.3 actor 训练时用的是融合 logits，不是单独 model2 logits

`dp_actor.py` 里 `_forward_micro_batch()` 最终直接调用 `self.actor_module(...)`，然后从 `output.logits` 里取 response 部分再算 `log_probs`：

- `verl/workers/actor/dp_actor.py:359-377`

由于 `self.actor_module` 在 joint training 下就是 `QwenJointForCausalLM`，所以这里得到的 `log_probs` 已经是融合策略 `pi_mix` 的 `log_probs`。

接下来：

- `compute_log_prob()` 用同一个 actor module 重算 old log probs: `verl/workers/actor/dp_actor.py:483-569`
- `update_policy()` 也是基于同一个 actor module 算当前 `log_prob`: `verl/workers/actor/dp_actor.py:571-749`

这说明：

- old log prob 是融合策略的 old log prob；
- new log prob 是融合策略的 new log prob；
- MiniRL 的 clip ratio 也是围绕融合策略定义的。

### 2.4 vLLM rollout 侧也确实在用融合策略

vLLM 侧的 joint model 在 `verl/models/joint_model/vllm_modeling_joint_qwen3.py`。

它的逻辑和 HF 侧一一对应：

- 正常模式下，`compute_logits()` 返回
  $$
  (1-\lambda)\,\text{logits}_0+\lambda\,\text{logits}_1
  $$
  对应 `verl/models/joint_model/vllm_modeling_joint_qwen3.py:167-180`
- 如果 `_use_model2_only=True`，则只调用 `sub_models[1]`
  对应 `verl/models/joint_model/vllm_modeling_joint_qwen3.py:138-145` 和 `verl/models/joint_model/vllm_modeling_joint_qwen3.py:168-169`

所以 rollout 端和训练端的策略定义是一致的：训练时就是 fused policy。

### 2.5 训练时 reward 是不是对“融合 rollout 的结果”打分？

是的。当前代码路径很清楚：

1. rollout 先生成 response；
2. 然后 reward 才对这些 response 打分；
3. 然后再算 old log probs、advantage、actor update。

训练主循环里顺序如下：

- reward 提取: `verl/trainer/ppo/ray_trainer.py:1668-1675`
- old log prob 计算: `verl/trainer/ppo/ray_trainer.py:1691-1719`
- token rewards / rollout correction / advantage: `verl/trainer/ppo/ray_trainer.py:1739-1784`
- actor 更新: `verl/trainer/ppo/ray_trainer.py:1794-1823`

因此当前训练优化的确实是：

> “融合策略采样出来的轨迹拿去算 reward，再用融合策略的 log_prob 回传梯度”

这正是你怀疑的那种目标，而不是“model2 单独 rollout，但训练时再拿 fusion 去算 loss”。

### 2.6 validation 和 offline eval 怎么切成 model2-only

当前代码库里，这两件事是分开的：

#### 在线 validation

`ray_trainer._validate()` 一进来就判断 joint mode，如果是 joint，就先：

```python
self.checkpoint_manager.update_weights(eval_only=True)
```

对应：

- `verl/trainer/ppo/ray_trainer.py:678-683`

之后 validation 生成阶段就使用 `model2-only` 权重。结束以后再恢复成完整 joint 权重：

- `verl/trainer/ppo/ray_trainer.py:831-847`

在 trainer -> rollout 的同步链路里，`eval_only=True` 会一路传到 FSDP worker，然后在 `rollout_mode()` 里做特殊处理：

- 对 vLLM：从 joint state dict 里抽取 `sub_models.1.*`
  - `verl/workers/fsdp_workers.py:791-800`
  - `verl/models/joint_model/weight_utils.py:25-46`
- 对 HF rollout：直接设置 `_eval_only_mode=True`
  - `verl/workers/fsdp_workers.py:763-771`
  - `verl/models/joint_model/modeling_joint_qwen3.py:58-73`

#### 离线 offline eval

离线评测走的是另外一条链路：

1. 从 joint HF 权重目录里提取 `sub_models.1`
   - `recipe/joint_training/extract_sub_model.py:36-90`
2. 把 config 改回普通 `Qwen3ForCausalLM`
   - `recipe/joint_training/extract_sub_model.py:64-77`
3. 再用 `offline_eval.py` 把提取出的 model2 当普通单模型跑 vLLM
   - `recipe/joint_training/offline_eval.py:153-233`

这意味着 offline eval 完全是 `model2-only`，不再经过 joint model 的融合逻辑。

### 2.7 一个必须点明的核心事实：训练目标和部署目标不一样

把上面的代码事实合并起来，就得到最重要的一句话：

- 训练目标是 `pi_mix`
- 部署 / offline eval 目标是 `pi_2`

也就是：

$$
J_{\text{train}}=J_{\text{mix}}(\theta_1,\theta_2),\qquad
J_{\text{deploy}}=J_2(\theta_2).
$$

这不是“同一个目标的两种实现”，而是**两个不同的优化目标**。

这件事在理论上比“有没有梯度放大”还重要，因为它直接决定了：

- 即使 joint training 对 `pi_mix` 有帮助；
- 它也不一定对 `model2-only` 的最终评测有帮助。

如果只记住本报告一个结论，应该记这个。

### 2.8 一个版本差异提醒

当前代码在 validation 时已经切成 `model2-only`，但 `recipe/joint_training/INFERENCE_RESULTS.md` 中 EVAL-05 的备注写的是：

> step 200 的 training validation 使用的是 fused joint model

见：

- `recipe/joint_training/INFERENCE_RESULTS.md:175-178`

而当前代码路径显示，**截至 2026-03-18 的代码**，`_validate()` 已经明确切成 `eval_only=True`。

这说明至少存在一种可能：

- 要么 EXP-07 当时跑实验时用的是更早的代码版本；
- 要么结果文档的说明没有随着代码更新同步修改。

做理论分析时，我以下都以**当前代码**为准；解释历史实验时，会把这个版本差异当作不确定因素单独指出。

---

## 3. 把论文里的符号和当前代码对齐

你要求参考 `Stabilizing Reinforcement Learning with LLMs`。这一步非常有必要，因为它给出了一个很干净的“sequence objective -> token surrogate”解释框架。

### 3.1 论文中最重要的三件事

论文主张可以概括成三句话：

1. 真正想优化的是 sequence-level reward objective。
2. 实际训练常用的是 token-level surrogate objective。
3. 这个 token-level surrogate 可以被看作 sequence objective 的一阶近似，但前提是 rollout policy 和 training policy 足够接近，同时训练引擎和推理引擎也足够接近。

对应 TeX 源码位置：

- sequence-level reward 设定: `docs/joint_training/references/external/stabilizing_rl_with_llms_source/colm2024_conference.tex:108-145`
- token-level surrogate: `.../colm2024_conference.tex:150-170`
- first-order approximation: `.../colm2024_conference.tex:172-209`
- mismatch 分解为 training-inference discrepancy 和 policy staleness:
  `.../colm2024_conference.tex:213-244`
- MiniRL 公式: `.../colm2024_conference.tex:344-373`

### 3.2 一个非常实用的“符号到代码”映射表

| 论文符号 | 含义 | 当前代码里的对应物 |
|---|---|---|
| \(\mu_{\theta_{\text{old}}}\) | rollout policy | vLLM rollout 的 `rollout_log_probs`，对应 `ray_trainer.py` / `rollout_corr_helper.py` |
| \(\pi_{\theta_{\text{old}}}\) | training engine 里的 old policy | `old_log_probs`，由 `compute_log_prob()` 重算 |
| \(\pi_\theta\) | 当前正在更新的 policy | actor 当前 forward 得到的 `log_prob` |
| \(R(x,y)\) | sequence reward | `reward_tensor` / `token_level_scores` |
| \(\widehat A(x,y)\) | advantage | `advantages` |
| \(M_t\) | MiniRL 的二值 clip mask | `core_algos.py` 里 `mask[(advantages > 0) & ...] = 0` |
| rollout IS | training-inference mismatch 修正 | `rollout_is_weights` |

关键代码：

- old/new log prob 和 reward/advantage 串起来的主循环:
  `verl/trainer/ppo/ray_trainer.py:1677-1784`
- MiniRL loss:
  `verl/trainer/ppo/core_algos.py:1782-1841`
- rollout correction:
  `verl/trainer/ppo/rollout_corr_helper.py:717-831` 和 `943-997`

### 3.3 当前 recipe 在论文框架下属于哪一类

当前 recipe 不是纯 REINFORCE，而是：

- REINFORCE-style `log_prob` loss
- GRPO-style outcome advantage
- MiniRL-style binary clip mask
- token-level rollout IS correction

如果把一个 token 的 loss 写成统一形式，就是：

$$
\ell_t = - c_t \,\widehat A_t \,\log \pi_\theta(y_t\mid h_t),
$$

其中

- 纯 REINFORCE 时，\(c_t=1\)
- MiniRL 时，\(c_t=M_t \bar\rho_t\)
- RLOO / GRPO / REINFORCE++ 的差别主要体现在 \(\widehat A_t\) 怎么构造

只要 \(c_t\) 和 \(\widehat A_t\) 在 backward 时被当成 `detach` 的标量，这些算法在**logit 层面的梯度方向**是同一个模板。

而当前代码里这一点正好成立：

- `advantages.detach()`: `verl/trainer/ppo/core_algos.py:1818-1824`
- `mask = mask.detach()`: `verl/trainer/ppo/core_algos.py:1813-1816`
- `rollout_is_weights.detach()`: `verl/trainer/ppo/core_algos.py:1822-1824`

这也是为什么“先从 REINFORCE 分析”是正确入口。

---

## 4. 从最简单的 REINFORCE 开始推导

这一节是整份文档的数学核心。

### 4.1 先定义 joint policy

记：

- \(z_1(h_t)\): 弱模型在前缀 \(h_t=(x,y_{<t})\) 上的 logits
- \(z_2(h_t)\): 强模型在前缀 \(h_t\) 上的 logits
- \(z_{\text{mix}}(h_t)=(1-\lambda)z_1(h_t)+\lambda z_2(h_t)\)
- \(\pi_{\text{mix}}(\cdot\mid h_t)=\mathrm{softmax}(z_{\text{mix}}(h_t))\)

当前 recipe 下，\(\lambda=0.5\)。

### 4.2 单个 token 的 REINFORCE loss

先考虑最干净的 REINFORCE 情形。对于一条 rollout 轨迹中第 \(t\) 个 token，采样动作记为 \(a_t\)。

单 token loss 写成：

$$
\ell_t = - A_t \log \pi_{\text{mix}}(a_t\mid h_t),
$$

其中 \(A_t\) 可以是：

- 纯 sequence-level reward
- reward 减 baseline
- RLOO advantage
- GRPO advantage

对下面的 logit 推导来说，唯一重要的是：**它在 backward 时是一个标量系数，不继续对 logits 求导。**

### 4.3 对融合 logits 的梯度

softmax 的标准恒等式是：

$$
\frac{\partial \log \pi(a)}{\partial z} = e_a - \pi,
$$

所以

$$
\frac{\partial \ell_t}{\partial z_{\text{mix}}}
=
-A_t\,(e_{a_t}-\pi_{\text{mix}}).
$$

这一步很重要，因为它告诉你：

> 在 logit 层，REINFORCE 的结构和 SFT 的 cross-entropy 几乎一模一样，只不过 SFT 里的“目标 one-hot”是真实标签 \(e_y\)，这里变成了“当前采样到的 token 的 one-hot” \(e_{a_t}\)，同时前面多了一个 advantage 符号。

### 4.4 通过 chain rule 分到两个子模型

由于

$$
z_{\text{mix}}=(1-\lambda)z_1+\lambda z_2,
$$

所以直接有：

$$
\frac{\partial z_{\text{mix}}}{\partial z_1}=1-\lambda,\qquad
\frac{\partial z_{\text{mix}}}{\partial z_2}=\lambda.
$$

于是：

$$
\frac{\partial \ell_t}{\partial z_1}
=
-(1-\lambda)A_t\,(e_{a_t}-\pi_{\text{mix}}),
$$

$$
\frac{\partial \ell_t}{\partial z_2}
=
-\lambda A_t\,(e_{a_t}-\pi_{\text{mix}}).
$$

如果你把 MiniRL / RLOO / GRPO 一起包进来，只要把前面的 detached 标量系数写成 \(c_t\)，那么统一公式就是：

$$
\ell_t = - c_t A_t \log \pi_{\text{mix}}(a_t\mid h_t),
$$

$$
\frac{\partial \ell_t}{\partial z_i}
=
- s_i\,c_t A_t\,(e_{a_t}-\pi_{\text{mix}}),
\qquad
s_1=1-\lambda,\ s_2=\lambda.
$$

这就是本报告最关键的公式。

它的含义非常直接：

- joint training 并没有创造一种全新梯度；
- 它只是把“融合策略的 REINFORCE 残差”按 \((1-\lambda)\) 和 \(\lambda\) 分配给两个子模型。

### 4.5 如果进一步看参数空间的一阶近似

如果像你之前做 SFT 分析那样，再做一次局部线性化：

$$
z_i(\theta_i+\Delta\theta_i)\approx z_i(\theta_i)+J_i\Delta\theta_i,
$$

定义 \(K_i=J_iJ_i^\top\)，则一阶 logit 更新近似为：

$$
\Delta z_i
\approx
\eta\, s_i\, c_t A_t\, K_i (e_{a_t}-\pi_{\text{mix}}).
$$

这一步的意义是：

- 上一节给的是“loss 对 logits 的梯度”
- 这一节给的是“参数更新最终在 logits 空间产生的有效位移”

如果 `model2` 的 \(K_2\) 沿当前残差方向更“敏感”，那么即使两个模型吃到的是同一个 fused residual，它也可能在有效更新上占主导。这一点和你 SFT 里分析过的“strong model directional sensitivity 更大”是同一条思路。

---

## 5. 在 REINFORCE 情形下，能不能得到“梯度放大”？

答案分成两层：

- 对**融合策略自己的残差**来说，可以。
- 对**model2 相比于单独训练 strong model 的梯度幅度**来说，只能得到条件性的结论，不能得到无条件“必然放大”。

### 5.1 什么叫“放大”

如果拿 `model2` 单独训练时的 REINFORCE 梯度作比较，那么 strong-only 的单 token 梯度是：

$$
\frac{\partial \ell_t^{\text{solo}}}{\partial z_2}
=
-A_t (e_{a_t}-\pi_2).
$$

joint training 下 `model2` 的对应梯度是：

$$
\frac{\partial \ell_t^{\text{joint}}}{\partial z_2}
=
-\lambda A_t (e_{a_t}-\pi_{\text{mix}}).
$$

所以你真正想比较的是：

$$
\lambda (e_{a_t}-\pi_{\text{mix}})
\quad \text{vs.} \quad
(e_{a_t}-\pi_2).
$$

### 5.2 对选中 token 的“上推”是否放大

对采样到的 token \(a_t\)，梯度绝对值由 \(1-\pi(a_t)\) 决定。

于是 `model2` 的 joint-vs-solo 放大条件是：

$$
\lambda \big(1-\pi_{\text{mix}}(a_t)\big)
>
1-\pi_2(a_t).
$$

这是什么意思？

- 如果强模型 `model2` 已经很饱和，\(\pi_2(a_t)\) 非常接近 1；
- 而弱模型把这个 token 的置信度明显拉低；
- 那么 \(1-\pi_{\text{mix}}(a_t)\) 会比 \(1-\pi_2(a_t)\) 大很多；
- 只要这个“去饱和增量”足够大，就能盖过前面的 \(\lambda\) 缩放。

所以：

> RL 版的“梯度放大”本质上仍然是去饱和：强模型太自信时，弱模型把 fused policy 拉回更高熵区域，从而重新放大 `log π` 的梯度残差。

### 5.3 对 hard negative 的“下压”是否放大

对任意负 token \(k\neq a_t\)，梯度绝对值由 \(\pi(k)\) 决定。

joint-vs-solo 放大条件变成：

$$
\lambda \pi_{\text{mix}}(k) > \pi_2(k).
$$

如果弱模型在某个 hard negative 上的概率明显高于强模型，那么 mixing 会提高 \(\pi_{\text{mix}}(k)\)。一旦提高得足够多，就会让 `model2` 在这个 negative 上吃到更大的 suppress gradient。

这就是 SFT 里“hard-negative amplification”的直接 RL 对应物。

### 5.4 为什么这是“条件性结论”，不是无条件定理

因为前面始终有一个 \(\lambda\)。

对于当前 recipe，\(\lambda=0.5\)。这意味着如果拿 `model2` 和“单独 strong model 训练”对比，想要真正放大，必须让 fused residual 相关坐标增大到**超过两倍**左右的量级，才可能盖过 `0.5` 这个缩放。

所以严谨的表述应该是：

> 在 REINFORCE 情形下，joint logit mixing 可以产生“残差放大 / 去饱和放大”，从而为 `model2` 提供更大的有效梯度；但这只在强模型已经饱和、弱模型又足够分歧的区间里成立，并不是对所有 token、所有阶段都成立。

这和你之前对 SFT 的理解是兼容的，但比 SFT 更弱一些，因为 RL 这里额外多了一个 \(\lambda\) 和 advantage 的随机性。

### 5.5 和 SFT 的根本差别

SFT 时，one-hot target 是真实标签，方向永远是“把正确 token 往上、错误 token 往下”。

RL 时，one-hot target 是“当前采样到的 token”：

- \(A_t>0\) 时，强化它；
- \(A_t<0\) 时，抑制它。

所以 joint mixing 带来的不是“总是有利的额外梯度”，而是：

- 放大有用轨迹时，确实是额外信号；
- 放大坏轨迹时，也会更强烈地把策略往相反方向推；
- 同时 advantage 噪声也会被放大。

因此在 RL 里，更准确的说法不是“梯度放大一定提升性能”，而是：

> joint mixing 提供了一种**高熵化 / 去饱和化**的梯度机制，它既可能帮助 model2 走出饱和区，也可能放大 rollout 噪声和错误轨迹带来的更新。

---

## 6. 为什么这个结论可以直接推广到 MiniRL、GRPO、RLOO

这部分其实非常干净。

### 6.1 MiniRL 的代码形式

当前 `verl` 里的 MiniRL loss 是：

$$
\ell_t^{\text{MiniRL}}
=
- M_t \,\widehat A_t \,\log \pi_\theta(y_t\mid h_t),
$$

如果开启 rollout correction，就再乘一个 detached 的 `rollout_is_weights`：

$$
\ell_t^{\text{MiniRL}}
=
- M_t \,\bar\rho_t\, \widehat A_t \,\log \pi_\theta(y_t\mid h_t).
$$

代码对应：

- ratio 和 binary mask: `verl/trainer/ppo/core_algos.py:1806-1816`
- `pg_losses = -mask * advantages.detach() * log_prob`: `.../core_algos.py:1818-1820`
- rollout IS correction: `.../core_algos.py:1822-1824`

所以在 joint model 下：

$$
\frac{\partial \ell_t^{\text{MiniRL}}}{\partial z_i}
=
- s_i\,M_t\,\bar\rho_t\,\widehat A_t\,(e_{a_t}-\pi_{\text{mix}}).
$$

### 6.2 这意味着什么

MiniRL 相对 REINFORCE 只多了两个东西：

1. 一个 detached 的二值门 \(M_t\)
2. 一个 detached 的 rollout IS 权重 \(\bar\rho_t\)

它们都只是标量系数，不改变 logit 梯度的基本方向结构。

所以：

- 你用 REINFORCE 推出来的 mixing 结论；
- 对 MiniRL 仍然成立。

### 6.3 GRPO / RLOO 改的是 baseline，不改 mixing 的链式结构

`compute_advantage()` 里可以看到：

- `GRPO` 是 group mean baseline: `verl/trainer/ppo/ray_trainer.py:283-295`
- `RLOO` 是 leave-one-out baseline: `verl/trainer/ppo/core_algos.py:476-520`

但只要这些 advantage 在 backward 时被当成 detached 标量，joint mixing 的链式法则就不变。

也就是说：

- REINFORCE / RLOO / GRPO 的差别，主要在 advantage 方差和偏差；
- joint logit mixing 的差别，主要在 policy residual `e_a - pi_mix` 和 rollout occupancy。

这两件事是可以解耦分析的。

所以对你的原问题，可以给一个很明确的回答：

> 是的，在“更接近 RLOO / REINFORCE”的情形下，joint training 的 logit-level 推导更干净，而且结论能直接迁移到当前 MiniRL recipe，因为当前 MiniRL 的 policy gradient 核心就是一个乘了 detached 标量的 REINFORCE 结构。

---

## 7. joint RL 的真正机制，不只是“放大梯度”

如果只盯着单 token 梯度公式，你会漏掉最重要的 RL 现象：**rollout 分布本身被 joint policy 改了。**

### 7.1 SFT 和 RL 的本质差异

SFT 时，训练数据是固定的。joint mixing 只改变：

- 同一条监督样本上的 residual
- 同一条样本上的梯度分配

RL 时，joint mixing 还会改变：

- 采样到什么 response
- 访问到哪些 prefix
- 后续 reward / advantage 的统计分布

也就是说，joint RL 同时有两个效应：

1. **estimator effect**: 在给定 sampled token 的情况下，`e_a - pi_mix` 和 `lambda` 改变了梯度。
2. **occupancy effect**: `pi_mix` 改变了 rollout 轨迹分布，从而改变了“你到底在训练哪些轨迹”。

第二件事在 RL 里经常比第一件事更强。

### 7.2 一个更接近 policy gradient 本质的写法

对某个前缀 \(h_t\)，定义 action-value：

$$
Q_{\text{mix}}(h_t,a)
=
\mathbb E[R(x,y)\mid h_t,\ a_t=a].
$$

那么对融合 logits 的期望梯度可以写成：

$$
\nabla_{z_{\text{mix}}(h_t)} J_{\text{mix}}
=
\sum_a \pi_{\text{mix}}(a\mid h_t)\,Q_{\text{mix}}(h_t,a)\,(e_a-\pi_{\text{mix}}).
$$

于是：

$$
\nabla_{z_i(h_t)} J_{\text{mix}}
=
s_i \nabla_{z_{\text{mix}}(h_t)} J_{\text{mix}}.
$$

这个式子揭示了 joint RL 的两层机制：

- 残差项 `(e_a - pi_mix)` 在变；
- 但更重要的是，前面的采样分布 \(\pi_{\text{mix}}\) 和 action-value \(Q_{\text{mix}}\) 也在变。

所以 weak model 不是只在“帮 model2 放大梯度”，它还在“帮整个 ensemble 改变探索方向”。

### 7.3 这也解释了为什么训练目标和部署目标会错位

当前训练优化的是：

$$
J_{\text{mix}}(\theta_1,\theta_2)
=
\mathbb E_{y\sim \pi_{\text{mix}}}[R(x,y)].
$$

部署时真正评估的是：

$$
J_2(\theta_2)
=
\mathbb E_{y\sim \pi_2}[R(x,y)].
$$

除非以下条件近似成立：

- \(\lambda \approx 1\)
- 或者 \(\pi_1 \approx \pi_2\)
- 或者训练中显式把 `model2-only` 目标也纳入优化

否则：

$$
\nabla_{\theta_2}J_{\text{mix}} \neq \nabla_{\theta_2}J_2.
$$

所以 joint training 的收益能不能转移到 `model2-only`，不是数学上自动成立的。

这正是当前实验里最需要警惕的地方。

---

## 8. 用这个框架解释你观察到的现象

你观察到的几个现象是：

- `model2` 在末尾重复输出；
- `extraction_fail` 明显升高；
- 训练过程中 `jointTraining/answer_extraction_failure_rate` 从某一处开始快速抬升；
- 最终 offline eval 的 `model2-only` 表现没有超过训练前的强模型。

这些现象用上面的框架都能解释，而且彼此之间是一致的。

### 8.1 现象一：为什么可能出现重复输出 / EOS 被抑制

你之前就提出了一个很对的问题：

> 如果 model1 很想输出 EOS，而 model2 不想，或者反过来，fused logits 会不会把 EOS 的概率压下去？

答案是：当然会。

因为 rollout 时真正采样的是：

$$
\pi_{\text{mix}}(\text{EOS}\mid h_t).
$$

如果在某些前缀上：

- `model2` 倾向于收尾；
- `model1` 倾向于继续生成；

那 fusion 完全可能把 EOS 概率压低，从而让 rollout 进入“继续写”的轨道。

在当前 recipe 里，这个效应又会被两个因素放大：

1. 训练 reward 是对 fused rollout 的结果打分，而不是对 `model2-only` 轨迹打分；
2. `loss_agg_mode="seq-mean-token-sum"` 不做 per-token 长度归一化，正 advantage 的长回答会积累更多 token-sum 梯度。

`agg_loss()` 的实现里，`seq-mean-token-sum` 是“先 token-sum，再按序列平均”：

- `verl/trainer/ppo/core_algos.py:1059-1065`

这意味着：

- 一个更长的正样本，会给更多 token 贡献正梯度；
- 如果 fused policy 把 EOS 延后，长回复的训练影响会更大；
- 这会形成“继续写”方向的正反馈。

所以“末尾重复输出”并不一定是一个独立 bug，也可能是：

> fused rollout 让策略进入了延迟 EOS 的 occupancy region，而 MiniRL 的 token-sum 聚合又把这类长轨迹的梯度放大了。

### 8.2 现象二：为什么 `extraction_fail` 会飙升

当前 validation 里，`answer_extraction_failure_rate` 的定义就是：

- 预测结果被解析成 `[NO_BOXED]` 的比例

对应：

- `verl/trainer/ppo/ray_trainer.py:893-898`

而 `INFERENCE_RESULTS.md` 里，EXP-07 的 `model2-only` offline eval 确实出现了显著更高的 extraction_fail：

- MATH-500: 16.2%
- AIME-2025: 61.3%
- AMC-2023: 34.1%
- OlympiadBench: 38.1%

见：

- `recipe/joint_training/INFERENCE_RESULTS.md:163-180`

这件事从 joint RL 的视角看，很像下面这条因果链：

1. 融合策略在 rollout 时改变了“格式 token / 收尾 token / `\boxed{}` 相关 token”的概率。
2. reward 是 sequence-level 的，credit assignment 很粗；一个回答即使格式差，只要 reasoning 部分偶尔还行，也不一定立刻收到足够强的惩罚。
3. MiniRL / GRPO 把同一个 sequence advantage 广播到很多 token 上，早期格式 token 的责任会被稀释。
4. 两个模型都在学习一个“ensemble 才成立”的行为分布。
5. 最后把 model1 拿掉后，`model2-only` 失去 ensemble 的补偿，格式脆弱性就暴露出来了。

更简单地说：

> 训练时能拿到 reward，不等于 `model2` 学会了“单独把答案格式输出正确”。

### 8.3 现象三：为什么 joint training 后 `model2-only` 没有超过原强模型

从 `INFERENCE_RESULTS.md` 看，EXP-06 step 680 的 baseline 强模型在 offline eval 上仍然整体更强，而 EXP-07 的 final `model2-only` 没有设新高：

- baseline step 680 的 MATH-500 `mean@8 = 66.9%`
- joint final `model2-only` 的 MATH-500 `mean@8 = 66.1%`

见：

- `recipe/joint_training/INFERENCE_RESULTS.md:129-146`
- `recipe/joint_training/INFERENCE_RESULTS.md:163-180`

这和本文前面的主结论是完全一致的：

- joint training 优化的是 `J_mix`
- 你最终关心的是 `J_2`

所以即使 joint training 真的在某些阶段通过去饱和获得了更好的训练信号，也完全可能：

- 对 ensemble 有利；
- 对 `model2-only` 无利，甚至有害。

因此当前结果并不反驳“存在条件性梯度放大”，它更像是在说：

> 梯度放大这件事即使存在，也还没有压过“训练目标和部署目标不一致”这件更大的问题。

---

## 9. 用阶段动力学来理解 joint RL

如果把训练想成一个动态过程，可以分成四个阶段理解。

### 阶段 A：初始差异阶段

当前 recipe 不是双克隆，而是：

- model1 = base 弱模型
- model2 = step680 强模型

所以一开始 \(\pi_1\) 和 \(\pi_2\) 就已经存在明显差异。

这意味着：

- mixing 一开始不是“几乎等于 model2”；
- 它确实会改变 rollout 行为和梯度残差。

这也是 joint training 真正有东西可学的前提。

### 阶段 B：去饱和与探索扩张阶段

强模型如果已经在很多 token 上过于自信，它的 solo REINFORCE 梯度会偏小。弱模型的存在可以：

- 降低 chosen token 的 fused confidence；
- 增加若干 hard negatives 的 fused probability；
- 让 rollout 访问到一些强模型自己不太会走的轨迹。

这是 joint training 最可能带来收益的阶段。

如果只从局部梯度上看，这就是“去饱和放大”。

如果从 RL 轨迹分布上看，这就是“扩张 exploration support”。

### 阶段 C：协同适配阶段

随着训练继续，两个模型都在更新，于是会出现一个新的现象：

- model2 学到的，不一定是“单独最优策略”；
- 它可能是在学“如何和 model1 一起组成一个高 reward 的 ensemble policy”。

这一阶段最容易出现的风险是：

- reasoning token 是 jointly 学到的；
- 格式 token / EOS token / 结束行为也是 jointly 学到的；
- 但最后部署时只保留 model2。

于是很多“ensemble 才成立”的行为，在 `model2-only` 下就塌了。

### 阶段 D：部署落差显性化阶段

当你做 `model2-only` offline eval 时，本质上是在问：

> 把训练时的协作伙伴 model1 删除之后，model2 还剩下什么？

如果它剩下的是：

- reasoning 还可以；
- 但 termination 和 formatting 崩了；

那表现出来就正是你看到的：

- 有些 benchmark 的 raw reasoning 可能没彻底坏；
- 但 `extraction_fail` 很高；
- pass / maj 等最终指标被明显拖累。

这和当前 EXP-07 的结果是高度一致的。

---

## 10. 对你两个核心问题的正式回答

### 问题一：在 REINFORCE 情形下，能不能得到梯度放大的结论？

正式回答：

可以，但只能得到**条件性的、局部的、相对于强模型饱和区的梯度放大结论**。

更精确地说：

1. 在 fused policy 层面，弱模型确实可以通过拉平强模型分布来增大 `e_a - pi_mix` 的残差。
2. 对 `model2` 自己而言，梯度前面还会乘一个 `λ`，所以不是必然放大。
3. 真正的放大条件是：
   $$
   \lambda(1-\pi_{\text{mix}}(a_t)) > 1-\pi_2(a_t)
   $$
   或
   $$
   \lambda\pi_{\text{mix}}(k)>\pi_2(k).
   $$
4. 这在“强模型已饱和、弱模型又足够分歧”的区域是可能成立的。
5. 因此 joint logit mixing 在 RL 中保留了 SFT 的“去饱和放大”机制，但比 SFT 更脆弱，因为多了 `λ` 衰减和 advantage 噪声。

### 问题二：在 REINFORCE 情形下，联合训练的机理和动力学行为是什么？

正式回答：

它有三层同时发生的机制：

1. **梯度残差机制**
   - `e_a - pi_mix` 变了，强模型被去饱和，局部梯度可能变大。

2. **轨迹分布机制**
   - rollout 用的是融合策略，所以 joint training 改变了采样到的 response 分布，而不仅仅是“同一 response 上的梯度大小”。

3. **目标错位机制**
   - 训练优化的是 `J_mix`，部署评估的是 `J_2`。
   - 这意味着 model2 可能学会“在 ensemble 中好用”，却不一定学会“单独部署时也好用”。

如果用一句最凝练的话说：

> REINFORCE 下的 joint logit mixing，本质上是在优化一个由弱模型参与定义的 ensemble policy；它既能通过去饱和带来额外训练信号，也会通过改变 rollout occupancy 和部署目标，带来 `model2-only` 的评测落差。

---

## 11. 这对当前项目意味着什么

基于当前代码和你提供的实验现象，我认为现在最应该接受的判断是：

### 11.1 可以接受的判断

- 当前 joint MiniRL 训练确实在数学上存在“REINFORCE 版的去饱和梯度放大”机制。
- 这个机制不是幻觉，也不是只能在 SFT 里成立。
- 但它不是当前实验结果的决定性正面因素，因为它要和一个更大的负面因素竞争：
  - `J_mix` 和 `J_2` 的目标不一致。

### 11.2 更可能的当前主导矛盾

就当前 EXP-07 的表现看，更像是下面这个矛盾在主导：

> joint training 确实改变了 rollout 和梯度，但 model2 最终学到的更像是“和 model1 协作的策略”，而不是“单独部署时稳健的策略”。

这能同时解释：

- 为什么 reasoning 指标没有全面崩掉；
- 为什么 `extraction_fail` 却大幅变差；
- 为什么会出现重复输出和收尾问题；
- 为什么最终 `model2-only` 没有稳定超过训练前的强模型。

### 11.3 一个基础设施层面的额外提醒

如果未来切换到 colocated reward model，要特别注意 `_validate()` 里有一条路径会在 reward 计算后调用普通的 `update_weights()`，这可能把 `model2-only` validation 中途切回 full joint weights：

- `verl/trainer/ppo/ray_trainer.py:750-760`

在当前这份报告的主结论里，这不是核心问题，但它是一个值得记住的 infra caveat。

---

## 12. 下一步最值得做的验证

如果你接下来要继续推进，我建议优先验证下面五件事。它们都不是“泛泛建议”，而是直接对应本文的理论结论。

### 12.1 直接测 `pi_mix` 和 `pi_2` 的 gap

在同一批 prefix 上同时记录：

- EOS token log prob gap
- `\boxed` 相关 token 的 log prob gap
- top-1 token 是否一致
- KL\((\pi_2 \| \pi_{\text{mix}})\) 或 KL\((\pi_{\text{mix}} \| \pi_2)\)

如果这些 gap 在训练中变大，那么 `model2-only` 部署退化就不是偶然，而是目标错位的直接证据。

### 12.2 单独监控 response length 分布

如果 joint training 真的在压低 EOS，你会看到：

- 平均 response length 上升
- 长尾显著变重
- repetition 主要发生在长回答末尾

这和 `seq-mean-token-sum` 的 token-sum 聚合是相互强化的。

### 12.3 同时评估 fused policy 和 model2-only policy

在相同 validation prompt 上同时做：

- fused rollout + reward
- model2-only rollout + reward

如果 fused 明显优于 model2-only，那么就说明当前训练更多是在优化 `J_mix` 而不是 `J_2`。

### 12.4 做 `lambda` sweep

如果本文分析是对的，那么：

- \(\lambda\) 太小：model2 吃到的梯度缩放太强，部署目标也更偏
- \(\lambda\) 太大：弱模型几乎不起作用，去饱和收益消失

所以应该存在一个 tradeoff，而不是“0.5 必然最好”。

### 12.5 区分“reasoning 退化”和“formatting 退化”

当前 `extraction_fail` 非常高，很容易掩盖真实 reasoning 水平。

最值得做的是把错误拆成两类：

1. reasoning 本身错
2. reasoning 可能对，但 boxed/termination/format 错

否则你很难判断 joint training 到底是在“学坏思维过程”，还是“学坏输出接口”。

---

## 13. 最后的结论

把整份报告压缩成最终版本，就是下面四句话：

1. 当前 `verl` 的 joint MiniRL 实现，训练时确实使用融合策略 rollout，并对融合 rollout 的结果计算 reward。
2. 从 REINFORCE 出发，joint logit mixing 的梯度可以严格写成
   $$
   -s_i c_t A_t (e_{a_t}-\pi_{\text{mix}})
   $$
   ，所以它确实保留了 SFT 式的“去饱和 / 增残差”机制。
3. 但对 `model2` 而言，这个“梯度放大”只是条件成立，因为它始终要和前面的 \(\lambda\) 缩放竞争。
4. 当前实验里更大的问题不是“有没有放大”，而是训练目标 `J_mix` 和部署目标 `J_2` 不一致，这很可能正是重复输出、EOS/格式脆弱、`extraction_fail` 升高、`model2-only` 评测不升反降的根源。

如果后续你愿意，我可以继续把这份报告往前推一步，补成一版更正式的“定理 + 命题 + 推论”风格，或者直接基于这份报告再写一份“该怎么改算法”的设计建议稿。
