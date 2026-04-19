# WDL-SFT-IS Spec — v1 vs v2 Loss Comparison

- Status: **SPEC (ACTIVE)** — 对应 plan: `docs/joint_training/plans/active/wdl_sft_is.md`
- 创建日期：2026-04-19
- 两个版本在代码库中并存；past experiments（EXP-12~15）锁定 v1，future experiments 选 `loss_mode=wdl_sft_is` 走 v2

本文档只对比 **loss 函数层**的差异。Weak-Driven rollout（fused sampling，λ=0.5）和 joint model 结构在 v1 和 v2 里完全一致。

---

## 1. 形式对比

### v1：`compute_wdl_sft_loss` (`core_algos.py:1920`)

$$
L^+_{\text{v1}} = -\frac{1}{|C|} \sum_{i \in C} \sum_t m^i_t \log P_\theta(y^i_t \mid x, y^i_{<t})
$$

$$
L^-_{\text{v1}} = \frac{1}{|I|} \sum_{j \in I} \sum_t m^j_t \log P_\theta(y^j_t \mid x, y^j_{<t})
$$

$$
L_{\text{v1}} = L^+_{\text{v1}} + \beta \cdot L^-_{\text{v1}}
$$

- $C$ = reward +1 集合，$I$ = reward -1 集合
- $m^i_t$ = response mask（padding=0，真 token=1）
- **不使用** `old_log_prob`、`rollout_is_weights`
- **无 clipping**

### v2：`compute_policy_loss_wdl_sft_is`（新增）

$$
r^i_t = \exp(\text{clamp}(\log P_\theta(y^i_t) - \log P_{\theta_{\text{old}}}(y^i_t),\; -20,\; 20))
$$

**正样本 mask**（上界）：$M^+_{i,t} = \mathbb{1}[i \in C] \cdot \mathbb{1}[r^i_t \leq 1 + \varepsilon_{\text{high}}]$

**负样本 mask**（下界）：$M^-_{j,t} = \mathbb{1}[j \in I] \cdot \mathbb{1}[r^j_t \geq 1 - \varepsilon_{\text{low}}]$

$$
L^+_{\text{v2}} = -\frac{1}{|C|} \sum_i \sum_t M^+_{i,t} \cdot m^i_t \cdot w_{i,t} \cdot \log P_\theta(y^i_t)
$$

$$
L^-_{\text{v2}} = +\frac{1}{|I|} \sum_j \sum_t M^-_{j,t} \cdot m^j_t \cdot w_{j,t} \cdot \log P_\theta(y^j_t)
$$

$$
L_{\text{v2}} = L^+_{\text{v2}} + \beta \cdot L^-_{\text{v2}}
$$

- $w_{i,t}$ = `rollout_is_weights`（vLLM↔FSDP 修正），若未启用为 1
- $\varepsilon_{\text{low}}=0.2,\; \varepsilon_{\text{high}}=0.27$（与 actor config 一致）
- Mask 是 binary（0/1），被 clip 的 token 完全不回传梯度（MiniRL 风格，非 PPO min-clip）
- Advantage 被 detach，gradient 只经过 log_prob

## 2. 关键差异（逐项）

| | v1 | v2 |
|---|---|---|
| `old_log_prob` | 忽略 | 用于计算 ratio |
| Ratio clip | 无 | 正样本上界 `1+ε_high`，负样本下界 `1-ε_low` |
| Clip 类型 | — | Binary mask (0/1)，被 clip token 零梯度 |
| `rollout_is_weights` | 忽略 | 乘进 token-level loss |
| Advantage | 用于筛选 C/I | 用于筛选 C/I + detached |
| Per-group 归一化 $1/k$, $1/(N{-}k)$ | 有 | 有 |
| 聚合模式 | `seq-mean-token-sum` | `seq-mean-token-sum`（暂保持一致以做 A/B） |
| `β=0` 边界条件 | k=0 时跳过整个 prompt | k=0 时 L+ = 0，L- 仍正常 |

## 3. 对三个 π_old ≠ π_new 来源的覆盖

| 来源 | v1 处理 | v2 处理 |
|---|---|---|
| A. 一次 rollout → 多个 mini-batch 更新 | ❌ 无修正 | ✅ IS ratio clip |
| B. vLLM ≠ FSDP 数值差异 | ❌ 无修正 | ✅ `rollout_is_weights` |
| C. Fused P_mix vs per-submodel P_θ | — (算法设计意图) | — (算法设计意图) |

C 不修复是 feature 不是 bug——WDL-SFT 的核心就是用 fused 分布采样来放大 model2 的梯度，model1 是牺牲品。

## 4. 数值一致性保证

**v2 在 `old_log_prob == log_prob` 且 `rollout_is_weights=None` 且 β=0 时，应与 v1 在数值上一致。**

这是 v1 ↔ v2 的 sanity check：等价于"没有任何 IS 信号进入时，v2 退化为 v1"。实现时写单元测试强制 `torch.allclose(loss_v1, loss_v2, rtol=1e-5)` 通过。

例外：v1 在 k=0 时 early return zero；v2 在 k=0 时 L+=0 但仍计算 L-。在 β=0 且 k=0 时两者都应返回 0，这个边界要显式测。

## 5. 代码位置

- v1 代码：
  - Wrapper: `verl/trainer/ppo/core_algos.py:1861`
  - Core:    `verl/trainer/ppo/core_algos.py:1920`
- v2 代码（待实现）：
  - 同文件内新增 `compute_policy_loss_wdl_sft_is`，registered as `wdl_sft_is`
- 参考实现：
  - MiniRL binary mask: `verl/trainer/ppo/core_algos.py:1782-1841`
  - Vanilla PPO clip: `verl/trainer/ppo/core_algos.py:1159-1250`

## 6. 不在本 spec 范围内的后续问题

- 长度归一化（`seq-mean-token-sum` vs `token-mean`）：v2 保留 v1 的聚合方式做干净 A/B，之后如果仍有长度漂移再单独改
- Model2-only online validation：算法逻辑之外的 infra 问题，见 plan §5
- KL 锚正则：暂不引入，防止一次加太多变量
