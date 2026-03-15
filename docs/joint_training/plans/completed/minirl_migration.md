# Plan: MiniRL Loss + MATH Dataset Migration for 1.7B Joint Training

**状态**：已完成 ✅
**日期**：2026-03-15
**完成日期**：2026-03-15
**前置**：Stage 2 进行中；依赖 MiniRL vs GRPO 对比分析报告

## Context

当前 1.7B 联合训练脚本 (`run_joint_grpo_qwen3_1.7b.sh`) 使用标准 GRPO + vanilla PPO loss，在 GSM8K 上训练。存在以下问题：
1. 无训推不一致的重要性采样修正（vLLM BF16 vs FSDP FP32）
2. PPO 式 ratio-based 梯度流不够稳定
3. GRPO z-score 归一化在 σ=0 时有数值问题
4. 数据集需从 GSM8K 升级为 MATH（更难），验证需同时评估 MATH-500 和 AIME-2025

目标：基于 MiniRL 论文（Qwen Team, COLM 2024），在 CISPO 基础上实现 MiniRL 损失函数，并配置新的 1.7B MATH 训练脚本。

---

## 关键发现（已探索确认）

| 组件 | 状态 | 说明 |
|------|------|------|
| Token 级 IS | **已实现** | `rollout_is="token"` 一行配置切换 |
| Dr.GRPO 优势估计 | **已实现** | `algorithm.norm_adv_by_std_in_grpo=False` |
| 多数据集 Validation | **已原生支持** | `data.val_files` 接受列表，按 `data_source` 分组报告 |
| MiniRL 损失函数 | **已实现** ✅ | `@register_policy_loss("minirl")` in `core_algos.py`，10/10 测试通过 |
| `seq-mean-token-sum` 聚合 | **已实现** | MiniRL 不做 per-token 长度归一化 |

---

## Step 0: 测试先行（TDD）

**文件**: `tests/joint_training/test_minirl_loss.py`（新建）

编写以下测试用例：

1. **test_minirl_basic_positive_advantage** — 正优势 + ratio 在范围内 → mask=1，loss < 0（最大化方向）
2. **test_minirl_basic_negative_advantage** — 负优势 + ratio 在范围内 → mask=1，loss > 0（惩罚方向）
3. **test_minirl_clip_mask_positive** — 正优势 + ratio > 1+ε_high → mask=0，该 token loss=0
4. **test_minirl_clip_mask_negative** — 负优势 + ratio < 1-ε_low → mask=0，该 token loss=0
5. **test_minirl_zero_advantage** — 优势=0 → loss=0（无论 ratio/mask）
6. **test_minirl_is_weights_applied** — 传入 rollout_is_weights → loss 被加权
7. **test_minirl_is_weights_none** — rollout_is_weights=None → 等价于权重全 1
8. **test_minirl_loss_agg_seq_mean_token_sum** — 验证 `seq-mean-token-sum` 聚合正确
9. **test_minirl_metrics_returned** — 返回 ppo_kl, pg_clipfrac, mask_fraction 等指标
10. **test_minirl_gradient_only_through_logprob** — 验证梯度仅通过 `log_prob` 流动（IS weight 和 mask 无梯度）

---

## Step 1: 实现 MiniRL 损失函数

**文件**: `verl/trainer/ppo/core_algos.py`（在 CISPO 之后注册）

```python
@register_policy_loss("minirl")
def compute_policy_loss_minirl(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode="seq-mean-token-sum",  # MiniRL: 无 per-token 长度归一化
    config=None, rollout_is_weights=None,
):
    # 1. 策略陈旧度 ratio（用于裁剪判断）
    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -20, 20))
    ppo_kl = masked_mean(old_log_prob - log_prob, response_mask)

    # 2. 二值裁剪掩码（MiniRL 核心：替代 PPO min-clip）
    mask = torch.ones_like(ratio)
    mask[(advantages > 0) & (ratio > 1 + clip_high)] = 0
    mask[(advantages < 0) & (ratio < 1 - clip_low)] = 0
    mask = mask.detach()

    # 3. MiniRL loss: -mask * advantage * log_prob
    pg_losses = -mask * advantages.detach() * log_prob

    # 4. 应用 rollout IS 权重（train-inference 修正，来自 rollout_corr 系统）
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights  # 已 detach

    # 5. 聚合 + 6. 指标
    ...
    return pg_loss, pg_metrics
```

**与 CISPO 的关键差异**：

| 维度 | CISPO | MiniRL |
|------|-------|--------|
| 裁剪方式 | `sg(clip(ratio))` 作为连续权重 | 二值 mask（0 或 1） |
| IS 修正 | `rollout_is_weights` 乘在 loss 上 | 同上（复用现有基础设施） |
| 梯度流 | 通过 `clipped_ratio_sg * log_prob` | 仅通过 `log_prob`（advantage 也 detach） |
| 默认聚合 | `token-mean` | `seq-mean-token-sum`（无长度归一化） |

---

## Step 2: 新建训练脚本

**文件**: `recipe/joint_training/run_joint_minirl_qwen3_1.7b_math.sh`
（复制自 `run_joint_grpo_qwen3_1.7b.sh`）

### 主要修改：

| 配置项 | 原值 | 新值 | 原因 |
|--------|------|------|------|
| `RUN_PREFIX` | `Joint-GRPO-Qwen3-1.7B-GSM8K` | `Joint-MiniRL-Qwen3-1.7B-MATH` | — |
| `TRAIN_FILE` | gsm8k/train | Maxwell-Jia-MATH train | MATH 数据集 |
| `TEST_FILES` | gsm8k/test | [MATH-500, AIME-2025] | 多数据集验证 |
| `loss_mode` | (无/vanilla) | `minirl` | MiniRL 损失 |
| `loss_agg_mode` | `token-mean` | `seq-mean-token-sum` | 无长度归一化 |
| `clip_ratio_high` | `0.28` | `0.27` | MiniRL 论文推荐 |
| `rollout_is` | (无) | `token` | Token 级 IS 修正 |
| `rollout_is_threshold` | (无) | `5.0` | MiniRL 论文推荐 |
| `norm_adv_by_std_in_grpo` | (默认 True) | `False` | Dr.GRPO = MiniRL 优势估计 |
| `max_response_length` | `1024` | `4096` | MATH 需要更长推理 |
| `max_prompt_length` | `512` | `500` | MATH 题目长度 |
| `n_resp_per_prompt` | `4` | `8` | 增大 group size |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | `0.75` | `0.60` | 4096 response 需更多 KV cache |
| `total_training_steps` | `100` | `200` | MATH 需要更多步 |

### Batch Size 分析

| 配置 | Prompts (B) | Group (G) | 总 Responses | 可行性 |
|------|:-----------:|:---------:|:------------:|:------:|
| 用户期望 | 64 | 16 | 1024 | 几乎不可能（1.7B+4096 resp） |
| 保守起步 | **32** | **8** | **256** | 可行（参考 4B 的 B=16,G=8） |
| OOM 后备 | 16 | 8 | 128 | 安全 |

1.7B 模型约为 4B 内存的 40%，但 response_length 从 1024→4096 增加 4x KV cache。
4B 脚本在 4096 resp 下用 B=16, G=8 + offload=True。
1.7B 不需要 offload，但 B=32, G=8 是合理上限。

---

## Step 3: 运行测试验证

```bash
# 1. MiniRL 单元测试
python -m pytest tests/joint_training/test_minirl_loss.py -v

# 2. 现有 rollout correction 测试（确认无回归）
python -m pytest tests/trainer/ppo/test_rollout_corr.py -v
python -m pytest tests/trainer/ppo/test_rollout_corr_integration.py -v

# 3. 现有 policy loss 测试（确认无回归）
python -m pytest tests/trainer/ppo/ -k "policy_loss" -v
```

---

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新建** | `tests/joint_training/test_minirl_loss.py` | MiniRL 测试用例（TDD，先于实现） |
| **修改** | `verl/trainer/ppo/core_algos.py` | 新增 `@register_policy_loss("minirl")` |
| **新建** | `recipe/joint_training/run_joint_minirl_qwen3_1.7b_math.sh` | 1.7B MATH 训练脚本 |

**不需要修改的文件**（已有功能直接复用）：
- `verl/trainer/ppo/rollout_corr_helper.py` — token 级 IS 已实现
- `verl/utils/dataset/rl_dataset.py` — 多 val 文件已支持
- `verl/trainer/ppo/metric_utils.py` — per-data_source 指标已支持
- `verl/trainer/ppo/ray_trainer.py` — 无需修改

---

## 风险与备选

| 风险 | 等级 | 应对 |
|------|------|------|
| B=32, G=8 OOM | 中 | 降至 B=16, G=8 或开启 offload=True |
| MiniRL 训练不稳定 | 低 | 降低 IS threshold 5→2，或回退 CISPO |
| MATH prompt 超长 | 低 | 已设 `filter_overlong_prompts=True` + `max_prompt_length=500` |
| 现有 metrics 不兼容 | 低 | MiniRL 返回相同 metric key（ppo_kl, pg_clipfrac 等） |
