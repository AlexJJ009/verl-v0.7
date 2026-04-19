# Project Status — On-Policy WDL-SFT

**Branch**: `feature/on-policy-wdl-sft`
**Last updated**: 2026-04-19
**Updated by**: Claude

---

## 1. 当前方向

**从 LR 搜参转向 method improvement（v1 → v2 loss）**。

EXP-12~15 四次 offline eval 完成后确认：WDL-SFT v1 的 model2 上限稳定在 MATH-500 mean@3 ≈ 79-80%，对 lr/β 都不敏感。同机 baseline MiniRL step 100 就能到 mean@1 ≈ 74%，v1 跑完 300 步 mean@1 ≈ 68%。差距根因：v1 loss 不使用 `old_log_prob`、无 clip、无 `rollout_is_weights` — 相当于没装任何 RL 稳定性机制的 on-policy SFT。

**下一步**：实现 `loss_mode=wdl_sft_is`（v2），补齐 IS 修正 + binary mask clipping + token-level rollout IS。计划文档：`docs/joint_training/plans/active/wdl_sft_is.md`。

## 2. v1 实验结果（锁定，作为 v2 对照基线）

| Experiment | lr | β | Best Step | Model2 MATH-500 mean@3 |
|---|---|---|---|---|
| EXP-12 M5 | 1e-6 | 0.1 | — (diverged) | — |
| **EXP-13 M5.5** | 5e-7 | 0 | 300 | **78.6%** |
| EXP-14 M5.6 | 5e-7 | 0.1 | 300 | 79.1% |
| EXP-15 LR3 | 1e-6 | 0 | 125 | 79.6% |

三者差异小于方差，证明 v1 的上限由 loss 决定。Model1 结果（weak anchor）在 M5.6 下塌掉 21%——这是算法设计意图（model1 作牺牲品），不是错误；但 v1 无 clip 保护让 β>0 的塌缩比必要的更大。

## 3. v2 实验计划

详见 `docs/joint_training/plans/active/wdl_sft_is.md`。三个 A/B 对照：

| 实验 | loss | lr | β | 要回答的问题 |
|---|---|---|---|---|
| EXP-16 (1a) | wdl_sft_is | 5e-7 | 0 | 补齐 IS/clip 后正向 SFT 的真实上限（>70% 表明方向靠谱）|
| EXP-17 (1b) | wdl_sft_is | 5e-7 | 0.1 | reverse SFT 在 IS/clip 保护下是否仍不稳 |
| EXP-18 (1c) | wdl_sft_is | 1e-6 | 0 | 高 LR 在有 IS/clip 下是否稳；区分 v1 漂移是 LR 问题还是 loss 问题 |

## 4. 代码与文档执行状态（本次会话）

| 任务 | 状态 |
|---|---|
| 新 plan `docs/joint_training/plans/active/wdl_sft_is.md` | ✅ 完成 |
| 归档 `docs/joint_training/plans/completed/lr_search.md`（加结论头） | ✅ 完成 |
| 新 spec `docs/joint_training/specs/wdl_sft_is.md` | ✅ 完成 |
| 更新 `CLAUDE.md`（v1/v2 区分、超参、training history） | ✅ 完成 |
| 更新 `STATUS.md`（本文件） | ✅ 完成 |
| 更新 auto-memory（`project_reverse_sft.md`, `project_training_status.md`） | 🔄 进行中 |
| 实现 `compute_policy_loss_wdl_sft_is` | 🔄 待做 |
| 单元测试 `tests/joint_training/test_wdl_sft_is_loss.py` | 🔄 待做 |
| pytest 验证通过 | 🔄 待做 |
| 训练脚本 `run_on_policy_wdl_sft_qwen3_4b_math_1a.sh` 等 | 🔄 待做 |

## 5. 关键文件路径

| 文件 | 说明 |
|---|---|
| `docs/joint_training/plans/active/wdl_sft_is.md` | **当前主计划** |
| `docs/joint_training/specs/wdl_sft_is.md` | v1 vs v2 loss spec |
| `docs/joint_training/plans/completed/lr_search.md` | 已归档的 LR 搜索计划 |
| `recipe/on_policy_wdl_sft/EXPERIMENT_INDEX.md` | EXP-12~15 实验记录 |
| `recipe/on_policy_wdl_sft/INFERENCE_RESULTS.md` | EVAL-10~15 评测结果 |
| `verl/trainer/ppo/core_algos.py:1861` | v1 loss 实现（保留不动）|
| `verl/trainer/ppo/core_algos.py:1782` | MiniRL 参考实现（binary mask + IS） |
| `verl/trainer/ppo/rollout_corr_helper.py` | rollout_is_weights 计算 |

## 6. 关键前提

- v1 ~ v2 只在 loss 函数层变化，Weak-Driven rollout（fused sampling, λ=0.5）和 joint model 结构保持不变
- Validation 只看 model2（model1 是设计上的牺牲品）
- 分支不变（`feature/on-policy-wdl-sft`），两版 loss 在代码库中并存，通过 `loss_mode` 切换
