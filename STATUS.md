# Project Status — On-Policy WDL-SFT

**Branch**: `feature/on-policy-wdl-sft`
**Last updated**: 2026-04-19 (EXP-16 1a 启动后)
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

| 实验 | loss | lr | β | 状态 | Run ID |
|---|---|---|---|---|---|
| **EXP-16 (1a)** | wdl_sft_is | 5e-7 | 0 | **运行中**（2026-04-19 16:57 启动） | `WDL-SFT-Qwen3-4B-MATH-1A_1776589025` |
| EXP-17 (1b) | wdl_sft_is | 5e-7 | 0.1 | 待 1a 有方向性结论后决策 | — |
| EXP-18 (1c) | wdl_sft_is | 1e-6 | 0 | 待 1a 稳定后并行 | — |

### EXP-16 (1a) 监控清单

| 项 | 路径 |
|---|---|
| 主训练 log | `recipe/on_policy_wdl_sft/WDL-SFT-Qwen3-4B-MATH-1A_1776589025.log` |
| Metrics JSONL | `recipe/on_policy_wdl_sft/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-1A_1776589025.jsonl` |
| Checkpoints | `/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-1A_1776589025/` |
| Val 原始 dump | `recipe/on_policy_wdl_sft/validation/WDL-SFT-Qwen3-4B-MATH-1A_1776589025/`（gitignore，本地保留） |
| WandB offline | `/data-1/wandb_runs/WDL-SFT-Qwen3-4B-MATH-1A/wandb/offline-run-20260419_090049-5ca47aqk/` |
| Tmux session | `wdl_sft_is_1a` |
| 预计结束时间 | ~24 小时（300 steps × ~90s/step + 12 次 val） |

关键监控指标（通过 WandB 或直接读 jsonl）：
- **IS 触发**：`actor/pg_clipfrac`（正样本上界 clip 比例），`actor/pg_clipfrac_lower`（负样本下界 clip 比例，1a β=0 时应始终为 0），`rollout_is_ratio_fraction_high/low`（vLLM↔FSDP IS 超阈值比例），`rollout_is_eff_sample_size`（ESS）
- **Drift 度**：`actor/ppo_kl`（ratio 偏离 1 的量）、`actor/entropy`（健康下降 vs 漂移上升）
- **Joint 训练**：`jointTraining/model{1,2}_grad_norm`, `model_grad_cosine_similarity`
- **Val**：`val-core/HuggingFaceH4/MATH-500/acc/mean@1`（对标 M5.5 step 300 = 67.94%，1a 目标 > 70%）

## 4. 本次会话执行状态

| 任务 | 状态 |
|---|---|
| 新 plan `docs/joint_training/plans/active/wdl_sft_is.md` | ✅ 完成 |
| 归档 `docs/joint_training/plans/completed/lr_search.md`（加结论头） | ✅ 完成 |
| 新 spec `docs/joint_training/specs/wdl_sft_is.md` | ✅ 完成 |
| 更新 `CLAUDE.md`（v1/v2 区分、超参、training history） | ✅ 完成 |
| 更新 `STATUS.md`（本文件） | ✅ 完成 |
| 更新 auto-memory（`project_reverse_sft.md`, `project_training_status.md`） | ✅ 完成 |
| 更新 v1 plan `on_policy_wdl_sft.md`（加 SUPERSEDED 头） | ✅ 完成 |
| 实现 `compute_policy_loss_wdl_sft_is` | ✅ 完成 |
| 单元测试 `tests/on_policy_wdl_sft/test_wdl_sft_is_loss.py` | ✅ 完成（12/12 通过） |
| pytest 验证通过（含 MiniRL 10/10 回归） | ✅ 完成 |
| 训练脚本 1a/1b/1c（`run_on_policy_wdl_sft_qwen3_4b_math_{1a,1b,1c}.sh`） | ✅ 完成 |
| Git commit + push（主 repo + recipe submodule） | ✅ 完成 |
| **EXP-16 (1a) 启动** | ✅ **运行中** |

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
