# Stage 2：算法正确性与鲁棒性

**状态**：进行中（2026-03-11 启动）
**最近审查**：2026-03-14

## 范围

1. 调试系统启动层之下的算法内部和逻辑级问题。
2. 添加联合训练专用指标，使融合策略的行为可观测（而不仅依赖通用 PPO/GRPO 指标）。
3. 识别联合训练固有的失败模式和降级行为（区别于通用运行时问题）。
4. 提升鲁棒性和兼容性，使联合训练成为 `verl` 中成熟的一等训练路径。

---

## 已完成工作

### 仪表化阶段（2026-03-11）✅

1. 验证生成结果已显示在 stdout/log、jsonl dump 和 tracking 表中。
2. 子模型梯度范数已实现：`jointTraining/model1_grad_norm`、`jointTraining/model2_grad_norm`。
3. 联合训练指标已包含在 validation-step 汇总中。
4. 修复 vLLM 启动端口套接字生命周期问题（`EADDRINUSE` 回归）。

### 关键发现

`-1` 奖励饱和问题并非单一 bug 类别，至少包含以下原因：

1. 答案提取/格式化失败（约 80% `[NO_BOXED]`）
2. 常规推理错误
3. 病理性响应漂移或损坏
4. 答案提取成功后验证器仍不匹配（约 10-12% `pred == gts` 但 `answer_correct = false`）
5. 以上失败模式在 100 步完整运行中持续存在

---

## 待完成工作

### P1：指标体系（优先级最高）

**目标**：实现联合训练专用指标，使训练行为可调试、可观测。

**当前状态**：11 项指标中已完成 10 项（约 91%）。

| # | 指标名称 | 状态 | 实施位置 |
|---|----------|------|----------|
| 1 | `jointTraining/model1_grad_norm` | ✅ | `dp_actor.py` |
| 2 | `jointTraining/model2_grad_norm` | ✅ | `dp_actor.py` |
| 3 | `jointTraining/answer_extraction_failure_rate` | ✅ | `ray_trainer.py` |
| 4 | `jointTraining/verification_method/*` | ✅ | `ray_trainer.py` |
| 5 | `jointTraining/fused_vs_eval_model2_acc_gap` | ❌ 未开始 | 需要双路 validation 对比 |
| 6 | `jointTraining/model_grad_norm_ratio` | ✅ | `dp_actor.py` |
| 7 | `jointTraining/model_grad_cosine_similarity` | ✅ | `dp_actor.py` |
| 8 | `jointTraining/submodel_logit_disagreement` | ✅ | `modeling_joint_qwen3.py` + `dp_actor.py` |
| 9 | `jointTraining/response_unprintable_or_non_ascii_ratio` | ✅ | `ray_trainer.py` |
| 10 | `jointTraining/verifier_pred_gt_disagreement_count` | ✅ | `ray_trainer.py` |
| 11 | `jointTraining/validation_tracking_payload_bytes` | ❌ 未开始 | 优先级低 |

---

### P2：语义正确性审计

**目标**：系统性验证联合训练的核心计算路径在语义上是正确的。

**当前状态**：诊断基础设施就绪，但尚未进行系统性审计（约 20%）。

| # | 审计项 | 状态 | 涉及文件 | 具体工作 |
|---|--------|------|----------|----------|
| 1 | Rollout logits 正确性 | ❌ 未审计 | `modeling_joint_qwen3.py` | 验证 `logits_fused = (1-λ)*logits_m1 + λ*logits_m2` 在 rollout 阶段的计算是否正确 |
| 2 | Old-log-prob 重计算 | 🔶 部分完成 | `dp_actor.py`, `core_algos.py` | 已有门控测试，但未验证重计算值与 rollout 时的一致性 |
| 3 | Reference-policy 比较 | ❌ 未审计 | `core_algos.py` | 验证 KL 散度和 ref log-prob 在联合训练下是否正确 |
| 4 | Validation/eval-only 权重提取 | 🔶 部分完成 | `modeling_joint_qwen3.py:60-72` | 已实现，但未覆盖边界情况测试 |

**建议的审计方法**：
- 编写单元测试，用已知输入验证 fused logits 的数值一致性
- 对比 rollout 阶段和 training 阶段的 log-prob 值，确保无精度漂移
- 验证 reference-policy 的梯度流在联合训练模式下被正确阻断

---

### P3：边界情况测试

**目标**：对联合训练进行边界条件压力测试。

**当前状态**：0% 完成。

| # | 测试场景 | 状态 | 具体工作 |
|---|----------|------|----------|
| 1 | Checkpoint 恢复 | ❌ 未开始 | 测试从训练中间保存的 checkpoint 恢复后，联合模型权重和优化器状态是否正确 |
| 2 | 部分 Checkpoint 保留 | ❌ 未开始 | 测试 checkpoint 清理逻辑在磁盘空间不足时的行为 |
| 3 | 无 `flash_attn` 降级 | ❌ 未开始 | 验证在没有 flash_attn 的环境下联合模型能否正常运行 |
| 4 | W&B 在线/离线切换 | ❌ 未开始 | 测试网络断开时 W&B 日志是否能平滑切换到离线模式 |
| 5 | vLLM 与 HF rollout 一致性 | 🔶 基础测试存在 | 已有 `test_vllm_joint_rollout.py` 和 `test_hf_rollout_joint.py`，但未做数值一致性对比 |

**测试文件位置**：`tests/joint_training/edge_cases/`（需新建）

---

### P4：框架内化

**目标**：将 recipe 中稳定的保护性代码迁移到框架层。

**当前状态**：0% 完成。

| # | 迁移项 | 状态 | 从 | 到 |
|---|--------|------|----|----|
| 1 | 答案提取逻辑 | ❌ 未开始 | `recipe/joint_training/custom_reward_function_latex_verify.py` | `verl/utils/reward/` |
| 2 | 验证方法降级链 | ❌ 未开始 | recipe reward function | `verl/utils/reward/` |
| 3 | 联合训练指标计算 | ❌ 未开始 | 取决于 P1 完成 | `verl/trainer/ppo/` 或 `verl/workers/` |

**注意**：P4 依赖 P1 指标实现稳定后再进行。

---

## 回归测试覆盖

当前测试套件：

1. `tests/joint_training/`（157 passed）
2. `tests/workers/actor/test_special_dp_actor.py`
3. `tests/joint_training/regression/test_validation_generation_logging.py`
4. `tests/joint_training/feat/test_vllm_joint_rollout.py`
5. `tests/workers/critic/test_dynamic_dp_critic.py`
6. `tests/workers/test_fsdp_workers.py`
7. `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py`
8. `tests/utils/test_attention_utils_on_cpu.py`
9. `tests/utils/test_torch_functional.py`

---

## 执行记录

### 2026-03-14：P1 指标体系 + P2 语义审计（已完成）

**P1 已实现指标**（198 测试全部通过，真实训练验证指标正常输出）：

| 指标 | 文件 | 真实运行值 |
|------|------|-----------|
| `model_grad_norm_ratio` | `dp_actor.py` | 0.0（FSDP 下 per-submodel grad 不可见，预存问题） |
| `model_grad_cosine_similarity` | `dp_actor.py` | 同上 |
| `submodel_logit_disagreement` | `modeling_joint_qwen3.py` + `dp_actor.py` | 模型层正确计算 |
| `answer_extraction_failure_rate` | `ray_trainer.py` | **82-83%** |
| `verification_method/*` | `ray_trainer.py` | **100% verl_math_verify** |
| `verifier_pred_gt_disagreement_count` | `ray_trainer.py` | **134-150** |
| `response_unprintable_or_non_ascii_ratio` | `ray_trainer.py` | **~0%** |

**P2 语义审计**（25 测试全部通过）：
- Fused logits 公式在 7 个 λ 值下验证正确
- Old-log-prob 重计算一致性确认
- Eval-only 权重提取正确
- Reference-policy 梯度阻断正确

### 2026-03-14：根因诊断 — reward 全为 -1 的三层问题

通过分析 step 0 的 1319 个验证样本，定位到以下因果链：

```
根因 1 (P0): verl_math_verify 验证器 bug
  → 134 个 pred==gt 的样本被判错 (answer_correct=False)
  → 本应 10.2% 正确率变成 0%

根因 2 (P1): Prompt 要求 "####" 格式，reward function 只提取 \boxed{}
  → 34.6% 输出用了 #### 格式（含 22 个正确答案），全部被忽略
  → 实际可用正确率 11.8%（合并两种提取方式）

根因 3 (P2): Logit fusion 首 token 噪声
  → 20.7% 输出有非 ASCII 前缀（emoji、韩文、希伯来文等）
  → 影响推理质量但非致命

后果: 所有 reward=-1 → GRPO advantage=0 → gradient=0 → 模型无法学习
```

**修复计划**：

1. **P0 修复**（验证器 bug）：调查 `verl_math_verify.compute_score` 对整数返回 False 的原因并修复
2. **P1 修复**（数据集格式）：将 GSM8K 数据集 prompt 改为 Qwen3 chat template + `\boxed{}` 格式
   - 参考 `/data-1/dataset/AIME-2025/aime-2025-example.json` 和 `/data-1/dataset/MATH-500/math500-test-example.json`
3. **P2 观察**（首 token 噪声）：P0+P1 修复后重新训练，观察是否仍有问题

---

## 建议执行顺序

```
P0 验证器 bug 修复
    ↓
P1 数据集格式修复 (GSM8K → \boxed{} + Qwen3 chat template)
    ↓
全量测试验证
    ↓
重新训练并观察
    ↓
P2 首 token 噪声（如仍存在则修复）
    ↓
P3 边界测试
    ↓
P4 框架内化
```
