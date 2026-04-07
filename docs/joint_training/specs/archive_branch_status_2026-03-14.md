# Joint Training 分支进展报告

> **日期**: 2026-03-14
> **分支**: `feature/joint-training`
> **仓库**: `https://github.com/AlexJJ009/verl-v0.7` (fork from `verl-project/verl`)

---

## 一、算法目标与核心方法

### 1.1 要解决什么问题

本分支实现的是 **联合训练（Joint Training）** 范式在强化学习（以 GRPO 为例）中的落地。

核心 idea：让两个 LLM 在训练过程中通过 **logit 融合** 进行协作，但最终只部署其中一个模型（model2）。model1 扮演"教师/协作者"角色，在训练和生成阶段通过 logit 加权来引导 model2 学习，但评测时只看 model2 的独立表现。

这个范式已经在 SFT（监督微调）中验证成功（同事使用 TRL 框架实现），本分支的任务是将其迁移到更复杂的强化学习训练框架 verl 中。

### 1.2 核心算法公式

```
logits_fused = (1 - λ) × logits_model1 + λ × logits_model2
```

其中 `λ`（fusion_lambda）是可配置的融合权重，默认 0.5。

**数据流：**

```
input_ids
    │
    ├──→ model1.forward() ──→ logits_1
    │
    └──→ model2.forward() ──→ logits_2
                                  │
    logits_fused = (1-λ)·logits_1 + λ·logits_2
                                  │
                    用于计算 loss / log_probs / 生成 token
```

**梯度传播：**

```
∂loss/∂logits_1 = (1-λ) × ∂loss/∂logits_fused  →  更新 model1（如果未冻结）
∂loss/∂logits_2 = λ × ∂loss/∂logits_fused       →  更新 model2
```

### 1.3 双模式推理设计

联合训练的推理分为两个场景，这是理解整个实现的关键：

| 场景 | 使用的 logits | 原因 |
|------|-------------|------|
| **Rollout 生成** | 融合 logits（两个模型） | RL 要求 on-policy：生成 response 的策略必须和训练策略一致 |
| **Evals 评测** | 仅 model2 logits | 评测目的是衡量 model2 独立能力，最终只部署 model2 |

> 为什么 Rollout 不能只用 model2？因为训练时 old_log_prob 和 new_log_prob 都基于融合策略计算，如果 Rollout 用的是不同策略，on-policy 假设被破坏，RL 训练将失效。

### 1.4 与标准 GRPO 的关系

GRPO（Group Relative Policy Optimization）是一种无 Critic 的 RL 算法，通过组内奖励标准化计算优势值。联合训练 **不修改 GRPO 算法本身**，而是在模型层面引入 logit 融合，使得：

1. 策略（policy）的定义从"单模型 logits"变为"融合 logits"
2. 所有依赖策略输出的计算（log_prob、KL、entropy）自动基于融合 logits
3. GRPO 的优势计算、loss 函数等核心算法逻辑完全不变

---

## 二、实现架构

### 2.1 模型层

**核心文件：**

| 文件 | 作用 |
|------|------|
| `verl/models/joint_model/modeling_joint_qwen3.py` | HuggingFace 联合模型类 `QwenJointForCausalLM` |
| `verl/models/joint_model/configuration_joint_qwen3.py` | 联合模型配置类 `QwenJointConfig` |
| `verl/models/joint_model/weight_utils.py` | 权重提取工具（从联合模型中提取 model2 权重） |
| `verl/models/joint_model/vllm_modeling_joint_qwen3.py` | vLLM 推理引擎的联合模型实现 |
| `verl/models/joint_model/vllm_registry.py` | vLLM 模型注册 |
| `verl/models/joint_model/prepare_joint_weights.py` | 从基座模型创建联合模型权重 |

**`QwenJointForCausalLM` 核心逻辑：**

```python
class QwenJointForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.sub_models = nn.ModuleList([
            Qwen3ForCausalLM(config) for _ in range(config.num_sub_models)
        ])
        self.fusion_lambda = config.fusion_lambda
        if config.freeze_model1:
            for param in self.sub_models[0].parameters():
                param.requires_grad = False

    def forward(self, ..., eval_only=False):
        if eval_only:
            return self.sub_models[1](...)  # 仅 model2

        # 两个子模型独立前向传播
        for sub_model in self.sub_models:
            out = sub_model(input_ids, ...)
            outputs_list.append(out)

        # 融合 logits（内存优化：就地运算避免额外全词表临时张量）
        logits = outputs_list[0].logits.mul(1 - lam)
        logits.add_(outputs_list[1].logits, alpha=lam)

        # 计算子模型 logit 分歧（分块 softmax，避免 151k 词表 OOM）
        # 每块 256 tokens，峰值内存 ≈ 440 MiB
        with torch.no_grad():
            flat0 = logits0.reshape(-1, vocab)
            flat1 = logits1.reshape(-1, vocab)
            total_diff = sum((softmax(flat0[i:i+256]) - softmax(flat1[i:i+256])).abs().sum()
                             for i in range(0, n_tokens, 256))
            self.last_logit_disagreement = total_diff / (n_tokens * vocab)

        return CausalLMOutputWithPast(logits=logits, ...)
```

**vLLM 侧的对应实现：**

vLLM 推理引擎通过 `vllm_modeling_joint_qwen3.py` 实现了同样的融合逻辑，并支持 `_use_model2_only` 标志在融合模式和 eval-only 模式之间切换。

### 2.2 权重同步与提取

**关键机制：** `weight_utils.py` 提供了从联合模型 state_dict 中提取单个子模型权重的工具：

```python
extract_sub_model_weights(state_dict, sub_model_index=1)
# 将 "sub_models.1.model.layers.0.weight" → "model.layers.0.weight"
```

这使得：
- 训练时：加载完整联合模型（两个子模型 + 融合逻辑）
- 评测时：从联合模型中提取 model2 权重，加载到标准的 Qwen3ForCausalLM 进行独立推理

### 2.3 训练器集成

**`verl/trainer/ppo/ray_trainer.py`** 中的联合训练集成点：

1. **配置检测**：`_is_joint_training` 标志识别是否启用联合训练
2. **Rollout 阶段**：使用融合 logits 的 vLLM 模型生成 response
3. **Validation 阶段**：调用 `checkpoint_manager.update_weights(eval_only=True)` 同步仅 model2 权重到 vLLM
4. **梯度监控**：额外输出 `jointTraining/model1_grad_norm` 和 `jointTraining/model2_grad_norm`

### 2.4 配方（Recipe）

启动脚本：`recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh`

关键配置：
- 模型：Qwen3-1.7B（两个子模型均从同一基座初始化）
- 数据集：GSM8K（数学推理任务）
- 奖励：LaTeX 语义验证（自定义 reward function）
- GPU：8 × H800
- 后端：FSDP（训练） + vLLM async（推理）
- 路径：所有检查点/临时文件/IPC 路径固定在 `/data-1` 下

---

## 三、开发阶段与进展

### 3.1 Stage 1：运行时稳定化（已完成）

**目标：** 让联合 GRPO 训练跑通完整的训练生命周期。

**退出条件（全部达成）：**
1. recipe 能完成完整训练（100 步）
2. 检查点成功保存到 `/data-1/checkpoints`
3. 验证按计划执行（每 5 步）
4. 指标在训练过程中可见（而非仅在结束时）

**Stage 1 解决的主要问题链：**

| 问题 | Commit | 根因 |
|------|--------|------|
| FSDP DP-group 不匹配 + NCCL 死锁 | `428a7e83` | 动态 batch 构造在分布式组内不确定 |
| vLLM 启动缓存预算失败 | `384804fe` | 联合模型双倍参数导致 GPU 内存估算不足 |
| Actor old-log-prob 和 entropy OOM | `428a7e83` | 全词表密集张量 (batch × seq × vocab) 开销在联合模型下翻倍 |
| flash_attn 缺失导致崩溃 | recipe 修复 | 可选依赖未做安全降级 |
| 检查点路径和磁盘压力失败 | `73404180` | 直接写入最终文件不够健壮，需原子写入 |
| /tmp 和根文件系统空间不足 | `5bd62896` | ZMQ/vLLM 运行时路径默认使用 /tmp |
| 周期性指标不可见 | `c4436d2b` | 关键指标仅在训练结束时输出 |
| HF Rollout 路径不兼容 | `648370bc` ~ `fd5a5cbf` | HFRollout 的 API 与 AgentLoopManager 不匹配（6 个连续修复） |

### 3.2 Stage 2：算法正确性与可观测性（进行中）

**目标：** 确保联合训练不仅"能跑"，而且"跑对"。

**已完成的工作：**

1. **观测性基础设施**（`a99ba83b`）：
   - 验证生成结果 dump 到 stdout/jsonl/tracking table
   - 联合训练专属指标：`model1_grad_norm`、`model2_grad_norm`
   - 验证步骤摘要包含 `jointTraining/` 指标

2. **首次成功运行的指标分析**（`joint_grpo_run_1773032262_metric_analysis.md`）：
   - 识别出 reward 配置错误（`overlong_buffer_len == max_response_length`）导致所有 response 获得惩罚
   - 确认梯度坍塌是 reward 坍塌的下游效应，而非联合训练逻辑本身的问题

3. **第二次运行的详细诊断**（验证 dump 分析）：
   - 发现 ~80% 的 response 缺少 `\boxed{}` 格式（`pred = [NO_BOXED]`）
   - 发现即使 `pred == gts` 也被标记为 `answer_correct = false` 的验证器不一致
   - 确认问题跨越整个 100 步训练持续存在

**当前发现的问题层级：**

```
Level 1（已确认）: Reward 配置错误 → 所有 response 获得负奖励 → 梯度坍塌
Level 2（已确认）: 答案提取/格式问题 → ~80% response 未被正确评估
Level 3（已确认）: 验证器不一致 → pred 匹配 gts 但仍判错
Level 4（待调查）: 联合训练固有问题：
    - 融合 logits 可能抑制 EOS token 和格式 token
    - 两个子模型从同一权重初始化，缺少对称性破缺
    - 训练策略（融合）vs 评测策略（model2-only）的差距
```

**Stage 2 未完成的优先事项：**

| 优先级 | 内容 | 状态 |
|--------|------|------|
| P0 | 修复 reward 配置，获得有效训练信号 | 待执行 |
| P0 | 修复验证器 pred/gts 不一致 | 待执行 |
| P1 | 新增联合训练专属指标（子模型 logit 分歧、融合 vs eval-only 差距等） | 已规划 |
| P1 | 审计 rollout logits / old-log-prob / reference-policy 语义正确性 | 未开始 |
| P2 | 子模型对称性破缺策略 | 未开始 |
| P2 | checkpoint resume 健壮性测试 | 未开始 |

---

## 四、近期 Commit 详解

以下是最近 20 个 commit 的分类汇总（从新到旧）：

### 4.1 观测性与指标（Stage 2）

| Commit | 描述 |
|--------|------|
| `a99ba83b` | 输出验证样本、添加 model1/model2 grad_norm 指标 |
| `c4436d2b` | 在每个 test-step 打印完整训练+验证指标摘要 |

### 4.2 运行时稳定化（Stage 1）

| Commit | 描述 |
|--------|------|
| `5bd62896` | 将 ZMQ socket 路径参数化，避免 /tmp 空间耗尽 |
| `73404180` | FSDP 检查点原子保存 + 磁盘压力预检 |
| `428a7e83` | FSDP actor rollout 路径稳定化（DP-group 一致性、entropy 门控、内存优化） |
| `384804fe` | 联合 vLLM rollout 支持 + eval-only 权重提取 |

### 4.3 HF Rollout 路径（早期 fallback 路径）

| Commit | 描述 |
|--------|------|
| `648370bc` | 通过 HFSyncRolloutManager 接入 HFRollout |
| `79c152f8` | 处理 async actor 中的事件循环冲突 |
| `f8d04127` | 为 HFRollout 在 gen_batch 中包含 tensor keys |
| `7e6f068e` | FSDP worker 中为 HFRollout 做 raw_prompt tokenization |
| `fd5a5cbf` | 修复 OOM、unequal-chunk、pad_token_id=None 三个 HFRollout bug |
| `562654ca` / `f498222c` | 修复 HF rollout 验证的 reward metadata 和日志 |

### 4.4 基础设施与文档

| Commit | 描述 |
|--------|------|
| `ec742937` | 标准化 tokenized chat template 输出 |
| `067b5412` | 修复自定义 reward function 在 validate/train 中的计算路径 |
| `0a72b713` | 服务器迁移经验文档 |
| `95abca6c` | Git submodule 工作流文档 |
| `59a4c534` | 刷新测试覆盖 |

---

## 五、测试覆盖

测试位于 `tests/joint_training/`，分为三类：

| 类别 | 文件数 | 覆盖范围 |
|------|--------|---------|
| `feat/` | 5 | 模型实例化、logit 融合、梯度流、冻结、eval_only、权重提取、vLLM checkpoint、GRPO 集成、HF rollout |
| `regression/` | 5 | entropy 门控、log_prob 跳过、验证生成日志、reward metadata、test-step 指标 |
| `reward/` | 1 | LaTeX 语义验证 reward function |

最近一次全量测试通过 **157 passed**。

---

## 六、关键困难与经验

### 6.1 工程困难

1. **双倍内存压力**：联合模型有两套完整参数，在已经很紧张的 H800 per-GPU 内存预算下需要精细管理。
2. **分布式一致性**：FSDP 要求 DP-group 内所有 rank 执行相同的 collective 操作，联合模型引入了额外的不一致风险。
3. **vLLM 双模式**：需要在同一个 vLLM 推理引擎中实现融合/eval-only 两种模式的切换，且权重同步必须正确。
4. **文件系统拓扑**：H800 服务器的 `/tmp` 和根文件系统空间有限，所有运行时路径必须显式指向数据盘。

### 6.2 算法层面的挑战（Stage 2 重点）

1. **EOS token 抑制**：两个模型对 EOS 的概率分歧可能导致融合后 EOS 概率降低，response 过长。
2. **对称性问题**：两个子模型从相同权重初始化时，融合 logits 等价于单模型，缺少对称性破缺。
3. **训练/评测策略差距**：训练用融合策略，评测用 model2-only 策略，这个差距是否会随训练进展而缩小尚待验证。
4. **Reward 信号有效性**：当前 GSM8K 上的 reward 配置和答案提取存在多个问题，导致无法获得有效训练信号。

### 6.3 核心设计原则（从经验中沉淀）

1. **保持上游行为不变**：除非联合训练确实需要，否则不修改框架核心逻辑。
2. **在模型层而非算法层实现融合**：GRPO 算法不需要知道 logits 来自联合模型还是单模型。
3. **可观测性优先**：如果一个指标对理解训练行为是必要的，就应该显式暴露。
4. **每个 bug 都写测试**：特别是那些首次在 E2E GPU 运行中发现的 bug。

---

## 七、项目结构速查

```
verl/models/joint_model/          # 联合模型实现（HF + vLLM）
verl/trainer/ppo/ray_trainer.py   # 训练主循环（联合训练集成点）
verl/trainer/ppo/core_algos.py    # GRPO 核心算法（不需要修改）
verl/workers/fsdp_workers.py      # FSDP 分布式训练 worker
verl/workers/rollout/vllm_rollout/# vLLM rollout 引擎
verl/checkpoint_engine/           # 检查点管理
tests/joint_training/             # 联合训练测试套件
recipe/joint_training/            # 启动脚本和配置
docs/joint_training/              # 文档（本报告所在目录）
```

---

## 八、参考文档

| 文档 | 内容 |
|------|------|
| `GRPO_Joint_Training_Target_v1.md` | 算法设计原始文档（中文） |
| `verl_joint_training_investigation_report.md` | 深度架构调研报告（中文，教学向） |
| `progress.md` | 逐阶段进度跟踪 |
| `stabilization_experience_notes.md` | Stage 1 工程经验总结 |
| `server_migration_pitfalls.md` | 服务器迁移踩坑记录 |
| `joint_grpo_run_1773032262_metric_analysis.md` | 首次运行指标坍塌分析 |
| `runReport/2026-03-11_*.md` | 验证日志和梯度范数专项报告 |
