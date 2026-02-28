# verl 联合训练强化学习（GRPO）—— 深度调研报告

> **目标读者**：有一定 Python 基础、能跑通训练脚本但不了解框架内部实现的开发者
> **阅读目标**：理解 verl 的代码架构哲学，弄清楚要改哪些文件、为什么要改这些文件，为后续实现联合训练 GRPO 做好知识储备

---

## 目录

1. [引言：从 SFT 联合训练到 RL 联合训练](#1-引言)
2. [verl 的架构哲学：为什么代码是这样组织的？](#2-架构哲学)
3. [一个 GRPO 训练 Step 的完整生命周期](#3-训练生命周期)
4. [深入解析：模型是如何被加载的？](#4-模型加载)
5. [深入解析：前向传播与 logits / log_probs 的计算](#5-前向传播)
6. [深入解析：策略更新（Actor Training）](#6-策略更新)
7. [深入解析：vLLM 推理引擎与权重同步](#7-vllm推理)
8. [深入解析：FSDP 分布式训练与权重管理](#8-fsdp分布式)
9. [深入解析：评测（Evaluation）与指标（Metrics）](#9-评测与指标)
10. [联合训练需要修改的位置与原因分析](#10-修改方案)
11. [关键文件索引表](#11-文件索引)

---

## 1. 引言：从 SFT 联合训练到 RL 联合训练 {#1-引言}

### 1.1 什么是联合训练（Joint Training / Vote-Base）？

联合训练的核心思想非常简单：

```
两个模型独立做前向传播 → 各自输出 logits → 加权融合 logits → 用融合后的 logits 计算 loss → 反向传播更新两个模型
```

用数学公式表达：

```
logits_fused = (1 - λ) × logits_model1 + λ × logits_model2
loss = CrossEntropy(logits_fused, labels)
```

联合训练的推理分两个场景：
- **Rollout 生成阶段**：使用两个模型融合的 logits 进行生成（与训练时保持一致）
- **Evals 评测阶段**：只用第二个模型（model2）进行推理和评测

这意味着 model1 像一个"协作者"，在训练和生成过程中通过 logit 融合来与 model2 协同工作。而最终评测时，我们只看 model2 独立的表现，以验证 model2 从联合训练中真正学到了什么。

### 1.2 SFT 联合训练 vs RL 联合训练

在 SFT 中，联合训练相对简单：
- 输入是 `(input_ids, labels)`
- 直接计算 `CrossEntropy(fused_logits, labels)`
- 用 TRL 的 SFTTrainer 即可实现

但在 RL（以 GRPO 为例）中，情况要复杂得多，因为 RL 训练涉及多个阶段：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     RL 训练的一个 Step                                │
│                                                                      │
│  1. Rollout（推理生成）: 用 vLLM 生成 response                        │
│  2. Reward（奖励计算）: 评估 response 质量                            │
│  3. compute_log_prob（推理）: 重新计算 old_log_probs                  │
│  4. compute_advantage（优势计算）: 计算每个 response 的优势值          │
│  5. update_policy（训练）: 用 PPO/GRPO 损失函数更新模型参数            │
│                                                                      │
│  其中步骤 1 使用 vLLM 推理引擎                                        │
│  步骤 3 和 5 使用 HuggingFace 模型（FSDP 包裹）做前向传播              │
│                                                                      │
│  另外还有：Evals（评测阶段）: 也使用 vLLM 推理引擎                    │
│  但 Evals 和 Rollout 的模型行为不同！                                 │
└──────────────────────────────────────────────────────────────────────┘
```

**核心挑战**：联合训练的 logit 融合需要在哪些步骤中生效？推理阶段的两个场景（Rollout vs Evals）分别应该怎么处理？

### 1.3 我们的目标

将"联合训练"的 logit 融合机制迁移到 verl 框架的 GRPO 训练中：
- **训练时**（compute_log_prob + update_policy）：两个子模型的 logits 加权融合，用融合后的 logits 计算 log_probs
- **Rollout 生成阶段**：使用两个模型融合的 logits 进行 response 生成（需要 vLLM 支持联合模型）
- **Evals 评测阶段**：只用第二个模型（model2）的权重进行推理和评测

> **为什么 Rollout 要用融合 logits？** 因为训练时的策略（policy）是基于融合 logits 定义的。如果 Rollout 生成用的是 model2 单独的策略，而训练时计算的 old_log_prob 和 new_log_prob 都是基于融合策略，那么 on-policy 的假设就被破坏了。RL 要求 Rollout 生成的 response 来自当前策略，而当前策略就是融合策略。
>
> **为什么 Evals 只用 model2？** 因为评测的目的是衡量 model2 本身的能力。联合训练是一种训练手段，最终部署时只会用 model2。所以评测需要反映 model2 独立运行时的真实表现。

---

## 2. verl 的架构哲学：为什么代码是这样组织的？ {#2-架构哲学}

### 2.1 分层解耦：verl 最重要的设计原则

verl 采用了"洋葱式"的分层架构，从外到内分别是：

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: 训练主循环（Trainer / ray_trainer.py）       │  ← 编排各个 Worker 的调用顺序
│  ┌─────────────────────────────────────────────────┐ │
│  │ Layer 2: Worker（fsdp_workers.py）               │ │  ← 管理模型生命周期（加载/卸载/同步权重）
│  │  ┌─────────────────────────────────────────────┐│ │
│  │  │ Layer 3: Actor（dp_actor.py）                ││ │  ← 实现前向传播、log_prob 计算、策略更新
│  │  │  ┌─────────────────────────────────────────┐││ │
│  │  │  │ Layer 4: 核心算法（core_algos.py）       │││ │  ← 纯函数：计算优势、计算 loss
│  │  │  │  ┌─────────────────────────────────────┐│││ │
│  │  │  │  │ Layer 5: 底层工具（torch_functional） ││││ │  ← logits→log_probs 转换等
│  │  │  │  └─────────────────────────────────────┘│││ │
│  │  │  └─────────────────────────────────────────┘││ │
│  │  └─────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**为什么这样设计？**
- **关注点分离**：每一层只关心自己的事情。ray_trainer.py 不关心模型怎么做前向传播；dp_actor.py 不关心权重怎么同步到 vLLM
- **可替换性**：你可以替换 Layer 4 的算法（PPO→GRPO→DAPO）而不需要改 Layer 2 或 3
- **可测试性**：core_algos.py 中的函数都是纯函数，可以独立做单元测试

**对我们的意义**：联合训练主要影响的是 **Layer 3（Actor的前向传播）** 和 **Layer 2（Worker的模型加载和权重同步）**，不需要改 Layer 1（训练主循环）和 Layer 4（核心算法）。

### 2.2 注册机制：verl 的"插件系统"

verl 大量使用注册（Registry）模式来实现可插拔性。这是你需要理解的关键设计模式：

```python
# 注册一个优势估计器
ADV_ESTIMATOR_REGISTRY = {}

def register_adv_est(name):
    def decorator(fn):
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

# 使用时通过名字获取
@register_adv_est("grpo")
def compute_grpo_advantage(...):
    ...

# 调用时
fn = get_adv_estimator_fn("grpo")  # 通过字符串名字获取函数
```

**同样的模式也用于**：
- 策略 loss 函数（`POLICY_LOSS_REGISTRY`）
- Rollout 引擎（`_ROLLOUT_REGISTRY`）

这意味着你可以通过注册新函数来扩展系统，而不需要修改已有代码。

### 2.3 DataProto：统一的数据传输协议

verl 中所有组件之间的数据传递都通过 `DataProto` 对象：

```python
class DataProto:
    batch: dict[str, torch.Tensor]           # 张量数据（input_ids, logits 等）
    non_tensor_batch: dict[str, Any]          # 非张量数据（uid, multi_modal_inputs 等）
    meta_info: dict[str, Any]                # 元信息（temperature, batch_size 等）
```

**为什么这很重要？**
- 所有 Worker 之间传递的数据都有统一的格式
- 你可以通过 `data.union(other_data)` 来合并数据
- 你可以通过 `data.select(batch_keys=[...])` 来选择需要的字段
- 这使得添加新的数据字段（如联合训练的第二组 logits）变得非常自然

---

## 3. 一个 GRPO 训练 Step 的完整生命周期 {#3-训练生命周期}

让我们跟踪一个完整的 GRPO 训练步骤，理解数据如何流经整个系统。

### 3.1 全局视图

```
                     ray_trainer.py (训练主循环)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Rollout Worker│ │ Actor Worker │ │ Reward Worker│
    │  (vLLM推理)   │ │ (FSDP训练)  │ │  (奖励模型)  │
    └──────────────┘ └──────────────┘ └──────────────┘
```

### 3.2 Step-by-Step 详解

**文件**: `verl/trainer/ppo/ray_trainer.py`，`fit()` 方法（line 1218）

#### Step 1: 生成响应（Rollout）

```python
# ray_trainer.py, 约 line 1275-1315
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

- 调用 vLLM 推理引擎
- 输入：一批 prompts
- 输出：每个 prompt 生成 `n` 个 responses（GRPO 需要多个样本进行组内比较）
- **联合训练关注点**：**Rollout 生成阶段需要使用融合 logits！** 因为 RL 要求 on-policy：生成 response 的策略必须和训练时计算 log_prob 的策略一致。联合训练的策略是基于融合 logits 定义的，所以 vLLM 也需要用融合 logits 来生成。这是本项目最大的技术挑战之一。

#### Step 2: 计算奖励（Reward）

```python
# ray_trainer.py, 约 line 1349-1374
batch.batch["token_level_scores"] = reward_tensor
```

- 使用奖励函数（规则型或奖励模型）计算每个 response 的分数
- **联合训练关注点**：无需修改

#### Step 3: 计算 old_log_probs（Actor 推理）

```python
# ray_trainer.py, 约 line 1391-1423
old_log_prob = self._compute_old_log_prob(batch)
batch = batch.union(old_log_prob)
```

**这是关键步骤！** 它做的事情是：
1. 用当前的 Actor 模型做一次**推理模式**的前向传播（`torch.no_grad()`）
2. 获取当前策略对每个 token 的 log 概率：`π_old(a|s)`
3. 这些 `old_log_probs` 将作为后续策略更新的 baseline

**调用链**：
```
ray_trainer._compute_old_log_prob()
  → actor_rollout_wg.compute_log_prob()  # 分发到各个 Worker
    → FSDPWorker.compute_log_prob()       # fsdp_workers.py:1082
      → self.actor.compute_log_prob()     # dp_actor.py:425
        → self._forward_micro_batch()     # dp_actor.py:113
          → self.actor_module(input_ids, ...) # HuggingFace 模型 forward()
            → output.logits               # 获得 logits
          → logprobs_from_logits(logits, labels) # logits → log_probs
```

**联合训练关注点**：这里使用的是 FSDP 包裹的联合模型（`QwenJointForCausalLM`），它的 `forward()` 内部会自动进行 logit 融合。所以这一步天然就是用融合 logits 计算的 old_log_probs，不需要额外修改。

#### Step 4: 计算优势（Advantage）

```python
# ray_trainer.py, 约 line 1437
batch = compute_advantage(batch, adv_estimator="grpo", ...)
```

GRPO 的优势计算核心逻辑（`core_algos.py:266-330`）：

```python
# 1. 将 token 级别的 reward 求和，得到每个 response 的总分
scores = token_level_rewards.sum(dim=-1)  # (batch_size,)

# 2. 按 prompt ID 分组
for i in range(batch_size):
    id2score[uid[i]].append(scores[i])

# 3. 计算组内统计量
for uid in unique_uids:
    group_mean = mean(id2score[uid])
    group_std = std(id2score[uid])

# 4. 标准化：advantage = (score - group_mean) / (group_std + ε)
# 这就是 "Group Relative" 的含义
```

**联合训练关注点**：无需修改（优势计算是纯函数，只依赖 rewards 和 response_mask）

#### Step 5: 更新策略（Actor Training）

```python
# ray_trainer.py, 约 line 1494
actor_output = self._update_actor(batch)
```

**这是第二个关键步骤！** 它做的事情是：
1. 将 Actor 模型设为训练模式（`model.train()`）
2. 做**带梯度**的前向传播，获得新策略的 log_probs
3. 用 PPO clip loss 计算策略损失
4. 反向传播并更新参数

**调用链**：
```
ray_trainer._update_actor()
  → actor_rollout_wg.update_actor()
    → FSDPWorker.update_actor()            # fsdp_workers.py:986
      → self.actor.update_policy()          # dp_actor.py:509
        → for epoch in range(ppo_epochs):
            for mini_batch in mini_batches:
              for micro_batch in micro_batches:
                → self._forward_micro_batch()  # 带梯度的前向传播
                → policy_loss_fn(old_log_prob, new_log_prob, advantages)
                → loss.backward()
              → optimizer.step()
```

**联合训练关注点**：同 Step 3，联合模型的 `forward()` 内部会自动融合 logits，不需要修改 `_forward_micro_batch()`。

#### Step 6: 评测（Evals）（周期性执行）

```python
# ray_trainer.py, 约 line 1531-1539
if self.global_steps % self.config.trainer.test_freq == 0:
    val_metrics = self._validate()
```

- 使用**同一个 vLLM 引擎**生成评测 response
- 通过 `meta_info["validate"] = True` 标记这是评测
- 计算准确率、奖励等评测指标
- **联合训练关注点**：**评测阶段只用 model2 的权重推理！** 需要在评测前将 vLLM 切换到 model2-only 模式（详见第 9 节）

### 3.3 小结：哪些步骤需要联合训练的 Logit 融合？

| 步骤 | 操作 | 引擎 | 是否需要 Logit 融合 | 原因 |
|------|------|------|---------------------|------|
| Step 1: Rollout | vLLM 生成 | vLLM | **是** ✓ | on-policy 要求：生成策略 = 训练策略 = 融合策略 |
| Step 2: Reward | 计算奖励 | — | **否** | 不涉及模型前向传播 |
| Step 3: compute_log_prob | Actor 推理 | FSDP | **是** ✓ | 需要用融合 logits 计算 old_log_probs |
| Step 4: Advantage | 计算优势 | — | **否** | 纯函数，不涉及模型 |
| Step 5: update_policy | Actor 训练 | FSDP | **是** ✓ | 需要用融合 logits 计算新的 log_probs |
| Step 6: Evals | vLLM 评测 | vLLM | **否** | 只用 model2 评测独立能力 |

> **注意对比 Rollout 和 Evals**：虽然它们都使用 vLLM 推理引擎，但联合训练对它们的要求完全不同。Rollout 需要融合 logits（两个模型协同生成），Evals 只需要 model2（评估独立能力）。这意味着 vLLM 需要支持两种模式切换。

---

## 4. 深入解析：模型是如何被加载的？ {#4-模型加载}

### 4.1 模型加载的入口

**文件**: `verl/workers/fsdp_workers.py`

当训练启动时，每个 Worker 进程会执行 `init_model()` 方法（line 848）来加载模型：

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    # 1. 获取模型路径
    local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

    # 2. 构建模型 + 优化器 + 学习率调度器
    (self.actor_module_fsdp,      # FSDP 包裹后的模型
     self.actor_optimizer,        # 优化器
     self.actor_lr_scheduler,     # 学习率调度器
     self.actor_model_config      # 模型配置
    ) = self._build_model_optimizer(
        model_path=local_path,
        fsdp_config=fsdp_config,
        optim_config=optim_config,
        ...
    )

    # 3. 用 Actor 封装模型（添加 compute_log_prob 等方法）
    self.actor = DataParallelPPOActor(
        config=actor_cfg,
        actor_module=self.actor_module_fsdp,
        actor_optimizer=self.actor_optimizer
    )
```

### 4.2 _build_model_optimizer() 的核心流程

**文件**: `verl/workers/fsdp_workers.py`，lines 328-674

这是整个模型加载的核心函数，让我们逐步解析：

#### 第一步：确定模型类

```python
# line 385-387: 加载模型配置
actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

# line 433-457: 自动选择模型类
# verl 使用 HuggingFace 的 AutoModel 机制来自动选择模型类
# 它会检查 config.json 中的 architectures 字段和 auto_map 字段
if has_remote_code:
    # 如果有自定义模型代码（如 QwenBoostForCausalLM），使用 auto_map
    actor_module_class = AutoModelForCausalLM  # 或其他 AutoModel 变体
else:
    # 否则使用标准的 model mapping
    if type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
        actor_module_class = AutoModelForCausalLM
```

**联合训练关键洞察**：如果我们创建一个自定义模型类（如 `QwenJointForCausalLM`），需要在模型目录的 `config.json` 中设置 `architectures` 字段，让 `AutoModelForCausalLM` 能自动找到它。这和你同事在 SFT 联合训练中做的方式一样！

#### 第二步：加载模型权重

```python
# line 459-465: 从预训练权重加载模型
actor_module = actor_module_class.from_pretrained(
    pretrained_model_name_or_path=local_path,
    torch_dtype=torch_dtype,
    config=actor_model_config,
    trust_remote_code=trust_remote_code,
)
```

#### 第三步：FSDP 包裹

```python
# line 568-580: 确定 FSDP 包裹策略
auto_wrap_policy = get_fsdp_wrap_policy(
    module=actor_module,
    config=fsdp_config.get("wrap_policy", None),
)

# line 591-604: 用 FSDP 包裹模型
actor_module_fsdp = FSDP(
    actor_module,
    auto_wrap_policy=auto_wrap_policy,    # 包裹策略
    device_id=get_device_id(),            # GPU 设备
    sharding_strategy=sharding_strategy,   # FULL_SHARD (ZeRO-3)
    mixed_precision=mixed_precision,       # 混合精度
    sync_module_states=True,               # 同步模型状态
    device_mesh=self.device_mesh,          # 设备网格
)
```

**FSDP 是什么？** FSDP（Fully Sharded Data Parallel）是 PyTorch 提供的分布式训练方案，它会：
1. 将模型参数分片存储在不同 GPU 上（每个 GPU 只存一部分参数）
2. 在前向传播时自动聚合需要的参数
3. 在反向传播时自动分片梯度

**联合训练关键洞察**：如果联合训练模型类（`QwenJointForCausalLM`）内部有两个子模型，FSDP 会自动处理两个子模型的参数分片。FSDP 包裹的是最外层的模型，内部的子模型参数都会被正确地分片。

#### 第四步：构建优化器

```python
# line 637-660: 构建优化器
actor_optimizer = build_optimizer(
    model=actor_module_fsdp,
    optim_config=optim_config,
)
```

### 4.3 模型加载总结

```
配置文件中的 model.path
       │
       ▼
  AutoConfig.from_pretrained()  ← 读取 config.json
       │
       ▼
  确定 AutoModel 类  ← 基于 architectures / auto_map
       │
       ▼
  AutoModelForCausalLM.from_pretrained()  ← 加载权重
       │
       ▼
  apply_monkey_patch()  ← 性能优化补丁
       │
       ▼
  FSDP()  ← 分布式训练包裹
       │
       ▼
  DataParallelPPOActor()  ← 封装为 Actor（添加 RL 训练接口）
```

---

## 5. 深入解析：前向传播与 logits / log_probs 的计算 {#5-前向传播}

### 5.1 _forward_micro_batch()：前向传播的核心

**文件**: `verl/workers/actor/dp_actor.py`，lines 113-389

这是所有前向传播的最终入口。不管是 `compute_log_prob`（推理）还是 `update_policy`（训练），最终都调用这个函数。

```python
def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
    with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        # 调用模型的 forward 方法
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        # output.logits 的 shape: [batch_size, seq_len, vocab_size]

        logits = output.logits
        logits.div_(temperature)  # 除以温度

        # 只取 response 部分的 logits
        logits = logits[:, -response_length - 1 : -1, :]

        # 将 logits 转换为 log_probs
        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
        # log_probs 的 shape: [batch_size, response_length]
```

**这里发生了什么？**

1. `self.actor_module(input_ids, ...)` 就是调用 HuggingFace 模型的 `forward()` 方法
2. 模型返回 `logits`，shape 是 `[batch_size, seq_len, vocab_size]`（vocab_size 通常是 151936 for Qwen）
3. `logits.div_(temperature)` 用温度缩放 logits（温度越高，分布越平滑）
4. `logprobs_from_logits()` 将 logits 转换为每个实际 token 的 log 概率

### 5.2 logprobs_from_logits()：从 logits 到 log_probs

**文件**: `verl/utils/torch_functional.py`，lines 72-200

```python
def logprobs_from_logits(logits, labels):
    """
    计算每个 token 的 log 概率。

    logits: [batch, seq_len, vocab_size] — 模型输出的原始分数
    labels: [batch, seq_len] — 实际的 token ID

    返回: [batch, seq_len] — 每个位置上实际 token 的 log 概率
    """
    # 本质就是: log_softmax(logits)[labels]
    # 即：先对 logits 做 softmax 得到概率分布，再取 log，再用 labels 做 gather
```

**为什么这一步很重要？** 在 RL 中，策略 loss 的计算需要两个关键量：
- `old_log_probs`: 旧策略的 log 概率（步骤 3 计算）
- `new_log_probs`: 新策略的 log 概率（步骤 5 计算）

两者的比值（重要性比率 ratio）就是 PPO 的核心：

```
ratio = exp(new_log_prob - old_log_prob) = π_new(a|s) / π_old(a|s)
```

### 5.3 联合训练如何修改前向传播

在联合训练中，`_forward_micro_batch()` 需要变成这样：

```python
# 伪代码 — 联合训练的前向传播
def _forward_micro_batch(self, micro_batch, temperature, ...):
    # 原来: output = self.actor_module(input_ids, ...)
    # 现在: self.actor_module 本身就是联合模型（QwenJointForCausalLM）
    #       它的 forward() 内部会自动处理两个子模型的 logit 融合
    output = self.actor_module(input_ids, attention_mask, position_ids, ...)

    # output.logits 已经是融合后的 logits（如果是联合训练模型的话）
    logits = output.logits
    logits.div_(temperature)
    log_probs = logprobs_from_logits(logits, labels)
```

**关键设计决策**：logit 融合应该在模型类的 `forward()` 方法内部完成，而不是在 `_forward_micro_batch()` 中完成。这样做的好处是：
1. **最小化修改范围**：不需要修改 `dp_actor.py` 的任何代码
2. **保持接口一致**：联合训练模型对外暴露的接口和普通模型完全一样（`input_ids` 进去，`logits` 出来）
3. **遵循 verl 的架构原则**：模型实现细节封装在模型类内部

---

## 6. 深入解析：策略更新（Actor Training） {#6-策略更新}

### 6.1 update_policy() 的完整流程

**文件**: `verl/workers/actor/dp_actor.py`，lines 509-676

```python
def update_policy(self, data: DataProto):
    self.actor_module.train()  # 设为训练模式

    # 准备数据
    mini_batches = data.split(self.config.ppo_mini_batch_size)

    for epoch in range(self.config.ppo_epochs):       # 多个 epoch
        for mini_batch in mini_batches:               # 多个 mini-batch
            self.actor_optimizer.zero_grad()           # 清零梯度

            for micro_batch in micro_batches:         # 梯度累积
                # ===== 带梯度的前向传播 =====
                outputs = self._forward_micro_batch(model_inputs, temperature)
                new_log_prob = outputs["log_probs"]

                # ===== 计算策略 loss =====
                old_log_prob = model_inputs["old_log_probs"]  # 来自 Step 3
                advantages = model_inputs["advantages"]        # 来自 Step 4

                # PPO clip loss
                ratio = exp(new_log_prob - old_log_prob)
                pg_loss = -advantages * clip(ratio, [1-ε, 1+ε])

                # ===== 反向传播 =====
                loss = pg_loss * scale_factor
                loss.backward()

            # ===== 优化器步骤 =====
            grad_norm = clip_grad_norm_(max_norm=1.0)
            optimizer.step()
```

### 6.2 PPO / GRPO 的损失函数

**文件**: `verl/trainer/ppo/core_algos.py`，`compute_policy_loss_vanilla()` (line 1159)

```python
@register_policy_loss("vanilla")  # 通过注册机制注册为 "vanilla"
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, ...):
    # 1. 计算重要性比率
    ratio = torch.exp(log_prob - old_log_prob)  # π_new / π_old

    # 2. 两个 loss 取最大值（PPO clip 的核心）
    pg_losses1 = -advantages * ratio                              # 无裁剪
    pg_losses2 = -advantages * clip(ratio, 1-ε, 1+ε)            # 有裁剪
    pg_loss = max(pg_losses1, pg_losses2)                         # 取较大的（较保守的）

    # 3. 在 response mask 上聚合
    pg_loss = masked_mean(pg_loss, response_mask)

    return pg_loss, metrics
```

**联合训练关注点**：loss 函数不需要修改！因为它只接收 `log_prob` 和 `old_log_prob`，不关心这些 log_prob 是怎么计算出来的（单模型还是融合模型）。

### 6.3 梯度如何流回联合训练的两个模型？

当使用联合训练模型时，梯度流是这样的：

```
loss.backward()
    │
    ▼
∂loss/∂logits_fused        ← loss 对融合 logits 的梯度
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
∂logits_fused/∂logits_0 = (1-λ)     ∂logits_fused/∂logits_1 = λ
    │                                          │
    ▼                                          ▼
∂loss/∂logits_0 = (1-λ) × grad      ∂loss/∂logits_1 = λ × grad
    │                                          │
    ▼                                          ▼
更新 model1 的参数                    更新 model2 的参数
（如果 model1 未冻结）                （始终更新）
```

这就是为什么 logit 融合可以自动传播梯度到两个子模型——PyTorch 的自动微分（autograd）会自动处理这一切。

---

## 7. 深入解析：vLLM 推理引擎与权重同步 {#7-vllm推理}

### 7.1 vLLM 在 verl 中的角色

在 verl 中，vLLM 负责两个推理场景：
1. **Rollout 生成**：在训练循环中生成 response（每个 Step 都会执行）
2. **Evals 评测**：周期性地评估模型性能（由 `test_freq` 控制）

**这两个场景使用同一个 vLLM 引擎实例**，但在联合训练中它们的行为需要不同：

```
┌─────────────────────────────────────────────────────────────────┐
│                    同一个 vLLM 引擎，两种模式                      │
│                                                                 │
│  Rollout 生成模式:                                               │
│    → vLLM 需要用联合模型（两个子模型融合 logits）                  │
│    → 因为 on-policy 要求生成策略 = 训练策略                        │
│                                                                 │
│  Evals 评测模式:                                                 │
│    → vLLM 只需要 model2 的权重                                    │
│    → 因为评测的是 model2 的独立能力                                │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 GPU 内存分时复用

vLLM 引擎和训练模型共享 GPU 内存，通过 sleep/wake 机制切换：

```
训练阶段:
  ┌──────────────────────────────────┐
  │ GPU 内存                          │
  │ [训练模型权重] [优化器] [梯度]     │
  │                                  │
  │ vLLM weights: SLEEPING (CPU)     │
  └──────────────────────────────────┘

推理阶段:
  ┌──────────────────────────────────┐
  │ GPU 内存                          │
  │ [vLLM 模型权重] [KV Cache]       │
  │                                  │
  │ FSDP weights: offloaded to CPU   │
  └──────────────────────────────────┘
```

**注意：训练模型和推理模型不是同一个对象！** 训练侧用 HuggingFace + FSDP 包裹的模型，推理侧用 vLLM 自己内部的模型实现。每次切换时需要通过权重同步保持一致。

### 7.3 权重同步的完整流程

**文件**: `verl/workers/fsdp_workers.py`，`rollout_mode()` (line 740)

当需要从训练模式切换到推理模式时：

```python
async def rollout_mode(self):
    # 1. 从 FSDP 模型中提取全量参数
    params = self.actor_module_fsdp.state_dict()  # DTensor → 全量 Tensor

    # 2. 转换参数名字（HuggingFace → vLLM 格式）
    params = convert_weight_keys(params, self.actor_module_fsdp)

    # 3. 将参数传输给 vLLM（通过 CUDA IPC 或共享内存）
    await self.rollout.update_weights(params)
```

**权重传输机制**（`vllm_rollout.py:153-246`）：

```
FSDP Worker                          vLLM Worker
    │                                    │
    │ ── ZMQ handshake ──────────────►   │
    │                                    │
    │ ── CUDA IPC handle ────────────►   │
    │    (GPU内存共享句柄)                │
    │                                    │
    │ ── 权重数据(分桶传输) ─────────►    │
    │    bucket1: {name, shape, dtype}   │
    │    bucket2: ...                     │
    │                                    │
    │ ◄── 确认接收 ──────────────────    │
    │                                    │
    │                               model.load_weights()
```

### 7.4 vLLM 使用的模型类

**关键发现**：vLLM 有自己独立的模型实现，不使用 HuggingFace 模型！

当 vLLM 初始化时，它根据 `config.json` 中的 `architectures` 字段选择自己内部的模型实现。比如对于 Qwen3，vLLM 会用自己的 `Qwen3ForCausalLM` 实现（这个和 HuggingFace 的 `Qwen3ForCausalLM` 是不同的类）。

**在 HYBRID 模式下**（训练和推理共享 GPU）：
- vLLM 以 `load_format="dummy"` 启动（不实际加载权重）
- 权重从 FSDP 训练模型同步过来
- 这就是为什么 `convert_weight_keys()` 函数很重要——它将 HuggingFace 格式的权重名转换为 vLLM 格式

**vLLM 也支持自定义模型类**：当 `trust_remote_code=True` 时，vLLM 可以从模型目录的 `config.json` 中的 `auto_map` 加载自定义模型。verl 在 `vllm_async_server.py:302` 处将 `trust_remote_code` 传递给 vLLM 引擎：

```python
# vllm_async_server.py, line 296-321
args = {
    "trust_remote_code": self.model_config.trust_remote_code,  # line 302
    "load_format": self.config.load_format,  # HYBRID 模式下为 "dummy"
    ...
}
```

### 7.5 联合训练对 vLLM 推理的影响（核心难点）

联合训练需要 vLLM 支持两种不同的推理模式。这是整个项目中最复杂的部分。

#### 场景一：Rollout 生成 — 需要融合 logits

```
训练模型: QwenJointForCausalLM (包含 model1 + model2)
                    │
                    │ 权重同步：需要同步 model1 + model2 的完整参数
                    ▼
vLLM 推理模型: 也需要是 QwenJointForCausalLM（或等效的融合模型）
                    │
                    │ 生成时：内部对两个子模型的 logits 融合后采样
                    ▼
                生成的 response（基于融合策略）
```

**这意味着**：
- vLLM 端也需要加载一个支持 logit 融合的模型类
- 权重同步时需要传输两个子模型的完整参数
- 参数名可能带有 `sub_models.0.` 和 `sub_models.1.` 前缀，需要正确处理

#### 场景二：Evals 评测 — 只用 model2

```
训练模型: QwenJointForCausalLM (包含 model1 + model2)
                    │
                    │ 权重同步：只提取 model2 的参数
                    ▼
vLLM 推理模型: 标准的 Qwen3ForCausalLM (只包含 model2)
                    │
                    │ 生成时：只用 model2 的标准推理
                    ▼
                生成的 response（基于 model2 独立策略）
```

**这意味着**：
- 评测前需要将 vLLM 的模型切换回标准架构（或切换权重）
- 只提取并同步 `sub_models.1.` 前缀的参数，去掉前缀后传给 vLLM

#### 两种模式切换的时机

```
Training Step N:
  1. rollout_mode() → 同步联合模型权重到 vLLM → Rollout 生成（融合 logits）
  2. trainer_mode() → 训练（FSDP 联合模型）
  3. （每 test_freq 步）eval_mode() → 同步 model2 权重到 vLLM → Evals（只用 model2）

Training Step N+1:
  1. rollout_mode() → 再次同步联合模型权重 → Rollout 生成（融合 logits）
  ...
```

### 7.6 vLLM 中实现联合模型的可能方案

#### 方案 A：自定义 vLLM 模型类（推荐）

创建一个 vLLM 兼容的联合模型类，在 `compute_logits()` 中实现融合：

```python
# 伪代码 — vLLM 端的联合模型
class QwenJointForCausalLM_vLLM(nn.Module):
    """vLLM 版本的联合模型"""
    def __init__(self, config):
        self.model1 = Qwen3Model(config)  # vLLM 的内部实现
        self.model2 = Qwen3Model(config)
        self.lm_head1 = ...
        self.lm_head2 = ...

    def compute_logits(self, hidden_states1, hidden_states2):
        logits1 = self.lm_head1(hidden_states1)
        logits2 = self.lm_head2(hidden_states2)
        return (1 - λ) * logits1 + λ * logits2

    def load_weights(self, weights):
        # 处理 sub_models.0.* 和 sub_models.1.* 前缀
        for name, tensor in weights:
            if name.startswith("sub_models.0."):
                self.model1.load_param(name.replace("sub_models.0.", ""), tensor)
            elif name.startswith("sub_models.1."):
                self.model2.load_param(name.replace("sub_models.1.", ""), tensor)
```

**优点**：最干净的实现，vLLM 的所有优化（PagedAttention, continuous batching）都能正常工作
**缺点**：需要编写和维护 vLLM 兼容的自定义模型，需要处理 vLLM 的权重加载接口

通过 `trust_remote_code=True`，vLLM 可以从模型目录加载这个自定义类。

#### 方案 B：Monkey-patch `compute_logits()`

利用 verl 已有的 monkey-patch 机制，在 vLLM worker 中拦截 logits 计算：

```python
# 参考: verl/workers/rollout/vllm_rollout/utils.py, lines 93-105
# verl 已经有 monkey_patch_compute_logits() 的先例

def monkey_patch_joint_logits(model1, model2, fusion_lambda):
    """在 vLLM 的 compute_logits 中注入 logit 融合"""
    original_compute_logits = model2.compute_logits

    def fused_compute_logits(self, hidden_states, ...):
        logits2 = original_compute_logits(hidden_states, ...)
        # 同时计算 model1 的 logits
        logits1 = model1.compute_logits(hidden_states_from_model1, ...)
        return (1 - fusion_lambda) * logits1 + fusion_lambda * logits2

    model2.compute_logits = MethodType(fused_compute_logits, model2)
```

verl 通过 `collective_rpc` 机制将 monkey-patch 广播到所有 vLLM worker（`vllm_async_server.py:450-452`）。可以扩展 `vLLMColocateWorkerExtension` 类添加新的 patch 方法。

**优点**：不需要创建完整的 vLLM 自定义模型
**缺点**：实现较为 hacky，需要在 vLLM 内部同时维护两个模型的前向传播状态

#### 方案 C：只在 HuggingFace 侧做融合，vLLM 用近似策略

Rollout 时仍然只用 model2 的权重通过标准 vLLM 生成，但接受这带来的 off-policy 偏差。然后在 compute_log_prob 阶段用融合 logits 重新计算 log_probs 进行修正。

**优点**：vLLM 完全不需要修改
**缺点**：违反 on-policy 假设，可能影响训练稳定性和收敛性。但在实践中，如果两个模型差异不大，off-policy 偏差可能可以接受。

### 7.7 Sleep/Wake 机制：GPU 内存管理

verl 使用 vLLM 的 sleep/wake 机制来管理 GPU 内存：

**Sleep Levels**：
- Level 1: 只释放 KV Cache（权重保留在 GPU）
- Level 2: 释放权重 + KV Cache（完全释放 GPU 内存）

**在联合训练中的意义**：如果 vLLM 需要加载两个子模型的权重（方案 A），GPU 内存压力会更大。Sleep Level 2 在释放时可以完全释放两个子模型的权重，但恢复时也需要重新传输两倍的权重数据。

### 7.8 `convert_weight_keys()`：参数名转换

**文件**: `verl/utils/model.py`，lines 230-248

```python
def convert_weight_keys(state_dict, model):
    """将 HuggingFace 内存中的参数名转换为 checkpoint 格式的参数名"""
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict  # 没有映射表就原样返回

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    # 使用正则表达式做名字替换
    ...
```

**联合训练的关键问题**：
- 如果自定义模型 `QwenJointForCausalLM` 没有定义 `_checkpoint_conversion_mapping`，这个函数会原样返回 state_dict
- 原样返回的 state_dict 中参数名会带有 `sub_models.0.` 和 `sub_models.1.` 前缀
- 这些带前缀的参数名需要与 vLLM 端模型的 `load_weights()` 期望的名字一致

---

## 8. 深入解析：FSDP 分布式训练与权重管理 {#8-fsdp分布式}

### 8.1 FSDP 基础知识（给小白的版本）

假设你有一个 7B 参数的模型和 4 张 GPU。

**没有 FSDP**：每张 GPU 都需要存储完整的 7B 参数 → 每张 GPU 需要 14GB（bf16）

**有 FSDP（FULL_SHARD = ZeRO-3）**：每张 GPU 只存 7B/4 ≈ 1.75B 参数 → 每张 GPU 只需要 3.5GB

```
GPU 0: [参数 0-1.75B]
GPU 1: [参数 1.75B-3.5B]
GPU 2: [参数 3.5B-5.25B]
GPU 3: [参数 5.25B-7B]

前向传播时: 需要某一层的完整参数 → 所有 GPU AllGather → 计算 → 释放
反向传播时: 同样 AllGather → 计算梯度 → ReduceScatter 梯度 → 释放
```

### 8.2 FSDP 包裹策略（Wrap Policy）

**文件**: `verl/utils/fsdp_utils.py`，`get_fsdp_wrap_policy()` (line 75)

FSDP 的 wrap policy 决定了哪些模块被独立分片。通常选择 Transformer 层作为分片单元：

```python
# 以 Qwen3 为例，每个 Qwen3DecoderLayer 被单独包裹
# 这意味着每个层的参数在需要时才会被 AllGather
Module: Qwen3ForCausalLM
  ├── FSDP(embed_tokens)        # 嵌入层
  ├── FSDP(layers.0)            # Transformer 第 0 层
  ├── FSDP(layers.1)            # Transformer 第 1 层
  ├── ...
  ├── FSDP(layers.N)            # Transformer 第 N 层
  ├── FSDP(norm)                # LayerNorm
  └── FSDP(lm_head)             # 语言模型头
```

**联合训练关键洞察**：如果联合训练模型有两个子模型，FSDP 需要能正确地包裹它们。有两种方式：
1. **让 FSDP 包裹整个联合模型**：FSDP 会递归地找到所有 Transformer 层并分片
2. **分别包裹两个子模型**：需要更复杂的配置

推荐方式 1，因为 verl 的 FSDP 包裹策略会自动根据 `_no_split_modules` 找到 Transformer 层。

### 8.3 权重的 CPU 卸载与 GPU 加载

**文件**: `verl/utils/fsdp_utils.py`

verl 支持将模型权重在 CPU 和 GPU 之间来回移动，以节省 GPU 内存：

```python
# 将 FSDP 模型卸载到 CPU
def offload_fsdp_model_to_cpu(model):
    for handle in model._all_handles:
        handle.flat_param_to(torch.device("cpu"))

# 将 FSDP 模型加载到 GPU
def load_fsdp_model_to_gpu(model):
    for handle in model._all_handles:
        handle.flat_param_to(torch.device(f"cuda:{device_id}"))
```

**这在联合训练中的意义**：联合训练模型的参数量是单模型的约 2 倍，CPU 卸载功能对于管理 GPU 内存变得更加重要。

---

## 9. 深入解析：评测（Evaluation）与指标（Metrics） {#9-评测与指标}

### 9.1 评测是如何触发的？

**文件**: `verl/trainer/ppo/ray_trainer.py`

评测在两个时机触发：

#### 训练前的初始评测

```python
# ray_trainer.py, 约 line 1246-1252
if self.config.trainer.get("val_before_train", True):
    val_metrics = self._validate()
    pprint(f"Initial validation metrics: {val_metrics}")
```

#### 训练中的周期性评测

```python
# ray_trainer.py, 约 line 1531-1539
if self.config.trainer.test_freq > 0 and (
    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
):
    val_metrics = self._validate()
```

评测频率由 `self.config.trainer.test_freq` 控制。设置为 0 则不评测。

### 9.2 `_validate()` 方法的完整流程

**文件**: `verl/trainer/ppo/ray_trainer.py`，lines 496-619

这是评测的核心方法，让我们逐步解析：

```python
def _validate(self):
    # 1. 遍历验证数据集
    for test_data in self.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        # 2. 重复每个样本 N 次（用于 best@N, maj@N 等统计）
        test_batch = test_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
            interleave=True
        )

        # 3. 设置评测专用的 meta_info（关键区别！）
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,     # 评测不需要重新计算 log_prob
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,                # ← 关键标记：这是评测！
            "global_steps": self.global_steps,
        }

        # 4. 使用同一个 vLLM 引擎生成
        test_output = self.async_rollout_manager.generate_sequences(test_gen_batch)

        # 5. 计算奖励
        reward_tensor, reward_extra_info = extract_reward(test_batch)

    # 6. 汇总评测指标
    return process_validation_metrics(data_sources, sample_uids, infos_dict)
```

### 9.3 Rollout vs Evals 的关键区别

**评测使用同一个 vLLM 引擎，但行为不同：**

| 方面 | Rollout（训练生成） | Evals（评测） |
|------|-------|-------|
| `validate` 标记 | `False`（默认） | `True` |
| 采样参数 | `config.rollout.temperature/top_p/top_k` | `config.rollout.val_kwargs.temperature/top_p/top_k` |
| 重复次数 | `config.rollout.n` | `config.rollout.val_kwargs.n` |
| log_prob 重计算 | 生成后会计算 | `recompute_log_prob: False` |
| vLLM 引擎 | **同一个引擎** | **同一个引擎** |
| 联合训练模型行为 | **融合 logits（两个模型）** | **只用 model2** |

**`validate` 标记如何影响采样参数？**

在 `AgentLoopManager`（`verl/experimental/agent_loop/agent_loop.py:416-429`）中：

```python
sampling_params = dict(
    temperature=config.temperature,
    top_p=config.top_p,
    top_k=config.top_k,
    ...
)
# 如果是评测，覆盖采样参数
if batch.meta_info.get("validate", False):
    sampling_params["top_p"] = config.val_kwargs.top_p
    sampling_params["top_k"] = config.val_kwargs.top_k
    sampling_params["temperature"] = config.val_kwargs.temperature
```

**注意**：目前 `validate` 标记只改变采样参数（温度、top_p、top_k），**不改变模型行为**。这意味着在联合训练中，我们需要额外的机制来让 vLLM 在评测时切换到 model2-only 模式。

### 9.4 评测指标（Metrics）详解

**文件**: `verl/trainer/ppo/metric_utils.py`

verl 计算三类指标：

#### 训练指标 `compute_data_metrics()` (line 81-225)

在每个训练 step 结束后计算：

| 指标类别 | 指标名 | 含义 |
|---------|--------|------|
| 奖励 | `critic/score/mean,max,min` | 每个 response 的总分 |
| 奖励 | `critic/rewards/mean,max,min` | 包含 KL 惩罚的奖励 |
| 优势 | `critic/advantages/mean,max,min` | response token 上的优势值 |
| 长度 | `response_length/mean,max,min` | 生成的 response 长度 |
| 长度 | `response/aborted_ratio` | 零长度 response 的比例 |
| 长度 | `prompt_length/mean,max,min` | prompt 长度 |

#### 时序/吞吐指标

| 函数 | 计算内容 |
|------|---------|
| `compute_timing_metrics()` | 每个阶段（gen, ref, update_actor 等）的耗时 |
| `compute_throughout_metrics()` | 总 token 数、每步耗时、tokens/second/GPU |

#### 评测指标 `process_validation_metrics()` (line 511-659)

这是评测阶段的指标，更加丰富：

| 指标名 | 含义 |
|--------|------|
| `mean@N` | N 个样本的平均得分 |
| `std@N` | N 个样本的标准差 |
| `best@N/mean` | Bootstrap 采样后最佳样本的平均 |
| `worst@N/mean` | Bootstrap 采样后最差样本的平均 |
| `maj@N/mean` | 多数投票的准确率（当有预测字段时） |

评测结果按前缀分类：
- `val-core/`：核心准确率/奖励指标
- `val-aux/`：辅助指标

### 9.5 联合训练对评测和指标的影响

#### 评测流程的修改

联合训练需要在评测前切换 vLLM 的模型行为：

```
正常训练流程:
  rollout_mode() → Rollout 生成（融合 logits） → 训练 → ...

评测时:
  eval_rollout_mode() → 只同步 model2 权重 → Evals 生成（只用 model2） → 计算评测指标
  rollout_mode()     → 恢复同步联合模型权重（为下一轮 Rollout 做准备）
```

这意味着每次评测前后都需要额外的权重同步操作。

#### 指标的额外考虑

联合训练可能需要新增一些比较指标：

| 新指标（建议） | 含义 | 用途 |
|---------------|------|------|
| `joint/fusion_lambda` | 当前的融合系数 | 记录超参数 |
| `joint/model1_logits_mean` | model1 logits 的均值 | 监控两个模型的分布差异 |
| `joint/model2_logits_mean` | model2 logits 的均值 | 监控两个模型的分布差异 |
| `joint/logits_kl_divergence` | 两个模型 logits 的 KL 散度 | 监控两个模型是否趋向一致 |

这些指标不影响训练本身，但对理解联合训练的动态非常有帮助。可以在 `_forward_micro_batch()` 或者联合模型的 `forward()` 中计算并通过 DataProto 传递。

---

## 10. 联合训练需要修改的位置与原因分析 {#10-修改方案}

### 10.1 修改点全局视图

```
需要修改的文件和原因：

1. 新建 HuggingFace 联合模型类（如 modeling_joint_qwen3.py）
   原因: 定义 QwenJointForCausalLM，实现 logit 融合的 forward()
   用途: FSDP 训练侧的前向传播（compute_log_prob + update_policy）

2. 新建 vLLM 联合模型类（或 monkey-patch 扩展）
   原因: vLLM Rollout 生成时也需要融合 logits
   用途: Rollout 阶段的推理

3. verl/workers/fsdp_workers.py 的 rollout_mode() / 权重同步
   原因: 需要两种权重同步模式 —
         Rollout: 同步完整联合模型权重
         Evals:   只同步 model2 权重

4. verl/workers/fsdp_workers.py 的 init_model()（可能需要小改动）
   原因: 可能需要配置冻结 model1 参数

5. 配置文件（新建一个 GRPO 联合训练的 YAML 配置）
   原因: 添加联合训练特定的配置项（fusion_lambda, freeze_model1 等）
```

### 10.2 修改点 1：创建 HuggingFace 联合训练模型类

**为什么需要**：verl 通过 `AutoModelForCausalLM.from_pretrained()` 加载模型。为了让 logit 融合发生在模型内部（对 verl 的其他代码透明），我们需要一个自定义模型类。

**参考实现**（来自 SFT 联合训练）：

```python
class QwenJointForCausalLM(PreTrainedModel):
    """联合训练模型：内部包含两个子模型，forward 时自动进行 logit 融合"""

    def __init__(self, config):
        super().__init__(config)
        # 两个子模型
        self.sub_models = nn.ModuleList([
            Qwen3ForCausalLM(config),
            Qwen3ForCausalLM(config),
        ])

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # 1. 两个子模型独立前向传播
        outputs_list = []
        for model in self.sub_models:
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       labels=None, **kwargs)
            outputs_list.append(out)

        # 2. 加权融合 logits
        fusion_lambda = getattr(self.config, "fusion_lambda", 0.5)
        logits = (1 - fusion_lambda) * outputs_list[0].logits + \
                     fusion_lambda * outputs_list[1].logits

        # 3. 返回标准格式的输出
        return CausalLMOutputWithPast(logits=logits, ...)
```

**如何让 verl 识别这个模型类？**

需要在模型目录的 `config.json` 中设置：
```json
{
    "architectures": ["QwenJointForCausalLM"],
    "auto_map": {
        "AutoModelForCausalLM": "modeling_joint_qwen3.QwenJointForCausalLM"
    },
    "model_type": "qwen3",
    "fusion_lambda": 0.5,
    ...
}
```

然后配置 `trust_remote_code=True`，HuggingFace 的 AutoModel 机制会自动从模型目录加载自定义模型类。

### 10.3 修改点 2：vLLM 端的联合模型推理支持（Rollout 生成）

**为什么需要**：Rollout 生成阶段需要用融合 logits 进行采样。vLLM 是推理引擎，它也需要理解联合模型。

**推荐方案**：自定义 vLLM 模型类 + `trust_remote_code`

1. 创建一个 vLLM 兼容的联合模型类，放在模型目录中
2. 实现 `load_weights()` 方法，能处理 `sub_models.0.*` 和 `sub_models.1.*` 前缀的参数
3. 实现 `forward()` 或 `compute_logits()`，在其中融合两个子模型的 logits
4. 通过 `config.json` 的 `auto_map` 让 vLLM 自动加载这个类

**权重名匹配问题的关键分析**：

当 FSDP 端调用 `state_dict()` 时，联合模型的参数名格式为：
```
sub_models.0.model.embed_tokens.weight
sub_models.0.model.layers.0.self_attn.q_proj.weight
...
sub_models.1.model.embed_tokens.weight
sub_models.1.model.layers.0.self_attn.q_proj.weight
...
```

`convert_weight_keys()` 函数（`verl/utils/model.py:230-248`）会检查模型是否有 `_checkpoint_conversion_mapping`。如果自定义联合模型**没有**定义这个映射，参数名会**原样传递**给 vLLM。

因此，vLLM 端的联合模型类的 `load_weights()` 需要能接受这些带 `sub_models.X.` 前缀的参数名。

**替代方案**：如果创建完整的 vLLM 联合模型过于复杂，可以考虑：

- 利用 `monkey_patch_compute_logits()` 的先例（`utils.py:93-105`），通过 monkey-patch 注入融合逻辑
- 扩展 `vLLMColocateWorkerExtension` 类（`utils.py:188-192`），添加一个 `load_second_model()` 方法来加载 model1
- 通过 `collective_rpc` 机制在所有 vLLM worker 上执行 patch

### 10.4 修改点 3：权重同步的两种模式

**为什么需要**：Rollout 和 Evals 需要同步不同的权重给 vLLM。

**需要修改的位置**：`fsdp_workers.py` 的 `rollout_mode()` 方法

```python
async def rollout_mode(self, eval_only=False):
    """
    切换到推理模式并同步权重到 vLLM。

    Args:
        eval_only: 如果 True，只同步 model2 的权重（用于 Evals）
                   如果 False，同步完整的联合模型权重（用于 Rollout）
    """
    full_state_dict = self.actor_module_fsdp.state_dict()

    if eval_only and is_joint_model(self.actor_module_fsdp):
        # Evals 模式：只提取 model2 的参数，去掉前缀
        params = {}
        for name, param in full_state_dict.items():
            if name.startswith("sub_models.1."):
                new_name = name.replace("sub_models.1.", "")
                params[new_name] = param
    else:
        # Rollout 模式：使用完整的联合模型参数
        params = full_state_dict

    params = convert_weight_keys(params, ...)
    await self.rollout.update_weights(params, ...)
```

**还需要在 `ray_trainer.py` 中协调**：在调用 `_validate()` 前后切换模式

```python
# 伪代码 — ray_trainer.py 中的评测流程
def _validate(self):
    # 评测前：切换到 model2-only 模式
    self.actor_rollout_wg.rollout_mode(eval_only=True)

    # ... 执行评测 ...

    # 评测后：如果下一步是 Rollout，需要恢复到联合模型模式
    # （这会在下一个训练 step 的 rollout_mode() 中自动处理）
```

### 10.5 修改点 4：冻结 model1 参数（可选）

**为什么可能需要**：在联合训练中，可能只想训练 model2，将 model1 作为固定的"老师"。

**在哪里修改**：`fsdp_workers.py` 的 `init_model()` 方法中，模型加载之后：

```python
# 在 init_model() 中，self._build_model_optimizer() 之后
if self.config.model.get("freeze_model1", False):
    # 冻结 model1 的所有参数
    unwrapped = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
    if hasattr(unwrapped, "sub_models"):
        for param in unwrapped.sub_models[0].parameters():
            param.requires_grad = False
```

**注意**：冻结参数需要在 FSDP 包裹之前或之后小心处理。如果在 FSDP 包裹之后冻结，可能需要考虑 FSDP 的 `use_orig_params=True` 设置。

### 10.6 修改点 5：Checkpoint 保存与加载

**为什么需要**：保存 checkpoint 时需要保存整个联合模型，加载时也需要正确恢复。

**当前实现**（`fsdp_workers.py:1166-1238`）：
```python
def save_checkpoint(self, local_path, ...):
    self.checkpoint_manager.save_checkpoint(local_path=local_path, ...)
```

**联合训练的考虑**：
- 保存：保存完整的联合模型 state_dict（默认行为，不需要修改）
- 加载：从两个独立模型的 checkpoint 初始化联合模型（需要在模型类的 `__init__` 或 `from_pretrained` 中处理）

### 10.7 修改点总结与风险评估

| 修改点 | 文件 | 难度 | 风险 | 理由 |
|--------|------|------|------|------|
| 创建 HF 联合模型类 | 新建文件 | ★★★☆☆ | 低 | 参考 SFT 实现，逻辑相同 |
| 创建 vLLM 联合模型类 | 新建文件 | ★★★★★ | 高 | 需要理解 vLLM 内部模型接口，需要实现 `load_weights()` |
| 权重同步双模式 | fsdp_workers.py | ★★★★☆ | 中 | 需要区分 Rollout 和 Evals 的权重同步逻辑 |
| 评测模式切换 | ray_trainer.py | ★★★☆☆ | 中 | 需要在评测前后切换 vLLM 模型状态 |
| 冻结 model1 参数 | fsdp_workers.py | ★★☆☆☆ | 低 | 简单的 requires_grad 设置 |
| Checkpoint 管理 | 新建 / 现有文件 | ★★☆☆☆ | 低 | 可以延后处理 |
| 配置文件 | 新建 YAML | ★☆☆☆☆ | 低 | 纯配置 |

### 10.8 不需要修改的部分（以及为什么）

| 组件 | 为什么不需要改 |
|------|---------------|
| `dp_actor.py` | 联合模型的 logit 融合在模型类内部完成，Actor 只看到标准的 `output.logits` |
| `core_algos.py` | 策略 loss 函数只接收 `log_probs`，不关心它们的来源 |
| `torch_functional.py` | `logprobs_from_logits()` 是纯数学函数，不涉及模型 |
| `metric_utils.py` | 指标计算只依赖 tensor 数据，不涉及模型架构 |

**这就是分层架构的力量**：通过在正确的层级（模型类 + Worker 权重同步）做修改，上层的代码完全不需要改变。

### 10.9 实现的建议优先级

```
Phase 1（核心功能 — 训练侧）:
  ✅ 创建 HuggingFace 联合模型类（QwenJointForCausalLM）
  ✅ 验证 FSDP 包裹、前向传播、反向传播都正常工作
  ✅ 用方案 C（Rollout 只用 model2）快速跑通端到端流程

Phase 2（完善推理侧）:
  ⬜ 创建 vLLM 联合模型类或 monkey-patch 方案
  ⬜ 实现 Rollout 阶段的融合 logits 推理
  ⬜ 实现 Evals 阶段的 model2-only 推理
  ⬜ 验证 on-policy 一致性

Phase 3（优化）:
  ⬜ 添加联合训练专用监控指标
  ⬜ GPU 内存优化（两个模型的显存管理）
  ⬜ Checkpoint 保存/加载优化
  ⬜ 配置文件和文档完善
```

> **为什么建议 Phase 1 先用方案 C？** 因为训练侧（FSDP 联合模型）是确定性的修改，风险可控。而 vLLM 联合模型（Phase 2）是最复杂的部分，可以先用"Rollout 只用 model2"的近似方案跑通端到端流程，确认训练能 work，再投入精力解决 vLLM 融合问题。这样可以更快地验证联合训练的 GRPO 在原理上是否有效。

---

## 11. 关键文件索引表 {#11-文件索引}

### 核心文件（需要理解的）

| 文件 | 行数 | 功能 | 联合训练相关度 |
|------|------|------|----------------|
| `verl/workers/fsdp_workers.py` | ~1724 | Worker 实现：模型加载、权重同步、训练/推理切换 | ★★★★★ |
| `verl/workers/actor/dp_actor.py` | ~677 | Actor 实现：前向传播、log_prob 计算、策略更新 | ★★★★☆ |
| `verl/trainer/ppo/ray_trainer.py` | ~1500+ | 训练主循环：编排各个 Worker，触发评测 | ★★★★☆ |
| `verl/trainer/ppo/metric_utils.py` | ~659 | 指标计算：训练指标、评测指标 | ★★★☆☆ |
| `verl/trainer/ppo/core_algos.py` | ~2100 | 核心算法：优势计算、策略 loss | ★★☆☆☆ |
| `verl/utils/torch_functional.py` | ~200 | logits→log_probs 转换 | ★☆☆☆☆ |

### 关键函数索引

| 函数 | 文件 | 行号 | 功能 |
|------|------|------|------|
| `init_model()` | fsdp_workers.py | 848 | 模型初始化入口 |
| `_build_model_optimizer()` | fsdp_workers.py | 328 | 构建模型+优化器 |
| `rollout_mode()` | fsdp_workers.py | 740 | 训练→推理模式切换，权重同步 |
| `trainer_mode()` | fsdp_workers.py | 719 | 推理→训练模式切换 |
| `update_actor()` | fsdp_workers.py | 986 | 策略更新入口（Worker 层） |
| `compute_log_prob()` | fsdp_workers.py | 1082 | 计算 log_probs 入口（Worker 层） |
| `generate_sequences()` | fsdp_workers.py | 1030 | 生成 response 入口（Worker 层） |
| `compute_log_prob()` | dp_actor.py | 425 | 计算 log_probs（Actor 层，推理模式） |
| `update_policy()` | dp_actor.py | 509 | 策略更新（Actor 层，训练模式） |
| `_forward_micro_batch()` | dp_actor.py | 113 | 前向传播核心（所有前向传播的最终入口） |
| `fit()` | ray_trainer.py | 1218 | 训练主循环 |
| `_validate()` | ray_trainer.py | 496 | 评测方法 |
| `_compute_old_log_prob()` | ray_trainer.py | ~1391 | 计算 old_log_probs |
| `_update_actor()` | ray_trainer.py | ~1494 | 触发策略更新 |
| `logprobs_from_logits()` | torch_functional.py | 72 | logits→log_probs 数学转换 |
| `compute_grpo_outcome_advantage()` | core_algos.py | 266 | GRPO 优势计算 |
| `compute_policy_loss_vanilla()` | core_algos.py | 1159 | PPO/GRPO clip loss |
| `compute_data_metrics()` | metric_utils.py | 81 | 训练指标计算 |
| `process_validation_metrics()` | metric_utils.py | 511 | 评测指标计算 |
| `get_fsdp_wrap_policy()` | fsdp_utils.py | 75 | FSDP 包裹策略 |
| `offload_fsdp_model_to_cpu()` | fsdp_utils.py | 146 | 模型权重卸载到 CPU |
| `load_fsdp_model_to_gpu()` | fsdp_utils.py | 180 | 模型权重加载到 GPU |
| `convert_weight_keys()` | model.py | 230 | 参数名格式转换 |
| `update_weights()` | vllm_rollout.py | 153 | 权重同步到 vLLM |
| `monkey_patch_compute_logits()` | utils.py | 93 | vLLM logits 拦截的先例 |
| `monkey_patch_model()` | utils.py | 188 | vLLM 模型 monkey-patch |

### vLLM 相关文件

| 文件 | 功能 | 联合训练相关度 |
|------|------|----------------|
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | vLLM 客户端适配器（ServerAdapter）| ★★★★☆ |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | vLLM HTTP 服务器 | ★★★★☆ |
| `verl/workers/rollout/vllm_rollout/utils.py` | vLLM Worker 扩展（权重加载、monkey-patch） | ★★★★★ |
| `verl/workers/rollout/base.py` | Rollout 基类和注册机制 | ★★☆☆☆ |

### 评测与指标相关文件

| 文件 | 功能 | 联合训练相关度 |
|------|------|----------------|
| `verl/trainer/ppo/ray_trainer.py` (`_validate()`) | 评测触发和流程编排 | ★★★★☆ |
| `verl/trainer/ppo/metric_utils.py` | 训练和评测指标计算 | ★★★☆☆ |
| `verl/experimental/agent_loop/agent_loop.py` | Rollout/Evals 采样参数切换 | ★★★☆☆ |
| `verl/workers/config/rollout.py` | Rollout + val_kwargs 配置 | ★★☆☆☆ |

### 配置文件

| 文件 | 功能 |
|------|------|
| `verl/workers/config/model.py` | HFModelConfig：模型配置 |
| `verl/workers/config/actor.py` | ActorConfig：Actor 配置 |
| `verl/trainer/config/algorithm.py` | AlgoConfig：算法配置 |
| `verl/workers/config/rollout.py` | RolloutConfig：Rollout + val_kwargs 配置 |

---

## 附录 A：PPO vs GRPO 对比表

| 组件 | PPO (GAE) | GRPO |
|------|-----------|------|
| 优势估计器 | `adv_estimator="gae"` | `adv_estimator="grpo"` |
| Critic 模型 | 需要 | 不需要 |
| 优势来源 | Value Function V(s) | 组内相对奖励 |
| 分组 | 无 | 按 prompt ID (uid) |
| 优势公式 | A(s,a) = r + γV(s') - V(s) | A = (r - μ_group) / σ_group |
| 策略 Loss | Clipped PPO | 同上（vanilla） |
| 每个 Prompt 的采样数 | 通常 1 | 多个（如 5-16） |
| 参考策略 | 可选 | 通常不需要 |

## 附录 B：已知的潜在问题

### B.1 EOS Token 问题

在联合训练中，两个模型对 EOS token 的预测可能不一致：
- model1 认为应该结束（EOS 概率高）
- model2 认为应该继续（EOS 概率低）
- 融合后的 logits 可能导致 EOS 无法正确触发

**可能的解决方案**：
1. 在 Rollout 生成阶段使用融合 logits 时，EOS 行为由两个模型的融合策略决定，这是合理的
2. 训练阶段（compute_log_prob + update_policy）使用已生成的 response（包含明确的 EOS），所以实际上不会"生成"新的 token
3. 如果 EOS 融合导致问题，可以在 logit 融合时对 EOS token 做特殊处理（如只用 model2 的 EOS 概率）

### B.2 GPU 内存压力

联合训练的两个模型参数量翻倍，可能导致 GPU OOM。而且现在 vLLM 也需要加载两个模型的权重（用于 Rollout 融合），内存压力更大：

**训练侧**：
- 解决方案 1：使用更多的 GPU（增加 FSDP 分片数量）
- 解决方案 2：冻结 model1 并使用 CPU offload（减少优化器状态）
- 解决方案 3：使用 gradient checkpointing 减少显存占用

**vLLM 推理侧**：
- 解决方案 1：Sleep Level 2 完全释放 vLLM 权重，训练完成后再重新加载
- 解决方案 2：如果 model1 被冻结，可以考虑用 FP8 量化 model1 在 vLLM 中的权重
- 解决方案 3：如果内存实在不够，退回方案 C（Rollout 只用 model2），牺牲 on-policy 一致性

### B.3 vLLM 的模型架构限制

vLLM 有自己的模型实现列表。联合训练模型的自定义架构不在 vLLM 的内置支持列表中。

**Rollout 阶段（需要联合模型）**：
- 通过 `trust_remote_code=True` 让 vLLM 加载自定义模型类
- 自定义模型类需要实现 `load_weights()` 方法来处理带 `sub_models.X.` 前缀的参数名
- 需要确保 vLLM 的 continuous batching、PagedAttention 等优化在自定义模型上正常工作

**Evals 阶段（只用 model2）**：
- 只同步 model2 的参数（去掉 `sub_models.1.` 前缀）
- vLLM 使用标准的 Qwen3 模型，没有架构限制

### B.4 Rollout 与 Evals 的模式切换开销

每次评测都需要切换 vLLM 的模型权重（从联合模型 → model2-only → 联合模型），这带来额外的时间开销：

- 每次切换需要传输模型权重（7B 模型约 14GB，两个模型约 28GB）
- 如果评测频率很高（`test_freq` 很小），切换开销可能显著
- **建议**：评测频率不要设得太高，或者考虑在权重同步时只传输 delta（差异部分）

### B.5 on-policy 一致性风险

如果最终选择方案 C（Rollout 只用 model2），存在 on-policy 一致性风险：

```
Rollout 策略: π_model2(a|s)           ← model2 单独的策略
训练 old_log_prob: log π_fused(a|s)   ← 融合策略
训练 new_log_prob: log π_fused(a|s)   ← 融合策略

重要性比率: ratio = π_fused_new / π_fused_old  ← 两个都是融合策略，比率合理
但 response 本身是由 π_model2 生成的，不是 π_fused 生成的
```

这可能导致：
- 训练样本的分布不匹配（off-policy）
- PPO clip 的约束可能不够紧
- 需要通过实验验证影响程度

---

> **文档版本**: v2.0
> **最后更新**: 2026-02-27
> **基于 verl 版本**: v0.7.0 (stable)
> **主要更新**: 修正推理场景描述（Rollout 用融合 logits，Evals 用 model2），新增评测与指标章节，重写 vLLM 推理和修改方案章节
