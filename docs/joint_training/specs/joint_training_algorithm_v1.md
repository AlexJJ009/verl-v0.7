> **Historical infrastructure investigation, not the live Stage2 contract.**
> This document records the original fused-rollout GRPO design discussion. The
> active Qwen3-1.7B Math causal-P60 Stage2 instead uses **Model2-only rollout**
> and applies fusion only while recomputing the teacher-forced WDL objective.
> It also adds `fusion_mode=strong_scaled` for the D0 matched-scale no-weak
> control. For the current protocol, use
> `plans/active/qwen3_1p7b_math_stage123.md`; do not infer current rollout
> semantics or available config fields from this historical document.

我们提出了一个联合训练的算法，它在标准的 SFT 上取得了成功。

现在我们想要知道这个范式能不能推广到强化学习中，那就以 GRPO 为例子，我们要试一下能不能推广到 GRPO 中。

下面是我同事用 TRL 这个强化学习框架（其中包含 SFT 组件）来做的 SFT 联合训练脚本，包含了启动逻辑和脚本内容。

用户使用 `vote-base` 模式进行联合训练，该模式应该是纯粹的 logits 融合（两个模型独立前向传播，只在输出层融合 logits）。

---

## 代码调用链（vote-base 模式）

```
llmboost_train.py::train_stage3()
  │
  ├─ fuse_submodels()                    # utils/fuse_models.py:108
  │   └─ 保存 config.json (architectures: ["QwenBoostForCausalLM"])
  │   └─ 保存 pytorch_model.bin (sub_models.0.*, sub_models.1.*)
  │
  ├─ load_fuse_model_tokenizer_vote()    # utils/fuse_models.py:768
  │   └─ QwenBoostForCausalLM.from_pretrained()  # EnsembleQwen3/modeling_qwen3.py
  │   └─ 可选：冻结 sub_models.0 的参数
  │
  └─ run_sft()                           # Trainer/sft_runner.py:73
      └─ SFTTrainer(model, tokenizer, args, train_dataset)  # from trl import SFTTrainer
         └─ trainer.train()
```

---

## 核心模型实现分析

### QwenBoostForCausalLM.forward() (EnsembleQwen3/modeling_qwen3.py:557-629)

```python
def forward(self, input_ids, attention_mask, ..., labels=None, ...):
    outputs_list = []

    # 两个子模型独立前向传播（无 hidden_states 交互）
    for idx, (model, pkv) in enumerate(zip(self.sub_models, past_key_values_list)):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ...,
            labels=None,  # 不传 labels 给子模型
            **kwargs,
        )
        outputs_list.append(out)

    # 加权融合 logits
    stacked_logits = torch.stack([o.logits for o in outputs_list], dim=0)
    fusion_lambda = getattr(self.config, "fusion_lambda", 0.5)

    if len(outputs_list) == 2:
        logits = (1 - fusion_lambda) * stacked_logits[0] + fusion_lambda * stacked_logits[1]
    else:
        logits = stacked_logits.mean(dim=0)

    # 用融合后的 logits 计算 loss
    loss = None
    if labels is not None:
        loss = self.sub_models[0].loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    return CausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

**确认**: vote-base 模式是**纯粹的 logits 融合**，没有 hidden_states 交互。

---

### 问题 4: fusion_lambda 的默认值

**位置**: `EnsembleQwen3/modeling_qwen3.py:601`

```python
fusion_lambda = getattr(self.config, "fusion_lambda", 0.5)
```

**问题**: 如果 config 中没有设置 `fusion_lambda`，默认值是 0.5（平均融合）。

**需要确认**: 用户在训练时是否正确设置了 `fusion_lambda`？

---

## 数据流图（vote-base 模式）

```
input_ids, labels
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              QwenBoostForCausalLM.forward()          │
│                                                      │
│   ┌─────────────┐         ┌─────────────┐           │
│   │  sub_model_0 │         │  sub_model_1 │           │
│   │    (m1)      │         │    (m2)      │           │
│   │              │         │              │           │
│   │  独立前向传播  │         │  独立前向传播  │           │
│   └──────┬──────┘         └──────┬──────┘           │
│          │                       │                   │
│          ▼                       ▼                   │
│     logits_0                logits_1                 │
│          │                       │                   │
│          └───────────┬───────────┘                   │
│                      ▼                               │
│   logits = (1-λ)*logits_0 + λ*logits_1              │
│                      │                               │
│                      ▼                               │
│         CrossEntropyLoss(logits, labels)             │
│                      │                               │
│                      ▼                               │
│                    loss                              │
└──────────────────────────────────────────────────────┘
       │
       ▼
  loss.backward()
       │
       ├──► ∂loss/∂logits_0 = (1-λ) * ∂CE/∂logits  →  更新 m1（如果未冻结）
       │
       └──► ∂loss/∂logits_1 = λ * ∂CE/∂logits      →  更新 m2
```

---
他在训练的时候，l两个子模型各自的 logits 会被融合用于预测下一个 token，两个模型的梯度都会被更新。
但在推理的时候，只用第二个模型自己 Logit 进行推理，计算 acc/pass 等。

所以这里存在一个过程：
1. 在训练的时候，会将 logit 合在一起用于预测下一个 token，计算得到的 log probs 用于更新梯度。
2. 在推理的时候，会将第一个模型的权重卸载掉。

由于我使用的强化学习框架和它的 TRL 框架是不一样的，目前我的目标是将这两个训练的逻辑迁移到 verl 这个强化学习框架中。
相似的地方是：我的同事也是用 vLLM 来做推理

我的同事告诉我，为了实现在 verl 中实现强化学习的联合训练，有两个关注点是需要考虑的：
第一个点就是自己新建一个模型类（他用了这个名字：QwenBoostForCausalLM，我们可以叫做 QwenJointForCausalLM） ， 加载的时候就是读取这个模型的路径。

第二个点是模型推理来进行 evals 的时候（例如在测试集上），只用第二个模型去进行推理，也就是梯度更新之后的第二个模型的权重去推理。

在 verl 中，目前我们用的是 FSDP 作为训练器，vLLM 作为推理器。

所以，目前我们针对这样一个 idea 已经设想到了可能会存在的问题：
1. Logit 融合可能会导致 End Of Sequence 这个特殊的 token 不能正确地出现。这是因为可能出现第一个模型输出“End of Sequence”的概率很高，但第二个模型输出“End of Sequence” 的概率很低。那么 logit 融合导致下一个预测的 token 就不是 end of sequence 了。

### 目录结构

```
verl/
├── verl/                    # 核心代码
│   ├── trainer/             # 训练器实现
│   │   ├── main_ppo.py      # PPO/GRPO/RLOO 等训练入口
│   │   ├── main_generation.py  # 生成模式入口
│   │   ├── ppo/             # PPO 算法实现
│   │   └── config/          # Hydra 配置文件
│   ├── workers/             # Worker 实现
│   │   ├── fsdp_workers.py  # FSDP 训练后端
│   │   └── rollout/         # Rollout 引擎 (vLLM/SGLang)
│   ├── single_controller/   # 单控制器模式
│   └── protocol.py          # 数据协议 (DataProto)
├── recipe/                  # 算法实现
│   ├── tspo/                # TSPO 算法 (当前开发重点)
│   │   ├── Qwen3-4B-base/   # 模型特定配置
│   │   ├── Qwen3-8B/
│   │   └── ...
│   ├── dapo/                # DAPO 算法
│   ├── grpo/                # GRPO 基线
│   └── ...
├── examples/                # 示例脚本
├── docs/                    # 文档
└── tests/                   # 测试用例
```


VERL 采用**分层解耦**的设计理念，将 RL 算法分为几个独立的层次：

```
训练主循环 (Trainer)
    ↓
算法核心逻辑 (Core Algorithms)
    ↓
分布式工作组 (Worker Groups)
    ↓
具体模型实现 (Model Implementations)
```

**关键设计原则：**
- **关注点分离**：算法逻辑与分布式实现解耦
- **可插拔组件**：通过注册机制实现算法变体的灵活替换
- **声明式配置**：通过配置文件控制算法行为，无需修改核心代码
- **函数式接口**：核心算法实现为纯函数，易于测试和组合

### 1.2 核心文件结构

```
verl/trainer/
├── ppo/
│   ├── core_algos.py           # 核心算法实现（重点）
│   ├── ray_trainer.py          # 训练主循环
│   ├── reward.py               # 奖励计算
│   └── metric_utils.py         # 指标计算
├── config/
│   ├── algorithm.py            # 算法配置定义
│   ├── actor/                  # Actor 配置
│   └── ...
└── main_ppo.py                 # 训练入口
```



```
┌─────────────────────────────────────────────────────────────────────────┐
│                           训练一个 step 的流程                            │
└─────────────────────────────────────────────────────────────────────────┘

1. VLLM Rollout (生成阶段)
   ├── 输入: prompts
   ├── 输出: response_ids, rollout_log_probs (可选)
   └── 特点: 高效推理引擎，专注生成，不返回 hidden states

2. Reward Computation (奖励计算)
   ├── 输入: prompts + responses
   └── 输出: token_level_rewards

3. Actor compute_log_prob (重计算 log_prob) ← 我们修改的地方
   ├── 输入: prompts + responses
   ├── 输出: old_log_probs, entropys, hidden_states (新增)
   └── 特点: 用 HuggingFace 模型做完整 forward pass

4. Advantage Computation (优势计算)
   ├── 输入: rewards, hidden_states
   └── 输出: advantages

5. Actor Training (训练)
   └── 更新模型参数
```

**关键点**：VLLM 是一个高度优化的推理引擎，它的设计目标是**快速生成 token**。

但是 verl 在第 3 步会用 **HuggingFace 模型重新计算 log_prob**（确保数值一致性），这个过程本质上就是一次完整的 forward pass。

所以你的任务是做一个探索：
在当前条件下，verl 是如何加载模型类来进行先前传播、获取 logits 以及 log probs、参数更新的、以及调用 vllm 来进行推理的？在 verl 中，具体是如何卸载掉一定权重的（问这个问题是我们想从联合训练加载的两个模型训练，只用一个模型做 evals）？
然后你需要探索：
在 verl 中如果我们想要实现“联合训练强化学习”这种范式（我们的目标是：把联合训练的这种 logit 融合机制，从 SFT 迁移到强化学习的训练中去），需要做哪些修改，以及在哪些流程中进行修改。

输出一份详细调查报告 md 文档到：verl07/verl/docs/joint_training 文件夹中。
这份调查报告的目标是让用户能够弄明白怎么上手在 verl 中改 infra 代码，因为这次修改涉及到的深度很深。
用户需要完成从初级程序员到资深程序员的过渡，所以这一份调查报告同时也会是一份教学文档，需要尽可能地详细，尽可能地将用户当做小白来对待。用户希望能从这份报告中不仅学到怎么上手改 Infra 代码（用户目前只会简单地改参数来启动训练，这种水平），还要学到一些项目中的代码架构的哲学。
这份调查报告要更侧重于原理的讲解，说清楚要改什么，为什么能够知道要改这些？


You should use subagent to probe to save context windows. 

当你有疑惑的问题，请及时询问用户，应该尽可能询问用户来获取那些无法自主得到的信息，减少幻觉。

目前我们只需要做详细的调研，不需要修改现有代码。

verl07/verl/docs/joint_training/GRPO_Joint_Training_Target_v1.md
verl07/verl/docs/joint_training/verl_joint_training_investigation_report.md
有一些不一致需要修正：
推理有两个场景：
1. rollout 生成阶段：这个阶段是用 vllm 来进行推理的，这个时候也是用两个模型融合的 logit 来进行生成的。
2. evals 评测阶段：这个阶段也是用 vllm 来进行推理的，但是这个时候是只用第二个模型的权重来推理和进行评测的。

这意味着我们在 vllm 的推理引擎中也需要实现一个机制，来支持在推理的时候只用第二个模型的权重以及 logit ，以及用融合后的的 logit 来进行推理。

这个修正是因为：在 sft 的联合训练中，损失函数的计算是用融合后的 logit 来计算的；但是在 evals 的评测阶段，是用第二个模型的 logit 来进行评测的（因为我们认为第二个模型才是训练之后性能更好的模型，我们实际也会让第二个模型一开始就用更尖锐的模型）。
所以在强化学习的联合训练中，我们也应该保持同样的逻辑：在训练阶段用融合后的 logit 来计算 loss，仅仅在 evals 评测阶段用第二个模型的 logit 来进行评测。
同时，这次还需要给出 metrics 的相关计算如何修改。

根据这一条修正，修改当前的调查报告，可以保留与此无关的讲解内容，仅仅修改与这次修正相关的部分。文档的要求仍然是：
修改当前已有的详细调查报告 md 文档：verl07/verl/docs/joint_training/verl_joint_training_investigation_report.md
这份调查报告的目标是让用户能够弄明白怎么上手在 verl 中改 infra 代码，因为这次修改涉及到的深度很深。
用户需要完成从初级程序员到资深程序员的过渡，所以这一份调查报告同时也会是一份教学文档，需要尽可能地详细，尽可能地将用户当做小白来对待。用户希望能从这份报告中不仅学到怎么上手改 Infra 代码（用户目前只会简单地改参数来启动训练，这种水平），还要学到一些项目中的代码架构的哲学。
这份调查报告要更侧重于原理的讲解，说清楚要改什么，为什么能够知道要改这些？

You should use subagent to probe to save context windows. 

当你有疑惑的问题，请及时询问用户，应该尽可能询问用户来获取那些无法自主得到的信息，减少幻觉。

目前我们只需要做详细的调研，不需要修改现有代码。
