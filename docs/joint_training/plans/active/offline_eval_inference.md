# 离线推理与评估脚本

**状态**：计划中（2026-03-16 创建）
**目标**：构建一个独立的 vLLM 离线推理 + 评估脚本，能对 merge 后的 checkpoint 在 MATH-500 和 AIME-2025 上计算 Pass@3、Maj@3、mean@3 等指标，与训练中 validation 的指标体系对齐。

---

## 背景

训练完成后，需要对保存的 checkpoint 进行独立评估：
1. 用 `python -m verl.model_merger merge --backend fsdp` 将分布式 checkpoint 合并为 HF 格式
2. 用 vLLM 加载合并后的模型权重，进行离线推理
3. 对生成的结果计算指标，与训练中 `_validate()` 的指标对齐

### 现有基础设施

| 组件 | 位置 | 可复用程度 |
|---|---|---|
| 模型合并 | `verl/model_merger/` | 直接使用 CLI |
| 奖励函数 | `recipe/joint_training/custom_reward_function_latex_verify.py` | 直接复用 |
| 指标计算 | `verl/trainer/ppo/metric_utils.py` + `recipe/rep_exp/metric_utils.py` | 复用核心函数 |
| 离线评估 | `verl/trainer/main_eval.py` | 仅评估已有生成结果，不含推理 |
| vLLM 推理 | `vllm.LLM` (offline) | 直接使用 vLLM 原生 offline API |

### 测试数据集

| 数据集 | 路径 | 样本数 |
|---|---|---|
| MATH-500 | `/data-1/dataset/MATH-500/math500-test.parquet` | 500 |
| AIME-2025 | `/data-1/dataset/AIME-2025/aime-2025.parquet` | ~30 |

数据集 schema：`data_source`, `prompt` (chat format), `reward_model.ground_truth`, `extra_info`

---

## 实现计划

### Step 1：创建推理脚本 `recipe/joint_training/run_offline_eval.sh`

Shell 入口脚本，接受以下参数：

```bash
# 用法示例
bash recipe/joint_training/run_offline_eval.sh \
    --model_path /path/to/merged_hf_model \
    --gpu_ids 4,5,6,7 \
    --tensor_parallel 4 \
    --n 3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --output_dir /path/to/eval_results
```

关键职责：
- 设置 `CUDA_VISIBLE_DEVICES` 限制使用指定 GPU
- 设置 vLLM 环境变量（`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_V1=1` 等）
- 调用 Python 推理脚本

### Step 2：创建 Python 推理+评估脚本 `recipe/joint_training/offline_eval.py`

这是核心实现，包含三个阶段：

#### Phase A：vLLM 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model=model_path,
    tensor_parallel_size=tp_size,     # 用户指定，默认 4
    dtype="bfloat16",
    gpu_memory_utilization=0.85,      # 离线推理可以用更多显存
    enforce_eager=True,
    trust_remote_code=True,
    max_model_len=max_prompt_len + max_tokens,  # ~4596
    enable_prefix_caching=True,
)

sampling_params = SamplingParams(
    temperature=temperature,           # 默认 1.0
    top_p=top_p,                       # 默认 0.95
    top_k=-1,
    max_tokens=max_tokens,             # 默认 4096
    n=n,                               # 每个 prompt 生成 n 条，默认 3
    stop=["<|endoftext|>", "<|im_end|>"],  # 根据 tokenizer 设置
)
```

**关键设计决策**：
- 使用 vLLM 原生 offline `LLM` 类（不是 verl 的 rollout server），因为这是独立推理，不需要 Ray
- `n` 参数直接在 `SamplingParams` 中设置，vLLM 会自动对同一 prompt 生成 n 条
- 需要从 parquet 读取 prompt，用 tokenizer 的 `apply_chat_template` 转换为 token ids

#### Phase B：奖励计算

复用 `custom_reward_function_latex_verify.py` 中的 `compute_score` 函数：

```python
from recipe.joint_training.custom_reward_function_latex_verify import compute_score

for prompt_idx, outputs in enumerate(all_outputs):
    ground_truth = dataset[prompt_idx]["reward_model"]["ground_truth"]
    data_source = dataset[prompt_idx]["data_source"]
    for output in outputs:
        response_text = output.text
        result = compute_score(data_source, response_text, ground_truth)
        # result = {"score": ±1.0, "acc": bool, "pred": str, ...}
```

#### Phase C：指标聚合

需要计算的指标（与训练 validation 对齐）：

| 指标 | 计算方式 | 来源 |
|---|---|---|
| `mean@3` | 3 条响应的 acc 平均值 | `metric_utils.py` |
| `pass@3` | 组合估计器 `1 - C(n-c,k)/C(n,k)` | `rep_exp/metric_utils.py: comb_estimator()` |
| `maj@3` | Bootstrap 多数投票（3 条中取多数） | `metric_utils.py: bootstrap_metric()` + majority vote |
| `answer_extraction_failure_rate` | `pred == "[NO_BOXED]"` 的比例 | reward 返回的 pred 字段 |
| `verification_method` 分布 | 各验证方法占比 | reward 返回的 verification_method 字段 |

**Pass@K 的组合估计器**（复用 `comb_estimator`）：
```python
def comb_estimator(n: int, c: int, k: int) -> float:
    """n=总样本数, c=正确数, k=选k条至少1条对的概率"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

**Maj@K**：Bootstrap 抽 k 条，多数投票，重复 `n_bootstrap` 次取均值。

#### Phase D：结果输出

1. **终端输出**：按数据集分表打印所有指标
2. **JSON 保存**：`{output_dir}/eval_metrics.json`
3. **Parquet 保存**：`{output_dir}/eval_generations.parquet`，包含每条 prompt 的所有生成结果和 reward 元数据，可用于后续分析

---

## 采样参数对齐

| 参数 | 训练 Validation (val_kwargs) | 离线推理（默认值） | 说明 |
|---|---|---|---|
| `temperature` | 1.0 | 1.0 | 采样温度 |
| `top_p` | 0.95 | 0.95 | Nucleus sampling |
| `top_k` | -1 | -1 | 不限制 |
| `n` | 1 (训练中) | 3 (离线) | 离线生成 3 条算 pass@3 |
| `max_tokens` | 4096 | 4096 | MATH 任务的最大响应长度 |
| `do_sample` | True | True (temperature>0) | vLLM 自动处理 |

**注意**：训练 validation 中 `n=1`，只能算 `mean@1`。离线推理设 `n=3` 可以计算 pass@3 和 maj@3。

---

## 指标与训练 Validation 的对齐关系

训练 validation 输出（`ray_trainer.py: _val_metrics_update`）：

```
val-core/{data_source}/acc/mean@N     ←→  离线 acc/mean@N
val-core/{data_source}/acc/maj@K/mean ←→  离线 acc/maj@K/mean
val-core/{data_source}/acc/best@K/mean ←→ 离线 acc/best@K/mean
```

离线新增（训练中因 n=1 无法计算）：

```
pass@3                               # 组合估计器（3 条中至少 1 条对）
maj@3                                # 3 条多数投票
```

---

## 端到端使用流程

```bash
# 1. 合并 checkpoint（假设 FSDP 后端）
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/global_step_100/actor \
    --target_dir /path/to/merged_model

# 2. 离线推理 + 评估（使用 GPU 4-7，n=3 计算 pass@3）
bash recipe/joint_training/run_offline_eval.sh \
    --model_path /path/to/merged_model \
    --gpu_ids 4,5,6,7 \
    --tensor_parallel 4 \
    --n 3 \
    --temperature 1.0 \
    --output_dir /path/to/eval_results
```

---

## 文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `recipe/joint_training/offline_eval.py` | 新建 | 核心推理+评估 Python 脚本 |
| `recipe/joint_training/run_offline_eval.sh` | 新建 | Shell 入口，设置环境变量和参数 |

**不需要修改的文件**：
- `verl/model_merger/` — 直接使用现有 CLI
- `custom_reward_function_latex_verify.py` — 直接 import 复用
- `metric_utils.py` — 复用 `comb_estimator`, `bootstrap_metric` 等函数

---

## vLLM Attention Backend 与 Flash Attention

vLLM 0.8.5 的 attention backend 选择逻辑：

1. **如果设置 `VLLM_ATTENTION_BACKEND=FLASH_ATTN`**：强制使用 FlashAttention 2，要求 `flash_attn` 包可用
2. **如果不设置该环境变量**：vLLM 自动探测，优先级为 FlashAttention > FlashInfer > xFormers
3. **训练脚本中统一设置了 `VLLM_ATTENTION_BACKEND=FLASH_ATTN`**，离线推理脚本也应保持一致

当前状态：`flash_attn` 包存在 ABI 兼容问题（`undefined symbol`），正在重新安装。安装完成后，离线推理脚本中设置 `export VLLM_ATTENTION_BACKEND=FLASH_ATTN` 即可。

如果 flash_attn 暂时不可用，可以不设 `VLLM_ATTENTION_BACKEND`，vLLM 会回退到其他后端（不影响正确性，可能影响性能）。

---

## 风险与注意事项

1. **显存**：4 张 GPU TP=4 推理 1.7B 模型绰绰有余，但 4B 模型需注意 `gpu_memory_utilization` 和 `max_model_len` 的设置
2. **Tokenizer 一致性**：离线推理必须使用与训练相同的 tokenizer 和 chat template，确保 `apply_chat_template` 参数一致
3. **Stop tokens**：需要与训练时 vLLM rollout 使用的 stop tokens 一致，避免生成过长或过短
4. **Joint model vs 单模型**：merge 后的模型是标准 HF 格式的单模型权重，vLLM 可以直接加载，不需要 joint model 的特殊处理
5. **`compute_score` import 路径**：需要确保 `recipe/` 在 Python path 中，或使用相对 import
