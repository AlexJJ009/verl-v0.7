# Qwen3-1.7B 数学与代码 Offline Pass@k / 回答多样性评测方案

- 文档职责：冻结 A/C/D0/fixed-M1/GRPO checkpoint 的共同 offline generation、pass@k、回答多样性和多机并行合同
- 当前状态：Math A/C/D0、fixed-M1 Stage1 P60、strict-scorer Cold/Stage1/C-P60→GRPO 已完成共同 `n=256`；DynPerm offline 与 Code matrix 尚未开始
- 当前修订：2026-08-23
- 目标硬件：多台 8 × NVIDIA L40S 46 GB；单机和美团路径均由环境变量覆盖
- 关联方案：[`qwen3_1p7b_standard_rlvr_grpo_matrix.md`](qwen3_1p7b_standard_rlvr_grpo_matrix.md)

## 0. Math common n=256 结果状态（2026-08-23）

Math 正式入口已在每个 checkpoint 上完成八个 TP=1、每 shard `n=32` 的生成与
exact-coverage merge。每个 arm 均覆盖 `2,798 prompts / 716,288 responses / n=256`，并使用
同一组 shard seeds `20260811...20260818`、`temperature=0.6`、`top_p=0.95`、`top_k=20`、
`min_p=0`、`max_tokens=4096`、`enable_thinking=true` 和同一 strict scorer。

| Arm | pass@1 | pass@128 | pass@256 | maj@256 | 当前判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| A P60 | 69.802% | 86.142% | 87.882% | 79.710% | 标准 On-Policy SFT anchor |
| C P60 | 71.974% | 87.958% | 88.944% | 80.946% | 主 WDL 方法 |
| D0 P60 | 68.733% | 88.719% | 90.393% | 78.346% | pass@1 较低、high-k coverage 最高 |
| fixed-M1 Stage1 P60 | 71.713% | 85.604% | 86.452% | 82.330% | 复现 C 的 pass@1，但 high-k 较弱 |
| Stage1→GRPO effective P200 | 70.682% | 85.235% | 86.037% | 81.717% | 未追平 C |
| Cold→GRPO P200 | 71.242% | 87.086% | 88.335% | 81.152% | 未追平 C |
| C-P60→GRPO effective P200 | 72.379% | 87.493% | 88.759% | 81.749% | pass@1 最高，但只比 C 高 0.405 pp |

10,000-rep within-dataset prompt-stratified paired bootstrap 给出：`C-A` pass@1
`+2.172 pp [1.631, 2.738]`，`C-D0` pass@1 `+3.242 pp [2.239, 4.307]`，但
`C-D0` pass@256 为 `-1.448 pp [-3.242, -0.045]`。因此 weak-logit treatment 的主要确认结果是
提高单样本质量，而不是全 `k` 支配；D0 的高预算 oracle coverage 更高。

`C-P60→GRPO - C` 的 pass@1 为 `+0.405 pp [0.172, 0.648]`，pass@128/256 的区间均跨零。
这支持“小幅 pass@1 refinement”，不支持额外 100 个 GRPO step 带来新的 high-k 能力突破。

fixed-M1 的快速 2,000-rep paired audit 显示，`fixed-C` pass@1 为
`-0.261 pp [-0.886, 0.348]`，而 pass@128/256 分别为
`-2.355 pp [-3.746, -1.103]`、`-2.492 pp [-4.631, -0.507]`。正式报告前应按主协议重跑
10,000-rep，但方向已经说明：冻结 weak guidance 足以解释主要 pass@1 收益，joint adaptation
仍可能影响 tail coverage、答案锐化和稳定性。

## 1. 实验目的

本实验不是另一轮训练，而是对相同 checkpoint 建立一套共同冻结的 test-time sampling 合同，回答：

1. 单次采样能力 `pass@1` 是否提升；
2. 增加采样预算后，`pass@2...pass@256` 能达到多高；
3. 方法改善的是平均单样本正确率、回答多样性，还是两者都有；
4. 数学和代码上的 online `mean@3` 排序能否在统一 offline harness 中复现。

## 2. Loss-mask 事故与推理阶段的边界

历史 CoT-v1/v2 事故发生在 SFT dataset tokenization：thinking 内容没有进入 supervised loss mask，
因此模型参数没有从这部分 token 获得梯度。纯推理不计算 loss，所以不会再次出现“CoT token 没算
loss”的同类错误。

推理仍有另一类风险：chat template 可能因为 `enable_thinking`、预渲染 prompt、`/no_think` 或
stop token 配置而关闭 thinking。该风险必须独立 fail closed：

- 显式传入 `enable_thinking=true`，不依赖 tokenizer 默认值；
- 同一 prompt 分别以 true/false 渲染，要求文本 SHA-256 不同；
- receipt 记录 chat-template kwargs、true/false prompt hash 和 tokenizer/model identity；
- raw output 保留 `<think>`、`</think>`、finish reason、长度和 sample index；
- 汇总报告 think-complete、format、EOS 与 truncation rate。

当前 Qwen3-1.7B CoT-v3 历史 `T=0.2` validation 输出中 `<think>` 开始标记为 100%，说明当前
lineage 确实能生成 thinking；这不是新 offline 入口已正确配置的替代证据。

## 3. Decoder 参数

### 3.1 历史配置回顾

| 场景 | 历史参数 | 用途与边界 |
| --- | --- | --- |
| 当前 A/C/D0 online validation | `T=0.2, top_p=0.95, n=3` | 低方差 checkpoint 曲线，不是高预算多样性合同 |
| 旧 math offline eval | `T=1.0, top_p=0.95, top_k=-1` | 历史 n=3/n=64 结果，未显式记录 thinking switch |
| 旧 code deterministic pass@1 | `T=0, n=1` | 非 thinking 模型常用口径；Qwen3 官方不建议 thinking mode 使用 greedy |

### 3.2 `T=0.6` 的证据边界与本轮主合同

`T=0.6` 不是所有任务、所有 `pass@k` 的统一同行标准，但它是 Qwen3 官方对 thinking mode 的
model-native 推荐：`temperature=0.6, top_p=0.95, top_k=20, min_p=0`，并明确不建议 thinking
mode 使用 greedy。代码领域的经典 HumanEval/Codex 协议则按目标 k 使用不同温度，常见的是
`pass@1/10/100` 分别使用 `T=0.2/0.6/0.8`；因此不能把 `0.6` 描述成代码 benchmark 的唯一标准。

本实验的目标是从同一个 `n=256` 样本池得到一条内部一致的 `pass@1...256` 曲线，而不是分别为
每个 k 挑最优温度。主报告采用 Qwen3-native `T=0.6`，并保持所有 checkpoint 完全一致；训练
rollout 的 `T=1.0` 是数据探索/优化分布，不要求与冻结的 test-time decoder 相同。只要不在测试集
上按模型挑温度，训练 `1.0`、在线监控 `0.2`、正式 offline `0.6` 并不构成协议错误，但三者的
绝对分数不能直接拼成同一条曲线。

主配置不在 final test benchmark 上调温度：

| 参数 | 数学 | 代码 |
| --- | ---: | ---: |
| `enable_thinking` | `true` | `true` |
| `temperature` | `0.6` | `0.6` |
| `top_p` | `0.95` | `0.95` |
| `top_k` | `20` | `20` |
| `min_p` | `0` | `0` |
| `max_tokens` | `4096` | `8192` |
| total samples / prompt | `256` | `256` |
| generation seed family | `20260811 + shard_index` | `20260811 + shard_index` |

主报告中的 `pass@1` 是从同一个 `T=0.6, n=256` 样本池用无偏组合估计器计算的 expected
single-sample success，不等同于 greedy accuracy，也不等同于在线 `T=0.2 mean@3`。

### 3.3 参数敏感性与历史桥接

不通过比较 A/C/D0 的完整 test 分数来选择温度。正式矩阵前只在预注册 anchor checkpoint 和
独立 calibration subset（每任务 64–128 prompts）运行 `T={0.2,0.6,1.0}, n=16`，其余参数
都保持 `top_p=0.95, top_k=20, min_p=0`，报告正确率、unique response、format、EOS、truncation、
平均输出 token 和 tokens/s。这一 canary 用于发现明显退化或无限重复，不用于事后改主合同。

如果需要与历史 online 数字精确连接，另做极小的 compatibility slice：完全复刻
`T=0.2, top_p=0.95, top_k=-1, n=3`。它只说明 harness 是否对齐，不进入主 `pass@k` 表。

## 4. Checkpoint 矩阵

第一批覆盖现有完整结果：

| Task | Checkpoint | View | 状态 |
| --- | --- | --- | --- |
| Math | `CS0` cold-start selected checkpoint | single | 需评测 |
| Math | `S1-P0` Stage1 final Model2 | single | 需 merge/核验后评测 |
| Math | A P60 Standard On-Policy SFT | single | 需 merge/核验后评测 |
| Math | C P60 | extracted Model2 | 需 merge + extract |
| Math | D0 P60 | extracted Model2 | 需 merge + extract |
| Code | `CS0` cold-start selected checkpoint | single | 需评测 |
| Code | `S1-P0` Stage1 final Model2 | single | 已有 HF-format source，需核验 |
| Code | A P60 Standard On-Policy SFT | single | 需 merge |
| Code | C P60 | extracted Model2 | 需 merge + extract |
| Code | D0 P60 | extracted Model2 | 需 merge + extract |

后续 GRPO、peak 和 terminal checkpoint 使用同一 manifest schema 追加，不能另建一套 decoder。
默认只评测 Model2；Model1 是 mechanism diagnostic，不进入第一批主表。

## 5. 数据集与 evaluator

### 5.1 数学

- 完整 Math-7：AIME-2025、MATH-500、AMC23、AQUA、GSM8K、MAWPS、SVAMP；
- 使用当前 schema-aligned system-prompt manifests；
- 同一 strict math scorer revision；
- 报告每个数据集和七数据集等权 macro。

### 5.2 代码

- HumanEval+：EvalPlus official evaluator；
- MBPP+：EvalPlus official evaluator；
- LiveCodeBench：frozen release-v5 official evaluator；
- 保持当前 `1024+8192` prompt/response contract；
- raw generation、代码抽取、官方 scorer 分阶段保存；
- `code_reward_dependency_error` 必须读取数值，不能按指标名误判。

## 6. 单机 8×L40S 与多机并行

每个 checkpoint/dataset 的 256 samples 切成 8 个 `n=32` sample shards：

| Shard | Global sample index | Seed |
| ---: | --- | ---: |
| 0 | 0–31 | 20260811 |
| 1 | 32–63 | 20260812 |
| ... | ... | ... |
| 7 | 224–255 | 20260818 |

### 6.1 单机默认拓扑

Qwen3-1.7B BF16 权重约 3.8 GB，可以完整放进单张 46 GB L40S。对这种小模型，8 卡 tensor
parallel 会引入不必要的 NCCL 通信；单机默认采用 **8 replicas × TP=1**：GPU 0–7 分别运行
shard 0–7，每个 shard 生成同一 checkpoint/dataset 的 `n=32`，最终 exact-coverage merge 成
`n=256`。vLLM 官方也把 data parallel 定义为各 GPU 复制模型、处理独立 request batch。

单进程 engine 初始合同：

| 参数 | Math | Code | 原因 |
| --- | ---: | ---: | --- |
| replicas / TP | `8 / 1` | `8 / 1` | 避免小模型 TP 通信，吃满 8 卡 |
| `gpu_memory_utilization` | `0.90` | `0.90` | 专用空闲节点，保留约 10% headroom |
| `max_num_seqs` | `64` | `32` | 分别按 4K/8K worst-case KV cache 保守起步 |
| `max_num_batched_tokens` | `8192` | `8192` | vLLM 0.12 在 <70 GB GPU 上的稳健基线 |
| CUDA graph | 开启 | 开启 | 默认不使用 `enforce_eager`，提升 steady-state throughput |

按本模型 28 层、8 KV heads、head_dim 128、BF16 粗算，KV cache 约为 112 KiB/token；因此 Math
64 条 worst-case 4.6K context 和 Code 32 条 worst-case 9.2K context 都约占 32 GiB KV，连同
3.8 GB 权重及 runtime overhead 可放入 46 GB。该估算只用于选择 canary 起点，正式上限由实际
preemption/OOM/throughput receipt 决定。

每个任务先用 64 prompts、`n=4` 做 scheduler canary：Math 比较
`max_num_seqs={64,96}`，Code 比较 `{32,48}`，并比较
`max_num_batched_tokens={8192,16384}`。只选择无 OOM、无 KV preemption/recompute、输出覆盖完整且
tokens/s 最高的 cell；一旦选择，对所有模型冻结，不允许按模型单独调 batch。batch/scheduler
参数理论上不改变目标采样分布，但仍记录 resolved engine config 和 seed，防止数值/调度漂移。

### 6.2 工作量和时间量级

完整 Math-7 是 2,798 prompts，即每 checkpoint 生成 716,288 responses；Code-3 是 1,422
prompts，即 364,032 responses。历史本机 `T=0.2,n=3,TP=4` 的 Code generation 约为 3.1k
output tokens/s，且平均输出约为 HumanEval+ 3.5k、MBPP+ 2.8k、LCB 6.2k tokens。按 8×TP1
目标总吞吐 12k–20k output tokens/s 粗估：

- Math：约 5–12 小时/checkpoint；
- Code generation：约 25–45 小时/checkpoint；
- Code official sandbox scoring 另计，特别是 LCB 的约 225k completions，必须 CPU 并行并与
  GPU generation 解耦，不能让 8 张 GPU 等待代码执行。

这只是容量规划区间，不作为结果承诺。正式 ETA 必须由上述 canary 的真实 output-token 数、
tokens/s、preemption count 和 scorer throughput 计算。若同时有多台 L40S，优先
checkpoint-parallel；单机内部始终保持 8×TP1 sample-parallel。

### 6.3 多机组合与合并

并行方式按资源选择：

1. checkpoint-parallel：不同机器处理不同 checkpoint；
2. sample-parallel：多台机器处理同一 checkpoint 的不同 shard；
3. 两者可以组合，但同一 cell 的 `(prompt_id, sample_index)` 必须唯一。

合并前强制验证：

- 所有 shard 的 model、dataset、temperature/top-p/top-k/min-p、max tokens、thinking contract 一致；
- 每个 prompt 恰好覆盖 sample index `0..255`；
- 无重复、无缺失、无越界；
- prompt ID、task ID、数据源和 evaluator revision 一致；
- thinking true/false template canary 均通过。

客户端 timeout、机器掉线或缺失 shard 都保持 `INCOMPLETE`，不得以现有样本数量冒充 n=256。

## 7. 指标与多样性解释

从合并后的 256 samples 一次性计算：

$$
k \in \{1,2,4,8,16,32,64,128,256\}.
$$

必须报告：

- `mean@256`；
- `pass@1...pass@256`；
- math 的 `maj@k`；
- `pass@256-pass@1` oracle uplift；
- 每 prompt unique normalized prediction 数；
- exact-response distinct rate；
- think/answer/format/extraction/EOS/truncation；
- output-token 分布和 generation/scoring GPU-hours。

`pass@k` 上升快不自动等价于“语言表面更丰富”。它表示正确解覆盖率随采样预算增加；因此同时
报告 unique prediction/response，区分“重复很多种错误”“多样且覆盖正确解”和“单样本本来就强”。

## 8. 公平性和统计协议

- 所有模型共享 prompt manifest、sample index、decoder 和 evaluator revision；
- 不使用某个模型的 test 结果选择温度；
- 同一 prompt 下对模型差值做 paired bootstrap，10,000 次；
- Math-7 先在数据集内 resample，再七数据集等权；
- 训练 seed 方差与 generation/sample uncertainty 分开报告；
- pass@k 曲线既报告绝对值，也报告相对 `S1-P0`、A、D0 和 GRPO 的差值。

## 9. 代码入口

已准备的核心入口：

- `recipe/joint_training/offline_eval.py`：math generation/scoring，显式 thinking、top-k/min-p、
  stable prompt/sample IDs；
- `recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py`：code generation，支持同一 shard 合同；
- `recipe/on_policy_wdl_sft/offline_eval/run_qwen3_1p7b_passk_shard.sh`：数学/代码统一 shard 入口；
- `recipe/on_policy_wdl_sft/offline_eval/run_qwen3_1p7b_passk_8gpu.sh`：单机 8×TP1 并发运行
  shard 0–7，默认吃满一台 8×L40S；
- `recipe/joint_training/merge_passk_shards.py`：exact-coverage 合并和 math multi-k 汇总；
- `recipe/on_policy_wdl_sft/offline_eval/merge_qwen3_1p7b_passk_shards.sh`：8-shard 合并入口。

所有路径和执行参数均由环境变量提供，核心入口也已登记到 training script index。正式多机运行前
还需要生成 checkpoint manifest、增加 Meituan `env.sh/jupyter.sh/run.hope` 薄包装，并完成
2 prompt × 2 shard 的真实 vLLM + merge + official code scorer smoke。

## 10. Admission 与完成标准

正式 shard 运行前必须通过：

1. checkpoint merge/extract identity 与 release gate；
2. tokenizer/chat-template thinking canary；
3. 两个 prompt 上的 `enable_thinking=true` raw-output inspection；
4. `n=2 + n=2 -> n=4` exact-coverage merge negative/positive canary；
5. math scorer 与 Code official evaluator readiness；
6. 真实 scheduler grid canary，记录 tokens/s、GPU memory、KV preemption/recompute、磁盘和
   输出大小 receipt；
7. 多机回传使用 rsync/共享持久盘，SHA-256 核验后才合并。

只有 merge summary 为 `status=pass`、每 prompt 完整 256 samples、official evaluator 完成且原始
outputs 可追溯，才可写入结果文档或 registry。

## 11. 主要外部依据

- Qwen3 官方 model card：thinking mode 推荐 `T=0.6, top_p=0.95, top_k=20, min_p=0`，且不建议 greedy；
- OpenAI Codex / HumanEval：用大量样本估计 pass@k，并按目标 k 使用不同 temperature；
- EvalPlus：额外测试会改变绝对结果，且不同模型/目标 k 的最优 temperature 不完全相同；
- vLLM 0.12 offline data-parallel 文档：小模型可按 GPU 复制 engine 处理独立 batch，并显式调节
  `max_num_seqs`、`max_model_len` 和 `gpu_memory_utilization`。
