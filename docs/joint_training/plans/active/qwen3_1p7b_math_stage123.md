# Qwen3-1.7B 数学任务 Cold-Start 与 Stage123 矩阵

- 状态：V3 16-RUN 矩阵已完成；WDL-first causal P60 已启动；D0 首步门禁通过
- 初始设计日期：2026-07-20
- 结果与后续方案更新：2026-07-23
- 目标硬件：8 × NVIDIA L40S 46 GB

## 实验目标

先确定 Qwen3-1.7B 所需的最小监督格式 cold-start，再针对 beta `0.0` 和 `0.1`
运行数据严格匹配的 Stage1 -> Stage2 -> Stage3 矩阵。矩阵包括 Stage2 的 no-KL
与 model2-only KL 对比，以及从两个 extracted submodel 分别继续训练的 Stage3。

只有重新确认 GPU 空闲后，集成队列才允许启动。队列依据用户预先授权、预注册的规则，
自动完成 cold-start 选择与 Stage123 handoff，中途无需再次人工暂停确认。

## 2026-07-20 Cold-Start Loss-Mask 事故

Cold-start V1（`lr=2e-5`）和 V2（`lr=5e-6`）不能作为有效训练证据。SFT parquet
包含完整的 `<think>...</think>` 推理，但 `MultiTurnSFTDataset` 会分别渲染每一轮消息。
Qwen3 的 Thinking template 依赖上下文；单独渲染 assistant 消息时会移除推理内容，
导致最终的 `loss_mask` 只监督 `<answer>...</answer>` 和 `<|im_end|>` token。
设置 `data.ignore_input_ids_mismatch=True` 只是隐藏了警告，并没有修复错误的逐轮序列。

修复后的 cold-start 路径设置 `data.tokenize_whole_message=True` 和
`data.ignore_input_ids_mismatch=False`。数据集现在一次性渲染完整对话，再从整条消息的
token 序列中确定 assistant span。使用真实 Qwen3-1.7B tokenizer 的 regression test
和完整 1,100 行 probe 已验证：`<think>`、完整推理正文、`</think>`、`<answer>` 与
`<|im_end|>` 都会进入 loss，而 system/user 内容不会进入 loss。

V3 必须使用全新的路径和 run identity，对应 manifest 为
`math_qwen3_1p7b_cold_start_cotmask_v3.yaml` 和
`math_qwen3_1p7b_stage123_cotmask_v3.yaml`。V3 代码与队列验证完成后，获准执行的
V3 矩阵也已完整跑完。V1/V2 checkpoint 仍然无效，不得选作 Model1，也不得发布到
DB/W&B。已完成的 V3 结果附件见下文索引。

### 机器强制门禁

目前从四层阻断这一故障模式：

1. `MultiTurnSFTDataset` 配有确定性的 CPU regression test，覆盖依赖上下文的
   Qwen Thinking template，包括多轮 assistant span。
2. `scripts/check_sft_loss_mask_policy.py` 会拒绝 frozen allowlist 以外任何新增的
   `ignore_input_ids_mismatch=True` launcher，并要求数学任务 launcher 使用
   whole-message tokenization。
3. 已纳入版本控制的 `.githooks/pre-push` 和专用 GitHub Actions workflow 会运行
   policy canary 与 regression test。`scripts/install_repo_hooks.sh` 通过
   `core.hooksPath=.githooks` 安装仓库 hook。
4. 在 V3 执行任何 optimizer step 之前，`scripts/math_cold_start_queue.py` 会调用
   `scripts/validate_sft_loss_mask.py`，使用真实 Qwen3 tokenizer 和生成的 SFT parquet
   验证全部 1,100 个样本，并写出绑定 SHA256 的结构化 receipt。只要有任何样本遗漏
   reasoning/answer/EOS 监督，或把 system/user 文本计入 loss，训练就不能启动。

公开的 cold-start 与 Stage123 queue wrapper 默认只使用 V3 manifest。历史上的
fail-closed 文本作为 launch 前 contract 保留；获准执行的 V3 已于 2026-07-22 完成，
证据链接见下文。

## 冻结的数据契约

数据源：`/data-1/dataset/math/train_rl_format.parquet`，恰好包含 7,500 条
Hendrycks MATH 数据。准备脚本使用 seed `20260719` 做一次 permutation，并在每个
shard 内保留该顺序。

| Shard | 行数 | 用途 |
| --- | ---: | --- |
| `cold_start` | 1,100 | 仅用于格式 SFT |
| `stage1` | 2,560 | beta `0.0` 和 `0.1` 的初始 Stage1；恰好 `40 x 64` 个 prompt |
| `stage2` | 1,280 | Stage2 treatment 与 matched control 的前 20 steps；恰好 `20 x 64` 个 prompt |
| `stage3` | 2,560 | Stage3 treatment 与 matched control 的后 40 steps；恰好 `40 x 64` 个 prompt |

四个主 shard 两两不重叠。其大小与固定 prompt batch 和各 phase 长度对齐，因此
Stage1、Stage2、Stage3 都会恰好完整消费各自 shard 一次，不发生 wraparound。
`stage1_control` 是唯一有意复用数据的分支：它不 reshuffle，直接按 `stage2` 再
`stage3` 的顺序拼接，使 60-step pure-Stage1 continuation 看到的 3,840 个有序 prompt
与 20-step Stage2 加 40-step Stage3 treatment 完全一致。

生成的 `dataset_receipt.json` 记录 source/shard hash、source row index、数据顺序和
overlap policy。所有正式队列都必须在训练前验证该 receipt。

## Cold-Start 流程

队列先在 step `0` 评测原始 post-trained Qwen3-1.7B。若未通过，SFT 每次增加五个
optimizer step，最多训练到 step `40`。Cold-start 格式筛选使用完整 Math-7 validation
set，`n=1`，不允许使用子集；Stage1、Stage2、Stage3 的性能评测仍使用 `n=3`。

Validation 数据集：

1. AIME-2025
2. MATH-500
3. AMC23
4. AQUA
5. GSM8K
6. MAWPS
7. SVAMP

每个 checkpoint 都记录 accuracy、response-level complete intersection，以及：

- `<think>` 完整率；
- `<answer>` 完整率；
- `\boxed{}` 提取成功率；
- reward grader 成功率；
- EOS rate；
- truncation rate。

预注册 admission gate：

| 指标 | 阈值 |
| --- | ---: |
| 完整格式契约 | >= 95% |

完整契约要求以下条件同时成立：恰好一对顺序正确的 `<think>` tag、恰好一对顺序正确的
`<answer>` tag、boxed extraction 成功、reward grader 执行成功，以及 EOS 且未截断。
各项单独比例只用于诊断，不能代替这个交集门禁。

队列在第一个通过门禁的 checkpoint 后停止 cold-start，写出所有候选指标，自动选择
最早通过的 checkpoint，并写入不可变的 `model1_selection.json`，随后把 Model1 交给
Stage123。这个过程是确定性的，因为用户已预先授权选择最早满足冻结的 95% 完整格式
门禁的 checkpoint。

### Step-0 结果

2026-07-19 使用完整 Math-7、`n=1`、temperature `0.2`、top-p `0.95` 和
`MAX_RESPONSE_LENGTH=4096` 评测了原始 Qwen3-1.7B post-trained/Instruct 模型。
全部 2,798 个 prompt 均成功完成。

| 指标 | Math-7 macro | Response-level micro | 门禁 |
| --- | ---: | ---: | ---: |
| `<think>` 完整 | 73.0% | 89.9% | >= 95% |
| `<answer>` 完整 | 67.4% | 85.1% | >= 98% |
| boxed extraction | 40.7% | 50.2% | >= 95% |
| reward grader 成功 | 100.0% | 100.0% | >= 99% |
| EOS | 72.1% | 90.2% | >= 98% |
| truncation | 27.9% | 9.8% | <= 2% |

全部 response 中只有 44.4% 同时通过两项 tag 检查、boxed extraction、reward-grader
执行和 EOS。即使只看未截断的 response，2,798 个 prompt 中仍有 1,140 个未能完成
boxed extraction，主要原因是模型返回 `<answer> 81 </answer>`，而不是
`<answer> \\boxed{81} </answer>`。竞赛级 prompt 的长度问题也很严重：AIME-2025、
AMC23 和 MATH-500 的 truncation 分别为 93.3%、52.5% 和 25.8%。

结论：step `0` 不可准入。必须进行监督格式 cold-start，从最小候选 step `5` 开始，
每次增加五步，直到第一个完整 Math-7 checkpoint 的 response-level 完整格式成功率达到
95%。

## 统一训练与 Validation 配置

- `MAX_PROMPT_LENGTH=500`
- `MAX_RESPONSE_LENGTH=4096`
- rollout 最大模型长度 `4596`
- prompt batch size `64`
- rollout `N=8`
- cold-start 格式 validation 使用 `n=1`；Stage1/2/3 validation 使用 `n=3`
- validation temperature `0.2`、top-p `0.95`，启用 sampling
- train/validation/save interval：每 `5` steps
- data seed `20260719`，`data.shuffle=False`
- fusion lambda `0.8`
- model2-only KL 类型 `low_var_kl`，系数 `0.01`
- probe 通过后 rollout GPU utilization 设为 `0.55`
- 禁用 actor entropy 计算

所有训练 phase 都使用完整 Math-7 列表做 validation。选择最佳 checkpoint 的主指标是
七个数据集各自 `acc/mean@3` 的无权 macro average：

`val-core/math7_macro/acc/mean@3`

joint Stage2 的 checkpoint 选择使用 Model2 视角：

`val-core/model2/math7_macro/acc/mean@3`

## 实验矩阵

beta `0.0` 和 beta `0.1` 使用完全 matched 的矩阵。

对每个 beta：

1. Stage1 在 `stage1` shard 上训练 40 steps。
2. Stage1 control 按顺序在 `stage2 -> stage3` 数据上继续 60 steps，达到
   effective step 100。
3. Stage2 no-KL 在 `stage2` 上训练 20 steps。
4. Stage2 model2-KL 在同一 `stage2` 上训练相同的 20 steps。
5. 每个 Stage2 arm 分别提取 model1 和 model2。
6. 四个 Stage3 run 分别在 `stage3` 上训练各 extracted submodel 40 steps。

Stage1 control 从合并后的 Stage1 权重出发，使用新的 optimizer 和 warmup 做一次
Stage1-like restart，以匹配 Stage2 handoff 时的 optimizer reset；它不是对 Stage1
optimizer state 的无缝续训。

整个矩阵共 16 个 run：两个 Stage1 source、两个 control、四个 Stage2 run 和八个
Stage3 run。

## 执行门禁

1. launch 前必须重新确认 GPU 处于空闲状态。
2. Dataset receipt 必须通过验证，并确认没有 overlap。
3. 真实模型的 loss-mask preflight receipt 必须在训练前通过。
4. 后续 launch 只允许准入 V3 whole-message-mask manifest；V1/V2 是仅供诊断的无效 run。
5. 队列复用已经完成且 identity 匹配的 step-zero Math-7 `n=1` 结果，之后每五步 validation 一次。
6. 最早满足 95% 完整格式门禁的 checkpoint 成为 Model1。
7. cold-start 训练/validation 释放 GPU 后，Stage123 自动启动。
8. queue 与 monitor 分别运行在独立 tmux session 中，并使用 append-only event 和 notification ledger。

失败或未完成的 run 只能作为诊断证据；发布到 DB 或 W&B 前，必须通过 training result release gate。

## 结果附件与可追溯性

已完成的矩阵由一份主结果附件和可机器读取的支持证据表示：

| 关系 | 文档 | 状态 |
| --- | --- | --- |
| 设计 -> 结果 -> 分析 | [`../../reports/qwen3_1p7b_math_stage123_matrix_results_20260723.md`](../../reports/qwen3_1p7b_math_stage123_matrix_results_20260723.md) | 已完成；16 个 authoritative run、136 个 validation point |
| Validation 证据 | [`../../reports/data/qwen3_1p7b_math_stage123_all_validation_steps.csv`](../../reports/data/qwen3_1p7b_math_stage123_all_validation_steps.csv) | 已完成 |
| 分数据集证据 | [`../../reports/data/qwen3_1p7b_math_stage123_all_validation_steps_by_dataset.csv`](../../reports/data/qwen3_1p7b_math_stage123_all_validation_steps_by_dataset.csv) | 已完成 |
| 训练动态 | [`../../reports/data/qwen3_1p7b_math_stage123_training_history.csv`](../../reports/data/qwen3_1p7b_math_stage123_training_history.csv) | 已完成 |
| 固定数据顺序分析 | [`../../reports/data/qwen3_1p7b_math_stage123_training_data_step_features.csv`](../../reports/data/qwen3_1p7b_math_stage123_training_data_step_features.csv) | 已完成 |
| 本地实验 registry | `/data-1/experiment_registry/experiment_registry.sqlite`，training-run ID `80–95` | 已完成 |
| W&B 云端发布 | 项目 `OnPolicyWDLSFT-Math-1P7B` | 等待 credentials；已识别 16 个 offline run，尚无 cloud sync marker |

该附件是权威的 post-run 解释。它记录：Stage3 model2 的增益真实存在，并覆盖多个数据集；
但 matched Stage1 control 最终也达到几乎相同的范围。因此，已完成的矩阵不能证明 WDL、
joint Stage2、KL 或 handoff 具有因果优势。

## 后续 Stage2 延长实验设计

### 待回答的问题

1. Stage2 超过 local step `20` 后，是否会进入 effective step `75–85` 附近观察到的同类 transition？
2. 该 transition 是由累计 optimization state 触发，还是由对应窗口消费的固定数据行触发？
3. model2-only KL 是否会改变 transition point、peak 或 truncation cost？
4. 哪个延长后的 Stage2 checkpoint 最适合作为 Stage3 handoff source？

### 主要 step 预算

主 continuation 从每个已完成 Stage2 arm 的 local-step-20 **merged joint model weights**
出发，使用**全新 optimizer 再运行 20 个 optimizer step**，不恢复 Stage2 optimizer state。
因此，每个新 run 的物理 step 为 `training/global_step=0–20`；分析和 run metadata 同时记录
`logical_stage2_step=20+training/global_step`，得到逻辑 Stage2 step `20–40`。对应的
effective step 为 `40 + logical_stage2_step`，即 `60–80`。Checkpoint 和 run name 必须
各自唯一，并同时包含物理与逻辑 step identity，避免把 fresh run 误认为 optimizer-state resume。

Checkpoint/validation 候选点是物理 step `5`、`10`、`15`、`20`，报告为逻辑 Stage2
step `25`、`30`、`35`、`40`，分别对应 effective step `65`、`70`、`75`、`80`。
逻辑 Stage2 step `40` 恰好按顺序完整消费一次下文定义的 1,280 行 canonical
Stage2-extension shard。

把 `40` 设为主预算的原因：

- local step `20` 明显过短；
- 已完成的 control 和 Stage3 曲线从 effective step `75–80` 左右开始变化；
- 每五步 validation 一次，可以分辨 transition 前、transition 期间和刚结束后的窗口；
- 40-step Stage2 足以检验该假设，不必一开始就投入额外 40-step tail。

### 主实验矩阵

| Run ID | 来源 | beta | Stage2 KL | 继续训练的 local step | 用途 |
| --- | --- | ---: | --- | ---: | --- |
| `b0-stage2-nokl-p40` | 已完成的 `b0-stage2-nokl` step-20 joint artifact | 0.0 | none | 20 -> 40 | 主 no-KL continuation |
| `b0-stage2-m2kl-p40` | 已完成的 `b0-stage2-m2kl` step-20 joint artifact | 0.0 | model2-only `0.01` | 20 -> 40 | 主 KL continuation |
| `b01-stage2-nokl-p40` | 已完成的 `b01-stage2-nokl` step-20 joint artifact | 0.1 | none | 20 -> 40 | beta 对比 |
| `b01-stage2-m2kl-p40` | 已完成的 `b01-stage2-m2kl` step-20 joint artifact | 0.1 | model2-only `0.01` | 20 -> 40 | beta/KL 对比 |

其他设置继续冻结为已完成矩阵的配置：learning rate `1e-6`、warmup `0`、prompt batch
`64`、rollout `N=8`、fusion lambda `0.8`、Math-7 `n=3`、每五步 validation/save、
response length `4096`、rollout GPU utilization `0.55`，并禁用 actor entropy 计算。

### Canonical continuation 数据

主 continuation 不得静默重复原来的 1,280 行 Stage2 数据。应从 cold start、Stage1、
Stage2、Stage3 均未消费的 3,500 条 source row 中生成新的不可变 `stage2_extension`
shard：沿用原始 seeded permutation，按保留顺序取剩余数据的前 1,280 行。Dataset receipt
必须记录 hash、source index、数据顺序，以及与所有原始 shard 的 zero overlap。

这个干净的 continuation 用于回答：额外的 Stage2 optimization 能否在未见过的 MATH 数据上
产生 transition。它本身不能回答固定数据顺序的因果问题。

### Data-order ablation

在四个主 continuation 之后运行，使用 beta `0.0`、no-KL source：

| Run ID | 数据 | Steps | 解释 |
| --- | --- | ---: | --- |
| `b0-stage2-nokl-p40-replay-s3-prefix` | 原始 Stage3 rows `0–1279`，顺序不变 | 20 -> 40 | 检验 Stage3 prefix 是否会把 transition 移入 Stage2 |
| `b0-stage2-nokl-p40-shuffled-s3-prefix` | 同一批 1,280 行，用新注册 seed 做确定性 shuffle | 20 -> 40 | 分离 row identity 与 row order 的作用 |

若 transition 只出现在 Stage3-prefix arm，支持 data-content 解释；若两个 Stage3-prefix
arm 都出现，而 canonical extension 不出现，则更支持 row identity 而非精确顺序；若三个
arm 都在相近累计 step 出现，则支持 optimization-threshold 解释。

### Checkpoint 与 Stage3 handoff 策略

不能预设 local-step-40 checkpoint 最优。使用完整 Math-7 macro `mean@3` 曲线，按以下
顺序选择 Stage2 候选点：

1. 最早满足相对 local step `20` 提升至少 `+5pp`，且 truncation 增幅不超过 `+5pp` 的 checkpoint；
2. 若不存在，则选择 local step `25–40` 中 Math-7 最好的 checkpoint；
3. 若没有任何 checkpoint 提升至少 `+2pp`，将 Stage2 extension 标记为 negative result，
   不启动大范围 Stage3 fan-out。

主 P40 continuation 只会生成逻辑 Stage2 step `40` 这个 checkpoint。若 P40 按上述规则
为正，则在 beta `0.0`、no-KL arm 上运行条件式 secondary tail：先从 P40 merged weights
出发，用 fresh optimizer 增加五个物理 step，得到逻辑 P45；再从 P45 merged weights 出发，
用 fresh optimizer 增加十五个物理 step，得到逻辑 P60。两个 tail 都记录同样的物理/逻辑
step metadata。因此，P45 和 P60 是 secondary extension checkpoint，不是 P40 结束时已经
存在的输出。

这些 checkpoint 生成后，才运行 **beta `0.0`、no-KL model2 Stage3 handoff pilot**：
分别从总 Stage2 step `40`、`45`、`60` 出发，训练到 effective step `100`。这样比较的是
matched total budget 下的 handoff timing，不会给较晚 handoff 额外训练预算。

### 监测与停止规则

每五步 validation 记录：

- Math-7 macro `mean@3`、精确 `pass@3` 和全部七个数据集指标；
- joint Stage2 的 model1/model2 validation view；
- `wdl_sft/correct_ratio`、reward mean、response-length distribution；
- EOS、truncation/clip ratio、format intersection、boxed extraction；
- grad norm、policy loss、positive/negative WDL loss 和 KL 指标。

若以下任一情况连续两次 validation 都存在，则提前停止 continuation：

- truncation/clip ratio 超过 `25%`；
- complete format 或 boxed extraction 相比 local step `20` 下降超过 `5pp`；
- loss/gradient 非有限，或出现不可恢复的 distributed/runtime failure。

### 验收标准

只有至少一个主 arm 同时满足以下条件，才支持 Stage2-extension 假设：

1. 相对 local step `20`，Math-7 macro mean@3 提升 `>=5pp`；
2. 七个数据集中至少五个得到提升；
3. 所选 checkpoint 的 truncation 增幅不超过 `5pp`；
4. 在两个 beta 设置之间，或两个 KL 条件之间得到复现。

更小的增益仍是有用的诊断证据，但不能证明原实验的主要限制是 Stage2 预算不足。

## 2026-07-26 WDL-First 因果验证修订

### 优先级与替代关系

下一项主目标是判断 weak-driven joint Stage2 本身是否带来增益。已完成的 16-run 矩阵
证明 staged pipeline 能够学习，但没有把 WDL 与普通继续训练、joint-wrapper effect、
optimizer restart、submodel extraction 或 Stage3 的作用分离开。因此，Code Stage123
矩阵在产出两个 Stage1 source 后降低优先级；目前还没有 Code Stage1 control、Stage2
或 Stage3 结果，这些 run 也不是本次 Math-first 决策的前置条件。

本修订取代上文“后续 Stage2 延长实验设计”中所有与之冲突的 launch priority、primary
matrix、fresh-optimizer continuation、P40 futility decision 和大范围 beta/KL fan-out。
上文章节仍作为设计历史保留，其中的 data-order ablation 仍有参考价值，但四臂 P40
continuation 不再是主实验。

### 主要因果问题

在匹配的 post-Stage1 预算下，带 weak-logit contribution 的 joint WDL，能否同时优于
普通 Stage1 continuation，以及移除 weak-logit contribution 后相同的 joint on-policy
路径？direct-Model2 的 D arm 是 optional：只要 manipulation probe 证明其 objective/update
与现有 Stage1 control 等价，默认队列就复用 A，而不再额外消耗 60-step 预算重跑 D。

第一轮决策固定 `beta=0.0`、禁用 KL，并使用已完成 `b0-stage1` 的 final Model2 source。
所有 arm 都消费现有 `stage2 -> stage3` shard 中同样的 3,840 行有序数据。Launch manifest
必须绑定精确的 source、row index、row order、seed、model identity、implementation identity
和 evaluator contract。任何 arm 都不得静默替换为未见数据，也不得 reshuffle 这些行。

复用 Arm A 和 B 前，必须验证新的 C/D/D0 实现与已完成矩阵具有相同的相关 training-plane
语义和 evaluator contract。任何会影响 joint forward pass、loss、data cursor、reward 或
validation 的代码变更都必须明确分类。若无法证明等价，就应在新的冻结实现下重跑受影响的
control，而不是跨不兼容实现比较。

| Arm | Stage1 后的训练 | 状态与解释 |
| --- | --- | --- |
| A：pure Stage1 | 现有 `b0-stage1-control`，60 steps | 现有 ordinary-training baseline，effective step 40 -> 100 |
| B：short WDL handoff | 现有 `b0-stage2-nokl` 先运行 20 个 joint WDL step，再让 extracted Model2 运行 40-step Stage3 | 现有 staged-pipeline treatment；它本身不能隔离 WDL |
| C：continuous WDL | 新增不间断 60-step joint Stage2，`beta=0.0`、no KL、`fusion_lambda=0.8` | 核心方法的主 arm |
| D：strong-only equivalence arm | Optional 60-step joint-wrapper replication，`beta=0.0`、no KL、`fusion_lambda=1.0` | probe 证明 direct-Model2 behavior 后默认省略；只有明确需要 same-wrapper replication 时才运行 |
| D0：matched-scale no-weak control | 新增不间断 60-step joint path，使用 `fusion_mode=strong_scaled`，$z_{train}=0.8z_2$ | H1 的主 control：移除 $0.2z_1$，同时保留 C 的 $0.8z_2$ 系数 |

现有 Stage2 topology 使用 **Model2-only rollout**；joint fused logits 只影响 teacher-forced
WDL training objective。因此 C/D/D0 共享 rollout 的 *source、prompt/data order、reward
function 和 generation configuration*。但 update 后它们的 Model2 权重会逐渐分化，所以
越过 zero-step manipulation probe 后，实际采样的 response 和 reward label 不应被假设为
仍然相同。本实验不检验 weak-driven rollout sampling。

D 是 optional diagnostic，且与 C 不满足 scale matching：它比较 $0.2z_1+0.8z_2$ 与
$z_2$。因此 H1 使用 D0，把 C 的 objective 与 $0.8z_2$ 比较。没有 D0 时，C-D 会同时
加入 weak logits 并改变 strong-logit scale/effective temperature；entropy 或 logit 指标
可以暴露这个 confound，但不能消除它。只有 machine-readable zero-step/backward/update
probe 验证精确公式、Model1 invariance、Model1 gradient 为零以及 Model1 hash 不变后，
D 和 D0 才能准入。对 D 而言，该 receipt 同时也是省略完整 run 的 fail-closed 条件：
`D_is_direct_model2`、`D_ignores_and_does_not_update_model1` 和
`D_and_D0_are_model1_invariant` 必须全部通过。任何检查失败，都不得静默省略 D。

### Continuation 语义

优先从已完成 Stage1 source 出发，原生、不间断地重跑 60-step Stage2。它必须维持单一
Stage2 optimizer trajectory，并明确记录 data cursor、RNG state、scheduler/global-step
语义和 checkpoint resume contract。当前默认使用 constant LR、zero warmup，因此
scheduler restore 不太可能解释差异，但 data position 和 RNG 差异仍可能造成影响。

用 fresh optimizer 从 merged weight restart，不等价于 continuous Stage2。Fresh-restart
和真实 optimizer-state continuation 可以保留为 secondary ablation，但都不能替代 Arm C
或 D。尤其不能把此前计划在 P40、P45、P60 串联 fresh optimizer 的方案描述为一次连续
60-step Stage2 run。

### 训练长度与 checkpoint

Stage2 local step 到 effective step 的映射为 `40 + local_stage2_step`。

| Stage2 checkpoint | Effective step | 决策作用 |
| ---: | ---: | --- |
| P40 | 80 | 只用于 onset probe，不足以得出负面结论 |
| P45 | 85 | 最小可解释终点，因为已完成矩阵的 transition 覆盖 effective step 75–85 |
| P50 | 90 | 预注册的 interim efficacy/futility 决策点；不能用于最终正面结论 |
| P60 | 100 | 主要 confirmatory matched-budget 与 late-stability 终点 |

从 P30 到 P60 每五步 validation 并保存一次。至少在 P20、P40、P45、P50、P60 保留
optimizer-resumable anchor。P40 时健康的 run 应继续执行；P50 前不得做 futility 判断。
除非触发 hard safety stop，否则 matched-budget 的正面结论仍必须以 P60 为依据。

### 评测与结论标准

主要 efficacy endpoint 是提取并评测的 **Model2-only** view；fused joint score 仅用于
诊断，不能替代它。在每个选定 checkpoint 记录 Math-7 macro `mean@3`、精确 `pass@3`、
全部七个数据集指标、paired task-level output、Model1/Model2 validation view、WDL correct
ratio 与 loss component、response-length distribution、EOS/truncation、complete-format、
boxed extraction、gradient norm，以及即使关闭 KL 后仍被输出的任何 KL 指标。主比较使用
paired task-level bootstrap resampling；不得从各 arm 独立挑选的 aggregate peak 推断因果。

只有 Arm C 在共同 effective step 同时满足以下条件，第一轮实验才支持核心 WDL effect：

1. Math-7 macro `mean@3` 同时比 Arm A 和 Arm D0 高约至少 `2pp`；
2. 相对两个 control 的 paired task-level bootstrap 95% confidence interval 下界都高于零；
3. 相对两个 control，七个 Math 数据集中至少五个得到提升；
4. truncation 增幅不超过 `5pp`，complete-format 或 boxed extraction 的降幅也不超过 `5pp`；
5. 在提出一般性方法结论前，随后用独立第二 seed、`beta=0.0` 完成复现。后续
   `beta=0.1` run 检验 treatment-effect heterogeneity，不能替代 seed replication。

### 预注册假设与 estimand

令 $M_X(s)$ 表示 arm $X$ 在 Stage1 后继续优化 $s$ 个 step 时，Model2-only Math-7
macro `mean@3` 的值。所有比较使用相同 effective step、冻结的 evaluator、prompt set、
sample count 和 generation setting。对 Arm B，$s \le 20$ 对应 joint Stage2，$s > 20$
对应 extracted-Model2 Stage3；对必跑的 Arm C 和 D0，$s$ 对应 continuous joint Stage2；
只有实际启用 optional D arm 时，D 才使用相同定义。Confirmatory endpoint 为 $s=60$
（effective step 100）；$s=50$ 是预注册的 interim decision。

实验检验四个彼此不同的假设，不能把它们合并成一句泛化的“WDL 有效”。

| ID | 假设 | Estimand | 正面结果能够证明什么 |
| --- | --- | --- | --- |
| H1：additive weak-logit contribution | 在 matched 的 $0.8z_2$ training scale 下，加入 $0.2z_1$ 能提升 Model2 | $\Delta_{weak}(s)=M_C(s)-M_{D0}(s)$ | weak-logit content 在 matched strong-logit coefficient 之外产生贡献；这还不能证明必须 joint update Model1 |
| H2：practical method value | Continuous WDL 优于普通 Stage1 继续训练 | $\Delta_{stage1}(s)=M_C(s)-M_A(s)$ | 完整 WDL treatment 相比更简单 baseline 具有实际价值 |
| H3：pipeline allocation | Continuous WDL60 优于原来的 WDL20 -> extracted-Model2 Stage3-40 pipeline | $\Delta_{allocation}(60)=M_C(60)-M_B(60)$ | continuous-joint allocation 优于原 handoff allocation；它**不能**单独隔离 Stage2 duration |
| H4：late-window stability | P45 后出现的 WDL 增益能保持到 matched final endpoint，且不发生质量崩塌 | P45-P60 trajectory、P60 non-inferiority margin 和 P60 quality gate | late-window gain 不只是短暂的长度/格式 regime shift 或不稳定 peak |

H1 是当前对论文 weak-driven training objective 最干净的检验。C-D 仍只是实用层面的
mixture-versus-strong-only 比较，不是 H1 estimand。H2 检验额外机制在实际中是否值得使用。
H3 同时改变 duration allocation、P20 后的 objective/architecture、extraction 和 optimizer
handoff；它是 pipeline-allocation 结果，不是纯 Stage2-dose estimate，也不能单独证明 weak
guidance 有效。若要主张 Stage2 dose-response curve，未来还需要 matched
`WDL40 -> Stage3-20` arm。H4 是 H1/H2 的必要安全条件，不是独立 accuracy claim。

主 null hypothesis 是 $\Delta_{weak}(60) \le 0$ 和
$\Delta_{stage1}(60) \le 0$。方法级正面结论要求两个 delta 均约至少 `+2pp`，且
stratified paired-bootstrap 95% confidence interval 下界高于零。这只能支持“观测到的正向
effect magnitude 达到 `2pp`”；若要声称真实 effect 至少为 `2pp`，则 CI 下界必须超过
`+2pp`。若要声称 effect 在实际意义上不存在，应使用 $\pm1$ pp 的 equivalence margin：
90% confidence interval 完全落在 $[-1,+1]$ pp 内才支持 practical equivalence。仅仅没有
通过 superiority test 属于**结论不确定**，不能证明没有 effect。H3 只有在
$\Delta_{allocation}(60) \ge 2$ pp、stratified paired-bootstrap 95% CI 下界高于零，且
七个数据集中至少五个支持 C 优于 B 时才为正。H4 要求最终 H1/H2 superiority gate 通过，
预先指定的 P50-to-P60 paired non-inferiority test 的 95% CI 下界高于 `-2pp`，并通过全部
format/truncation quality gate。P45/P50/P55 的最大值只作描述；把有噪声的 selected peak
当作参照会引入 winner's curse。

### Treatment manipulation 与准入指标

只有先证明 intended treatment difference，accuracy 比较才具有因果解释。第一个
optimizer step 前，必须输出 machine-readable receipt，覆盖：

- 精确的 Model1/Model2 source hash、tokenizer/chat-template hash、joint config、
  implementation identity、reward/evaluator identity 和全部 optimizer hyperparameter；
- 精确的 3,840 个 source-row index 和有序 `stage2 -> stage3` hash；C、D、D0 使用相同
  seed 和 `data.shuffle=False`，并验证与 A/B 等价；
- `beta=0.0`、关闭 KL、Model2-only rollout、C `mixture/0.8`、D `mixture/1.0`、
  D0 `strong_scaled/0.8`；
- zero-step manipulation probe：替换 Model1 会改变 C 的 fused training logits，但不会
  改变 D 或 D0；在同一 frozen Model2 上，三个 arm 的 rollout logits 都保持 Model2-only
  且完全一致；
- backward/update probe：C 的 Model1 与 Model2 gradient 都非零；D/D0 的 Model1 gradient
  为零，且完成一个 optimizer step 后 Model1 parameter hash 不变；
- 对同一个固定 microbatch，在 tolerance 内，D 的 logits/objective 等于 direct Model2，
  D0 logits 等于 $0.8z_2$。

任何 manipulation check 失败都会阻断受影响 arm。尤其是 `fusion_lambda=1.0` 只能建立
direct strong-only D arm；它不是 matched-scale no-weak control，不能替代 D0。

### 训练信号与 loss 指标

每个 optimizer step 都记录下列指标。这些指标用于解释 run *为什么*学习或失败，但都不能
替代 Model2-only evaluation endpoint。

| 指标 | 预期解释 | 失败或告警特征 |
| --- | --- | --- |
| `wdl_sft/correct_ratio` | 能提供 positive SFT signal 的 sampled rollout 比例 | 接近零说明 beta-zero WDL 几乎没有学习信号；上升可能标记 transition，但不能证明 generalization |
| `critic/rewards/mean`, `critic/score/mean` | reward function 下的 online rollout correctness | 若与长度一起上升但 held-out accuracy 不升，更像 reward/length regime change，而非方法成功 |
| `actor/wdl_sft_loss_positive` | correct rollout 上实际的 beta-zero optimization term | 非有限值是 fatal；correct count 非零但该项为零说明 plumbing error |
| `actor/wdl_sft_loss_total` | `beta=0.0` 时必须等于 positive term | 任何实质差异都说明 beta/KL/loss 配置与实验声明不符 |
| `actor/wdl_sft_loss_negative` | 仅用于诊断 incorrect rollout 的 likelihood；此处乘以零 | 不得解释为 active training force 或提升证据 |
| `actor/pg_loss` | aggregation/scaling 后 trainer-level policy loss | 若持续与配置的 WDL total 不一致，继续前必须调查 |
| `actor/grad_norm` | 全局 optimization signal 与 numerical health | 非有限是 fatal；有 positive example 时持续为零说明没有 update；数量级突然跳变属于告警 |
| `jointTraining/model1_grad_norm`, `jointTraining/model2_grad_norm` | 确认哪个 submodel 收到学习信号 | C 应同时 update 两者；D/D0 的 Model1 必须严格为零且保持不变 |
| `jointTraining/model_grad_norm_ratio` 与 gradient share | 显示 weak/strong update 哪一方占主导 | 持续极端失衡可以解释 collapse 或 guidance 无效 |
| `jointTraining/model_grad_cosine_similarity` | 两个 submodel 的 update direction 是一致还是冲突 | 持续强负 cosine 是机制告警，本身不是停止或成功标准 |
| `jointTraining/submodel_total_variation`, `jointTraining/submodel_js_divergence`, `jointTraining/submodel_top1_agreement` | 可按 scale 解释的 weak/strong distribution disagreement | 接近零说明可利用的 weak signal 很少；disagreement 本身既不代表有益，也不代表有害 |
| `jointTraining/fused_vs_model2_chosen_token_logprob_{delta,abs}_mean` | mixture 对 response token 的直接影响；positive/negative split 用于诊断 | D 必须为零；C 对 D0 分离 mixture effect 与 strong-logit scaling |
| `actor/lr` | 验证 constant `1e-6`、zero-warmup 契约 | 任何 scheduler drift 都会使 matched comparison 失效 |

实现必须输出 `wdl_sft/all_incorrect_group_ratio`、`wdl_sft/mixed_group_ratio`、
`wdl_sft/all_correct_group_ratio`，以及 positive supervised response/token count。
当 `beta=0` 时，all-incorrect group 不产生有效 gradient。还要记录
`actor/grad_clip_event` 和 `actor/optimizer_step_applied`，由 run-level monitor 计算
clipping rate。

`jointTraining/model{1,2}/wdl_sft_loss_*` 是在**同一批 Model2 rollout 与 reward label**
上计算的 counterfactual teacher-forced NLL diagnostic。它们不是两个分别优化的 loss，
也不得表述成两个模型各自的 on-policy training reward。每五步的 Model1-only 与
Model2-only validation view 才提供独立的 capability/reward/format 证据。

Full-vocabulary entropy 仍为 optional，主 run 中保持禁用。它可以诊断 C-D 的 logit-scale
差异，但不能修复 confound。Fixed-token logit reduction 和 chosen-token log-prob delta
属于必需指标；不保留原始 full logits。

### Rollout、格式与 run-health 指标

每个 optimizer step 记录 response 与系统健康状况；每个 validation checkpoint 计算
held-out output quality。

| 类别 | 必需指标 | 门禁或解释 |
| --- | --- | --- |
| Response regime | `response_length/mean`、p50、p95、`response_length/clip_ratio`、`response/aborted_ratio`、EOS rate | 长度增加不等于提升。Truncation 超过 25% 触发 hard stop；在声明结论的 endpoint，C 相对 matched control 的 truncation 增幅不得超过 5pp |
| Output contract | complete-format rate、boxed-extraction success/failure、malformed-answer rate | 相比 matched start 或 control 下降超过 5pp 会阻断正面结论，并触发基于持续性的停止规则 |
| Reward integrity | scorer error/timeout rate、reward/answer-extraction disagreement | 任何 arm-specific scorer failure 都会使 reward 与 loss signal 不可比；重跑前，受影响比较均无效 |
| Numerical health | loss 与 gradient 有限、grad clipping rate、optimizer-step success、parameter-update norm | NaN/Inf 或 optimizer step 跳过/失败是 fatal；反复 clipping 或 parameter 不连续跳变需要诊断 |
| Distributed/runtime | GPU utilization/memory、MFU、step 与 validation latency、OOM、NCCL/Ray/vLLM error、worker restart | 用于区分 validation 缓慢和 run 卡死/部分失败；runtime recovery 不得静默重复或跳过数据 |
| Data/checkpoint integrity | global/logical step、data cursor、RNG state、consumed-row receipt、checkpoint 与 optimizer-shard 完整性 | 任何行重复、跳过、重排，或缺少可 resume 的必要 anchor，都会使 matched trajectory 失效 |

Monitor 必须报告共同 effective step 上的 absolute value，以及 C-minus-D0/D/A delta。
绝不能仅凭 loss、reward、response length 或 fused-joint validation score 宣布成功。

### 评测协议与最终决策指标

对必跑的新 Arm C 和 D0，online evaluation 在 P0 以及其后每五步直到 P60，使用完整
Math-7 suite，并同时评测 Model1/Model2 view。只有显式启用 optional D arm 时，D 才执行
相同协议。Model2 是 primary endpoint；Model1 用于 collapse/health 诊断。Common-step
描述表在 P30、P35、P40、P45、P50、P55、P60 纳入 A、B、C、D0；若 D 实际运行，也纳入
D。A/B 使用保留的历史 validation 证据。因为并非所有 A/B intermediate weight 都被保留，
这些中间比较只能描述 trajectory，不能支持 paired causal inference。到 P60 时，使用与
C/D0（以及实际运行的 D）完全相同的 evaluator 和 generation seed，重新评测保留的 A/B
final model；confirmatory superiority/equivalence test 只使用这次共同 P60 re-evaluation。
若任何必需的 P60 final model 不可用或 identity check 失败，就在冻结协议下重跑该 arm。

最终必需指标包括：

- primary：P60 的 `val-core/model2/math7_macro/acc/mean@3`；
- secondary capability：Math-7 macro `pass@3` 和分数据集 `pass@3`；
- consistency：`pass@3 - mean@3`，与 extraction failure 一起报告，避免把 reasoning
  variance 与 format failure 混淆；
- 七个数据集各自的 `mean@3`，并统计相对 A 和 D0 的方向数量；只有 optional D 实际运行时才加 D；
- 每个 arm 的 complete-format、boxed extraction、EOS、truncation 与 response-length distribution；
- $\Delta_{weak}$、$\Delta_{stage1}$、$\Delta_{allocation}$ 的 stratified
  paired-bootstrap confidence interval。

Math-7 是对规模差异很大的七个数据集做无权 macro average，因此应在**每个数据集内部**
bootstrap prompt，再对七个 resampled dataset score 等权平均。使用 10,000 次冻结的 bootstrap
resample。把所有 prompt 混在一起 bootstrap 会估计另一个由大数据集加权的 target。预注册
generation seed，并保留所有 per-prompt/per-sample output。精确 Math `pass@3` 由分析脚本
从这些 JSONL output 计算，trainer scalar 不会原生输出它。Prompt bootstrap 不包含
training-seed uncertainty。

P50 是 interim decision：只有当 C 比 A 和 D0 至少低 `2pp`（若 optional D 实际运行，也要
低于 D）、held-out accuracy 和 learning-signal metric 都没有出现 transition，且不存在能
触及 superiority region 的合理 confidence interval 时，才因 futility 停止。否则继续到 P60。
P50 不作任何方法级正面声明。

Confirmatory reporting 使用冻结的 offline `n=8` evaluation，在 P60 评测 A、B、C、D0；
只有 optional D 实际运行时才加入 D。`n=8` 用于复现 `n=3` 的方向，不得在看到哪个指标更
有利后把它选作替代结果。若不运行该 confirmatory evaluation，结论只适用于预注册的
Math-7 `n=3` 协议。

### 结果决策表

| Matched endpoint 的观测结果 | 结论 |
| --- | --- |
| C 按 superiority/quality 标准同时优于 D0 和 A，且随后复现 | 支持 additive weak-logit training contribution，完整方法具有实际价值 |
| C > D0，但 C 与 A 近似相等或低于 A | weak logits 会影响 joint objective，但方法尚未证明优于更简单的继续训练 |
| C 在 equivalence margin 内与 D0 近似相等 | 该配置下没有具实际意义的 additive weak-logit contribution |
| C > D，但 C 与 D0 近似相等 | 表面的 C-D 增益来自 strong-logit scale/effective-temperature 变化，而不是 weak logits |
| C 和 D0 都优于 A（若 D 运行，D 也优于 A） | 收益更可能来自 on-policy forward-SFT/joint execution package，而不是 weak guidance |
| C 按 H3 规则优于 B，但不优于 A/D0 | Continuous WDL60 优于旧 WDL20 -> Stage3-40 allocation，但既没有验证纯 Stage2-dose effect，也没有验证 WDL 本身 |
| B > C | short WDL intervention 加 single-model Stage3 优于 continuous joint WDL |
| C 在 P60 前达到 peak，随后违反 stability/quality gate | WDL 可能只适合作为短暂 intervention；不支持 continuous objective |
| Confidence interval 仍很宽，同时跨过零和 superiority margin | 结论不确定；应增加 evaluation sample 或 seed，而不是宣布成功或失败 |

### Hard stop 与执行顺序

若某 arm 持续出现非有限 loss/gradient、不可恢复的 distributed/runtime failure、truncation
超过 `25%`，或 complete-format/boxed extraction 相比 matched start 下降超过 `5pp`，则停止
该 arm。除非继续运行可能破坏 run，否则单次短暂 validation 异常不触发 quality stop；一般
要求连续两次 validation 都存在。

只有 Arm C 显示核心信号后，实验才按以下顺序扩展：

1. `fusion_lambda=0.8` 且冻结 Model1，以分离 fixed weak guidance 与 joint update weak model；
2. 使用独立第二 seed、`beta=0.0` 做必需复现；
3. 运行 matched `beta=0.1`，作为单独的 heterogeneity/robustness test；
4. 若需要 Stage2 dose-response 结论，增加 `WDL40 -> Stage3-20` matched allocation arm；
5. 比较 optimizer-state continuation 与 fresh restart，并执行前述 data-order intervention；
6. 最后再做 model2-only KL 和更广泛的 handoff/duration sweep。

在核心比较通过前，KL、大范围 beta grid 和额外 Code Stage123 分支都属于次要实验，不得
占用主实验预算。

### 2026-07-27 launch 与首步准入记录

获准的 causal queue 最初以 `D0 -> D -> C` 加载到 live process。2026-07-27，在 D 启动前，
contract 修订为 `D0 -> C`：D 改为 optional equivalence arm，其 wrapper 改为 fail-closed，
因此已运行的 process 也会在 equivalence receipt 通过后省略 D。只有显式设置
`--include-optional-d` / `RUN_OPTIONAL_D=1` 才会恢复 D run。

第一次 launch attempt 在任何 optimizer step 之前的 Hydra composition 阶段失败，因为
`trainer.protected_ckpt_steps` 需要显式 `+` override。该 diagnostic log 被保留，但不属于
实验结果。修复 override，并避免把空 checkpoint directory 误判为 resume source 后，第二次
launch 进入真实 D0 training flow。

当前 D0 run 为 `MATH-WDL-CAUSAL-P60-ARM-D0-QWEN3-1P7B_1785119985`。Model1 与
Model2 的 P0 Math-7 validation view 均已完成。第一个 optimizer step 通过以下
machine-readable gate：
`/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/admission/first_step_receipt_1785119961.json`。

- `actor/optimizer_step_applied=1`，`actor/grad_norm=88.6049`；
- `jointTraining/model1_grad_norm=0`，符合 D0 要求；
- `jointTraining/model2_grad_norm=87.2251`；
- 88 个 positive supervised response、26,152 个 positive supervised token；
- gate 要求的全部 group-composition、response-quantile 和 chosen-token logit diagnostic
  都存在且为有限值；
- admission 时没有出现 `CUDA out of memory`、`NCCL ERROR`、`ActorDied` 或
  `WorkerCrashed` marker。

该 receipt 证明获准的 D0 treatment 正在执行，并产生真实 Model2 update。它只属于
launch/health 结果，不能支持 H1–H4；H1–H4 仍取决于直到 P60 的共同 step held-out comparison
和预注册 confirmatory evaluation。
