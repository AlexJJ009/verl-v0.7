# Qwen3-1.7B 数学与代码 Standard RLVR / GRPO 对比实验方案

- 文档职责：冻结 Standard On-Policy SFT、WDL 与 canonical GRPO 的对比问题、训练预算、公平性和评测协议
- 当前状态：设计已按连续 GRPO 纠偏；本地核心入口、AFO 四层薄包装、数据合并与配置/loss-mask 测试已准备，GPU admission 与训练未开始
- 当前修订：2026-08-11
- 目标硬件：本地 8 × NVIDIA L40S 46 GB；训练入口必须同时支持美团 AFO
- 关联方案：[`qwen3_1p7b_math_stage123.md`](qwen3_1p7b_math_stage123.md)、[`qwen3_1p7b_code_acd0_p60_beta0.md`](qwen3_1p7b_code_acd0_p60_beta0.md)

本文中的 RLVR 是研究范式，canonical GRPO 是本轮采用的标准 RLVR 算法。仓库中没有名为
`ILVR` 的独立算法实现；若口头讨论中的 “ILVR” 指的就是 RLVR，则不把两者拆成两个实验。

## 1. 术语和已有实验重命名

### 1.1 两个零点 checkpoint

本矩阵有两个不同的初始化锚点，必须分开命名：

| 名称 | checkpoint | 含义 | 是否训练 |
| --- | --- | --- | --- |
| `CS0` | 通过格式门禁的 cold-start checkpoint | Stage1 之前的能力与格式锚点 | 否，只做 evaluation |
| `S1-P0` | Stage1 结束后导出的 Model2 | post-Stage1 矩阵的 step 0；现有 A/C/D0 的 P0 | 否，只做 evaluation |

`S1-P0 evaluation` 的含义是：对 Stage1 完成后的 Model2 做与其他最终 checkpoint 完全相同的
evaluation。在线曲线先沿用对应任务的 `n=3` 配置；正式结果使用共同冻结的 `n=256 pass@k`。
它不是新增训练 run，也不能只复用历史在线分数代替共同冻结重评。

### 1.2 A 的正式名称

现有 A 正式命名为 **Standard On-Policy SFT continuation**，保留 `A` 作为短标签。其机制是：

1. 当前 Model2 对每个 prompt 生成 8 个 on-policy responses；
2. reward/evaluator 选出正确且满足格式与完成条件的 responses；
3. 只对这些正样本做 forward SFT；错误 response 不产生负向梯度；
4. 不使用 group-relative advantage、PPO clipping 或 KL penalty。

`RFT` 或 `filtered SFT` 可以描述其所属方法族，但不是本项目的正式实验名。Stage1 与 A 使用同一
Standard On-Policy SFT 算法；Stage1/A 的分界只表示数据窗口、checkpoint 和 fresh-optimizer
handoff，不表示算法发生变化。

## 2. 实验问题

本轮优先回答三个问题：

1. **外部算法对比**：从相同 Stage1 Model2 出发，WDL C 是否优于 canonical GRPO？
2. **Stage1 是否必要**：比较连续 100-step `Cold Start + GRPO` 与复用既有 40-step Stage1 后再跑
   60-step `Stage1 + GRPO`；前者只保存 step 40 checkpoint，不重启 optimizer。
3. **最终能力是否稳健**：在线 `mean@3` 的排序能否在共同冻结的 `n=256 pass@k` 上保持？

## 3. 主实验矩阵

### 3.1 Pipeline 定义

| ID | Pipeline | 前 40 step：`stage1` shard | 后 60 step：`stage2 -> stage3` shard | 主要用途 |
| --- | --- | --- | --- | --- |
| `P0-CS` | Cold-start anchor | 不训练 | 不训练 | cold-start 绝对能力锚点 |
| `P0-S1` | Stage1 Model2 anchor | Standard On-Policy SFT | 不训练 | post-Stage1 的 P0 锚点 |
| `A` | Standard On-Policy SFT | 复用已完成 Stage1 | Standard On-Policy SFT | practical baseline |
| `C` | Continuous WDL | 复用已完成 Stage1 | joint WDL，$0.2z_1+0.8z_2$ | 核心方法 |
| `D0` | Matched-scale no-weak | 复用已完成 Stage1 | $0.8z_2$，只更新 Model2 | weak-logit causal control |
| `Stage1 + GRPO` | Stage1 -> GRPO | 复用已完成 Stage1 | canonical GRPO | **主外部 Baseline** |
| `Cold Start + GRPO` | Cold-start -> GRPO | 连续 canonical GRPO | 连续 canonical GRPO | 检验是否需要 SFT Stage1 |

`Cold Start + GRPO` 是单个 100-step run：依次消费 stage1/stage2/stage3 数据，在 step 40 保存
checkpoint 后原 optimizer 继续训练，不能人为拆成两个 run。`Stage1 + GRPO` 从既有 S1-P0
checkpoint 以 fresh GRPO optimizer 消费相同的后 3,840 prompts。两条 pipeline 的差别正是前
40 step 使用 Standard On-Policy SFT 还是 GRPO；不再增加 optimizer-restart diagnostic。

### 3.2 三个主比较

| 比较 | 只允许回答的问题 | 不能直接回答的问题 |
| --- | --- | --- |
| `C - (Stage1 + GRPO)` | 相同 Stage1 初始化和 60-step post-Stage1 数据预算下，WDL 是否优于标准 GRPO | weak logits 的纯因果贡献；该问题仍由 `C-D0` 回答 |
| `(Stage1 + GRPO) - A` | 相同初始化和数据下，group-relative RL 是否优于正样本 On-Policy SFT | Stage1 是否有价值 |
| `(Cold Start + GRPO) - (Stage1 + GRPO)` | 相同 100-step 总 prompt/rollout budget 下，前 40 step 使用 GRPO 还是 Standard On-Policy SFT 更好 | GRPO 与 WDL 的直接优劣 |

## 4. 假设、预期结果和反证解释

| 假设 | 符合预期时的结论 | 不符合预期时的结论 |
| --- | --- | --- |
| H1：`Stage1 + GRPO > A` | 标准 GRPO 利用组内相对 reward 和错误样本信息，比只学习正确 response 的 Standard On-Policy SFT 更有效 | A 已经吸收了大部分可用信号，或 GRPO 在当前 reward/data/budget 下没有优势；不能仅凭“GRPO 是标准方法”假定它更强 |
| H2：`C > Stage1 + GRPO` | WDL 不仅优于内部 control，也在 matched Stage1/post-Stage1 合同下优于标准 RLVR Baseline，支持方法具有外部竞争力 | 若 `C <= Stage1 + GRPO`，weak logits 仍可能有因果贡献，但当前 WDL 不能声称优于标准 GRPO |
| H3：`Stage1 + GRPO > Cold Start + GRPO` | Standard On-Policy SFT Stage1 提供了更好的初始化、格式或 reward density，提高后续 GRPO 的样本效率 | 若 `Cold Start + GRPO >= Stage1 + GRPO`，Stage1 不是必需步骤，甚至可能限制探索；应优先采用更短的 direct-GRPO pipeline |
| H4：online 排序在 offline pass@k 保持 | 当前在线 `mean@3` 足以作为 checkpoint selection signal | 若排序反转，在线 n=3 只能用于训练监控，最终结论必须以后续共同冻结 pass@k 为准 |
| H5：C 的绝对分数更高且 cost-normalized 不劣 | WDL 同时具有质量和计算竞争力 | 若 C 更高但 GPU-hours/token 成本显著更大，只能声称质量更高，不能声称效率更高 |

第一轮不预注册“C 必须胜过 GRPO”的方向性结论。当前合理先验是 GRPO 应明显强于或至少不弱于
A，但 C 与 GRPO 的排序未知；这正是该矩阵需要实测的问题。

## 5. GRPO 算法合同

第一轮使用仓库已有 single-model canonical GRPO 实现，不另写 trainer：

- 复用 `recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh`；
- 参考 `run_2g_base.sh`、`run_2g_sft.sh` 与现有 `run_2g_math_*.sh`；
- `algorithm.adv_estimator=grpo`；
- `policy_loss.loss_mode=vanilla`；
- group size / rollout `N=8`；
- `norm_adv_by_std_in_grpo=true`；
- symmetric PPO clip：low/high 均为 `0.2`；
- actor PPO epochs `1`；
- `algorithm.use_kl_in_reward=false`，避免把 KL 混入 reward/advantage；
- actor loss 中启用固定 reference-policy KL：`use_kl_loss=true`、`kl_loss_coef=0.001`、
  `kl_loss_type=low_var_kl`；reference 是每条 GRPO run 的初始化 checkpoint；
- no WDL loss、no joint model、no weak logits；
- no rollout IS correction；
- Model2/single-model rollout 与更新。

`N=8` 既符合现有 A/C/D0 的 rollout budget，也与本仓库 canonical GRPO 历史入口一致，因此是
本轮唯一 group-size 主设置。group-size ablation 后置，不与第一轮方法有效性验证混跑。

这里必须区分三个 batch 概念：每 step 是 `64` 个 prompt；每 prompt 生成 `N=8` 个 response，
所以 reward/policy batch 是 `512` 条 trajectory；actor 的 global PPO mini-batch 也固定为 `512`，
与当前 1.7B on-policy 合同一致。预算比较以 prompt、generated response、token 和 GPU-hours 同时
记录，不能只写一个含混的 “batch size”。

代码审计确认旧 `run_2g_*` 存在 resolved-config 漂移：注释和索引称 no-IS/no-KL，实际 common
默认 `ROLLOUT_IS=token` 且 KL 关闭。新 1.7B 入口已显式冻结 `ROLLOUT_IS=null` 和上述 actor-KL
合同，并由 config-only 测试断言。若后续评估 token correction 或 no-KL，必须单独命名为
`GRPO + rollout-IS` / `GRPO no-KL` sensitivity arm，不能混入主 Baseline。

## 6. 任务合同

### 6.1 共同条件

数学和代码分别保持自己现有的 frozen contract：

| 配置 | 数学 | 代码 |
| --- | --- | --- |
| Model | Qwen3-1.7B | Qwen3-1.7B |
| Prompt / response length | `500 / 4096` | `1024 / 8192` |
| Prompt batch / rollout N | `64 / 8` | `64 / 8` |
| Data shuffle | `False` | `False` |
| Stage1 rows / steps | `2,560 / 40` | `2,560 / 40` |
| Post-Stage1 rows / steps | `3,840 / 60` | `3,840 / 60` |
| Online validation | Math-7，`n=3` | Code-3，`n=3` |
| Validation / save interval | 每 5 step | 每 5 step |
| Reward/evaluator | 现有 strict math scorer | 现有 official-aligned code scorer |

不同任务的最大长度不能为了“参数看起来一致”而强行统一。公平性要求同一任务内各方法保持完全
相同的 prompt/response contract；数学与代码之间允许使用各自已经冻结的长度。

两项数据 receipt 都已验证相同结构：stage1 `2,560` 行、stage2 `1,280` 行、stage3 `2,560`
行。`batch=64` 且 `shuffle=False` 时分别精确支持 `40 + 20 + 40 = 100` step，不需要重复样本。
`Stage1 + GRPO` 使用既有 `stage2 -> stage3` 合并 shard，共 `3,840/64=60` step；
`Cold Start + GRPO` 使用新生成并带 SHA-256 receipt 的 `stage1 -> stage2 -> stage3` 6,400-row
连续 shard，共 100 step。若继续训练超过 P100，就必须预注册新数据窗口或明确进入第二 epoch，
不能静默循环当前 6,400 rows。

### 6.2 Thinking 与 loss-mask 合同

GRPO 不使用 Cold Start SFT 的 assistant-label mask。它对模型自己生成的整个 response 计算
policy-gradient loss：`response_mask` 直接取 response 区间的 attention mask，因此从 `<think>`
开始、thinking body、`</think>` 到最终 answer 的所有非 padding token 都参与 loss；不会只训练
answer token。历史 “CoT token 没进入 SFT loss mask” 的事故不能原样复发，但 prompt template
仍可能关闭 thinking，因此新入口同时强制：

- `data.apply_chat_template_kwargs.enable_thinking=True`；
- resolved config 中该字段必须为 true；
- 单测构造 `<think> / reasoning / </think> / answer / pad` 五个位置，断言前四个位置梯度非零、
  只有 padding 梯度为零；
- 1-step GPU smoke 还必须检查原始 rollout 的 `<think>` 非空率、闭合率和 response-mask token 数。

### 6.3 Learning rate

第一轮 GRPO learning rate 只使用 `5e-7`。这是本仓库已有 GRPO 实验的稳定主设置，也避免把
偏 SFT 的 `1e-6` 直接移植到 policy-gradient baseline。只有当训练健康但学习信号明显不足时，
才新增预注册的 LR sensitivity；第一轮不并行跑 `1e-6`，也不事后挑最好 LR。

## 7. 训练 budget 与 checkpoint 规则

主预算先按 optimizer steps 冻结，因为 batch 固定为 64：

- `Stage1 + GRPO`：60 step，3,840 prompts，30,720 generated responses；
- `Cold Start + GRPO`：连续 100 step，6,400 prompts，51,200 generated responses；
- 每 5 step validation；`Stage1 + GRPO` 保护 P20/P40/P45/P50/P60；
- `Cold Start + GRPO` 保护 P20/P40/P60/P80/P100，其中 P40 只是同一 run 的 checkpoint；
- 第一轮先完整跑到 60/100，不用在线分数做 winner-only early stop；
- 随后的 saturation sweep 与 A/C/D0 使用同一 P80/P100/P120、20-step chunk 和 P180 hard-cap 规则。

step-matched 不是唯一公平口径。每个 run 还必须记录：

- 实际 GPU-hours 与 wall-clock；
- train/validation generated tokens；
- rollout、reward、update、validation 各阶段耗时；
- 峰值显存、checkpoint/storage；
- 相对 `S1-P0` 的 gain/GPU-hour。

## 8. Online 与 offline evaluation

### 8.1 Online

- `CS0`、`S1-P0` 和每 5-step checkpoint 使用同一任务现有 `n=3` validation；
- online 只用于健康监控、曲线和 checkpoint 候选选择；
- 同时记录 reward、advantage mean/std、zero-variance group ratio、clip fraction、KL diagnostic、
  entropy、grad norm、optimizer applied、response length、EOS/truncation 和格式指标。

### 8.2 共同冻结 offline pass@k

所有矩阵产出的最终 checkpoint 都进入同一轮评估，而不是只评估赢家：

- checkpoint：`CS0`、`S1-P0`、A/C/D0、`Stage1 + GRPO`、`Cold Start + GRPO` 的 P40/P60、peak 和 terminal；
- 第一阶段 `n=8`，用于检查 evaluator、样本对齐和方向；
- 正式阶段 `n=256`；
- $k \in \{1,2,4,8,16,32,64,128,256\}$；
- 固定 prompt manifest、tokenizer/evaluator/sandbox revision、解码参数、generation seed list 和
  per-prompt sample index；
- 保存 per-prompt/per-sample outputs；
- 数学报告 Math-7 各数据集和 macro；代码报告 HumanEval+、MBPP+、LiveCodeBench；
- 同一 frozen evaluation 内做 prompt-level paired bootstrap；多 training seed 单独估计优化方差。

代码 `n=256` 的 sandbox 执行成本可能高于模型生成成本。正式全量前必须用 `n=8` 实测
responses/s、执行吞吐和失败率，再由实测值生成资源 receipt；不能直接按数学任务线性外推。

## 9. 多 seed 计划

按以下顺序控制快速验证成本：

1. seed 1：数学/代码同时完成 `Stage1 + GRPO` 与连续 `Cold Start + GRPO`，GRPO LR 固定 `5e-7`；
2. 共同冻结 `n=8`，检查与 A/C/D0 的方向；
3. seed 2：至少复现 `A/C/D0/Stage1 + GRPO`；`Cold Start + GRPO` 若 seed 1 显示 Stage1 起点效应，也必须复现；
4. 最终目标 3 个 training seeds，再报告 seed-level 或 hierarchical interval；
5. 不允许只给 seed 1 的赢家补 seed，避免 winner-selection bias。

## 10. 入口和美团可迁移要求

实现复用已有 GRPO trainer/common launcher，并新增任务级薄包装，不复制 trainer 逻辑。当前本地
入口已经准备：`standard_grpo/run_{math,code}_{stage1,cold_start}_grpo.sh`；共同入口显式冻结
KL、thinking、rollout-IS、batch/group 和数据顺序。`standard_grpo/meituan/` 与
`platform/hope_standard_grpo/` 提供 credential-free AFO 薄包装；真实 endpoint、queue、image 与
不可变输入 manifest 必须由每次 submission 注入，不写入 Git。
每个任务均提供四层入口：

1. `run_*.sh`：本地默认路径全部可由环境变量覆盖；
2. `meituan/env.sh`：只放 dolphinfs/model/data/output 等平台路径；
3. `meituan/jupyter.sh`：校验输入、绑定 revision/receipt 并启动同一个 `run_*.sh`；
4. `meituan/run.hope`：AFO/Hope dispatcher。

还必须满足：

- 同一个实验脚本可在本地和美团运行，不维护 host-specific branch；
- `ROOT`、model、dataset、checkpoint、log、W&B 路径均可覆盖；
- Meituan worker 默认 `WANDB_MODE=offline`，不得自动 `wandb sync`；
- 启动前绑定 Git commit、recipe gitlink、Docker image、模型/数据/evaluator SHA-256；
- 新增或实际使用的 launcher 更新 `docs/joint_training/guides/training_script_index.md`；
- CPU dry-run、shell syntax、reward smoke、1-step GPU smoke 和 admission 全部通过后才允许正式训练。
- resolved-config canary 必须拒绝 canonical GRPO 中非 `null` 的 `rollout_is`。

## 11. 本机资源初估

2026-08-11 设计时本机为 8 × L40S 46 GB，查询时 8 卡均空闲。已有实测参考：

- Code A 单模型 60 step、含每 5-step Code-3 validation：约 18 小时；
- Math joint C 60 step：约 9.3 小时；该值不能直接当作 single-model GRPO 精确耗时。

因此第一轮先按以下区间预留，不作为调度承诺：

| Run | 数学 8×L40S | 代码 8×L40S |
| --- | ---: | ---: |
| `Stage1 + GRPO` 60 step | 约 6–10 小时 | 约 18–24 小时 |
| `Cold Start + GRPO` 100 step | 约 10–17 小时 | 约 30–40 小时 |

每个任务两条 pipeline（不是两条 LR）的 seed-1 总量约为数学 16–27 小时、代码 48–64 小时的
8-GPU wall-clock；串行总计约 64–91 小时。实际排期必须以 5-step throughput probe 重估。若有两台同规格机器，
优先按任务拆分并行；不要在同一 8-GPU 节点上并发两个正式 run。

## 12. 第一阶段完成标准

只有以下证据齐全，才允许对“WDL 是否优于标准 RLVR”下结论：

1. 数学和代码 `Stage1 + GRPO` / `Cold Start + GRPO` 均完成并通过 training release gate；
2. 所有 run 的真实 resolved config、数据顺序、模型 identity 和 cost receipt 齐全；
3. A/C/D0/GRPO 至少完成共同冻结 `n=8`，正式结论再使用 `n=256`；
4. 至少第二 training seed 复现核心方向；
5. 同时报告 `C-D0`、`C-A`、`C-(Stage1+GRPO)`、`(Stage1+GRPO)-A`、`(Cold Start+GRPO)-(Stage1+GRPO)`；
6. 不把单个 P60 点、单个 seed 或公共论文 headline score 写成最终竞争性结论。
