# Qwen3-1.7B 数学与代码 Standard RLVR / GRPO 对比实验方案

- 文档职责：冻结 Standard On-Policy SFT、WDL 与 canonical GRPO 的对比问题、训练预算、公平性和评测协议
- 当前状态：2026-08-18 scorer 审计撤回旧 Math GRPO 的正式公平 baseline 资格；Math 三条 pipeline 必须从原始初始化用 strict scorer 重训；Code Job 38 因 canonical 配置漂移也需重训，Code Jobs 43/63 保留为独立诊断/有效 arm
- 当前修订：2026-08-18
- 目标硬件：本地 8 × NVIDIA L40S 46 GB；训练入口必须同时支持美团 AFO
- 关联方案：[`qwen3_1p7b_math_stage123.md`](qwen3_1p7b_math_stage123.md)、[`qwen3_1p7b_code_acd0_p60_beta0.md`](qwen3_1p7b_code_acd0_p60_beta0.md)

本文中的 RLVR 是研究范式，canonical GRPO 是本轮采用的标准 RLVR 算法。仓库中没有名为
`ILVR` 的独立算法实现；若口头讨论中的 “ILVR” 指的就是 RLVR，则不把两者拆成两个实验。

### 2026-08-18 strict-scorer 重训执行合同

本节覆盖下文所有历史扩展预算/resume 叙述，是当前可执行合同：

- primary LR 为 `1e-6`；`5e-7` 只作为预注册 sensitivity arm；
- Math 使用 `recipe/joint_training/custom_reward_function_latex_verify.py`，Code 使用
  `official_aligned_reward.py`；两者的 SHA-256 进入 admission receipt；
- fresh retrain 固定 `resume_mode=disable`；复现 A/C/D0 的实际 resolved randomness：
  actor/FSDP 与 actor data-loader seed `42`、vLLM rollout base seed `0`、Math data seed
  `20260719`（Code 为 `20260706`）；PPO epoch `1`、
  prompt batch `64`、group `N=8`、clip `0.2/0.2`、actor KL `0.001`；
- formal launch 必须核对 clean root/recipe commits、recipe gitlink、image digest、init model、
  train parquet 和 scorer hash，并在 repo 外写唯一 admission receipt；
- `C-P60 -> GRPO` 从原始 C-P60 Model2 出发，使用和 C 相同的 post-Stage1
  `stage1_control_stage2_then_stage3.parquet`，由一个 fresh process 直接跑到 GRPO local P100；
  禁止 P98 停止后再 resume P100，也禁止使用完整 Stage1->2->3 cold-start parquet。

旧 Math GRPO 和旧 C-P60->GRPO 只保留为宽松 reward 下的 diagnostic evidence，不能进入正式
fair-baseline 结论。

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
4. **WDL 是否又快又强**：分别在 step/prompt、实测 GPU-hours/generated tokens 和训练饱和点三个
   口径下比较 C 与 GRPO，区分短程样本效率、等算力效果和最终能力上限。

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
- 原始 GRPO 的 sequence-level aggregation：`loss_agg_mode=seq-mean-token-mean`；
- group size / rollout `N=8`；
- `norm_adv_by_std_in_grpo=true`；
- symmetric PPO clip：low/high 均为 `0.2`；
- actor PPO epochs `1`；
- actor gradient clip `1.0`；
- `algorithm.use_kl_in_reward=false`，避免把 KL 混入 reward/advantage；
- actor loss 中启用固定 reference-policy KL：`use_kl_loss=true`、`kl_loss_coef=0.001`、
  `kl_loss_type=low_var_kl`；reference 是每条 GRPO run 的初始化 checkpoint；
- no WDL loss、no joint model、no weak logits；
- no rollout IS correction；
- Model2/single-model rollout 与更新。

`N=8` 既符合现有 A/C/D0 的 rollout budget，也与本仓库 canonical GRPO 历史入口一致，因此是
本轮唯一 group-size 主设置。group-size ablation 后置，不与第一轮方法有效性验证混跑。

这里必须区分三个 batch 概念：每 step 是 `64` 个 prompt；每 prompt 生成 `N=8` 个 response，
所以 reward/policy batch 是 `512` 条 trajectory。VERL 的 `ppo_mini_batch_size` 外部配置单位是
prompt group，因此固定为 `64`；worker 内部乘以 `N=8` 后对应完整的 `512` 条 trajectory，
每个 outer step 执行一次 actor optimizer update。预算比较以 prompt、generated response、token 和
GPU-hours 同时记录，不能只写一个含混的 “batch size”。

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

两项数据 receipt 都已验证相同 raw 结构：stage1 `2,560` 行、stage2 `1,280` 行、stage3
`2,560` 行。但 VERL 训练入口会先执行 `filter_overlong_prompts=True`，再用
`drop_last=True` 构造 dataloader；因此训练预算必须按实际 dataloader batches 记录，而不能只按
raw parquet 行数推断。当前 Math frozen tokenizer/template 下：

- post-Stage1 `stage2 -> stage3` raw `3,840` 行过滤为 `3,792` 行，每个 dataloader epoch 是
  `59` 个 batch，尾部 `16` 条被 `drop_last` 丢弃；
- cold-start `stage1 -> stage2 -> stage3` raw `6,400` 行过滤为 `6,324` 行，每个 dataloader
  epoch 是 `98` 个 batch，尾部 `52` 条被 `drop_last` 丢弃。

第一轮 `Stage1 + GRPO` 仍以 hard cap local P60 停止，`Cold Start + GRPO` 仍以 hard cap
P100 停止；两者都需要 `total_epochs=2` 才覆盖 VERL 实际 dataloader。超过一个 epoch 的扩展训练
必须继续使用同一冻结行序：每个完整 dataloader epoch 结束后才从 epoch 起点继续。admission
receipt 必须同时记录 `raw_rows/raw_steps_per_epoch` 与
`effective_rows/steps_per_epoch/full_epochs/trailing_steps`，并要求 `total_epochs` 恰好等于覆盖
目标 step 所需的实际 dataloader epoch 数；禁止 shuffle、跳行或以 resume 改变边界。

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

第一轮已覆盖 `5e-7` 和 `1e-6`，且部分数学 run 在 `1e-6` 下更好。第二阶段的 learning-rate
sensitivity 只保留这两个 LR，不新增第三档：`5e-7 / 1e-6`。任务、pipeline、初始化、seed、数据
顺序和其他超参完全一致；两个 LR 都报告完整曲线，不能只挑最好结果。

必须区分两种执行方式：

- **同 LR 延长训练**：从现有 terminal checkpoint resume，保留 optimizer、scheduler 和 RNG state，
  只把总训练步数延长到第二 epoch；
- 两个 LR 各自只能 resume 自己的 terminal checkpoint；不得在 resume 时互换 LR。

所有新 GRPO run 固定 `lr_warmup_steps=0`。这里的依据不是“已有 cold start 所以理论上永远不需要
warmup”，而是本轮要测 canonical constant-LR baseline，并避免第二 epoch resume 时重新 warmup。

## 7. 训练 budget 与 checkpoint 规则

第一轮的原始预算如下，因为 batch 固定为 64。这里同时记录 GRPO local step 和包含既有 40-step
Stage1 的 pipeline effective step，避免把两种坐标混在一起：

- `Stage1 + GRPO`：40-step 单模型 Stage1 + 60-step GRPO；GRPO local terminal 是 P60，pipeline
  effective terminal 是 P100；GRPO 部分消费 3,840 prompts、生成 30,720 responses；
- `Cold Start + GRPO`：连续 100 step，6,400 prompts，51,200 generated responses；
- 每 5 step validation；`Stage1 + GRPO` 只显式保护 local P20（Stage2 -> Stage3 边界），terminal
  local P60 由 latest 规则保留；
- `Cold Start + GRPO` 只显式保护 P40/P60（Stage1 -> Stage2、Stage2 -> Stage3 边界），terminal
  P100 由 latest 规则保留；不再保留无语义边界的 Stage1+GRPO local P40 或 Cold Start P80；
- checkpoint retention 固定为“阶段关键点 + best/latest”：latest 保留 optimizer 以支持连续
  saturation sweep，较早的 protected/best checkpoint 只保留 model weights；best/latest 与关键点
  重合时去重。不得默认保留每 5 step 的全部 full checkpoint；
- 第一轮先完整跑到 60/100，不用在线分数做 winner-only early stop；
- 数学 `Cold Start + GRPO`：从 P100 原 optimizer resume 到 P200；完整 `stage1 -> stage2 -> stage3`
  6,400-prompt 数据集按相同 seed、`shuffle=False` 和相同顺序再跑 100 step；第二 epoch 显式保护
  source P100、Stage1 end P140、Stage2 end P160，terminal P200 由 latest 保留；
- 数学 `Stage1 + GRPO`：从现有 GRPO local P60 / pipeline effective P100 checkpoint resume，保留
  GRPO optimizer，随后用相同的完整 6,400-prompt `stage1 -> stage2 -> stage3` 数据集再跑 100 step。
  终点是 GRPO local P160 / pipeline effective P200；显式保护 source local P60、第二轮 Stage1 end
  local P100、Stage2 end local P120，terminal local P160 由 latest 保留；
- 因而本轮“两 epoch”按完整 100-step pipeline 数据预算定义：两条 pipeline 都从 effective P100
  延长到 effective P200。`Stage1 + GRPO` 的第二轮全部使用 GRPO，不重跑最初 40-step SFT；
- 第二 epoch 之后，若曲线仍持续上升且没有出现 peak/回落，再以 20-step chunk 延长到预注册 hard cap；
  不得因为某个中间点有利于任一方法而临时停止。

step-matched 不是唯一公平口径。每个 run 还必须记录：

- 实际 GPU-hours 与 wall-clock；
- train/validation generated tokens；
- rollout、reward、update、validation 各阶段耗时；
- 峰值显存、checkpoint/storage；
- 相对 `S1-P0` 的 gain/GPU-hour。

### 7.1 与双模型 WDL 的 budget 对齐

“WDL 更新两个模型，所以 GRPO 直接乘二 step”只能作为粗略的 model-update proxy，不能作为最终
算力结论：GRPO 还包含 rollout、reward、actor backward 和 frozen reference forward；WDL 则包含
双模型 rollout/forward/backward。正式比较同时报告三种横轴：

1. **第一轮 pipeline matched**：C 的 GRPO/WDL-local P60（含共同 Stage1 时 effective P100）对
   `Stage1 + GRPO` local P60 / effective P100；回答相同 pipeline 数据预算下谁学得更快；
2. **扩展预算比较**：C effective P100 对 GRPO effective P200；这是本轮用于检验 GRPO 是否只是
   收敛更慢的预注册扩展点，但明确标注它不是精确 FLOPs 对齐；
3. **实测 compute matched**：根据 GPU-hours、generated tokens、训练/rollout/validation 分项耗时和
   profiler receipt，在 GRPO 曲线上插值到与 C 相同预算；这是主要的等算力比较。

最终还要报告各自 saturation peak。若 GRPO 在 effective P200 追平 C，只支持“WDL 更快/更省样本”；
若 GRPO 饱和后仍低于 C，才支持“WDL 又快又强”；若 GRPO 后期超过 C，则结论是 WDL 短程更快、
GRPO 上限更高。

### 7.2 Math 第二 epoch 历史诊断结果（2026-08-14；2026-08-18 撤回正式比较资格）

2026-08-18 scorer 审计确认，本节所有 Math GRPO 轨迹使用了旧版
`recipe/on_policy_wdl_sft/custom_reward_function_latex_verify.py`。该 scorer 允许 `<answer>` 外的
正确 `\boxed{}` 获得正 reward，与 A/C/D0 冻结的格式合同不一致。以下数字只保留为旧 reward
条件下的诊断结果，不能再支持 canonical GRPO 与 A/C/D0 的正式公平比较。重训配置审计见
[`../../reports/qwen3_1p7b_grpo_retrain_config_review_20260818.md`](../../reports/qwen3_1p7b_grpo_retrain_config_review_20260818.md)。

以下是在线 `Math-7 mean@3` 的 source-backed 运行快照。由于四个 GRPO run 的 deterministic
release gate 目前仍为 `pending -> BLOCKED`，本表只作为本地诊断与方案结果记录，不代表 DB/W&B
正式发布。

| Arm | Slurm Job | 实际终点 / effective 口径 | 最新验证 | 峰值 | 当前状态 |
| --- | ---: | --- | ---: | ---: | --- |
| Cold Start + GRPO，LR `1e-6` | 68 | actual P198 / effective ≈P200 | P195：68.54% | P160：68.82% | 进程正常退出；latest full checkpoint P195；gate BLOCKED |
| Stage1 + GRPO，LR `1e-6` | 72 | local P160 / effective P200 | terminal P160：68.37% | local P155 / effective P195：68.90% | terminal artifact 完整；gate BLOCKED |
| Cold Start + GRPO，LR `5e-7` | 73 | actual P198 / effective ≈P200 | P195：50.70% | P195：50.70% | 进程正常退出；latest full checkpoint P195；gate BLOCKED |
| Stage1 + GRPO，LR `5e-7` | 74 | receipt snapshot local P99 / effective P139 | P95 / effective P135：46.95% | P90 / effective P130：47.06% | 正在健康运行；8 卡有持续负载；gate BLOCKED |

Job 68/73 与 Job 72 使用的是同一份原始 6,400-row prompt 序列；差异不是 prompt 不一致。
`max_prompt_length=500` 过滤后实际保留 6,324 行，在 batch 64、drop-last 语义下每个 dataloader
epoch 只有 98 个 update。resume checkpoint 又没有可恢复的 dataloader cursor，trainer 使用
`current_epoch = global_steps // len(train_dataloader)` 反推 epoch：Cold Start 从 P100 恢复时得到
`100 // 98 = 1`，在 `total_epochs=2` 下只剩一个 98-step epoch，因此停在 P198；Stage1 从 local
P60 恢复时得到 `60 // 98 = 0`，循环仍允许两个 epoch，最终由 hard step cap 正常截在 local P160。
这是 resume/epoch bookkeeping 缺陷，不是数据内容或 scorer 差异。

在 compute/budget 分析中，P198 与目标 P200 只差 2 个 update（1%），可以预注册地记作
“effective ≈P200”近似点；但 artifact 事实仍必须写 actual terminal P198，且由于没有 P198 validation，
最终指标使用 P195。不得把 checkpoint、latest validated step 或 release-gate 终态改写为 P200。

以下是历史诊断现象，不是正式方法结论：

1. LR `1e-6` 的两条 GRPO pipeline 在 effective 约 P195–P200 收敛到相近峰值 68.82%–68.90%，
   Stage1 初始化对最终峰值的差异只有约 0.08 pp；
2. WDL C 的 P60 endpoint 为 70.80%、peak 为 71.16%，分别比 Job 72 的 terminal/peak 高
   2.43/2.26 pp；扩展到约两个 epoch 后，GRPO 仍未追平 C；
3. LR `5e-7` 的 Cold Start arm 到 P195 仅 50.70%，显示当前 canonical GRPO baseline 对 LR
   高度敏感，不能只报告一个 LR；
4. 这些旧轨迹不能支持“WDL 在公平 canonical GRPO 对比下更快或 endpoint 更高”；正式结论必须
   等 strict scorer 从原始初始化重训、共同冻结 offline evaluation 和多 seed；
5. effective step 只是粗略 model-update proxy。当前已新增
   `docs/joint_training/reports/data/qwen3_1p7b_math_grpo_wdl_budget_estimate.csv`
   作为第一版 GPU-hours、train/validation generated tokens、rollout/old/reference forward 与训练 FLOPs
   receipt；严格结论还需要 GRPO 更长 saturation、共同冻结 offline evaluation 和多 seed。

### 7.4 第一版 compute receipt（2026-08-14）

预算与曲线脚本：
`docs/joint_training/reports/scripts/plot_qwen3_1p7b_math_grpo_wdl_results.py`。输出：

- `docs/joint_training/reports/figures/qwen3_1p7b_math_grpo_internal_curve.{png,pdf}`；
- `docs/joint_training/reports/figures/qwen3_1p7b_math_grpo_vs_wdl_curve.{png,pdf}`；
- `docs/joint_training/reports/data/qwen3_1p7b_math_grpo_online_validation.csv`；
- `docs/joint_training/reports/data/qwen3_1p7b_math_grpo_wdl_budget_estimate.csv`。

估算采用 Qwen3-1.7B 的精确参数量 `1,720,574,976`，dense forward 为
`2 * params * tokens`、训练 forward/backward 为 `6 * params * tokens`。GRPO 计
single-model rollout、single-model old-log-prob、single-model
KL reference 和 single-model actor update。WDL C 配置为 `fusion_mode=mixture`、`freeze_model1=false`，
rollout 按两模型 fused forward 估；WDL D0 配置为 `fusion_mode=strong_scaled` 且 rollout source
为 model2-only，rollout 按单模型估；C/D0 的 old-log-prob 与训练都按 joint 双模型估。GPU-hours
来自 metrics timing：训练列为 `sum(timing_s/step) * 8`，在线评测列为
`sum(timing_s/testing) * 8`，总计为二者之和；不是 Slurm allocated wall-clock。FLOPs 是模型级
算法估算，不包含 attention 二次项、padding、重计算、通信和 kernel 利用率，不能冒充 profiler trace。

| Arm | peak / latest | train generated tokens | online val tokens target / all | 8×GPU-hours train / val / total | rollout / old / ref forward FLOPs | train FLOPs | total FLOPs train-pipeline / incl. val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WDL C P60 | 71.16 / 70.80 | 23.57M | 59.41M / 119.54M | 17.67 / 61.46 / 79.13 | 0.162 / 0.196 / 0 e18 | 0.589e18 | 0.947 / 1.358 e18 |
| WDL D0 P60 | 67.86 / 67.39 | 18.11M | 42.70M / 91.42M | 13.86 / 59.38 / 73.24 | 0.062 / 0.159 / 0 e18 | 0.476e18 | 0.697 / 1.011 e18 |
| GRPO Stage1 LR `1e-6` effective P200 | 68.90 / 68.37 | 103.21M | 178.64M / 178.64M | 83.15 / 49.21 / 132.36 | 0.355 / 0.401 / 0.401 e18 | 1.202e18 | 2.358 / 2.972 e18 |
| GRPO Cold Start LR `1e-6` ≈P200 | 68.82 / 68.54 | 175.74M | 248.29M / 248.29M | 131.88 / 67.06 / 198.94 | 0.605 / 0.661 / 0.661 e18 | 1.983e18 | 3.909 / 4.764 e18 |
| GRPO Cold Start LR `5e-7` ≈P200 | 50.70 / 50.70 | 94.69M | 127.64M / 127.64M | 85.60 / 51.51 / 137.11 | 0.326 / 0.382 / 0.382 e18 | 1.146e18 | 2.236 / 2.675 e18 |
| GRPO Stage1 LR `5e-7` running snapshot | 47.06 / 46.95 | 26.99M | 63.07M / 63.07M | 31.51 / 25.72 / 57.23 | 0.093 / 0.121 / 0.121 e18 | 0.363e18 | 0.698 / 0.915 e18 |

本表的成本与分数仍可用于估算重训资源和诊断旧 reward 轨迹，但不得再把它解释为 WDL C 相对
公平 canonical GRPO 的质量或效率优势。strict scorer 重训完成后，必须用新轨迹重算本表。

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
