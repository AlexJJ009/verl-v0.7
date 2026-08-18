# Qwen3-1.7B GRPO 重训前配置审计（2026-08-18）

## 结论

Math GRPO 必须从各自原始初始化重训。现有 Math Cold Start、Stage1 和 C-P60 -> GRPO
都使用了旧版 `recipe/on_policy_wdl_sft/custom_reward_function_latex_verify.py`。这个 scorer
会在整个 response 中查找 `\boxed{}`，没有执行 A/C/D0 已冻结的
`<think>...</think><answer>...</answer>` 格式门禁。严格 scorer 的 fail-fast canary 已证明：

- 完整且正确的结构化回答为 `+1`；
- 缺 `<answer>`、缺 `<think>`、标签重复或乱序均为 `-1`；
- 截断或无 EOS 为 `-1`。

Code GRPO 不能因 Math scorer 问题整批判废。Code Jobs 38/43/63 实际使用的
`official_aligned_reward.py / compute_score_code_official_aligned` 与 Code A/C/D0 相同。
不过 Job 38 还存在独立的 canonical GRPO 配置漂移，因此需要重训；Jobs 43/63 的训练
配置没有同类漂移。

## 已完成 run 的判定

| 任务 / run | scorer | 其他关键偏差 | 正式用途判定 | 动作 |
| --- | --- | --- | --- | --- |
| Math Cold Start + GRPO，Jobs 37/41 及其 P200 continuation | 旧 Math scorer | Job 37 还使用 `token-mean`、`grad_clip=500`; Job 41 及 continuation 的 GRPO 核心配置正确 | 旧 reward 条件下诊断结果 | 从原 Cold Start step20 重训，不从任何旧 GRPO checkpoint resume |
| Math Stage1 + GRPO，Jobs 39/40 及其 continuation | 旧 Math scorer | canonical GRPO 核心配置正确 | 旧 reward 条件下诊断结果 | 从同一 S1-P0 Model2 重训 |
| Math C-P60 -> GRPO，Jobs 84/89 | 旧 Math scorer | canonical GRPO 核心配置正确 | 旧 reward 条件下诊断结果 | 从原始 C-P60 Model2 重训，不从旧 GRPO P100 resume |
| Code Stage1 + GRPO，Job 38 | strict code scorer | `loss_agg_mode=token-mean`、`grad_clip=500`，不符合冻结的 `seq-mean-token-mean` / `1.0` | 非 canonical 的诊断结果 | 从同一 Code S1-P0 Model2 重训 |
| Code Cold Start + GRPO，Job 43，LR `5e-7` | strict code scorer | 训练核心配置正确；但 best key 错指 Math-500，导致真实 P90 peak 未保留 | final/曲线可保留；缺 peak artifact | 不因 scorer 重训；若要补齐完整 checkpoint-selection 合同，再从 Cold Start 重跑 |
| Code Cold Start + GRPO，Job 63，LR `1e-6` | strict code scorer | 未发现核心配置漂移，best key 为 Code-3 | canonical single-seed 结果 | 保留；没有配置依据要求重训 |
| Math/Code A/C/D0 | 各任务 strict scorer | 方法本身的 loss/joint-model 差异是预注册 treatment | 正式内部对照仍有效 | 不重训 |
| Math Fixed-Model1 | strict Math scorer，非旧 GRPO 链路 | 无本问题 | 正式结果仍有效 | 不重训 |

Training release gate 的 `success_complete/ALLOWED` 只证明进程完成、checkpoint 与 metrics
齐全，不证明 scorer 与公平比较合同一致。因此旧 run 的 operational gate 不应伪改成“训练失败”；
应额外标成 `scientific_eligibility=diagnostic_only`，禁止用于正式公平 baseline。

## 必须完全一致的变量

以下项目在同一任务、同一对比问题内必须通过 SHA-256 或 resolved config 精确核对：

1. **初始化模型**：`C vs Stage1+GRPO` 必须使用同一 S1-P0 Model2；Cold arm 必须使用同一
   Cold Start step20；C-P60 -> GRPO 必须使用原始 C-P60 Model2。
2. **训练数据与顺序**：同一 parquet SHA-256、prompt/schema、过滤规则、`shuffle=False`、
   每阶段行数和消费顺序。Stage1 arm 为 post-Stage1 3,840 rows / 60 steps；Cold arm 为
   6,400 rows / 100 continuous steps。
3. **reward 合同**：Math 固定
   `recipe/joint_training/custom_reward_function_latex_verify.py` /
   `compute_score_latex_verify`；Code 固定 official-aligned scorer。记录 scorer 文件 SHA-256。
4. **rollout 合同**：prompt batch `64`、`N=8`、thinking enabled、训练采样
   `temperature=1.0, top_p=1.0, top_k=-1`；Math 长度 `500/4096`，Code `1024/8192`。
5. **canonical GRPO 合同**：`loss_mode=vanilla`、`seq-mean-token-mean`、
   `norm_adv_by_std_in_grpo=True`、clip `0.2/0.2`、PPO epoch `1`、prompt-group mini-batch `64`、
   `grad_clip=1.0`、`weight_decay=0.1`、warmup `0`、rollout IS `null`、reward KL disabled、
   actor reference KL `0.001/low_var_kl`。
6. **validation/selection**：相同数据、scorer、`n=3, T=0.2, top_p=0.95`、每 5 step validation/save，
   best key 分别为 Math-7 或 Code-3 macro mean@3；最后统一执行相同 manifest 的 n=256 offline eval。
7. **随机性与 provenance**：training seed、generation seed、Git commit、recipe gitlink、image digest、
   model/dataset/scorer identity 和硬件类型均写入 admission receipt。

## 允许不同但必须提前声明的变量

- A/C/D0 与 GRPO 的 loss、single/joint model、weak logits、rollout source、reference KL、
  grad clip 和 loss aggregation 属于方法定义，不应为了“参数看起来相同”强行统一；但同一方法的
  所有 arm 必须使用各自冻结的同一合同。
- Cold、Stage1 和 C-P60 三种初始化及其数据窗口是实验问题本身，不能混成一个总体分数。
- Math 与 Code 的最大长度和 evaluator 按任务合同不同。
- FSDP offload、vLLM memory utilization、worker 数、节点等资源参数可以因机器调整，只要不改变
  batch、样本、optimizer update 或解码语义；必须记录真实 GPU-hours/tokens。
- LR `5e-7` 与 `1e-6` 只能作为预注册 sensitivity arms。建议重训先完成 `1e-6` primary，随后
  若要声称“tuned canonical GRPO”，必须补齐两条 pipeline 的 `5e-7`，不能只给赢家补跑。

## 重训批次与保留 checkpoint

| 优先级 | 新 run | 原始初始化 | 有效训练步数 | 必保留 checkpoint |
| ---: | --- | --- | ---: | --- |
| 1 | Math Stage1 -> GRPO，LR `1e-6` | 同一 S1-P0 Model2 | local 60 / effective 100 | local P20、P60(final)，另保留真实 best |
| 2 | Math Cold Start -> GRPO，LR `1e-6` | Cold Start step20 | continuous 100 | P40、P60、P100(final)，另保留真实 best |
| 3 | Math C-P60 -> GRPO，LR `1e-6` | 原始 C-P60 Model2 | GRPO local 100 | P20、P40、P60、P100(final)，另保留真实 best |
| 4 | Code Stage1 -> GRPO，LR `5e-7` | 同一 Code S1-P0 Model2 | local 60 / effective 100 | local P20、P60(final)，另保留真实 best |
| 5 | 统一 offline evaluation | 上述新 checkpoint | n=256 | raw generations、scorer receipts、merged summaries |

若继续第二 epoch，必须从**新 strict-scorer run** 的 terminal checkpoint 连同 optimizer、scheduler
和 RNG state resume；旧 GRPO checkpoint 不得进入新链路。

## 启动门禁

当前入口已增加三层 fail-fast：

1. Math reward canary 会拒绝旧 scorer，并覆盖正确结构、缺/乱序/重复标签、截断/无 EOS；
2. canonical GRPO allowlist 会拒绝 `token-mean`、`grad_clip=500`、mini-batch `512`、
   rollout IS 非 `null`、PPO epoch 非 `1`、`resume_mode` 非 `disable` 和 seed 漂移；
3. admission checker 核对 clean root/recipe commits、recipe gitlink、image digest、init model、
   train parquet 与 scorer SHA-256，并在 repo 外写不可复用的 receipt。

`C-P60 -> GRPO` 已改成单进程 fresh 100-step：使用与 C 相同的 post-Stage1
`stage1_control_stage2_then_stage3.parquet`，`TOTAL_EPOCHS=2` 只提供足够的 dataloader loop
budget，hard cap 在 P100 停止。旧 P98 -> P100 两次调用/resume 策略已删除。

正式 Slurm submission 还必须保存最终 Hydra resolved config 并与 allowlist 逐项比较；只检查 shell
默认值不够，因为 submission 尾部 override 仍可能改变实际训练配置。

## 旧结果的允许表述

允许：在旧版宽松 Math reward 下训练得到的 GRPO checkpoints 上，统一严格 offline evaluation
显示 C 更强；这些结果用于诊断 scorer 与训练轨迹的影响。

禁止：把旧 Math GRPO 分数、Job 38 Code Stage1 分数或 C-P60 -> 旧 GRPO 的提升，写成
与 A/C/D0 完全公平的 canonical GRPO 结论。
