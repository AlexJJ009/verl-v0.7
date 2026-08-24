# Qwen3-1.7B On-Policy SFT Baseline Extension：文献调查与实验方案

> 2026-08-23 current evidence：2026-08-18 之前的历史 Math GRPO 比较仍为
> diagnostic-only；strict-scorer aligned retrain 与共同 `n=256` 已完成并单独汇报。

- 文档职责：给当前 1.7B Math-first WDL-SFT 工作补充外部 baseline、机制 ablation 和后续实验队列。
- 创建日期：2026-08-15；实验状态更新于 2026-08-23
- 状态：fixed-M1 P60 与 common `n=256` 已完成；strict-GRPO retrain/offline 已完成；DynPerm partial matrix 已完成。
- 关联方案：
  - `qwen3_1p7b_math_stage123.md`
  - `qwen3_1p7b_code_acd0_p60_beta0.md`
  - `qwen3_1p7b_standard_rlvr_grpo_matrix.md`
  - `qwen3_1p7b_offline_passk_diversity_eval.md`
- 关联飞书文档：`https://ocnwds5io8yp.feishu.cn/docx/NEIvdnwU0o0vszxi2wycfcTHnjd`

## 0. 结论

当前应把 baseline 分成三层，而不是只找“另一个 on-policy SFT 名字”的方法。

1. **主外部 baseline 仍是 canonical GRPO/RLVR**。strict-scorer aligned common `n=256` 已确认：Cold P200 和 Stage1 effective P200 都未追平 effective P100 的 WDL C；C+GRPO 只比 C 增加 `0.405 pp` pass@1，未带来 high-k 增益。多 training seed 仍未完成。
2. **Standard On-Policy SFT continuation A 是最重要的内部 practical baseline**。它几乎就是“当前模型采样、正确样本过滤、正样本 SFT”，直接回答“WDL 是否比同预算 on-policy selected SFT 更有用”。
3. **GFT 可以做外部 published baseline，但不是首轮最高性价比 baseline**。它已经发表于 ACL 2026 Findings，且有代码；但它混合 expert / teacher distillation / self-sampled responses，并使用 group advantage 与 DCR，和我们的方法在计算结构、teacher 依赖和目标问题上不完全对齐。
4. **《On-Policy Supervised Fine-Tuning for Efficient Reasoning》截至本调查时只能按 arXiv preprint 处理**。它很适合做 related work / background；实验 A 已覆盖 correctness-filtered selected-CE 这一核心学习形式，但没有复现其 length-efficiency 目标、rollout 数和 aggregation，是否还需 GPU sensitivity 应先由 loss/gradient 等价审计决定。
5. **fixed-Model1 已成为 conclusion-changing arm**。它与 C 的 pass@1 近似相同、明显高于 D0，但 high-k coverage 低于 C；静态 guidance 是主要 pass@1 来源，joint adaptation 的剩余作用仍需多 seed 和 geometry controls 判断。
6. **Dynamic Perturbation 不能回答“是否可以去掉 WM1”**。当前 keyed cyclic DynPerm 保留 WM1 forward/backward；中剂量显著破坏训练，说明真实 assignment 或其绑定 geometry 重要，但还不能归因为 token semantics。下一步应是 Align/true-Random/Anti，而不是直接宣称 semantic dark knowledge。

### 0.1 下一轮两个实验问题必须分开

`lambda` sweep 与 permutation geometry 是两套独立实验，首轮不能交叉成一个大网格。

当前 Qwen3-1.7B Math 仍没有 `lambda=0.4/0.5` 的正式结果或 checkpoint；C-only 的
launch-ready wrapper、manifest 和静态测试已经补齐，并于 2026-08-23 以 Jobs 232/233
提交正式 P60 搜参。仓库中其他旧 `lambda=0.5` 属于 4B 或 Code/KodCode，不能复用。A 是 direct
Model2 On-Policy SFT，不含 fusion `lambda`，因此正确矩阵不是八条新训练，而是共享 A anchor，
并在每个 `lambda` 下新增 C、D0、fixed-M1 三条，共六条新 run：

| fusion lambda | C | D0 scale control | fixed-M1 |
| --- | --- | --- | --- |
| 0.5 | `0.5z1 + 0.5z2` | `0.5z2` | `0.5z1 + 0.5z2`, freeze M1 |
| 0.4 | `0.6z1 + 0.4z2` | `0.4z2` | `0.6z1 + 0.4z2`, freeze M1 |

若要求同一 code revision，只需额外重跑一次 A，仍是七个 unique arms。所有新 run 必须继承
causal-P60 的 Model1/Model2 hashes、3,840-row ordered shard、P60、strict scorer、LR、batch/N、
seed `42`、rollout seed `0`、data seed `20260719`、`shuffle=false`，并使用 lambda-tagged
run/cache/artifact identity。Jobs 232/233 只回答 C 的 primary lambda reconnaissance；D0 与
fixed-M1 尚未实现或提交，因此完整的 scale/static-guidance 因果矩阵仍未完成，不能仅凭 C-only
结果归因 weak guidance、joint adaptation 或 logit scale。

geometry pilot 则固定 `lambda=0.8`，先在 fixed-M1 上比较 `Real/AlignSort/true
RandomPerm/AntiAlignSort` 的 P20/P30 trajectory 与 affinity、fused target probability、
gradient telemetry。只有 validity 与预注册方向成立才延长 P60。这样不会把“增加 weak 权重”
与“改变 non-target assignment”混成一个无法解释的 treatment。

## 1. 文献调查

### 1.1 On-Policy Supervised Fine-Tuning for Efficient Reasoning

一手来源：

- arXiv: `https://arxiv.org/abs/2602.13407`
- 官方代码：`https://github.com/EIT-NLP/On-Policy-SFT`

截至 2026-08-16，本调查未在 arXiv 页面、官方 repo、作者主页、OpenReview 或会议 proceedings 中找到该论文已接收会议或期刊的信息。arXiv 页面显示 v1 提交于 2026-02-13；官方 repo 的 News 只列出 2026-02-13 arXiv preprint 和 2026-03-12 code release，BibTeX 也标为 `@misc`。GitHub 页面快照为 7 stars、0 forks；这只能说明公开采用度尚低，不能代替论文质量判断。该工作太新，长期影响力目前无法可靠评估。

方法本身可概括为：

1. 当前 policy 生成 $G$ 个 responses；
2. 用 verifier 选择正确 responses，并由最大生成长度排除过长/截断输出；released filter 本身只检查正 reward，没有额外的显式“越短越优”阈值；
3. 只对过滤后的 response 做 teacher-forced maximum-likelihood；
4. 去掉 GRPO 的 KL、group-wise normalization 和 policy-gradient surrogate；
5. 重点目标是 efficient reasoning，即保留/提升准确率，同时显著压缩 CoT 长度。

它和我们的实验 A 很接近，但并不完全一致。

| 维度 | 该 paper | 我们的实验 A |
| --- | --- | --- |
| 模型结构 | single model | single Model2 continuation |
| 采样策略 | 当前 policy rollout；论文强调 $T=1$ | Model2-only rollout，训练 $T=1$ |
| 样本选择 | `score > 0`；全错 prompt 丢弃，最大长度控制截断 | correctness + format/EOS/非截断；全错 prompt 不贡献正向梯度 |
| loss | selected responses 上的正样本 CE/SFT | selected responses 上的正样本 CE/SFT；`beta=0` 时错误样本无负向梯度 |
| 主要配置 | released example：prompt batch 32、$N=32$、max response 3500、LR $10^{-6}$、token-mean | prompt batch 64、$N=8$、max response 4096、LR $10^{-6}$、`seq-mean-token-sum` |
| 研究问题 | efficient reasoning Pareto frontier | WDL 是否优于同预算 selected-SFT continuation |
| 能否当“我们基于它做” | 不建议 | 我们的 A 是同类方法实例，但 WDL 的 weak-logit treatment 是额外机制 |

因此 A 不只是模糊的 “OPSFT-like”：它覆盖了 correctness-filtered On-Policy SFT 的核心 objective，但 aggregation、batch、rollout 数、format 合同和长度目标不同。论文口径建议称为 **Correctness-filtered On-Policy SFT (our implementation)**，并说明与 Zhao et al. 的配置差异。除非能证明时间与代码依赖，不写“我们基于 Zhao et al.”；WDL 的 weak-logit fused objective 也不是参数微调。

在消耗新 GPU 前，先对同一 frozen batch 做 objective / selected-token / gradient 等价审计：若差异仅为可解释的全局 scalar 或聚合方式，则现有 A 已是主 baseline；只有 aggregation 改变优化行为时才补 `A-OPSFT-loss`，只有论文要主张长度效率时才补 paper-aligned concise/length arm。

### 1.2 这个方向不是 2026 才出现

“当前 policy 采样、用 reward/verifier 筛选、再做 SFT”这个骨架早已有先例：

- STaR, 2022：生成 rationale，保留能得到正确答案的 rationale，再 fine-tune，迭代自举 reasoning。
- RAFT, 2023：从当前/候选生成中用 reward model 排序选高质量样本，再 fine-tune；官方实现也把它称为 iterative best-of-n / rejection sampling fine-tuning。
- ReST, 2023：由当前 policy 生成数据，再用 offline RL / filtering 方式改进 policy，强调数据可复用和 sample efficiency。
- Llama 2, 2023：alignment pipeline 中使用 rejection sampling fine-tuning，sample 多个 outputs，经 reward model 选择后再 SFT。
- GKD / On-Policy Distillation, ICLR 2024：student 生成序列、teacher 在 student states 上提供 distributional feedback；它证明“on-policy”术语在 LLM fine-tuning 中更早出现，但属于 teacher KD，不是 hard selected-response CE。

更早的概念祖先是 DAgger (AISTATS 2011)：在 learner-induced state distribution 上向 expert 查询标签，解决 imitation learning 的 covariate shift；它不是 LLM 自生成答案后按 verifier 自训练。

所以 2026 OPSFT 的贡献更像是把这个老骨架放到 efficient reasoning / GRPO 简化视角下系统化，并给出长度控制实验；不是首次提出“on-policy selected SFT”。

### 1.3 Towards On-Policy SFT / DDT / IDFT

一手来源：

- arXiv: `https://arxiv.org/abs/2602.12222`
- 官方代码：`https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT`

这篇的核心不是“从当前 policy rollout 后过滤正样本再 SFT”，而是用 DDT 衡量数据和模型诱导分布的匹配，并提出：

- IDFT：loss-level 方法，提高 SFT 的 in-distribution / generalization 表现；
- Hinted Decoding：data-level 方法，把训练数据重新对齐到模型分布。

它对我们有两个用途：

1. related work：说明 on-policy / distribution alignment 是合理研究问题；
2. future work：用 continual learning / OOD 序列实验检验 WDL 是否也改善分布迁移和遗忘。

不建议把 IDFT 放进当前第一批 baseline。它主要解决 distribution mismatch，不直接对照 WDL 的 weak-logit mechanism；实现和解释成本都高。

### 1.4 GFT

一手来源：

- ACL Anthology: `https://aclanthology.org/2026.findings-acl.1444/`
- 官方代码：`https://github.com/ZJU-OmniAI/GFT`

GFT 已发表于 ACL 2026 Findings，正式题名是 **GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification**。ACL 页面给出 pages 28909-28922 和 DOI `10.18653/v1/2026.findings-acl.1444`。

它不是普通 SFT。作者把 SFT 解释为一种极稀疏 implicit reward 下的 policy-gradient special case，然后提出：

- Group Advantage Learning：构造 response group 并做 normalized contrastive supervision；
- Dynamic Coefficient Rectification：对 inverse-probability weight 做 adaptive bound；
- 数据组分包括 expert / teacher distillation / self-sampled responses。

这就是为什么它会涉及 teacher。论文中的 faithful $K=8$ group 由 1 条 expert response、3 条 Qwen2.5-Math-72B teacher distillation 和 4 条 student self-generated responses 组成；repo launcher 分别暴露 teacher/student count（示例配置可能与论文主表不同）。teacher 输出可以预生成，但计算账本仍必须计入 teacher generation / distillation 数据成本。

GFT 适合作为第二批 external baseline，原因是：

- 优点：已发表、有代码、和 SFT/RL 统一视角相关；
- 缺点：teacher/off-policy 数据让它不是纯 on-policy SFT baseline；计算预算与我们 A/C/D0/GRPO 很难自然对齐；
- 解释风险：如果 GFT 赢，可能来自 teacher data 或 group advantage；如果 GFT 输，也不直接支持 WDL mechanism。

推荐做法：若排期，在 Math 上做 faithful minimal reproduction，明确 report 为 “published teacher-assisted group fine-tuning baseline”。必须使用完整 expert reasoning trace，而不能把只有 final answer 的数据误当 expert response；不要把改成纯 student $N=8$ 的变体仍称为 faithful GFT，也不要把它列为最直接的 WDL ablation。

## 2. Baseline 优先级

### 2.1 第一批：已完成的核心 baseline

| Baseline / control | 作用 | 当前状态 | 下一步 |
| --- | --- | --- | --- |
| A: Standard On-Policy SFT continuation | practical baseline；同预算正样本 selected SFT | Math P60 + common n=256 complete | 第二 training seed |
| D0: matched-scale no-weak | weak-logit causal control | Math P60 + common n=256 complete | 第二 training seed；high-k trade-off 已显现 |
| Stage1 + GRPO | 主外部 RLVR baseline | strict-aligned effective P200 + common n=256 complete | 补 terminal release event 与第二 seed |
| Cold Start + GRPO | 检验 Stage1 是否必要 | strict-aligned P200 + common n=256 complete | 补 terminal release event 与第二 seed |

### 2.2 第二批：机制 ablation

| 实验 | 优先级 | 回答的问题 | 推荐范围 |
| --- | --- | --- | --- |
| C-fixed-M1-CS0 / C-fixed-M1-S1 | 已完成主 arm | fixed weak guidance 是否足够、是否依赖 Stage1 起点、还是必须 jointly update WM1 | Stage1 P60 + common n=256 已完成；CS0 保留为次要参照 |
| Dynamic Perturbation | 进行中 | real assignment 或其绑定 geometry 是否重要 | `rho=0/0.25/0.5` 双臂完成；`rho=1` running；之后做 geometry controls |
| Synthetic / cached surrogate weak logits | 中低 | WM1 是否可被别的结构替代 | 只有 DynPerm 结果提示 token identity 不重要时再做 |
| fusion-lambda grid | 新增高优先级 | weak coefficient 增大后 static guidance / co-adaptation / scale 如何变化 | 共享 A；每个 lambda 只跑 C/D0/fixed，先实现正式门禁 |

### 2.3 第三批：外部 published baselines

| 方法 | 是否建议做 | 理由 |
| --- | --- | --- |
| OPSFT paper reproduction | 先审计，默认不新增完整 run | A 已覆盖核心 objective；只为 loss aggregation 或长度效率补敏感性 arm |
| GFT | 可做第二批 | published + code；但 teacher/off-policy data 使公平预算复杂 |
| IDFT | 暂不做当前 baseline | 问题指向 distribution mismatch；更适合 future continual-learning 实验 |

## 3. fixed-Model1 实验设计

### 3.1 实验问题

在 C 中，WM1 的作用可能有两部分：

1. 初始 weak prior / guidance：$0.2z_1$ 给 fused objective 提供 token distribution bias；
2. adaptive weak update：训练中 WM1 也被 loss 更新，可能与 Model2 共同形成 trajectory-level effect。

`freeze_model1=True` 可以分离这两个因素。

### 3.2 Arm 定义

| Arm | 训练目标 | Model1 | Model2 | 主要比较 |
| --- | --- | --- | --- | --- |
| C-joint | $0.2z_1 + 0.8z_2$ | 更新 | 更新 | 现有核心方法 |
| C-fixed-M1-CS0 | $0.2z_1^{CS0} + 0.8z_2^{CS0}$ | frozen CS0 | 从 CS0 更新 | 固定 weak guidance；与 A 共享起点 |
| C-fixed-M1-S1 | $0.2z_1^{CS0} + 0.8z_2^{S1}$ | frozen CS0 | 从 Stage1 更新 | 固定 weak guidance；与 C/D0 共享起点 |
| D0 | $0.8z_2$ | 无梯度 / hash 不变 | 更新 | no-weak matched-scale control |
| A | selected SFT | 不存在 | 更新 | practical on-policy SFT |
| GRPO | canonical GRPO | 不存在 | 更新 | external RLVR baseline |

核心报告比较：

1. `C-fixed-M1-S1 - C-joint`：在同一 Stage1 起点下，WM1 adaptive update 是否有净收益；
2. `C-fixed-M1-S1 - D0`：在同一 Stage1 起点下，固定 weak logits 是否足以优于 no-weak control；
3. `C-fixed-M1-CS0 - A`：在同一 CS0 起点下，固定 weak guidance 是否比普通 selected SFT 更有 practical value；
4. `C-fixed-M1-CS0 - C-fixed-M1-S1`：Stage1 Model2 source 是否仍提供净收益；
5. 两个 fixed arms 分别与 matched GRPO endpoint 比较相对 RLVR baseline 的竞争力。

这里的 estimand 是 **fixed weak-guidance procedure 与 joint-WM1 procedure 的总差异**，不能把差值全部归因于“WM1 学到了更好的语义”。冻结还会改变 global gradient norm、clip 事件、optimizer state 和通信。为分离这一混杂，必须同步记录 Model2 pre-clip grad norm、post-clip update norm、global clip frequency，以及 joint arm 中 Model1/Model2 分支各自的 update norm。

### 3.3 执行合同

第一轮 Math P60 已完成。二者共享下列训练合同；唯一 treatment 是 `freeze_model1=true`，另一个预注册因素是 Model2 source（CS0 或 Stage1）：

- model family / size：Qwen3-1.7B；
- Model1 init：两组都使用现有 CS0 weak source，并保持 frozen；
- Model2 init：`C-fixed-M1-CS0` 使用同一 CS0 source，`C-fixed-M1-S1` 使用现有 C/D0 的 Stage1 Model2 source；
- data：同 `stage2 -> stage3` 3,840 prompts，`shuffle=False`；
- step：60；
- `fusion_lambda=0.8`；
- `fusion_mode=mixture`；
- `loss_mode=wdl_sft`；
- `beta=0.0`；
- rollout source：Model2-only；
- validation：Math-7 online `n=3` 每 5 step；
- validation views：每次分别加载并评估 Model1、Model2；原始 generations 分目录保存为 `model1/`、`model2/`，指标分别进入 `val-core/model1/...` 与 `val-core/model2/...`；
- final：共同冻结 n=8 confirmation，然后 n=256 official pass@k。

该 arm 仍需执行 WM1 forward，因此只能预期节省 WM1 backward、gradient communication 和 optimizer state/update；不能写成“去掉一半模型计算”或“无需 WM1”。

### 3.4 现有代码状态

代码现已具备：

- `QwenJointConfig` 有 `freeze_model1` 字段；
- `QwenJointForCausalLM.__init__` 在 `config.freeze_model1` 为真时将 `sub_models[0]` 参数设为 `requires_grad=False`；
- `prepare_joint_weights.py` 已支持 `--freeze_model1`，并把该字段写入 `config.json`；
- 单测已有 `test_freeze_model1` 和 `test_frozen_model1_no_gradients`；
- common launcher 用 `FREEZE_MODEL1=true` 向 joint preparation 传入 `--freeze_model1`，并对已有 cache 的 config fail closed；
- `run_math_qwen3_1p7b_wdl_fixed_m1_{cold_start,stage1}.sh` 提供两个独立 run/cache identity。

正式运行已经按 admission 记录 `freeze_model1=true`、Model1 param hash before/after、Model1 grad norm 为 zero/None、Model2 grad norm 非零并核对 source identities；Stage1 fixed arm 的 common `n=256` 结果已进入主结论。

## 4. Dynamic Perturbation 与 WM1 替代问题

本节保留 baseline-extension 阶段的证据边界和 surrogate 梯子。完整的理论分解、
Chernoff-affinity telemetry、同熵 directional controls、反向实验和分波执行合同见
[`qwen3_1p7b_wdl_mechanism_program.md`](qwen3_1p7b_wdl_mechanism_program.md)。

飞书 Dynamic Perturbation 文档定义的是 **step-resampled target-preserving non-target permutation**：

- 保留当前 weak distribution 的 entropy；
- 保留 target token probability；
- 保留完整 logit/probability value multiset；
- 打乱 non-target logits 和 token identity 的对应关系；
- 保留 weak/strong joint update 和双分支反向传播；
- 每个 optimizer step 重采样 permutation，避免 WM1 学会固定 inverse mapping。

因此它的证据边界是：

| 结果 | 支持的解释 | 不能推出 |
| --- | --- | --- |
| `DynPerm-0 > DynPerm-100` | token-specific weak assignment 对完整 joint-training procedure 有用 | 每个 hard negative 都有语义意义；WM1 本体不可替代 |
| `DynPerm-0 ≈ DynPerm-100` | token assignment 证据变弱，weak 的 distributional shape / optimization perturbation 可能足够 | generic sharpening 已被证明；可以去掉 WM1 |
| `DynPerm-100` 训练不稳 | 置换破坏了某种必要结构，或引入 stochasticity/optimization cost | 不能直接归因于 hard-negative 机制 |

DynPerm 已按显式预注册语义运行。由于 fixed-M1 接近 C-joint，正式矩阵直接交叉 joint/frozen 与 `rho={0,0.25,0.5,1}`，不再把 frozen 条件作为事后可选补充：

| WM1 更新条件 | `rho=0` | `rho=1` | 回答的问题 |
| --- | --- | --- | --- |
| joint | C 原方法 | joint DynPerm | token assignment 在 co-adaptive joint training 中是否重要 |
| frozen | fixed weak guidance | frozen DynPerm | 固定 weak distribution 的 token assignment 是否重要 |

如果目标是证明“不必一直保留 WM1”，需要在 DynPerm 之后加一组真正的 no-WM1 surrogate：

1. **calibrated entropy/spectrum-matched synthetic weak logits**：在独立 calibration pass 中，按 strong confidence / token position 拟合 weak target probability、entropy 和 centered spectrum；正式训练不跑 WM1。若 calibration 使用过 WM1，结论只能是“不需要 online WM1”，不能说从未需要 weak teacher；
2. **cached weak logits**：对固定训练 prefix 缓存 WM1 logits；局限是 on-policy 新 response prefix 会离开 cache，适合做局部 teacher-forced diagnostic，不适合完整 rollout；
3. **detached strong-logit perturbation**：用 `stopgrad(z2)` 构造同熵扰动，控制计算便宜，但容易把结论变成 strong self-regularization；
4. **uniform / Gaussian reference**：只作为粗 reference，不是 same-entropy joint-training control。

当前顺序：

1. fixed-M1 Stage1 P60 与 common `n=256` 已完成；
2. 完成 DynPerm `rho=1` terminal 与 common offline；
3. 在 `lambda=0.8`、fixed-M1 上做 Align/true-Random/Anti P20/P30 pilot；
4. 只有 geometry controls 表明低维统计足够时，才设计 no-online-WM1 surrogate。

## 5. Continual learning future experiment

IDFT 提到的 distribution / OOD / continual-learning 问题可以转化成 WDL 的 future experiment，但不进入当前第一批 baseline。

建议的 future queue：

1. 顺序训练 domains：logic reasoning -> math -> code -> medicine/science QA；至少增加 reverse 或 balanced order，避免把单一顺序效应误判为方法效应；
2. 每个 domain 用相同 prompt budget / rollout N / validation cadence；
3. 每个 stage 后评估所有已见和未见 domains；
4. 指标：
   - current-domain score；
   - previous-domain retention；
   - average forgetting: $F_i=\max_{t<k}S_i^t-S_i^k$；
   - backward transfer；
   - forward transfer；
   - format/EOS/truncation；
   - GPU-hours / generated tokens。
5. 第一版比较 A、C、D0、两个 C-fixed-M1 arms；GRPO/GFT 会显著放大预算，建议等主结论稳定后再做。

状态合同：第一版让 Model1/Model2 参数和 optimizer state 跨 domain 持续继承；`C-fixed-M1` 使用同一个 frozen WM1 贯穿所有 domains。每个 domain 重置 WM1 属于另一项 ablation，不能与 persistent setting 合并报告。

关键设计点：不要混入 replay，先测 replay-free catastrophic forgetting；如果后续加入 replay，A/C/D0 必须使用完全相同 replay budget。

## 6. 推荐执行顺序

Math-first，全部 1.7B：

1. 完成运行中的 DynPerm `rho=1` 与 common offline evaluation；
2. 实现并验证 fixed-M1 `AlignSort / true RandomPerm / AntiAlignSort` P20/P30 pilot；
3. 单独实现 fusion `lambda=0.5/0.4` 的 C/D0/fixed 六条 formal wrappers、manifests 与门禁，再启动 sweep；
4. A 共享现有 matched anchor；若要求同 revision，只额外重跑一次 A；
5. 第二 training seed 优先复制 C/fixed-M1/D0，估计 optimization uncertainty；
6. GFT 作为 published external baseline 单独排期；OPSFT 只在等价审计发现结论相关差异时补 sensitivity arm；
7. IDFT/continual-learning 留作下一轮机制扩展。

## 7. 论文写法建议

可写：

> Our Standard On-Policy SFT continuation belongs to the family of selected self-training / rejection-sampling fine-tuning methods: the current policy generates multiple responses, verifiable correct responses are selected, and the model is updated by teacher-forced likelihood on the selected trajectories. WDL extends this baseline by injecting a weak-model logit branch into the teacher-forced objective, while keeping the rollout, data order, and evaluation contract matched.

避免写：

- “On-Policy SFT 是 2026 年才提出的新方法”；
- “我们的实验 A 完全复现了 Zhao et al. 2026”；
- “Dynamic Perturbation 去掉了 weak model”；
- “GFT 是纯 SFT baseline”；
- “GRPO 已经被 offline eval 证明不如 WDL”。
