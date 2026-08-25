# On-Policy WDL-SFT 文档单一真相源规则

## 1. 权威关系

Git 中受版本控制的 Markdown 是可审计、可回滚的正文单一真相源；飞书文档是面向读者的完整发布副本。两侧必须包含相同的实验合同、主要结果表、结论边界和当前状态。链接只能用于补充导航或复现，不能替代正文。

| 文档 | Git 正文 | 飞书完整副本 |
| --- | --- | --- |
| Math 主实验方案 | `docs/joint_training/plans/active/qwen3_1p7b_math_stage123.md` | [Qwen3-1.7B Math 主实验方案](https://ocnwds5io8yp.feishu.cn/docx/Zh7fdwipKoqofZxQ6uDcCWZcnkc) |
| Math 实验结果 | `docs/joint_training/reports/qwen3_1p7b_math_stage123_matrix_results_20260723.md` | [Qwen3-1.7B Math 实验结果](https://ocnwds5io8yp.feishu.cn/docx/CFx6dw2YsoFpqzxGl61c2HRNnlh) |
| A/C/D0 综合方案与结果 | `docs/joint_training/reports/qwen3_1p7b_code_acd0_p60_feishu_cn.md` | [Qwen3-1.7B A/C/D0 综合文档](https://ocnwds5io8yp.feishu.cn/docx/Jl3sdHVuuoRsHAxvsT3cM80Bn4b) |
| WDL 机制探索 | `docs/joint_training/reports/qwen3_1p7b_wdl_mechanism_discovery_story_20260816.md` | [On-Policy WDL-SFT 机制探索](https://ocnwds5io8yp.feishu.cn/docx/X6ONdRRLlooN1PxBqfJcotiNnSh) |

## 2. 写入规则

1. 新实验现象、结果或解释先写入对应 Git Markdown，或在飞书修改后于同一轮回写 Git；不得只存在于飞书历史版本。
2. Git 更新后同步飞书，并回读目录、关键数值、结论措辞和附件；飞书不能只列服务器本地路径。
3. 飞书中的 CSV 必须作为真实文件附件上传。正文同时保留必要的汇总表和解释，附件只承载逐步、逐数据集或机器可读明细。
4. 运行状态以 scheduler、日志、checkpoint、raw evaluation 和 release-gate receipt 为权威证据；文档只发布已核验状态。
5. 发现两侧冲突时按证据新旧与完整性合并，不机械覆盖；过时的重复章节从发布副本移除，并在 Git 历史中保留可追溯性。

## 3. 每次同步的完成门槛

- Git 工作树中对应 Markdown 已更新并被版本跟踪；
- 飞书正文可以独立阅读，不依赖本地文件路径；
- 所有正文提到的 CSV 要么已转成正文表格，要么已作为飞书附件上传；
- 飞书回读后，章节结构与关键数值和 Git 一致；
- 变更已经 commit 并 push，且本地分支与远端无未推送提交。
