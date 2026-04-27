---
name: n=1 与 n=3 不可混比
description: 对比实验做 L-term / 涨跌分析时，online mean@1 与 offline mean@3 不能混用，必须 n=1 vs n=1 或 n=3 vs n=3
type: feedback
originSessionId: 0fcea846-0b57-4d6a-b278-52e0aa3b5619
---
不能把 online MATH-500 mean@1（单次 rollout 验证）和 offline mean@3（3 次 rollout 平均）放进同一张 L_loss / L_fusion / L_init 表里相减。

**Why**：mean@3 的方差比 mean@1 小一个 √3 倍，且这个项目里 online mean@1 与 offline mean@3 已多次出现 5–10 pp 量级的背离（v1 EXP-15 online 68% / offline 79%；1B online 70.97% / offline 82.9%）。两条曲线在不同噪声水平和不同 decoding 配置（val top_p=0.95 单次 vs 评估 pipeline 的 mean@3）下取数，符号都可能反过来。混着减相当于把指标差当效果差。

**How to apply**：
- 做 §5/§9.2 那种 L_loss = score(2X) − score(2Z) 的拆解时，先声明列名是 "online mean@1 step-300" 还是 "offline mean@3 best-step"，全表只用一种。
- 如果两边数据不齐（例如 7 个 run 都有 online、只有 1 个 run 有 offline），就只输出能配对的那一格，缺的写 "pending offline" 而不是把 offline 数值塞进 online 表里凑齐。
- 报告里引用 §9.3 已确认的 offline 结论时，单独成段、明确标 "offline n=3"，不要插进 online 决算表的同一行。
