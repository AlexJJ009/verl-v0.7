# Qwen3-1.7B 数学任务 Stage123 16-Run 矩阵：结果与分析

日期：2026-07-23

实验方案：[`../plans/active/qwen3_1p7b_math_stage123.md`](../plans/active/qwen3_1p7b_math_stage123.md)

> 指标口径：`mean@3` 是每题 3 次采样的平均准确率，再对 Math-7 七个数据集做 macro average。本文 `pass@3` 从原始 validation JSONL 精确重算：每题 3 次中任意一次正确即通过，再对七个数据集做 macro average。训练日志中的 `best@3/mean` 是 1000 次 bootstrap 估计，数值接近但不完全等同于精确 pass@3。

## 1. 数据与运行权威性

- 设计矩阵共 16 个 run：2 个 Stage1、2 个 matched-data Stage1 control、4 个 Stage2（beta × KL）、8 个 Stage3（beta × KL × extracted model）。
- 16 个 authoritative run 均有 `stage_run_completed` 事件与最终模型；失败的早期重复 run、port-collision attempt 未纳入。
- Validation：完整 Math-7，`n=3`，每 5 step 一次；effective step 映射为 Stage1 `0–40`、Stage2 `40–60`、Stage3 `60–100`、control `40–100`。
- Stage1/2/3 的学习率均为 `1e-6`，warmup `0`。Stage2 `m2kl` 为 model2-only KL，系数 `0.01`。

## 2. 汇总表

| 实验 | Phase | beta | KL | 评测视角 | mean@3 起点→终点 | Δ mean@3 | 最佳 step/数值 | pass@3 起点→终点 | Δ pass@3 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `b0-stage1` | stage1 | 0 | na | model2-init | 38.67% → 42.25% | +3.58pp | 40 / 42.25% | 49.13% → 53.66% | +4.53pp |
| `b01-stage1` | stage1 | 0.1 | na | model2-init | 38.47% → 41.47% | +3.01pp | 40 / 41.47% | 49.16% → 52.36% | +3.20pp |
| `b0-stage1-control` | stage1_control | 0 | control | single | 42.48% → 66.35% | +23.87pp | 95 / 67.70% | 53.36% → 74.62% | +21.27pp |
| `b01-stage1-control` | stage1_control | 0.1 | control | single | 41.01% → 68.24% | +27.22pp | 100 / 68.24% | 52.81% → 74.33% | +21.52pp |
| `b0-stage2-nokl` | stage2 | 0 | nokl | default | 41.76% → 43.14% | +1.38pp | 60 / 43.14% | 52.68% → 54.55% | +1.88pp |
| `b0-stage2-m2kl` | stage2 | 0 | m2kl | default | 42.12% → 43.41% | +1.29pp | 60 / 43.41% | 54.26% → 55.72% | +1.47pp |
| `b01-stage2-nokl` | stage2 | 0.1 | nokl | default | 41.54% → 41.10% | -0.44pp | 55 / 41.72% | 53.76% → 53.70% | -0.07pp |
| `b01-stage2-m2kl` | stage2 | 0.1 | m2kl | default | 41.15% → 42.02% | +0.87pp | 50 / 42.04% | 52.44% → 53.36% | +0.92pp |
| `b0-stage3-nokl-model1` | stage3 | 0 | nokl | model1 | 40.98% → 57.03% | +16.05pp | 100 / 57.03% | 52.24% → 68.63% | +16.39pp |
| `b0-stage3-nokl-model2` | stage3 | 0 | nokl | model2 | 44.43% → 68.53% | +24.09pp | 100 / 68.53% | 56.55% → 76.24% | +19.68pp |
| `b0-stage3-m2kl-model1` | stage3 | 0 | m2kl | model1 | 40.06% → 52.12% | +12.06pp | 100 / 52.12% | 51.05% → 65.77% | +14.71pp |
| `b0-stage3-m2kl-model2` | stage3 | 0 | m2kl | model2 | 42.93% → 67.36% | +24.43pp | 95 / 68.51% | 54.92% → 72.55% | +17.63pp |
| `b01-stage3-nokl-model1` | stage3 | 0.1 | nokl | model1 | 39.89% → 47.34% | +7.45pp | 100 / 47.34% | 51.07% → 60.98% | +9.91pp |
| `b01-stage3-nokl-model2` | stage3 | 0.1 | nokl | model2 | 42.17% → 66.57% | +24.41pp | 90 / 67.17% | 54.76% → 74.00% | +19.24pp |
| `b01-stage3-m2kl-model1` | stage3 | 0.1 | m2kl | model1 | 40.97% → 51.24% | +10.27pp | 100 / 51.24% | 53.01% → 65.08% | +12.07pp |
| `b01-stage3-m2kl-model2` | stage3 | 0.1 | m2kl | model2 | 40.76% → 67.57% | +26.81pp | 100 / 67.57% | 52.63% → 74.60% | +21.97pp |

## 3. 每个 validation step

### `b0-stage1`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 0 | 38.67% | 49.13% |
| 5 | 5 | 39.72% | 50.76% |
| 10 | 10 | 39.25% | 49.69% |
| 15 | 15 | 39.77% | 50.67% |
| 20 | 20 | 40.24% | 51.05% |
| 25 | 25 | 40.61% | 51.90% |
| 30 | 30 | 40.48% | 51.46% |
| 35 | 35 | 41.74% | 53.23% |
| 40 | 40 | 42.25% | 53.66% |

### `b01-stage1`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 0 | 38.47% | 49.16% |
| 5 | 5 | 38.49% | 50.03% |
| 10 | 10 | 39.68% | 51.22% |
| 15 | 15 | 39.79% | 50.36% |
| 20 | 20 | 39.94% | 51.71% |
| 25 | 25 | 39.96% | 50.61% |
| 30 | 30 | 40.01% | 51.36% |
| 35 | 35 | 41.12% | 53.19% |
| 40 | 40 | 41.47% | 52.36% |

### `b0-stage1-control`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 40 | 42.48% | 53.36% |
| 5 | 45 | 42.26% | 54.21% |
| 10 | 50 | 41.48% | 52.74% |
| 15 | 55 | 42.81% | 53.85% |
| 20 | 60 | 43.17% | 54.68% |
| 25 | 65 | 43.00% | 54.94% |
| 30 | 70 | 43.32% | 54.76% |
| 35 | 75 | 49.65% | 63.51% |
| 40 | 80 | 61.84% | 70.85% |
| 45 | 85 | 67.00% | 73.51% |
| 50 | 90 | 67.56% | 74.11% |
| 55 | 95 | 67.70% | 74.94% |
| 60 | 100 | 66.35% | 74.62% |

### `b01-stage1-control`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 40 | 41.01% | 52.81% |
| 5 | 45 | 41.58% | 54.35% |
| 10 | 50 | 41.20% | 52.33% |
| 15 | 55 | 41.90% | 53.91% |
| 20 | 60 | 43.13% | 55.32% |
| 25 | 65 | 42.18% | 54.33% |
| 30 | 70 | 44.07% | 56.89% |
| 35 | 75 | 51.90% | 64.56% |
| 40 | 80 | 65.32% | 74.60% |
| 45 | 85 | 66.90% | 73.75% |
| 50 | 90 | 67.31% | 75.07% |
| 55 | 95 | 66.92% | 75.00% |
| 60 | 100 | 68.24% | 74.33% |

### `b0-stage2-nokl`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 40 | 41.76% | 52.68% |
| 5 | 45 | 42.46% | 52.95% |
| 10 | 50 | 42.28% | 53.02% |
| 15 | 55 | 42.92% | 53.94% |
| 20 | 60 | 43.14% | 54.55% |

### `b0-stage2-m2kl`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 40 | 42.12% | 54.26% |
| 5 | 45 | 41.16% | 52.77% |
| 10 | 50 | 42.27% | 53.00% |
| 15 | 55 | 42.58% | 53.97% |
| 20 | 60 | 43.41% | 55.72% |

### `b01-stage2-nokl`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 40 | 41.54% | 53.76% |
| 5 | 45 | 41.31% | 53.10% |
| 10 | 50 | 41.46% | 52.20% |
| 15 | 55 | 41.72% | 54.10% |
| 20 | 60 | 41.10% | 53.70% |

### `b01-stage2-m2kl`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 40 | 41.15% | 52.44% |
| 5 | 45 | 41.70% | 54.14% |
| 10 | 50 | 42.04% | 54.54% |
| 15 | 55 | 41.51% | 53.61% |
| 20 | 60 | 42.02% | 53.36% |

### `b0-stage3-nokl-model1`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 40.98% | 52.24% |
| 5 | 65 | 41.30% | 53.40% |
| 10 | 70 | 41.56% | 52.70% |
| 15 | 75 | 41.55% | 54.20% |
| 20 | 80 | 41.87% | 53.30% |
| 25 | 85 | 41.85% | 53.54% |
| 30 | 90 | 42.20% | 54.73% |
| 35 | 95 | 45.59% | 58.54% |
| 40 | 100 | 57.03% | 68.63% |

### `b0-stage3-nokl-model2`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 44.43% | 56.55% |
| 5 | 65 | 43.16% | 55.41% |
| 10 | 70 | 44.57% | 57.09% |
| 15 | 75 | 46.71% | 59.24% |
| 20 | 80 | 53.60% | 66.45% |
| 25 | 85 | 64.27% | 73.04% |
| 30 | 90 | 67.40% | 74.51% |
| 35 | 95 | 67.47% | 75.47% |
| 40 | 100 | 68.53% | 76.24% |

### `b0-stage3-m2kl-model1`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 40.06% | 51.05% |
| 5 | 65 | 41.18% | 52.92% |
| 10 | 70 | 40.48% | 51.94% |
| 15 | 75 | 41.05% | 52.94% |
| 20 | 80 | 41.80% | 53.37% |
| 25 | 85 | 42.44% | 53.86% |
| 30 | 90 | 41.38% | 53.00% |
| 35 | 95 | 43.86% | 57.09% |
| 40 | 100 | 52.12% | 65.77% |

### `b0-stage3-m2kl-model2`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 42.93% | 54.92% |
| 5 | 65 | 43.57% | 55.74% |
| 10 | 70 | 44.08% | 57.51% |
| 15 | 75 | 46.03% | 59.10% |
| 20 | 80 | 54.27% | 66.01% |
| 25 | 85 | 65.74% | 73.88% |
| 30 | 90 | 66.96% | 72.74% |
| 35 | 95 | 68.51% | 75.87% |
| 40 | 100 | 67.36% | 72.55% |

### `b01-stage3-nokl-model1`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 39.89% | 51.07% |
| 5 | 65 | 39.72% | 51.02% |
| 10 | 70 | 40.21% | 51.94% |
| 15 | 75 | 40.66% | 52.05% |
| 20 | 80 | 41.49% | 53.68% |
| 25 | 85 | 41.58% | 54.16% |
| 30 | 90 | 40.14% | 52.14% |
| 35 | 95 | 42.04% | 54.45% |
| 40 | 100 | 47.34% | 60.98% |

### `b01-stage3-nokl-model2`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 42.17% | 54.76% |
| 5 | 65 | 42.05% | 54.66% |
| 10 | 70 | 42.62% | 55.87% |
| 15 | 75 | 42.67% | 55.95% |
| 20 | 80 | 48.64% | 62.55% |
| 25 | 85 | 61.25% | 71.42% |
| 30 | 90 | 67.17% | 74.17% |
| 35 | 95 | 66.86% | 74.70% |
| 40 | 100 | 66.57% | 74.00% |

### `b01-stage3-m2kl-model1`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 40.97% | 53.01% |
| 5 | 65 | 39.45% | 51.01% |
| 10 | 70 | 40.46% | 51.95% |
| 15 | 75 | 39.46% | 51.22% |
| 20 | 80 | 40.11% | 51.62% |
| 25 | 85 | 40.91% | 53.40% |
| 30 | 90 | 40.73% | 54.35% |
| 35 | 95 | 42.83% | 55.41% |
| 40 | 100 | 51.24% | 65.08% |

### `b01-stage3-m2kl-model2`

| Local step（局部步数） | Effective step（有效步数） | Math-7 mean@3 | Math-7 pass@3 |
|---:|---:|---:|---:|
| 0 | 60 | 40.76% | 52.63% |
| 5 | 65 | 41.67% | 54.48% |
| 10 | 70 | 41.61% | 54.84% |
| 15 | 75 | 44.07% | 57.09% |
| 20 | 80 | 50.79% | 64.33% |
| 25 | 85 | 64.20% | 74.41% |
| 30 | 90 | 67.26% | 74.17% |
| 35 | 95 | 67.32% | 74.33% |
| 40 | 100 | 67.57% | 74.60% |

## 4. Stage3 model2 分数据集结果

### `b0-stage3-nokl-model2`

| 数据集 | Stage3 前 | Stage3 后 | Δ mean@3 |
|---|---:|---:|---:|
| AIME-2025 | 1.28% | 17.95% | +16.67pp |
| MATH-500 | 34.68% | 71.17% | +36.49pp |
| GSM8K | 54.26% | 81.91% | +27.65pp |
| AQUA-RAT | 43.44% | 75.98% | +32.55pp |
| SVAMP | 74.67% | 89.44% | +14.78pp |
| MAWPS | 87.70% | 92.39% | +4.69pp |
| AMC23 | 15.00% | 50.83% | +35.83pp |

### `b0-stage3-m2kl-model2`

| 数据集 | Stage3 前 | Stage3 后 | Δ mean@3 |
|---|---:|---:|---:|
| AIME-2025 | 1.28% | 12.82% | +11.54pp |
| MATH-500 | 33.33% | 67.81% | +34.48pp |
| GSM8K | 51.76% | 82.39% | +30.63pp |
| AQUA-RAT | 38.06% | 76.25% | +38.19pp |
| SVAMP | 74.78% | 91.33% | +16.56pp |
| MAWPS | 87.98% | 92.58% | +4.60pp |
| AMC23 | 13.33% | 48.33% | +35.00pp |

### `b01-stage3-nokl-model2`

| 数据集 | Stage3 前 | Stage3 后 | Δ mean@3 |
|---|---:|---:|---:|
| AIME-2025 | 1.28% | 12.82% | +11.54pp |
| MATH-500 | 32.19% | 69.09% | +36.90pp |
| GSM8K | 49.68% | 81.68% | +31.99pp |
| AQUA-RAT | 38.71% | 75.33% | +36.61pp |
| SVAMP | 72.11% | 90.33% | +18.22pp |
| MAWPS | 84.51% | 92.58% | +8.08pp |
| AMC23 | 16.67% | 44.17% | +27.50pp |

### `b01-stage3-m2kl-model2`

| 数据集 | Stage3 前 | Stage3 后 | Δ mean@3 |
|---|---:|---:|---:|
| AIME-2025 | 0.00% | 14.10% | +14.10pp |
| MATH-500 | 32.93% | 69.62% | +36.69pp |
| GSM8K | 48.45% | 82.41% | +33.97pp |
| AQUA-RAT | 36.35% | 75.98% | +39.63pp |
| SVAMP | 70.56% | 90.22% | +19.67pp |
| MAWPS | 87.04% | 93.15% | +6.10pp |
| AMC23 | 10.00% | 47.50% | +37.50pp |

## 5. 结论

### 得到支持的结论

1. **修复后的 CoT-v3 训练链路确实能学习。** Stage1 只有约 `+3pp`，但 control 和所有 Stage3 分支都出现明确增长；Stage3 model2 的训练 correct ratio 也随训练上升，排除了“只有 validation 偶然波动”的解释。
2. **Stage3 对 extracted model2 的增益稳定且普遍。** 四条 model2 Stage3 最终 mean@3 增益为 `+24.09pp`、`+24.43pp`、`+24.41pp`、`+26.81pp`；七个数据集均增长，不是单个数据集拉动。
3. **增长存在清晰的 phase transition。** model2 在 Stage3 local step `0–15` 基本平缓，主要跃升集中在 local step `20–30`；最大单个 5-step 增量出现在 step `20→25`，约 `+10.67` 到 `+13.40pp`。因此不是一开始即持续线性增长，也不是最后一点偶然跳变。
4. **model2 明显比 model1 更容易从 Stage3 获益。** model1 最终增益为 `+7.45` 到 `+16.05pp`，显著低于 model2 的 `+24.09` 到 `+26.81pp`。Stage2 联合训练后两个 extracted submodel 的可学习性并不对称。
5. **Stage2 20-step 预算偏短。** 四条 Stage2 final mean@3 变化仅 `+1.38pp`、`+1.29pp`、`-0.44pp`、`+0.87pp`；曲线尚未进入 Stage3 在 local step 20–30 出现的跃升区间。

### 被否定或未得到支持的结论

1. **“三阶段 WDL 路径在 matched 100-step budget 下优于继续 Stage1”没有得到证明。** Stage1 control 在 effective step 100 达到 `66.35%/68.24%` mean@3；Stage3 model2 final 为 `66.57–68.53%`，整体相当而非稳定领先。最好的 staged final `68.53%` 只比对应 beta=0 control final `66.35%` 高 `2.18pp`，但 control 自身 best 为 `67.70%`，差距缩至 `0.83pp`；beta=0.1 control final `68.24%` 反而高于两个 staged final。当前证据只能说明 staged path 可达到 control 水平，不能归因出 WDL 本身的独立优势。
2. **“model2-only KL 稳定有效”没有得到支持。** Stage2 m2kl 相比 nokl 仅有小且不一致的差异；Stage3 model2 中 beta=0 的 KL/no-KL final 基本持平（KL 略低），beta=0.1 的 KL 略高。没有跨 beta 的一致优势。
3. **“beta=0.1 会阻止 Stage3 大幅增长”被本轮完整结果否定。** beta=0.1 的两条 model2 Stage3 同样增长 `+24.41pp/+26.81pp`，与 beta=0 一致甚至略高。此前只看中途 step 得出的“没有复现大跳”是 premature conclusion，因为跃升主要发生在 local step 20–30。
4. **“beta=0.1 优于 beta=0”也没有成立。** beta=0.1 在 model2 上不破坏学习，但 final 与 beta=0 互有胜负；在 model1 上，nokl 分支 beta=0.1 明显更弱。reverse-SFT 的总体收益仍不确定。

### 实验设计中仍无法回答的问题

1. **WDL 的因果增益没有被纯净隔离。** control 是继续单模型 Stage1，treatment 中间经过 joint Stage2、submodel extraction，再单模型 Stage3；二者结构不同。要验证 WDL，需要增加 matched 初始化/数据/step 的 no-WDL Stage2→Stage3 或 single-model loss ablation。
2. **Stage2 是否只差训练步数仍需延长实验。** 当前证据强烈提示 20 steps 太短，但不能保证延长到 30/40 一定复现 Stage3 跃升，因为 Stage2 使用 joint model2-rollout/fused-loss，Stage3 使用 extracted single model，优化几何不同。
3. **单次 seed 不能确认 KL 或 beta 的小差异。** 大幅跃升是 robust signal；`1–3pp` 的差异仍需多 seed 或独立 offline eval。

## 6. 产物

- [`data/qwen3_1p7b_math_stage123_all_validation_steps.csv`](data/qwen3_1p7b_math_stage123_all_validation_steps.csv)：16 个实验的所有 validation step。
- [`data/qwen3_1p7b_math_stage123_all_validation_steps_by_dataset.csv`](data/qwen3_1p7b_math_stage123_all_validation_steps_by_dataset.csv)：逐数据集、逐 step 指标。
- [`data/qwen3_1p7b_math_stage123_experiment_summary.csv`](data/qwen3_1p7b_math_stage123_experiment_summary.csv)：实验汇总。
- [`figures/qwen3_1p7b_math_stage123_effective_step_mean3.png`](figures/qwen3_1p7b_math_stage123_effective_step_mean3.png)：mean@3 合并曲线。
- [`figures/qwen3_1p7b_math_stage123_effective_step_pass3.png`](figures/qwen3_1p7b_math_stage123_effective_step_pass3.png)：精确 pass@3 合并曲线。
- [`figures/qwen3_1p7b_math_stage123_final_mean3_comparison.png`](figures/qwen3_1p7b_math_stage123_final_mean3_comparison.png)：effective step 100 final 对比。

## 6.1 发布状态

- Deterministic training release gate：`16/16` 个 authoritative run 均以
  `success_complete` 通过。
- 本地 experiment registry：已导入
  `/data-1/experiment_registry/experiment_registry.sqlite`，training-run ID
  为 `80–95`。
- W&B offline run：Math project 下的 `16/16` 个 run 均已定位，并映射到预期的
  remote run ID。
- W&B cloud sync：**等待中**。当前环境没有提供 `WANDB_API_KEY`，因此尚无
  `.synced` marker，也没有经过验证的 cloud run。
- 没有上传 checkpoint 或 model artifact payload。

Release 证据：
`/data-1/tmp/verl_agent_scratch/math_stage123_release_20260723/release_summary.json`.

## 7. 固定顺序训练数据分析

### 7.1 问题

Stage1 control 和 Stage3 model2 run 的 validation 跃升都出现在 effective step
`75–85` 附近。由于 dataset permutation 和 row order 已冻结，一个合理的替代解释是：
对应训练行异常有用，而不是 optimizer 跨过了由 step 决定的阈值。

相关 fixed-order window 如下：

| Effective window | Stage1 control local step | Stage3 local step | Source row |
|---|---:|---:|---|
| `61–70` | `21–30` | `1–10` | Stage3 rows `0–639` |
| `71–75` | `31–35` | `11–15` | Stage3 rows `640–959` |
| `76–80` | `36–40` | `16–20` | Stage3 rows `960–1279` |
| `81–85` | `41–45` | `21–25` | Stage3 rows `1280–1599` |
| `86–90` | `46–50` | `26–30` | Stage3 rows `1600–1919` |

### 7.2 描述性结果

根据 Hendrycks MATH 可用的 metadata，Stage3 local `16–25` 对应的数据没有构成一个
明显的单独 outlier：

| Window | 平均难度等级 | Level >=4 | Level 5 | 平均题目字符数 | 平均 reference solution 字符数 |
|---|---:|---:|---:|---:|---:|
| Stage3 `11–15` | 3.46 | 50.94% | 30.63% | 229.41 | 532.84 |
| Stage3 `16–20` | 3.63 | 53.44% | 33.75% | 204.92 | 551.91 |
| Stage3 `21–25` | 3.49 | 50.63% | 29.69% | 199.19 | 543.83 |
| Stage3 `26–30` | 3.48 | 52.19% | 26.88% | 196.78 | 516.49 |

Local step `16–20` 的数据稍难，但四条 Stage3 model2 run 中 validation 跃升最大的
local step `21–25`，与相邻窗口相比并不更难、更长，也没有更集中于某一 subject。
Subject mix 会随 batch 波动，但不存在与跃升匹配的唯一共同 spike。

Stage3 只有八个 5-step window，因此相关性只作探索。平均难度等级与四模型平均
validation delta 的 Spearman correlation 为 `0.79`，但 Pearson correlation 只有
`0.52`，`p=0.19`；其他简单 metadata feature 也没有显示稳定关系。这些证据不足以把
跃升归因于数据难度或 subject composition。

![固定顺序数据窗口](figures/qwen3_1p7b_math_stage123_data_window_analysis.png)

结构化证据：

- [`data/qwen3_1p7b_math_stage123_training_data_step_features.csv`](data/qwen3_1p7b_math_stage123_training_data_step_features.csv)
- [`data/qwen3_1p7b_math_stage123_training_data_window_features.csv`](data/qwen3_1p7b_math_stage123_training_data_window_features.csv)
- [`data/qwen3_1p7b_math_stage123_stage3_data_metric_join.csv`](data/qwen3_1p7b_math_stage123_stage3_data_metric_join.csv)

### 7.3 当前解释

固定数据顺序仍是现实存在的 confounder，但描述性证据**没有**找到能够解释跃升的特殊
row block。当前更强的假设是：模型在处理这些数据行期间跨过了 optimization/state
threshold。要进行因果分离，仍需 data-order intervention。

### 7.4 2026-07-27 D0 在线复核：数据窗口解释未获支持

本节补充正在运行的 matched-scale no-weak control `D0` 的中期证据；它用于解释
跃升机制，不作为 `D0` 或后续 Arm C 的最终 efficacy 结论。`D0` 从完成 40 steps
的 `b0-stage1` Model2 初始化，因此 local step `35` 对应 effective step `75`。
它在 effective `70 -> 75` 的 Math-7 macro mean@3 从 `40.11%` 升至
`48.43%`，增加 `8.32pp`。此前 `b0-stage1-control` 和
`b01-stage1-control` 在同一区间分别增加 `6.33pp` 和 `7.83pp`。从 `D0`
step-40 原始 validation JSONL 直接重算的 effective-step-80 mean@3 为
`51.88%`，说明 step 35 不是立即回落的孤立 validation 点，但 `D0` 仍在运行，
最终比较必须等待预注册终点。

对 effective `71–75` 对应的 combined parquet rows `1920–2239`（Stage3
source rows `640–959`）重新审计后，没有发现数据源、schema、prompt、reward
style 或 shard 边界发生变化。真正的 Stage2-to-Stage3 shard 边界位于 effective
step `60 -> 61`；effective step `75` 只是同一 Stage3 shard 内的普通位置。

| Effective window | 平均难度等级 | Level >=4 | 平均题目字符数 | 平均 reference solution 字符数 |
|---|---:|---:|---:|---:|
| `66–70` | 3.53 | 53.8% | 210 | 556 |
| `71–75` | 3.46 | 50.9% | 229 | 533 |
| `76–80` | 3.63 | 53.4% | 205 | 552 |
| `81–85` | 3.49 | 50.6% | 199 | 544 |
| 全部 3,840 rows | 3.50 | 52.7% | 207 | 544 |

该窗口只表现为题目文字略长、标注难度略低，reference solution 长度、subject
mix，以及 numeric、fraction、expression 三类答案比例均与相邻窗口和全量数据
接近。3,840 个 source index 全部唯一，没有重复训练题；对规范化题面做 exact
match，也没有发现训练数据与 Math-7 的 2,797 个 unique validation prompts
重叠。静态特征因此不支持“effective `71–75` 恰好是一批显著更优质训练数据”
作为跃升的主要解释。

实际训练信号同样没有显示该窗口是孤立的 easy/high-reward batch。两个 Stage1
controls 在 local steps `31–35` 的平均 rollout correct ratio 分别只有 `21.2%`
和 `20.9%`；四条历史 Stage3 Model2 路径处理相同 source rows 时为
`18.6–20.5%`。当前 `D0` 在该窗口为 `30.9%`，随后 local steps `36–40`
继续升至约 `39.6%`，同时平均 response length 从 `391` tokens 增至约 `511`
tokens。变化更像跨多个相邻窗口持续展开的模型状态转变，而不是单批数据触发的
一次性奖励脉冲。

当前更可能的机制是：随着累计优化进入 transition regime，正确 rollout 比例
上升，使 `beta=0.0` 的 positive-only WDL/SFT objective 获得更多可监督 response
和 token；同时回答从偏短、推理不足进入更合适的长度区间，从而提高每步有效
梯度信号的密度与质量。correct ratio、positive supervised token count、response
length、grad norm 和 held-out accuracy 的同步变化支持这一解释。不过这仍是机制
假设，不是严格因果证明；回答继续变长也可能最终带来 truncation，不能把长度
上升本身当成质量提升。

因此，本轮复核否定的是“现有证据显示跃升区间的数据本身更优质”这一具体疑问，
而不是数学上排除任何 data-order effect。由于 fixed row order 与 effective step
仍然绑定，最终区分数据窗口和优化阈值仍需要 deterministic reshuffle 或 window
relocation intervention。

## 8. 训练指标分析

### 8.1 共同的 transition 特征

两个 Stage1 control 和四个 Stage3 model2 run 都呈现以下特征：

1. effective step `75` 前，Math-7 基本保持在 `41–47%`；
2. training correct ratio 在 effective step `75–80` 左右开始持续上升；
3. effective step `80–90` 期间，平均 rollout response length 从约 `400–500` token
   上升到 `600–1,000+` token；
4. validation accuracy 在同一区间上升；
5. truncation/clip ratio 也随之上升，到 effective step `100` 时约为 `15–18%`。

在六条 control/model2 曲线上，validation level 与 training correct ratio 高度相关
（`Pearson r=0.86`，`Spearman r=0.92`），与 rollout length 也高度相关
（`Pearson r=0.81`，`Spearman r=0.89`）。这些是 trajectory correlation，不能确定
因果方向；但它们表明跃升伴随真实的 rollout behavior 变化，而不只是 validation noise。

![Validation 与训练动态](figures/qwen3_1p7b_math_stage123_training_dynamics.png)

### 8.2 安全性解释

本轮实验中，更长的 response 与更好的 validation accuracy 相关，但持续上升的 truncation
rate 表明这并非没有代价的提升。Continuation 实验必须监测：

- Math-7 macro `mean@3` 和精确 `pass@3`；
- `wdl_sft/correct_ratio` 与 reward mean；
- response-length mean 与 distribution；
- response clip/truncation ratio 与 EOS rate；
- grad norm 和 policy/loss 指标；
- format completeness 与 boxed extraction。

不能仅凭 reward 和 response length 上升就宣布 run 更好。主结果仍是 Math-7 accuracy，
truncation 与 format 指标作为 hard health check。

结构化证据：

- [`data/qwen3_1p7b_math_stage123_training_history.csv`](data/qwen3_1p7b_math_stage123_training_history.csv)
- [`data/qwen3_1p7b_math_stage123_training_metric_correlations.csv`](data/qwen3_1p7b_math_stage123_training_metric_correlations.csv)

## 9. 更新后的结论

1. CoT-v3 训练链路健康，所有 treatment/control path 都能学习。
2. Stage3 model2 的大幅增益真实存在，覆盖多个数据集，并在 beta `0.0` 和 `0.1` 下复现。
3. 跃升集中在 effective step `75–85` 附近，pure Stage1 control 也在同一时段出现。因此，
   该跃升**本身不能证明** WDL、joint Stage2 或 handoff 具有优势。
4. 没有任何简单的 Stage3 data-window feature 能唯一解释跃升。固定顺序仍是 confounder，
   必须通过受控的数据重排来检验。
5. transition 伴随 correct ratio、response length 上升，并最终伴随 truncation 上升。
   模型改变了 rollout regime，而不只是落在 validation noise 内波动。
6. 在已完成矩阵中，Model2-only KL 没有稳定优势。
7. matched-budget control 阻止我们对 staged WDL design 作正面因果声明。仍需增加 matched
   loss 与 data-order ablation。

## 10. 后续实验问题

关联的实验方案已经预注册 continuation matrix，主要问题是：

1. Stage2 超过 local step `20` 后，是否进入同样的 transition？
2. 当 Stage2 使用相同 canonical data order 时，transition 是否仍绑定 effective step
   `75–85`？
3. 移动或 shuffle 疑似相关的 Stage3 row window，是否会移动 transition？
4. Stage2 总 step 为 `40`、`45`、`60` 时，哪个 checkpoint 能提供最好的 Stage3 handoff？
5. 能否在不让 response length 和 truncation 持续上升的情况下保留增益？
