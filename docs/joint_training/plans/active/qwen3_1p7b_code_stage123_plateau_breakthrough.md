# Qwen3-1.7B Code Stage1 -> Stage2 -> Stage3 Plateau Breakthrough

- Status: AUTHOR-SIGNATURE-V2 STEP-20 MODEL1 SELECTED; 16-RUN MATRIX GPU PROBE PENDING
- Date: 2026-07-11
- Machine target: 8 x NVIDIA L40S 46 GB, 582 GiB RAM, 176 vCPU

## Question

Test whether a short joint Stage2 intervention at a Stage1 plateau can move the
model into a better basin, after which a Stage1-like single-model Stage3 can
continue improving beyond a compute-aligned pure Stage1 continuation.

## 2026-07-20 CoT Training Correction

The historical FRAC25/FRAC50 Cold Start models and all Stage1/Stage2/Stage3
checkpoints derived from them are answer-only controls. The old KodCode SFT
conversion discarded the problem-specific `<think>` label, and the old
`MultiTurnSFTDataset` path did not supervise Qwen3 reasoning tokens.

The corrected path is isolated under `format_cold_start_fraction_cot_v3`, uses
the `code-cot-python-answer-v3` dataset contract, supervises the complete
assistant message, rejects silent truncation, and requires a non-empty ordered
`<think>...</think><answer>...</answer>` response plus correct executable code
for reward `+1`. A full real-tokenizer preflight passed all 9,417 retained
samples.

Code Cold Start, Stage1, and Stage2 fraction queues now use CoT-v3 paths and run
prefixes. The legacy Stage123 manifests remain preserved for historical
traceability but are explicitly `launch_allowed: false`; their queue entrypoints
fail closed in non-dry-run mode. A new CoT-v3 Cold Start and Stage1 step40 must
finish before regenerating Stage123 manifests, calibration, and formal admission.

## 2026-07-25 Author-Signature-V2 Step-20 Model1 Decision

The author-preprocessed KodCode rerun completed full HumanEval+, MBPP+, and
LiveCodeBench validation at steps `0/5/10/15/20/25/30`, with `n=3`. No
checkpoint reached the pre-registered 85% strict-format gate. The user therefore
authorized an explicit below-threshold selection rather than rewriting the gate
as passed.

Model1 is frozen to:

`/data-1/model_weights/code_task/qwen3_1p7b_cold_start_cotmask_v3_author_signature_v2_steps/candidates/step_20`

Step 20 has the highest micro strict-format rate among the evaluated candidates:
50.63% overall, 68.50% on HumanEval+, 72.13% on MBPP+, and 38.07% on
LiveCodeBench. LiveCodeBench remains limited by 8K truncation. The selection receipt records
`selection_policy=manual_format_gate_override` and preserves the failed gate.

Cold Start ran with fixed order and consumed exactly the first `20 * 64 = 1,280`
SFT rows. The Stage1/2/3 data receipt excludes those 1,280 source indices, not
all 9,417 rows present in the SFT parquet. The resulting fixed-seed shards remain
pairwise disjoint and runtime `data.shuffle=False` preserves their order.

## Formal 16-Run Matrix

The new manifest is
`recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml`.

| Phase | Runs | Budget | Purpose |
| --- | ---: | ---: | --- |
| Stage1 | beta `0`, beta `0.1` | 40 each | Test reverse-SFT beta before joint handoff. |
| Stage1 control | matched beta `0`, `0.1` | 60 each after Stage1 step 40 | Pure single-model continuation to effective step 100. |
| Stage2 | beta x `{no-KL, model2-only KL}` | 20 each | Test WDL joint handoff and model2 KL. |
| Stage3 | each Stage2 arm x extracted `{model1, model2}` | 40 each | Test whether either submodel preserves or extends Stage2 movement. |

Total: 16 formal runs. Stage1 uses 2,560 rows, Stage2 1,280 rows, and
Stage3 2,560 rows. These shards are pairwise disjoint and exclude exactly the
first `20 * 64 = 1,280` source rows consumed by Cold Start. Stage1 controls use the
exact Stage2-then-Stage3 row order, so both paths consume the same 3,840
post-Stage1 examples by effective step 100. The split is fixed by seed
`20260706`; runtime `data.shuffle=False` prevents order drift.

Frozen training parameters are `lr=1e-6`, `warmup_steps=0`, prompt batch 64,
rollout `n=8`, response length 8192, validation `n=3`, temperature 0.2,
`top_p=0.95`, `top_k=-1`, validation/save every 5 steps, validation before
training, and entropy calculation disabled. The online primary metric is the
unweighted HumanEval+/MBPP+/LiveCodeBench macro `acc/mean@3`; Stage2 selects on
the model2 view.

The Code 8K profile initially proposed vLLM utilization 0.55. A real Stage2
model2-KL backward pass reached 44,943 MiB and OOMed while requesting another
2.39--2.75 GiB. A second real probe at 0.50 still missed a 2.74 GiB allocation
by about 0.13 GiB. The 0.45 Stage2 model2-KL probe completed a real optimizer
step, but its minimum observed headroom was only 639 MiB, below the frozen
1,024 MiB admission floor. The 0.40 Stage2 probe also completed a real optimizer
step but had only 611 MiB minimum headroom, confirming that the peak is dominated
by the joint actor plus model2 KL reference rather than KV-cache size. The
admitted candidate was therefore lowered to 0.35. Formal training starts only
after one-step Stage1 and Stage2 model2-KL probes at 0.35 plus the full Code-3
validation generated at the stricter 0.40 profile.

The admitted minimum GPU headroom is an empirical `512 MiB` floor, not a
production safety margin. The historical step-25 0.35 Stage2 probe observed 717 MiB while exercising
an 8192-token maximum response. That result is resource evidence only and cannot
admit the new author-signature-v2 step-20 manifest; a fresh positive-reward
Stage1/Stage2 probe and immutable admission are required.
BigCodeBench remains an offline endpoint rather than part of every online
validation cycle.

## Historical Minimal Matrix

Primary chain:

| Chain | Stage1 source | Stage2 | Stage3 | Effective step |
| --- | --- | --- | --- | ---: |
| `FRAC25-P40-S2-20-S3-40` | beta `0.1`, step 40 | beta `0.1`, lambda `0.8`, 20 steps | Stage1-like beta `0.1`, 40 steps | 100 |

Approval-gated ablation:

| Chain | Purpose |
| --- | --- |
| `FRAC25-P60-S2-20-S3-40` | Test a later plateau trigger at effective step 120. |

Do not launch FRAC50 or a broader trigger/duration sweep before interpreting the
primary chain. Compare each chain against the matching pure Stage1 checkpoint at
the same effective step. Stage2 is deliberately only 20 steps because prior
Stage2 runs showed useful early movement but severe late instability.

## Handoff Contract

1. Stage1 source must be a newly trained CoT-v3 FRAC25 beta `0.1` actor at the
   requested trigger (`global_step_40`, optionally `global_step_60`).
2. Stage2 uses model2 rollout and joint fused loss, full HumanEval+, MBPP+, and
   LiveCodeBench validation.
3. The final Stage2 joint actor is merged and `sub_model_index=1` is extracted.
4. Stage3 accepts only that extracted model2 plus a provenance record containing
   the Stage1 source, trigger, Stage2 run, final Stage2 step, extracted path, and
   shared resource-profile hash.
5. Stage3 returns to the Stage1-like single-model objective. Only model/artifact,
   objective, step count, run/output names, and provenance may differ by phase.

## Single Resource Profile

All three stages source exactly:

`recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh`

Canonical defaults include:

| Setting | Value |
| --- | ---: |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | `1024` / `8192` |
| model/token limits | `9216` |
| train batch / rollout N / mini batch | `64` / `8` / `512` |
| generation / log-prob / KL-ref micro batch | `32` / `8` / `1` |
| vLLM GPU memory utilization / TP / GPUs | `0.35` after Stage2 OOM at `0.55`/`0.50` and sub-floor 639/611 MiB headroom at `0.45`/`0.40` / `1` / `8` |
| reward workers / concurrency per worker | `8` / `4` |
| manager / evaluator timeout | `30s` / `30s` |
| execution AS limit | `4096 MB` |
| Ray memory threshold | `0.90` |

The queue exports the canonical serialization and SHA256. Every phase
re-serializes all fields, compares them field-by-field, and checks the expected
SHA256 before invoking an existing launcher. Any mismatch exits nonzero.

The reward worker default is one. The first L40S full-LCB launch measured two
workers at about 191 GB RSS each and crossed the Ray host-memory gate before
step 1. Benchmark breadth remains full; scorer concurrency is serialized.

Only settings that are wired into the launcher's Hydra/Ray runtime or exported
runtime environment are included in the canonical profile. Host CPU count and
RAM are machine-gate facts, not pretend per-run limits: this launcher does not
claim to enforce `num_cpus` or object-store bytes unless the underlying VERL
entry point exposes and consumes those settings.

## Storage And Execution Gates

- New checkpoints are written under `/data-1/checkpoints`; merged durable
  Stage123 outputs and runtime state are written under `/data-2/model_weights/code_task`.
- Formal execution requires the frozen manifest to bind the selected GPU probe,
  Model1 receipt, fixed-data receipt, runtime files, and reviewer result in an
  immutable admission receipt. The queue fails closed before that receipt exists.
- The current queue intentionally stops before real execution until the full
  Code-3 zero-step plus one-optimizer-step probe passes.
- All future long runs must execute in tmux.

## Acceptance Before Formal Training

- `bash -n` passes for profile, wrappers, queue, and monitor.
- Dry-run prints Stage1, Stage2, and Stage3 with the same canonical SHA256.
- `MAX_RESPONSE_LENGTH=8192` appears in all three launcher configs.
- Full LiveCodeBench appears in all three validation lists.
- A deliberately changed profile field causes a nonzero consistency failure.
- No formal training process, container, or tmux session is started.
