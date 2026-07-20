# Qwen3-1.7B Code Stage1 -> Stage2 -> Stage3 Plateau Breakthrough

- Status: ACTIVE DESIGN, DRY-RUN READY, FORMAL TRAINING BLOCKED
- Date: 2026-07-11
- Machine target: 8 x NVIDIA L40S 46 GB, 582 GiB RAM, 176 vCPU

## Question

Test whether a short joint Stage2 intervention at a Stage1 plateau can move the
model into a better basin, after which a Stage1-like single-model Stage3 can
continue improving beyond a compute-aligned pure Stage1 continuation.

## Minimal Matrix

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

1. Stage1 source must be the retained cold-start FRAC25 beta `0.1` actor at the
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
| generation / log-prob micro batch | `16` / `4` |
| vLLM GPU memory utilization / TP / GPUs | `0.40` / `1` / `8` |
| reward workers | `1` |
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

- `/data-1/checkpoints` must resolve to `/data-2/checkpoints`.
- New checkpoints must physically land on `/data-2`.
- Formal execution requires `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1`.
- P60 additionally requires `ALLOW_STAGE123_P60_ABLATION=1`.
- The current queue intentionally stops before real execution until a bounded
  initialization smoke receives explicit approval.
- All future long runs must execute in tmux.

## Acceptance Before Formal Training

- `bash -n` passes for profile, wrappers, queue, and monitor.
- Dry-run prints Stage1, Stage2, and Stage3 with the same canonical SHA256.
- `MAX_RESPONSE_LENGTH=8192` appears in all three launcher configs.
- Full LiveCodeBench appears in all three validation lists.
- A deliberately changed profile field causes a nonzero consistency failure.
- No formal training process, container, or tmux session is started.
