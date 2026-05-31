# Goal: Stage 2 Model2-Rollout Fused-Loss Fast Validation

- Status: IMPLEMENTATION REVISION REQUIRED - the plan contract and acceptance
  criteria below are fixed; do not launch training until the scripts implement
  the matched-beta matrix and on-policy mini-batch gates, and dry-run
  verification records the rendered config.
- Target branch: `feature/on-policy-wdl-sft`
- Parent workflow: `docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search.md`
- Target recipe family: `recipe/on_policy_wdl_sft/staged_v1/`
- Target platform family: `platform/hope_staged_v1/`
- Current acceptance tier: local-only PASS. Meituan/AFO support is deferred and
  must be marked `NOT ACCEPTED` until explicitly implemented and reviewed.
- Created: 2026-05-29
- Last updated: 2026-05-30

## Acceptance Summary

This plan has an all-or-nothing local PASS gate. A completed PASS requires both
default runs to finish 75 training steps:

```text
Stage 1 beta 0.0 best -> Stage 2 beta 0.0
Stage 1 beta 0.1 best -> Stage 2 beta 0.1
```

The previous `Stage 1 beta 0.1 -> Stage 2 beta 0.0` attempt is a cross-beta
diagnostic only and cannot satisfy this plan.

For every PASS candidate:

1. Rollout is Model2-only.
2. Training loss is computed on fused Model1+Model2 logits.
3. Both Model1 and Model2 are trainable.
4. Stage 2 beta matches the Stage 1 beta that produced Model2.
5. Actor update uses one optimizer mini-batch per rollout batch:
   `TRAIN_PROMPT_BSZ=64`, `ROLLOUT_N=8`, `TRAIN_PROMPT_MINI_BSZ=512`,
   `ppo_epochs=1`, and `actor.shuffle=false`.
6. Training data is the non-overlap Stage 2 shard after the first 150 Stage 1
   steps for seed `20260528`.
7. Queue and WxPusher notifications cover queue start, launch start/confirmation,
   per-run completion, queue completion, and unrecoverable failure/blocker.

Any run that violates these gates can be recorded only as a diagnostic failure
or diagnostic warning, not as runtime PASS.

After this 2026-05-30 revision, this goal file is locked. Do not modify this
plan again without explicit user permission; continue implementation and
runtime tracking in scripts, logs, and the status file.

## 1. Objective

Run a minimal Stage 2 validation of On-Policy Weak-Driven Learning SFT after the
Stage 1 single-model On-Policy SFT beta search produced usable checkpoints.

The question for this plan is narrow:

```text
If Model 2 rolls out answers on held-out training prompts, and training computes
WDL-SFT on fused Model1+Model2 logits while updating both submodels, does the
method produce a useful online validation signal?
```

This is not a full Stage 2 beta search, not a checkpoint-search study, and not a
final generalization claim. It is a fast algorithm validation with strict
controls around model provenance, data non-overlap, and runtime evidence.

## 2. Algorithm Contract

Stage 2 uses two models:

| Role | Source | Trainable |
| --- | --- | --- |
| Model 1 / weak | Original Qwen3-4B-Base | yes |
| Model 2 / strong | Merged Stage 1 best checkpoint | yes |

Required behavior:

1. Rollout must use **Model 2 only**.
2. The same Model2-generated responses are scored by the reward function.
3. Actor training must teacher-force those responses through the joint model and
   compute loss on **fused Model1+Model2 logits**.
4. Gradients must update both Model 1 and Model 2. `freeze_model1` and
   `freeze_model2` must both be false or absent.
5. Evaluation target remains Model 2, matching the rest of this branch's
   joint-training convention.

This is intentionally described as:

```text
Model2-policy rollout + fused-policy WDL-SFT loss
```

It is not fused-policy rollout. The WDL mechanism is being tested as a gradient
amplifier for the Model2 trajectory, using the existing joint-logit fusion code
path during training.

## 3. Loss Contract

Use the existing v1 WDL-SFT loss:

```text
loss_mode=wdl_sft
```

Default Stage 2 loss knobs:

| Knob | Value |
| --- | --- |
| Stage 2 `wdl_sft_beta` | matched to the Stage 1 beta used to initialize Model 2 |
| Default beta pairs | Stage 1 `0.0` -> Stage 2 `0.0`; Stage 1 `0.1` -> Stage 2 `0.1` |
| Cross-beta wrappers | script-only diagnostics; not launched for this plan's PASS criteria |
| Importance sampling | disabled, `rollout_is=null` |
| Reward shaping correction | disabled, `rollout_rs=null` |
| KL in reward | disabled |
| Actor KL loss | disabled |
| Advantage normalization | disabled for this WDL-SFT path |
| Loss aggregation | `seq-mean-token-sum` unless the shared WDL-SFT launcher already overrides it differently |

The trainer must apply raw reward labels for `wdl_sft`, as in the current
branch's existing WDL-SFT path. Do not introduce GRPO group advantages,
`wdl_group_adv_is`, rollout IS weights, model2-vs-fused IS, or KL penalty in
this plan.

The matched-beta rule is part of the experiment definition. A run such as
Stage 1 beta `0.1` -> Stage 2 beta `0.0` can be recorded as a diagnostic
attempt, but it does not count toward this plan's runtime PASS.

## 4. Stage 1 Inputs

Use completed Stage 1 runs only.

Initial fast-validation inputs:

| Stage 1 beta | Stage 1 run | Required checkpoint | Current best metric |
| ---: | --- | --- | ---: |
| `0.0` | `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803` | `global_step_85` | `0.7325268817204301` |
| `0.1` | `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295` | `global_step_150` | `0.7573924731182795` |

The implementation must not rely on a loose latest-prefix search that can pick a
different Stage 1 run. Every Stage 2 wrapper must pin either:

```text
STAGE1_CKPT_DIR=/data-1/checkpoints/<exact-run-id>
STAGE1_STEP=<exact-best-step>
```

or an already merged explicit:

```text
MODEL2_PATH=/data-1/model_weights/staged_v1/<exact-run-id>/step_<step>
```

The merge step must produce a Hugging Face model directory before Stage 2
training starts. Reusing an existing merged directory is acceptable only if the
path matches the pinned Stage 1 run and step.

## 5. Data Protocol

Stage 2 must avoid prompt-level overlap with the Stage 1 beta runs used to build
Model 2.

Stage 1 beta-search settings consumed a deterministic shuffled prefix:

```text
DATA_SEED=20260528
TRAIN_PROMPT_BSZ=64
STAGE1_TOTAL_TRAINING_STEPS=150
STAGE1_CONSUMED_PROMPTS=9600
```

Stage 2 fast validation uses the next deterministic slice:

```text
STAGE2_TOTAL_TRAINING_STEPS=75
STAGE2_PROMPTS=4800
STAGE2_START_OFFSET=9600
STAGE2_END_OFFSET=14400
```

Required shard:

```text
/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet
```

Shard construction rule:

1. Load `/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet`.
2. Reproduce the Stage 1 eligible training set before sampling:
   - same tokenizer/model path;
   - same `data.max_prompt_length=500`;
   - same `data.filter_overlong_prompts=True`;
   - same prompt key and truncation settings.
3. Generate the same deterministic sampler order used by Stage 1 after filtering:
   `torch.randperm(eligible_row_count, generator=torch.Generator().manual_seed(20260528))`.
   The generator must not use numpy or pandas shuffle as a substitute.
4. Skip the first `9600` eligible-row positions in that permutation.
5. Write the next `4800` prompts in deterministic order.
6. Preserve the original schema and columns.
7. Write a manifest next to the shard recording source path, raw source row
   count, eligible row count, tokenizer/model path, filter settings, permutation
   implementation, seed, offset, length, sha256, generated row count, and
   generation command.

Stage 2 wrappers must set:

```text
TRAIN_FILE=/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet
TRAIN_MAX_SAMPLES=-1
DATA_SEED=20260528
```

If the implementation supports a clear `data.shuffle=False` override, use it for
the shard-backed Stage 2 runs so the shard order itself is the training order.
If not, the plan must record the limitation and demonstrate that the prompt set
is still non-overlapping even if the shard is shuffled internally.

## 6. Experiment Matrix

Run only these fast-validation candidates unless the user explicitly authorizes
more:

| Run label | Model 2 input | Stage 2 beta | Steps | Purpose |
| --- | --- | ---: | ---: | --- |
| `s2-from-s1-beta0-beta0` | Stage 1 beta `0.0` best, step `85` | `0.0` | `75` | Control candidate from no reverse-SFT Stage 1 checkpoint |
| `s2-from-s1-beta01-beta01` | Stage 1 beta `0.1` best, step `150` | `0.1` | `75` | Matched reverse-SFT candidate |

Prepare but do not launch cross-beta diagnostic wrappers:

| Run label | Model 2 input | Stage 2 beta | Launch policy |
| --- | --- | ---: | --- |
| `s2-from-s1-beta0-beta01` | Stage 1 beta `0.0` best | `0.1` | explicit authorization only |
| `s2-from-s1-beta01-beta0` | Stage 1 beta `0.1` best | `0.0` | explicit authorization only; previous partial attempt is not accepted |

The default first launch, after all acceptance checks pass, should be
`s2-from-s1-beta0-beta0`, followed by `s2-from-s1-beta01-beta01`. This order
keeps the beta-aligned control first and avoids interpreting cross-beta results
as the main Stage 2 validation.

## 7. Runtime Defaults

| Knob | Value |
| --- | --- |
| Base model / Model 1 | `/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Model 2 | merged Stage 1 best checkpoint |
| Fusion lambda | `0.5` |
| LR | `5e-7` |
| Total steps | `75` |
| Prompt batch | `64` |
| Rollouts per prompt | `8` |
| Actor mini-batch | `512` response samples (`TRAIN_PROMPT_BSZ * ROLLOUT_N`) |
| Actor PPO epochs | `1` |
| Actor mini-batch shuffle | `false` |
| Max prompt / response | `500 / 4096` |
| Actor entropy diagnostic | `CALCULATE_ENTROPY=False` for this fast validation |
| vLLM memory utilization | `0.35` |
| Validation | MATH-500 + AIME-2025 |
| Validation samples | `VAL_N=3` |
| Validation before train | `False` |
| Test frequency | `5` |
| Save frequency | `5` |
| Checkpoint retention | latest full checkpoint + best checkpoint |
| W&B mode | offline |
| W&B project | `OnPolicySFT-Then-WDLSFT-StagedV1` |
| Attention backend | vLLM `FLASHINFER`; FSDP `flash_attention_2` |

## 8. Required Implementation Artifacts

Data artifacts:

```text
recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py
/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet
/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.manifest.json
```

Recipe wrappers:

```text
recipe/on_policy_wdl_sft/staged_v1/_run_stage2_model2_rollout_common.sh
recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta0.sh
recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta01.sh
recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta01.sh
recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta0.sh
```

Operational files:

```text
recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.sh
recipe/on_policy_wdl_sft/staged_v1/monitor_stage2_fast_validation_queue_notify.sh
docs/joint_training/plans/active/stage2_model2_rollout_fused_loss_fast_validation_status.md
```

Meituan four-layer support must be updated if any of these scripts are intended
to run on Meituan. Local-only implementation may defer Meituan launch, but then
the status file must mark Meituan as not yet accepted and the training script
index must say local-only.

## 9. Implementation Tasks

1. Create the Stage 2 non-overlap shard generator.
2. Generate and validate the Stage 2 shard and manifest.
3. Implement or adapt the Stage 2 common wrapper so that:
   - rollout is Model2-only;
   - training loss is fused Model1+Model2 logits;
   - both submodels are trainable;
   - Stage 1 checkpoint provenance is pinned;
   - `TRAIN_FILE` points to the non-overlap shard.
4. Add the four candidate wrappers listed in Section 8.
5. Add a local sequential queue for the two default runs.
6. Add WxPusher notifications for queue start, individual training launch,
   individual run completion, full queue completion, and unrecoverable
   failure/blocker.
7. Update `docs/joint_training/guides/training_script_index.md`.
8. Update active plan index and bridge docs.
9. Run shell syntax checks.
10. Run a dry-run or config-render check that prints the exact Hydra overrides.
11. Run one short smoke only after user authorization or when the implementation
    contract explicitly calls for smoke validation.
12. Do not launch the 75-step fast-validation runs until all pre-launch
    acceptance criteria pass.

## 10. Pre-Launch Acceptance Criteria

Pre-launch PASS requires all of the following:

### 10.1 Data PASS

- Stage 2 shard exists at the required path.
- Shard row count is exactly `4800`.
- Manifest exists and records:
  - source parquet path;
  - raw source row count;
  - eligible row count after Stage 1-compatible filtering;
  - tokenizer/model path used for filtering;
  - prompt length and overlong-filter settings;
  - permutation implementation `torch.randperm` with a manually seeded
    `torch.Generator`;
  - seed `20260528`;
  - offset `9600`;
  - length `4800`;
  - output sha256;
  - command used to generate the shard.
- A verifier confirms no `extra_info.index` overlap between:
  - Stage 1 consumed prefix indices `[0, 9600)` in the deterministic order;
  - Stage 2 shard indices `[9600, 14400)` in the deterministic order.
- A verifier confirms the first and last 10 selected `extra_info.index` values
  match the manifest, so a different permutation implementation cannot silently
  pass only by row count.
- The shard preserves required columns: `prompt`, `reward_model`, `extra_info`,
  `ability`, `split`, and `data_source` if present in the source.

### 10.2 Model Provenance PASS

- Stage 1 beta `0.0` wrapper pins run
  `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803`, step `85`.
- Stage 1 beta `0.1` wrapper pins run
  `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295`, step `150`.
- The merge command succeeds or the merged Model2 directory already exists.
- The merged directory contains Hugging Face model weights and tokenizer/config
  files sufficient for joint weight preparation.
- The status file records the source FSDP actor checkpoint path, global step,
  merge command, merged output path, and a sha256 digest or file inventory for
  the merged model.
- A lightweight load check succeeds for both merged Model2 directories:
  `AutoConfig.from_pretrained`, `AutoTokenizer.from_pretrained`, and a model
  architecture/config check confirming the expected Qwen3-4B family.
- The status file records merged Model2 paths for both Stage 1 inputs.

### 10.3 Algorithm PASS

- Config/log evidence shows rollout uses Model2-only weights.
- Dry-run/config-render evidence prints both paths separately:
  - rollout weight source path equals the pinned/merged `MODEL2_PATH`;
  - if the rollout engine requires a joint architecture shell, its architecture
    path may equal the prepared joint model path, but the runtime weight-sync
    source must be Model2-only and must set the rollout to Model2-only mode;
  - actor training model path equals the prepared joint model path.
- Runtime logs must contain an explicit marker such as
  `rollout source: model2-only` or an equivalent code/config dump. A generic
  `joint_training=True` log is not sufficient evidence.
- Config/log evidence shows actor training uses a joint model path with
  `joint_training=True`.
- Config/log evidence shows loss mode `wdl_sft`.
- Config/log evidence shows the default runs use matched beta:
  - `s2-from-s1-beta0-beta0`: `WDL_SFT_BETA=0.0`;
  - `s2-from-s1-beta01-beta01`: `WDL_SFT_BETA=0.1`.
- Config/log evidence shows `rollout_is=null` and `rollout_rs=null`.
- Config/log evidence shows KL reward and actor KL loss are disabled.
- Config/log evidence shows actor update is single-mini-batch per rollout batch:
  - `data.train_batch_size=64`;
  - `actor_rollout_ref.rollout.n=8`;
  - `actor_rollout_ref.actor.ppo_mini_batch_size=512`;
  - `actor_rollout_ref.actor.ppo_epochs=1`;
  - `actor_rollout_ref.actor.shuffle=false`.
  In the FSDP actor path the rollout-augmented batch has
  `64 * 8 = 512` response samples, and `dp_actor.update_policy()` splits it by
  `ppo_mini_batch_size`. Therefore `512` is required to avoid multiple
  optimizer steps over the same Model2 rollout batch. Smaller values, including
  the historical `8`, are accepted only for diagnostics and do not satisfy this
  plan's on-policy gate.
- Code or runtime logs confirm both submodels are trainable and receive updates.
  If per-submodel gradient metrics are not already emitted, add or identify a
  lightweight evidence source before launch.
- Validation is model2-targeted, consistent with current joint-training
  convention.

### 10.4 Script PASS

- All required scripts exist.
- `bash -n` passes for all Stage 2 shell scripts.
- The shard generator has a `--verify-only` mode or equivalent verifier.
- A dry-run/config-render command records exact paths and knobs without starting
  a long training job.
- The dry-run/config-render output must include exact values for
  `TRAIN_FILE`, `MODEL2_PATH`, prepared joint `MODEL_PATH`, `BASE_MODEL_PATH`,
  `STAGE1_CKPT_DIR`, `STAGE1_STEP`, `LOSS_MODE`, `WDL_SFT_BETA`,
  `ROLLOUT_IS`, `ROLLOUT_RS`, `TRAIN_PROMPT_BSZ`,
  `TRAIN_PROMPT_MINI_BSZ`, `ROLLOUT_N`, `TOTAL_TRAINING_STEPS`, and
  validation cadence.
- Scripts are default-local and overridable by environment variables.
- Scripts fail fast if the pinned Stage 1 checkpoint, merged Model2 path, train
  shard, or base model is missing.
- No Stage 2 script can silently fall back to a loose latest Stage 1 checkpoint.

### 10.5 Ops PASS

- Queue runs default candidates sequentially, not in parallel.
- Queue checks disk free space, GPU utilization, conflicting tmux sessions, and
  conflicting Docker containers before launch.
- Training and queue launch commands are intended to run inside tmux.
- WxPusher is configured through the existing local notification script and
  usage guide on this server:
  `/root/agent-core/skills/wxpusher-notify/SKILL.md`.
- WxPusher sending must use the guarded local script unless the guide is later
  revised:
  `python3 /root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py --title ... --body ...`.
- WxPusher phone notifications are mandatory at these important task nodes:
  - queue start;
  - each individual training launch start;
  - each individual training launch confirmation after the tmux session is
    alive;
  - each individual training end, whether completed or failed;
  - full queue completion;
  - unrecoverable failure or blocker;
  - any problem the agent cannot solve without user decision or external state
    change.
- WxPusher messages must be concise and actionable, with status, what happened,
  evidence path/step/error, and next action. They must not include secrets or
  large raw logs.
- W&B defaults to offline mode.
- Checkpoint retention is latest plus best.
- Status file records exact command, tmux session, log path, checkpoint path,
  metrics path, W&B offline path, and current result.

## 11. Runtime Acceptance Criteria

For each 75-step default run, runtime PASS requires:

- The run reaches `training/global_step=75`.
- Checkpoint marker reaches `latest_checkpointed_iteration.txt=75`.
- `best_checkpoint.json` exists and points to the configured metric
  `val-core/HuggingFaceH4/MATH-500/acc/mean@3`.
- Metrics JSONL contains:
  - `training/global_step`;
  - `actor/wdl_sft_beta`;
  - `actor/wdl_sft_loss_positive`;
  - `actor/wdl_sft_loss_negative`;
  - `actor/wdl_sft_loss_total`;
  - `wdl_sft/correct_ratio`;
  - `actor/grad_norm`;
  - `response/aborted_ratio`;
  - MATH-500 `mean@3`;
  - AIME `mean@3` if validation ran.
- `actor/entropy` is optional for this fast validation. It is disabled by
  default because `entropy_coeff=0` and the first runtime attempt OOMed inside
  diagnostic entropy computation, which does not participate in the WDL-SFT
  training objective.
- `response/aborted_ratio` remains low enough to interpret validation; any value
  above `0.05` must be called out as a risk.
- No `CUDA out of memory`, `ActorDied`, `WorkerCrashed`, `NCCL ERROR`, or
  unhandled traceback is present before the final checkpoint.
- Cleanup-time W&B/DataLoader warnings after a saved final checkpoint may be
  recorded as warnings rather than failures, but only if metrics and checkpoint
  completion are verified.
- Queue/notification evidence exists:
  - runs are launched by `run_stage2_fast_validation_queue.sh`;
  - `monitor_stage2_fast_validation_queue_notify.sh` or the queue's built-in
    notifier remains active during unattended execution;
  - WxPusher notifications were sent for queue start, training launch,
    individual run completion, full queue completion, and failure/blocker if
    one occurs.

If a run fails before step 75, it may be marked `FAILED_WITH_DIAGNOSIS` or
`BLOCKED_WITH_DIAGNOSIS`, but not runtime PASS. A diagnosed failure can satisfy
handoff discipline; it cannot satisfy the run's success criterion.

## 12. Result Interpretation

This fast validation can support only the following conclusion types:

| Evidence | Allowed conclusion |
| --- | --- |
| Both runs fail before useful training | Implementation/runtime blocker, not method failure |
| Runs complete but validation collapses, response lengths saturate, or grad norms explode | Method risk; inspect rollout distribution and fused-loss gradient scale |
| A matched-beta run improves or preserves MATH-500 mean@3 over its Stage 1 Model2 baseline | Positive early signal for Model2-rollout fused-loss training for that beta |
| `s2-from-s1-beta0-beta0` and `s2-from-s1-beta01-beta01` diverge materially | Stage 1 beta and Stage 2 beta choice matter; schedule a follow-up Stage 2 matrix |
| Only online validation improves | Promising but not final; offline model2 eval is required before claiming a durable result |
| Cross-beta diagnostic runs differ from matched-beta runs | Exploratory signal only; not part of this plan's PASS criteria |

Do not claim final generalization from this plan alone. The non-overlap shard
removes prompt reuse with Stage 1, but the experiment is still a short
75-step online validation.

Default comparison rule:

```text
Stage 2 score = best_checkpoint.json metric_value for val-core/HuggingFaceH4/MATH-500/acc/mean@3
Stage 1 baseline = selected Stage 1 best-checkpoint metric for the same Model2 input
improve = Stage 2 score > Stage 1 baseline
preserve = Stage 2 score >= Stage 1 baseline - 0.005
regress = Stage 2 score < Stage 1 baseline - 0.005
```

Final-step metrics must still be reported, but the default success comparison is
best-vs-best by the configured online validation metric.

## 13. Done Definition

This plan is done only when:

- The data shard and manifest exist and pass verification.
- The Stage 2 wrappers pin the exact Stage 1 checkpoint inputs.
- The scripts prove Model2-only rollout and fused-logit training, or clearly
  record the implementation gap before launch.
- Both Model 1 and Model 2 are trainable.
- Stage 2 beta is matched to the selected Stage 1 beta for the two default
  acceptance runs.
- Actor update uses a single mini-batch per rollout batch:
  `TRAIN_PROMPT_MINI_BSZ = TRAIN_PROMPT_BSZ * ROLLOUT_N` as rendered into
  `actor_rollout_ref.actor.ppo_mini_batch_size`.
- The two default 75-step runs complete for runtime PASS. If either run fails,
  the plan can only be handed off as `FAILED_WITH_DIAGNOSIS` or
  `BLOCKED_WITH_DIAGNOSIS`, not completed.
- Results are summarized in one table against their Stage 1 baselines.
- The status file is current enough for another agent to resume without chat
  history.
- The training script index, active plan index, and bridge docs mention this
  Stage 2 plan and its runnable scripts.
