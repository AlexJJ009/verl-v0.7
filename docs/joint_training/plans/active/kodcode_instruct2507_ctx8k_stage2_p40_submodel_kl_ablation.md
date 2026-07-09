# KodCode Instruct2507 CTX8K Stage2 P40 Submodel KL Ablation

- Status: ACTIVE EXPERIMENT DESIGN, SCRIPT CHANGES PENDING MAIN AGENT DECISION
- Created: 2026-07-01
- Branch: `feature/on-policy-wdl-sft`
- Recipe family: `recipe/on_policy_wdl_sft/code_task/`
- Scope: design only; do not launch training from this document

## Objective

Test whether adding per-model KL regularization to Stage2 improves stability for
the current KodCode Instruct2507 CTX8K P40 handoff line.

The question is narrow:

```text
Given the existing Stage1 step40 -> Stage2 effective step100 no-KL baseline,
does constraining model1, model2, or both submodels with KL preserve the P40
Stage2 gain while reducing late Stage2 drift?
```

Stage1 is unchanged. It continues to use the existing single-model on-policy SFT
run. Only Stage2 receives the new KL intervention.

## Existing Baseline To Inherit

The no-KL baseline already exists and should be treated as the control rather
than rerun by default:

| Control | Setting |
| --- | --- |
| Handoff | Stage1 beta `0.1` step `40` |
| Stage2 duration | `60` steps, effective step `100` |
| Stage2 beta | `0.1` |
| `fusion_lambda` | `0.8` |
| KL | off for both submodels |
| Main comparison | P40 `lambda=0.8` fresh effective100 no-KL result |

This baseline is represented in the P40 lambda-sweep family, not the older
40-step P40 matched-beta wrapper. The formal ablation must therefore inherit the
fresh effective100 lambda-sweep settings:

- `STAGE2_HANDOFF_STEP=40`
- `TOTAL_TRAINING_STEPS=60`
- `WDL_SFT_BETA=0.1`
- `FUSION_LAMBDA=0.8`
- `LOSS_MODE=wdl_sft`
- `LR=5e-7`
- `TRAIN_PROMPT_BSZ=64`
- `ROLLOUT_N=8`
- `TRAIN_PROMPT_MINI_BSZ=512`
- `DATA_SEED=20260604`
- `DATA_SHUFFLE=False`
- `JOINT_TRAINING_ROLLOUT_SOURCE=model2`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=4096`
- `ROLLOUT_MAX_MODEL_LEN=8192`
- `LOG_PROB_MAX_TOKEN_LEN_PER_GPU=8192`
- `ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192`
- `ACTOR_PPO_MAX_TOKEN_LEN=8192`
- `VAL_N=1`
- `VAL_TEMPERATURE=0.2`
- `VAL_TOP_P=0.95`
- `TEST_FREQ=5`
- `SAVE_FREQ=10` unless storage forces a different explicit retention choice
- online validation on HumanEval+, MBPP+, and LiveCodeBench v5 subset128

The no-KL baseline must be reported from the already completed run with the same
online/offline eval protocol. Do not compare against a 40-step Stage2 run as if
it were the same effective-step control.

## Experiment Matrix

Run the three KL groups across the `fusion_lambda=0.5..0.9` grid:

| Run label | Model1 KL | Model2 KL | Purpose |
| --- | --- | --- | --- |
| `P40-SUBKL-M1` | on | off | Test whether stabilizing the weak/currently-sacrificed side is enough. |
| `P40-SUBKL-M2` | off | on | Test whether constraining the handoff Model2 preserves benchmark skill. |
| `P40-SUBKL-BOTH` | on | on | Test whether symmetric constraints improve final stability without blocking useful Stage2 movement. |

For each run label, sweep:

```text
fusion_lambda = 0.5, 0.6, 0.7, 0.8, 0.9
```

All 15 runs use the same Stage1 source, Stage2 data shard, rollout source, beta,
training length, validation, and decode settings as the no-KL baseline. The
`lambda=0.8` rows are the direct KL-vs-no-KL comparison against the existing
fresh effective100 no-KL baseline; the other lambda rows test whether KL changes
the best logit-fusion balance.

## KL Settings

Use per-submodel actor-loss KL, not KL reward penalty:

```text
L_total = L_wdl
        + c1 * KL(model1_current || model1_ref)
        + c2 * KL(model2_current || model2_ref)
```

Default reference paths:

| Submodel | Reference |
| --- | --- |
| model1 | Stage2 initial weak/base model, `BASE_MODEL_PATH` |
| model2 | Stage1 step40 merged Model2, `MERGED_MODEL2_DIR` |

Recommended first-pass KL type:

- `SUBMODEL_KL_*_TYPE=low_var_kl`

Recommended first-pass coefficient:

- `SUBMODEL_KL_MODEL1_COEF=0.01`
- `SUBMODEL_KL_MODEL2_COEF=0.01`

This coefficient is a design default, not a settled result. Main Agent should
confirm it from the 5-step submodel-KL smoke matrix before launching the formal
60-step ablation. If the smoke shows KL loss scale is too small to matter or
dominates `L_wdl`, adjust the coefficient before this formal matrix starts.

Concrete environment settings:

| Group | Top switch | Model1 switch/coef | Model2 switch/coef |
| --- | --- | --- | --- |
| `P40-SUBKL-M1` | `SUBMODEL_KL_ENABLED=true` | `true`, `0.01` | `false`, `0.0` |
| `P40-SUBKL-M2` | `SUBMODEL_KL_ENABLED=true` | `false`, `0.0` | `true`, `0.01` |
| `P40-SUBKL-BOTH` | `SUBMODEL_KL_ENABLED=true` | `true`, `0.01` | `true`, `0.01` |

Keep legacy global KL off:

- `actor_rollout_ref.actor.use_kl_loss=False`
- `algorithm.use_kl_in_reward=False`

## Variable Control

These must not change across the no-KL baseline and the three new KL runs:

- Stage1 source prefix:
  `ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1`
- Stage1 checkpoint root:
  `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871`
- handoff checkpoint: `global_step_40/actor`
- Stage2 train shard:
  `/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_p40_handoff_s2steps60.parquet`
- Stage2 non-overlap manifest matched to Stage1 step40 and Stage2 60 steps
- base/weak model path:
  `/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- merged Model2 provenance file: `stage1_source.json`
- W&B project and offline mode unless release-gate approval explicitly changes
  publication behavior

Use distinct `RUN_PREFIX`, `MODEL2_CACHE_TAG`, and checkpoint dirs for each KL
mode. Do not reuse the no-KL baseline prefix or the 5-step smoke prefixes.

## Metrics And Diagnostics

Report online validation separately from official offline eval.

Online training-health metrics:

- HumanEval+ `pass@1`
- MBPP+ `pass@1`
- LiveCodeBench v5 subset128 `pass@1`
- `wdl_sft/correct_ratio`
- `response_length/clip_ratio`
- `response/aborted_ratio`
- `actor/grad_norm`
- `actor/lr`
- `actor/submodel_kl/model1_loss`
- `actor/submodel_kl/model1_coef`
- `actor/submodel_kl/model2_loss`
- `actor/submodel_kl/model2_coef`
- `actor/submodel_kl/total_loss`

Official offline eval for candidate checkpoints:

- HumanEval+
- MBPP+
- BigCodeBench
- LiveCodeBench `release_v5`
- `N_SAMPLES=3`
- `TEMPERATURE=1.0`
- `TOP_P=0.95`
- `MAX_TOKENS=4096`
- `SEED=42`
- `ENABLE_THINKING=true`
- report both `mean@3` and `pass@3`

Evaluate at least:

- no-KL baseline final/best checkpoint already available;
- each KL run final checkpoint at Stage2 step60;
- each KL run best online checkpoint if it differs from final by a meaningful
  margin.

## Acceptance Criteria

### Implementation acceptance

Before launch, accept the script changes only if:

- wrappers exist for the formal KL lambda sweep or a queue can pass distinct
  env overrides without ambiguity for all 15 rows;
- dry-run prints all inherited P40 effective100 settings, including
  `FUSION_LAMBDA={0.5,0.6,0.7,0.8,0.9}` and
  `TOTAL_TRAINING_STEPS=60`;
- dry-run prints resolved model1/model2 KL reference paths;
- Stage2 provenance guard still verifies Stage1 prefix, checkpoint root,
  handoff step, merged Model2 dir, and train shard;
- each run has a distinct `RUN_PREFIX` and `MODEL2_CACHE_TAG`;
- the queue refuses non-dry-run launch unless an explicit allow flag is set;
- monitor support tracks all 15 prefixes to final step60;
- training script index is updated when runnable scripts are created or used;
- Meituan/AFO routing is updated if new runnable wrappers are added.

### Result acceptance

Treat a KL group as useful only if it beats the no-KL baseline on stability
without erasing the Stage2 benefit:

- final online validation does not show late collapse relative to its own peak;
- final checkpoint remains within `3 pp` of the run's best online average across
  HumanEval+, MBPP+, and LCB subset128;
- official offline eval is at least neutral versus no-KL on the main benchmark
  bundle, target `>= 0 pp` average delta and no broad per-benchmark regression;
- response health improves versus no-KL, especially lower
  `response_length/clip_ratio`, fewer aborted samples, and fewer malformed or
  low-quality completions;
- KL losses are nonzero and stable rather than missing, exploding, or dwarfing
  WDL loss.

Strong success requires:

- `>= +1 pp` average official offline gain over no-KL, or
- comparable official offline score with clearly better final stability and
  lower collapse indicators.

## Expected Outcomes

Interpretation table:

| Outcome | Interpretation | Next decision |
| --- | --- | --- |
| model1-only helps most | Stage2 instability likely comes from model1 drifting while fused loss trains both submodels. | Promote model1 KL as the default Stage2 stabilizer and test coefficient sensitivity. |
| model2-only helps most | Handoff Model2 needs anchoring; unconstrained Model2 update may destroy Stage1 code skill. | Keep model2 KL and test whether model1 can stay unconstrained. |
| both-on helps most | Both sides drift enough that symmetric constraints are needed. | Promote both-on, then sweep coefficient. |
| all KL modes underperform | KL either over-constrains useful Stage2 movement or wrong coefficient/type was chosen. | Inspect KL loss scale, try smaller coefficient, or change Stage2 method. |
| KL improves online but not offline | Online subset is not predictive enough for this intervention. | Use official eval as source of truth; do not publish online-only success. |

## Failure Diagnostics

Flag the run for diagnosis if any of these happen:

- KL metrics are absent while `SUBMODEL_KL_ENABLED=true`;
- model1/model2 reference path resolves to the wrong source;
- `SUBMODEL_KL_MODEL2_REF_PATH` points to an old or stale merged Model2 dir;
- no-KL and KL runs accidentally share `MODEL2_CACHE_TAG` or checkpoint prefix;
- final online validation drops by `>= 10 pp` from peak on any core online
  benchmark;
- `response_length/clip_ratio >= 0.35`;
- `response/aborted_ratio` rises materially versus no-KL;
- `actor/grad_norm` becomes persistently extreme after KL turns on;
- KL loss is near zero for the full run despite enabled KL;
- KL loss dominates total actor loss and online reward/correct ratio stalls.

When diagnosing, first compare the first 10 Stage2 steps against no-KL. If KL
changes behavior immediately, inspect loss scale and reference wiring. If early
behavior matches no-KL but final drift improves or worsens, treat it as a real
stability effect.

## Follow-Up Decision Logic

1. If exactly one KL placement wins, keep that placement and run a coefficient
   sweep around it: `0.003`, `0.01`, `0.03`.
2. If both-on wins but is slightly worse than no-KL on offline score, try a
   lower symmetric coefficient before rejecting KL.
3. If model2-only prevents collapse but reduces pass@k, test a smaller model2
   coefficient and keep model1 off.
4. If model1-only improves stability and offline score, make it the next
   default for Stage2 P40/P60 comparisons.
5. If no KL variant beats no-KL, do not spend more Stage2 budget on per-model KL
   until reference wiring, coefficient scale, and KL type are reviewed.

## Script Changes Needed

Minimum script plan:

| File | Required change |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_instruct2507_ctx8k_p40_beta01_subkl_smoke.sh` | Keep as implementation smoke only; do not reuse its 5-step defaults for the formal ablation. |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_instruct2507_ctx8k_p40_beta01_beta01.sh` | Either add formal-subKL override compatibility or leave unchanged and create thin wrappers. |
| `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` | Add a formal queue mode such as `kodcode_instruct2507_ctx8k_stage2_p40_subkl_ablation`, or create a dedicated queue that delegates here. |
| `recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh` | Add monitor mode for the three formal KL prefixes. |
| `recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_ablation_notify.sh` | Thin monitor wrapper if a dedicated entry point is preferred. |
| `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh` | Add experiment routing if new runnable wrappers are created. |
| `platform/hope_code_task/run.hope` | Document the new Meituan experiment names if routed. |
| `docs/joint_training/guides/training_script_index.md` | Register any new runnable wrappers, queue, and monitor after creation/dry-run. |

Recommended formal prefixes:

- `CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL1-LAMBDA08-V1`
- `CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-MODEL2-LAMBDA08-V1`
- `CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-SUBKL-BOTHON-LAMBDA08-V1`

Do not start training from this document alone. Main Agent must first confirm
the KL coefficient, script landing approach, dry-run output, and available disk
budget.
