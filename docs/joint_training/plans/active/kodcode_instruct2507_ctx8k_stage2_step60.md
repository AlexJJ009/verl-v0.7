# KodCode Instruct2507 CTX8K Stage2 Step60 Experiment

- Status: ACTIVE EXPERIMENT DESIGN, SCRIPT CHANGES PENDING APPROVAL
- Created: 2026-06-26
- Branch: `feature/on-policy-wdl-sft`
- Recipe family: `recipe/on_policy_wdl_sft/code_task/`
- Scope: design only; do not launch training from this document

## Objective

Run a matched-beta code-task Stage2 experiment:

```text
KodCode Instruct2507 CTX8K Stage1 step60 -> fixed Model2 handoff -> Stage2 40 steps
```

The narrow question is whether Stage2 can improve the KodCode Instruct2507 CTX8K
line at effective step 100, while preserving a fair comparison against the
same-budget Stage1 continuation checkpoint at step 100.

This is a script-landing plan, not a launch approval. The current repository
has generic code-task Stage2 retention scripts, but they target the older
retention line defaults (`step70`, `30` Stage2 steps). The Step60 CTX8K
experiment needs explicit wrappers, p60 non-overlap shards, provenance, and
monitor/index updates before any training starts.

## Current Evidence and Preconditions

Read-only checks on 2026-06-26 found local Stage1 checkpoint roots for both
matched-beta inputs:

| Beta | Stage1 run prefix | Observed checkpoint root | Observed step60 actor |
| ---: | --- | --- | --- |
| `0.0` | `ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1` | `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1_1782371396` | present |
| `0.1` | `ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1` | `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871` | present |

These paths are launch inputs only after Main Agent re-verifies them. They do
not by themselves satisfy the experiment gate.

Existing dataset shards under `/data-1/dataset/code/verl_rl/` include generic
and `p70` handoff files. There is no clearly named `p60` Stage2 shard yet. The
Step60 experiment must create or verify explicit p60 shards before launch.

## Primary Matrix

Keep the first pass small and matched-beta only:

| Run label | Stage1 beta | Weak/model1 init | Strong/model2 handoff | Stage2 beta | Stage2 steps | Effective step |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `KOD-I2507-CTX8K-P60-B0` | `0.0` | Instruct2507 init | Stage1 beta `0.0` model2 at step 60 | `0.0` | `40` | `100` |
| `KOD-I2507-CTX8K-P60-B01` | `0.1` | Instruct2507 init | Stage1 beta `0.1` model2 at step 60 | `0.1` | `40` | `100` |

Fixed defaults:

- `TOTAL_TRAINING_STEPS=40`
- `STAGE2_HANDOFF_STEP=60`
- `TRAIN_PROMPT_BSZ=64`
- `ROLLOUT_N=8`
- `TRAIN_PROMPT_MINI_BSZ=512`
- `DATA_SEED=20260604`
- `DATA_SHUFFLE=False`
- `JOINT_TRAINING_ROLLOUT_SOURCE=model2`
- `LOSS_MODE=wdl_sft`
- `LR=5e-7`
- `VAL_N=1`
- `VAL_TEMPERATURE=0.2`
- `VAL_TOP_P=0.95`
- `SAVE_FREQ=10`
- `TEST_FREQ=5`

Do not add mixed-beta Stage2 chains in the first pass. Cross-beta variants are
diagnostic follow-up only after the matched-beta result is interpreted.

## Comparison Protocol

### Online validation

Online validation is training-health evidence, not final benchmark truth.
Report it separately from offline official evaluation.

Use the same online validation files as the Stage1 KodCode Instruct2507 CTX8K
line unless Main Agent intentionally changes them:

- HumanEval+ full online parquet
- MBPP+ full online parquet
- LiveCodeBench v5 subset128 online parquet, if kept in the Stage1 control

The earlier Stage1 wrapper kept `MAX_RESPONSE_LENGTH=4096` while setting
`ROLLOUT_MAX_MODEL_LEN=8192`. For this Stage2 plan, keep CTX8K explicit and
verify that online validation also runs with an 8K max model length where the
runner supports it:

- `ROLLOUT_MAX_MODEL_LEN=8192`
- `LOG_PROB_MAX_TOKEN_LEN_PER_GPU=8192`
- `ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192`
- `ACTOR_PPO_MAX_TOKEN_LEN=8192`

If online validation falls back to a 4K max model length anywhere, mark those
numbers as non-comparable. Do not mix 4K-online and 8K-online curves as if they
were the same metric.

### Same-budget Stage1 continuation

The clean control for each beta is the Stage1 continuation checkpoint at step
100, not Stage1 step60 and not Stage1 latest/best:

| Stage2 result | Required same-budget control |
| --- | --- |
| `beta=0.0`, Stage1 step60 + Stage2 step40 | Stage1 `beta=0.0` step100 |
| `beta=0.1`, Stage1 step60 + Stage2 step40 | Stage1 `beta=0.1` step100 |

Interpret Stage2 as useful only if it beats the matching Stage1 step100 control
under the same eval protocol.

### Official offline eval

Final comparison must use official offline eval, reported separately from
online subset metrics. The default code-task official-eval口径 remains:

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

For CTX8K, Main Agent should decide whether official generation should keep
`MAX_TOKENS=4096` for continuity with prior fair evals, or add a second 8K
diagnostic pass. Do not compare a 4096-token offline result against an 8192-token
result without labeling the token budget difference.

## Expected Result Bands

Use same-budget official offline eval as the main decision criterion.

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| `>= +3 pp` over Stage1 step100 on the main official metric bundle | Strong evidence that Step60 Stage2 is worth continuing. | Promote Step60 as the primary Stage2 handoff for this line; run full official eval and release-gate bookkeeping. |
| `+1` to `+2 pp` | Weak positive. Could be noise or benchmark-specific. | Do not declare method success; inspect per-benchmark deltas, response health, and maybe rerun one adjacent handoff step. |
| `<= 0 pp` or mixed/unstable result | No useful Step60 Stage2 lift. | Search another handoff step or change the Stage2 method before spending more budget. |

Online validation can flag collapse or obvious improvement, but it cannot
replace official offline eval.

## Follow-up Search Plan

### If step60 is good

If at least one matched-beta run shows a clean `>= +3 pp` official offline gain
and no health regression:

1. Run official offline eval for Stage2 peak and final if they differ.
2. Compare against Stage1 step100 with the same decode settings.
3. Run the other beta if only one beta was launched first.
4. Consider a shorter Stage2 duration around the observed peak if final step40
   trails peak by more than 1 pp.
5. Only then test adjacent handoffs: `step70`, then `step90`.

### If step60 is weak or bad

If step60 gives `+1` to `+2 pp`, no gain, or visible format/length drift:

1. Check whether the Stage2 peak appears before step40.
2. If peak is early and then declines, test shorter Stage2, not a later handoff.
3. If Stage2 is flat from the start, test a stronger handoff: `step70` or
   `step90`.
4. If Stage2 destabilizes immediately, change Stage2 settings before searching
   more checkpoints.

### Candidate handoff steps

Use these steps as a bounded search order, not a launch queue:

| Step | Role |
| ---: | --- |
| `60` | Primary early handoff; this plan. |
| `70` | Closest existing retention precedent; useful if step60 is undertrained. |
| `90` | Mid-curve handoff; tests whether more Stage1 competence is needed. |
| `115` | Known beta `0.1` candidate region in prior code-task discussions; use only after early steps are interpreted. |
| `125` | Late-mid diagnostic for beta `0.0` if curves suggest delayed peak. |
| `145` | Near-best/late Stage1 diagnostic; higher risk of Stage2 overfit or no added value. |

Each new handoff step requires its own non-overlap Stage2 shard and provenance
manifest. Do not reuse the p60 shard for another step.

## Model Configuration Plan

### Default: homogeneous Instruct2507

Keep the main line homogeneous:

- model1 weak/init: Qwen3-4B-Instruct-2507 init
- model2 strong/handoff: model2 extracted from the matching Stage1 step60 actor
- both submodels trainable under the existing joint Stage2 fused-loss path

This keeps the Stage2 test focused on whether the same Instruct2507 family gains
from fixed-Model2 handoff, not on architecture or model-lineage changes.

### Weak=model init, strong=step60 model2

Default Stage2 construction should be:

| Role | Source |
| --- | --- |
| weak/model1 | clean Instruct2507 init |
| strong/model2 | Stage1 step60 model2 from the matching beta run |

This matches the intended handoff logic: model2 carries the Stage1 improvement,
while model1 remains the weak side of the fused distribution.

### Optional: Step0 weak + Step40/60 strong

If the default homogeneous Step60 plan is inconclusive, test whether the gap
between weak and strong is the useful signal:

- weak/model1: clean Instruct2507 init, i.e. Step0
- strong/model2: Stage1 Step40 or Step60 model2

Step40 is a useful diagnostic if step60 seems already too aligned with model1 or
if Stage2 needs a smaller competence gap. Do not add this before the primary
matched-beta Step60 result.

### Optional: Instruct init weak

If scripts reveal that model1 is accidentally loaded from a trained Stage1 actor
or old base-model path, force `INIT_MODEL_PATH` / model1 source back to the
Instruct2507 init and rerun dry-run checks. The model-lineage control matters
more than matching old KodCode Base-era defaults.

### Deferred: heterogeneous 1.7B weak

Do not include a Qwen3-1.7B weak model in this matrix yet.

Reasons to defer:

- It changes model size, tokenizer/model config assumptions, and possibly joint
  wrapper behavior at the same time as the Stage2 handoff test.
- It complicates Meituan path overrides and flat-model upload requirements.
- It makes any gain hard to attribute: handoff timing, CTX8K, Instruct2507, and
  heterogeneity would all change together.

Heterogeneous weak models belong in a separate ablation after homogeneous
Step60 either succeeds or fails cleanly.

## Proposed Script Landing

Do not create these files until Main Agent or the user approves script changes.

### New wrappers

| File | Purpose |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_instruct2507_ctx8k_p60_beta0_beta0.sh` | Stage2 from KodCode Instruct2507 CTX8K Stage1 beta `0.0` step60 to Stage2 beta `0.0`, 40 steps. |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_instruct2507_ctx8k_p60_beta01_beta01.sh` | Matched beta `0.1` version. |
| `recipe/on_policy_wdl_sft/code_task/run_code_task_kodcode_instruct2507_ctx8k_stage2_p60_queue.sh` | Host-side sequential queue for the two p60 matched-beta Stage2 runs. |
| `recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_p60_notify.sh` | Thin monitor entry point delegating to `scripts/training_queue_monitor.sh`. |

### Wrapper defaults

Recommended key overrides:

```bash
export RUN_PREFIX=${RUN_PREFIX:-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1}
export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1_1782371396}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export STAGE2_HANDOFF_STEP=${STAGE2_HANDOFF_STEP:-60}
export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta0_p60_handoff.parquet}
export MODEL2_LABEL=${MODEL2_LABEL:-kodcode_i2507_ctx8k_beta0_p60}
export STAGE1_MERGED_MODEL_ROOT=${STAGE1_MERGED_MODEL_ROOT:-/data-1/model_weights/code_task/kodcode_instruct2507_ctx8k_stage2_p60/${MODEL2_LABEL}}
```

The beta `0.1` wrapper should mirror this with:

- `RUN_PREFIX=CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA01-BETA01`
- `STAGE1_RUN_PREFIX=ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1`
- `STAGE1_CKPT_DIR=/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871`
- `WDL_SFT_BETA=0.1`
- `CODE_TRAIN_FILE=/data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta01_p60_handoff.parquet`
- `MODEL2_LABEL=kodcode_i2507_ctx8k_beta01_p60`

Use `run_s2_code_model2_rollout_common.sh` as the shared launcher. Keep
default-local, overridable-everything semantics for every path so Meituan can
override them from `recipe/on_policy_wdl_sft/code_task/meituan/env.sh`.

### Queue defaults

The p60 queue should:

- require `ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P60_TRAINING=1` for
  non-dry-run;
- default `DRY_RUN=1` behavior to print and validate wrappers only;
- set `STAGE2_TOTAL_TRAINING_STEPS=40`;
- set `STAGE2_BETA0_HANDOFF_STEP=60`;
- set `STAGE2_BETA01_HANDOFF_STEP=60`;
- create or verify p60 non-overlap shards with
  `create_code_stage2_nonoverlap_shard.py`;
- refuse partial checkpoint collisions unless `ALLOW_RESUME=1`;
- write a distinct status file, for example
  `run_code_task_kodcode_instruct2507_ctx8k_stage2_p60_queue_status.tsv`;
- use distinct tmux names from the older retention queue.

### Required p60 shard commands

The queue should create or verify the shards with arguments equivalent to:

```bash
python3 recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py \
  --output /data-1/dataset/code/verl_rl/kodcode_stage2_after_s1_seed20260604_beta0_p60_handoff.parquet \
  --seed 20260604 \
  --stage1-steps 60 \
  --stage1-train-batch-size 64 \
  --stage2-steps 40 \
  --stage2-train-batch-size 64
```

Repeat for beta `0.1` with the matching output path. The two shards can have
the same sampler geometry but should remain separately named so provenance
points to the correct beta chain and future searches do not collide.

### Meituan changes

If Main Agent decides this experiment must be Meituan-runnable immediately,
update the existing code-task Layer 3 files:

| File | Required change |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/meituan/env.sh` | Add p60 train-file env vars such as `CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA0_TRAIN_FILE` and `CODE_STAGE2_KODCODE_I2507_CTX8K_P60_BETA01_TRAIN_FILE`; add any p60 model-weight root override. |
| `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh` | Add `EXPERIMENT` cases for the two p60 wrappers and require `SANDBOX_FUSION_URL` for non-smoke Stage2. |
| `platform/hope_code_task/run.hope` | Only update if the submitted `EXPERIMENT` template needs a new default or documented example. |

Do not create a new Meituan family. This belongs in the existing `code_task`
family.

### Training script index

Do not update `docs/joint_training/guides/training_script_index.md` for this
design-only document. Update it when runnable wrappers, queue, or monitor files
are actually created or when an existing script is materially changed/used.

## Risks and Emergency Handling

| Risk | Symptom | Required handling |
| --- | --- | --- |
| OOM from CTX8K Stage2 | vLLM init failure, CUDA OOM, Ray worker death, or reward workers starving GPUs | Do not lower context silently. First record exact failing setting, then decide between lower `ROLLOUT_GPU_MEMORY_UTILIZATION`, smaller generation/logprob micro-batches, or reduced validation load. |
| Disk pressure | free space below queue gate, checkpoint write failures, stale merged dirs | Keep `MIN_FREE_GB>=300` for the queue. Do not delete W&B, checkpoints, or model weights without explicit verified cleanup approval. |
| Validation口径 mismatch | online 4K vs 8K mixed curves, subset vs official confusion | Label online metrics as online/subset; keep official offline eval separate. Re-run mismatched eval rather than normalizing by prose. |
| Checkpoint provenance drift | `stage1_source.json` mismatch, stale merged Model2 dir, wrong beta source | Stage2 must fail closed unless provenance matches `stage1_run_prefix`, `source_checkpoint`, `handoff_step=60`, actor dir, target dir, and train manifest. |
| Non-overlap failure | Stage2 shard overlaps Stage1 consumed prefix or row count differs | Treat as a launch blocker. Regenerate the shard; do not train on an ambiguous data split. |
| Release gate block | run stops before final checkpoint+metrics | Failed/incomplete training is local diagnostic evidence only. Do not import DB records or sync W&B cloud as a successful run. |
| W&B staging ambiguity | multiple offline-run dirs or missing final exit record | Preserve local W&B by default; sync only the final verified run after release-gate check. |
| Offline eval instability | BigCodeBench groundtruth issues, LCB scorer quiet CPU phase, extraction failures | Use official summaries as the success artifact, record `usable_with_caution` where stderr/harness health requires it, and report `mean@3` plus `pass@3`. |

## Acceptance Criteria

### Script readiness

Before any real training launch, Main Agent must verify:

- p60 wrappers exist and pass shell syntax checks;
- p60 queue exists and `DRY_RUN=1` validates both wrappers;
- no command writes scratch or durable artifacts into repo root;
- all path roots are overrideable for Meituan;
- `STAGE2_HANDOFF_STEP=60` and `TOTAL_TRAINING_STEPS=40` are printed in dry-run config;
- the queue refuses non-dry-run without
  `ALLOW_KODCODE_INSTRUCT2507_CTX8K_STAGE2_P60_TRAINING=1`;
- training script index is updated after runnable files are created.

### Checkpoint and provenance

Before launch:

- both Stage1 checkpoint roots exist;
- both `global_step_60/actor` dirs contain complete world-size-8 FSDP shards;
- merged Model2 target dirs are distinct from older generic/p70 retention dirs;
- `MERGED_MODEL2_PROVENANCE_FILE` is written or checked;
- stale merged dirs without weights fail unless explicit overwrite is approved;
- model2 extraction targets are reusable only if provenance exactly matches.

### Non-overlap data

Before launch:

- p60 beta0 shard exists with manifest;
- p60 beta0.1 shard exists with manifest;
- each manifest records `handoff_step=60`, `consumed_rows=3840`,
  `stage2.steps=40`, and `rows_requested=2560`;
- verifier reports zero overlap against the Stage1 consumed prefix;
- shard names include `p60` to prevent accidental p70 reuse.

### Monitoring and release gate

During and after launch:

- queue runs in tmux;
- monitor delegates to `scripts/training_queue_monitor.sh`;
- final completion requires checkpoint step40 plus metrics evidence;
- failed or incomplete attempts are recorded as local diagnostic evidence;
- DB import and W&B cloud sync require:

```bash
python3 scripts/training_result_release_gate.py check --run-name <RUN_NAME>
```

### Result decision

Stage2 Step60 is accepted as useful only if official offline eval shows a
meaningful same-budget gain and the run remains healthy:

- primary target: at least one beta chain improves `>= +3 pp` over matching
  Stage1 step100;
- cautious target: `+1` to `+2 pp` requires per-benchmark confirmation or an
  adjacent-step repeat;
- reject/redirect: no gain, obvious drift, or inconsistent official metrics;
- never claim success from online subset metrics alone.

## Main Agent Decisions Still Needed

1. Whether to implement p60 scripts now, or keep this document as a design
   checkpoint until the current offline eval queue finishes.
2. Whether official offline eval for CTX8K should stay at `MAX_TOKENS=4096` for
   continuity or add a separate 8K diagnostic pass.
3. Whether Meituan routing must be updated in the same patch as local wrappers.
4. Whether to launch beta `0.0` first as a single-run probe or queue both
   matched-beta chains sequentially after dry-run acceptance.
5. Which metric bundle becomes the primary result criterion if HumanEval+/MBPP+
   and LiveCodeBench move in different directions.
