# Plateau Handoff Stage1 -> Stage2 Experiment

- Status: ACTIVE EXPERIMENT PLAN, IMPLEMENTATION READY
- Created: 2026-06-02
- Branch: `feature/on-policy-wdl-sft`
- Recipe family: `recipe/on_policy_wdl_sft/staged_v1/`
- Parent evidence: `boxed_matched_stage1_stage2_chain.md`

## Objective

Test whether entering Stage2 earlier, near the Stage1 plateau rather than at
the Stage1 best checkpoint, preserves the early On-policy WDL-SFT gain while
reducing late SFT drift/collapse.

The intended claim is narrow:

```text
If Model2 is taken from an early Stage1 plateau checkpoint and Stage2 is kept
short, then Stage2 may deliver the observed +1 to +2 pp local intervention gain
without the severe late collapse seen in 75-step Stage2 runs.
```

This experiment is also a settings-debug run. It should tell us whether the
current Stage2 formulation is fundamentally unstable even under a shorter
handoff gap, or whether the prior collapse was largely an over-training /
too-late-handoff effect.

## Motivation

Recent boxed matched runs showed two facts at the same time:

1. Stage2 can improve online validation soon after handoff.
2. Stage2 collapses if trained too long.

The latest boxed matched chains used best-checkpoint handoff:

| Chain | Stage1 source | Stage2 peak | Stage2 final |
| --- | ---: | ---: | ---: |
| `beta=0.0 -> beta=0.0` | step 135, 71.03% | step 15, 70.77% | step 75, 0.20% |
| `beta=0.1 -> beta=0.1` | step 115, 70.90% | step 30, 72.24% | step 75, 2.35% |

The collapse was not a verifier-only artifact: raw validation outputs showed
long generations, repeated text, mixed-language/garbled content, url-like or
HTML/CSS fragments, and frequent missing `\boxed{}` answers. It should be
treated as distribution collapse / late-format drift, not merely answer-parser
noise.

Stage1 curves from the boxed rerun suggest that the plateau begins around
steps 50-60:

| Run | step 40 | step 50 | step 55 | step 60 | later peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage1 boxed `beta=0.0` | 64.11% | 67.20% | 68.28% | 68.62% | 71.03% at step 135 |
| Stage1 boxed `beta=0.1` | 63.51% | 65.93% | 65.86% | 69.56% | 70.90% at step 115 |

The hypothesis is that handoff near this plateau may reduce the Model1/Model2
gap while preserving enough Model2 competence for Stage2 to amplify.

## Primary Hypotheses

### H1: Earlier handoff reduces late collapse

Taking Model2 from Stage1 step 60 should reduce the Stage2 tendency to enter
long, malformed, non-boxed generation compared with best-checkpoint handoff.

Evidence to support H1:

- `response_length/clip_ratio` remains controlled through Stage2 step 40.
- extraction-failure rate stays low enough for validation metrics to remain
  meaningful.
- final Stage2 step 40 is close to peak rather than collapsed.

### H2: Short Stage2 preserves local intervention gain

Short Stage2 training should still reproduce or improve the early intervention
effect seen in the previous four Stage2 runs.

Evidence to support H2:

- Stage2 peak MATH-500 `mean@3` improves over its Stage1 source checkpoint.
- Stage2 peak improves over same-budget Stage1 continuation at the aligned
  effective step.
- The improvement is larger than noise, target `>= +1.0 pp`.

### H3: Collapse behavior diagnoses current Stage2 settings

If collapse still happens by Stage2 step 30-40, then the current Stage2
`model2 rollout + fused loss + both submodels trainable` setting is unstable
even under shorter, earlier handoff. In that case, future work should change
the Stage2 objective or constraints rather than only tune duration.

## Experiment Matrix

### Primary matrix

Run the smallest matrix that can answer the main question:

| Run | Stage1 beta | Stage1 handoff | Stage2 beta | Stage2 steps | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `P60-B0` | 0.0 | step 60 | 0.0 | 40 | Main early-plateau test without reverse term |
| `P60-B01` | 0.1 | step 60 | 0.1 | 40 | Main early-plateau test with matched reverse term |

Use the boxed-prompt training data and the same validation protocol as the
boxed matched chain. Keep matched beta only. Do not add mixed-beta chains in
the first pass.

### Optional ablations

Only run these after the primary matrix is interpreted:

| Run | Stage1 beta | Stage1 handoff | Stage2 beta | Stage2 steps | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `P50-B0` | 0.0 | step 50 | 0.0 | 40 | Test whether earlier-than-plateau handoff works |
| `P50-B01` | 0.1 | step 50 | 0.1 | 40 | Same for reverse-term chain |
| `P60-B0-S2-30` | 0.0 | step 60 | 0.0 | 30 | Check whether the best Stage2 budget is shorter |
| `P60-B01-S2-30` | 0.1 | step 60 | 0.1 | 30 | Same for reverse-term chain |

The optional matrix should not be launched automatically as part of the first
queue unless the primary results are inconclusive.

## Fixed Settings

Keep these fixed to isolate handoff timing and Stage2 duration:

- boxed-prompt Stage1 train data;
- Stage2 non-overlap train shard policy;
- `DATA_SEED=20260528`;
- `TRAIN_PROMPT_BSZ=64`;
- `ROLLOUT_N=8`;
- `TRAIN_PROMPT_MINI_BSZ=512`;
- `VAL_N=3`;
- `TEST_FREQ=5`;
- `SAVE_FREQ=5`;
- `MAX_RESPONSE_LENGTH=4096`;
- `VAL_BEFORE_TRAIN=False`;
- `JOINT_TRAINING_ROLLOUT_SOURCE=model2`;
- matched Stage1/Stage2 beta only.

If any setting differs, record it in the result table and do not compare it as
a clean plateau-handoff result.

## Metrics To Report

For each run, report:

- Stage1 source step and validation metrics;
- Stage2 peak step and metrics;
- Stage2 final step and metrics;
- same-budget comparison against Stage1 continuation;
- MATH-500 `mean@3`, `best@3`, `maj@3`;
- AIME-2025 `mean@3`, `best@3`;
- `wdl_sft/correct_ratio`;
- `response_length/clip_ratio`;
- `response/aborted_ratio`;
- `jointTraining/answer_extraction_failure_rate`;
- `actor/grad_norm`;
- `rollout_corr/kl`;
- `training/rollout_actor_probs_pearson_corr`;
- raw validation sample audit at peak and final.

The result should distinguish:

```text
best/peak checkpoint evidence
final checkpoint stability
raw-output health
```

Do not claim success from peak metrics alone if the final checkpoint has already
collapsed.

## Collapse Criteria

Mark a Stage2 run as collapsed if any of these hold after an initial healthy
window:

- MATH-500 `mean@3` drops by `>= 10 pp` from Stage2 peak;
- `response_length/clip_ratio >= 0.35`;
- `jointTraining/answer_extraction_failure_rate >= 0.20`;
- raw validation shows broad malformed-output patterns: long repetition,
  mixed-language/garbled text, url-like or HTML/CSS fragments, or widespread
  missing `\boxed{}` answers;
- `actor/grad_norm` becomes persistently extreme relative to earlier healthy
  steps.

If collapse happens before step 40, the final checkpoint should be treated as
negative evidence for stability even if an earlier peak improved.

## Acceptance Criteria

### Implementation acceptance

Before launching real training, the implementation is accepted only if:

- runnable wrappers exist for the primary matrix:
  - `run_s1_plateau_p60_beta_0.sh`;
  - `run_s1_plateau_p60_beta_01.sh`;
  - `run_s2_plateau_p60_beta0_beta0.sh`;
  - `run_s2_plateau_p60_beta01_beta01.sh`;
- the primary queue exists and runs only `P60-B0` then `P60-B01`:
  - Stage1 fixed to 60 steps;
  - Model2 merge fixed to Stage1 step 60, not `best_checkpoint.json`;
  - Stage2 fixed to 40 steps;
  - matched beta only;
- fixed Model2 directories are distinct from the prior best-checkpoint
  `boxed_matched` dirs;
- every merge writes/checks `stage1_source.json` with `stage1_step=60`;
- Stage2 wrappers set `REQUIRE_MERGED_MODEL2_PROVENANCE=True`;
- the queue refuses incomplete checkpoint collisions unless `ALLOW_RESUME=1`;
- the queue refuses stale merged Model2 dirs unless
  `ALLOW_OVERWRITE_MERGED_MODEL2=1`;
- the queue waits for final Stage2 checkpoint and final metrics before marking
  a chain complete;
- the training script index documents the new runnable wrappers and queue.

Implementation acceptance is a launch-readiness standard. It does not imply the
experiment succeeded.

### Result acceptance

The primary matrix is successful if at least one run satisfies all of:

- Stage2 peak improves over Stage1 source by `>= +1.0 pp` MATH-500 `mean@3`;
- Stage2 peak improves over same-budget Stage1 continuation by `>= +1.0 pp`;
- Stage2 final step 40 remains within `5 pp` of Stage2 peak;
- final extraction-failure rate `< 10%`;
- final response-length clip ratio `< 20%`;
- raw final validation samples are mostly coherent mathematical answers with
  extractable `\boxed{}` output.

The primary matrix is a useful negative result if:

- Stage2 still collapses by step 30-40 for both beta values; or
- no run improves over the Stage1 source / same-budget continuation, even
  before collapse.

## Implementation Notes

Do not launch this experiment until scripts have explicit provenance handling:

- Stage1 checkpoint selection must be fixed by step number, not by
  `best_checkpoint.json`.
- the merged Model2 directory must write `stage1_source.json` with the source
  step and checkpoint path;
- Stage2 scripts must refuse stale merged Model2 dirs unless overwrite is
  explicitly enabled;
- queue and monitor logs must identify `P60-B0` / `P60-B01` clearly;
- training script index must be updated when runnable scripts are created or
  used.

Primary runnable wrappers and the host queue were added on 2026-06-02. The
experiment has not yet produced training results; interpret this status as
launch-ready, not result-complete.

P50 optional-ablation runnable wrappers, host queue, and monitor were prepared
on 2026-06-03. They use Stage1 step 50 as the fixed handoff source, run
matched-beta Stage2 for 40 steps, write fixed Model2 dirs under
`/data-1/model_weights/staged_v1/plateau_handoff_p50/`, and are documented in
the training script index. Interpret this status as prepared, not launched.
