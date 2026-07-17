# Qwen3-1.7B Stage123 Model2-KL / Split-Stage3 Experiment

Status: implementation ready and CPU validated; no training launched.

## Pre-Registered Questions

This matrix answers exactly two questions. Other observations are exploratory and
must not replace these pre-registered questions after results are available.

1. **Primary — is model2-only KL effective?** Does a `model2-only` KL constraint
   preserve or improve Model2 during the 20-step joint/fused Stage2 compared with
   the otherwise identical no-KL arm, and does that difference persist after an
   independent Model2 Stage3 continuation?
2. **Secondary — what happens to model1?** During model2-rollout joint/fused Stage2,
   does model1 improve, degrade, remain unchanged, or change indirectly when KL is
   applied only to model2? Can the extracted model1 subsequently learn or retain
   capability in its own Stage3 continuation?

Stage3-model2 is a persistence test for the primary KL result. Stage3-model1 is an
endpoint and recoverability test for the secondary model1 observation. The matrix is
not a general search for whichever of four Stage3 runs happens to peak highest.

## Fixed Source And Controls

- shared Stage1 handoff: FRAC25 beta `0.1`, step `40`
- Stage2: `20` steps, model2-only rollout, joint/fused WDL-SFT objective
- Stage3: `40` steps, Stage1-like single-model training
- training data order: `DATA_SHUFFLE=False`
- rollout: `temperature=1.0`, `top_p=1.0`, sampled
- validation: `temperature=0.2`, `top_p=0.95`, sampled, `n=3`
- primary checkpoint metric: `acc/mean@3`
- secondary audited metrics: `acc/pass@3`, `acc/std@3`
- raw three generations, rewards, response lengths, task identities, and validation
  JSONL remain retained per validation view

Stage2 exposes separate metric namespaces:

```text
val-core/model1/<dataset>/acc/mean@3
val-core/model1/<dataset>/acc/pass@3
val-aux/model1/<dataset>/acc/std@3
val-core/model2/<dataset>/acc/mean@3
val-core/model2/<dataset>/acc/pass@3
val-aux/model2/<dataset>/acc/std@3
```

`std@3` is the per-task standard deviation across the three sampled correctness
labels, averaged across tasks. `pass@3` is exact any-of-three task success, while
`mean@3` is the mean correctness of all three samples. They are not aliases.

## Matrix

| Arm | Stage2 KL | Stage2 validation | Stage3 branches |
| --- | --- | --- | --- |
| `nokl` | model1 off, model2 off | model1 + model2 | extracted model1; extracted model2 |
| `m2kl` | model1 off, model2 `low_var_kl`, coef `0.01` | model1 + model2 | extracted model1; extracted model2 |

Execution order:

```text
frac25-stage2-nokl
frac25-stage2-m2kl
frac25-stage3-nokl-model1
frac25-stage3-m2kl-model1
frac25-stage3-nokl-model2
frac25-stage3-m2kl-model2
```

The two Stage2 arms run consecutively so the primary KL comparison is collected as
close in wall-clock time and machine state as possible. Model1 Stage3 branches run
next because model1 behavior is the designated secondary question. Model2 Stage3
branches then test whether the Stage2 KL effect persists.

The queue continues to the sibling Stage3 branch when one Stage3 child fails. If a
Stage2 arm fails, both dependent Stage3 runs are marked skipped and the next
independent Stage2 arm starts. There is no automatic retry, resume, or parameter
change.

## KL Diagnostic Semantics

For joint Stage2, model2 produces rollout responses while the fused joint actor is
the optimized policy. Therefore one generic `rollout_corr/kl` cannot be interpreted
as implementation mismatch.

The implementation now records:

```text
rollout_impl_corr/model2_vllm_vs_model2_fsdp/*
behavior_gap/model2_vs_fused/*
```

- `rollout_impl_corr/...` measures the same Model2 implementation across vLLM and
  FSDP and is the implementation-parity diagnostic.
- `behavior_gap/...` measures the intentional Model2-rollout versus fused-training
  policy gap.
- `rollout_corr/*` remains available for correction/backward compatibility.
- single-model training keeps the generic `rollout_corr/*` namespace because its
  rollout and training policies are not semantically ambiguous.

The configured `model2-only KL` is actor-loss KL against the Stage1 P40 Model2
reference. It is independent from rollout-correction diagnostics and does not enable
model1 KL, global actor KL, or KL-in-reward.

## Result Interpretation

### Primary KL Decision

The primary endpoint is the **final Stage2 step-20 Model2 macro `mean@3`**:

$$
M_{\mathrm{macro}} = \frac{M_{\mathrm{HumanEval+}} + M_{\mathrm{MBPP+}} + M_{\mathrm{LCB}}}{3}
$$

The reported treatment effect is:

$$
\Delta_{\mathrm{KL}} = M_{\mathrm{macro}}^{\mathrm{model2\ KL}} - M_{\mathrm{macro}}^{\mathrm{no\ KL}}
$$

Use paired task-level bootstrap over the retained raw `n=3` outcomes to construct a
95% confidence interval. The decision is frozen before launch:

- **KL effective:** $\Delta_{\mathrm{KL}} \ge 1.0$ percentage point, the paired 95%
  CI lower bound is above zero, and no individual benchmark regresses by more than
  1.0 point.
- **KL harmful:** $\Delta_{\mathrm{KL}} \le -1.0$ point, or at least two of three
  benchmarks regress by more than 1.0 point.
- **Inconclusive / no demonstrated benefit:** every other outcome. A higher peak or
  one isolated benchmark improvement is not enough to claim effectiveness.

`pass@3`, `std@3`, trajectory shape, response length, and Stage3-model2 are supporting
evidence. They cannot replace the final-step primary endpoint. Stage3-model2 determines
whether an accepted Stage2 effect persists; it does not retroactively make a failed
Stage2 primary test positive.

### Model1 Observation

For model1, report four pre-registered comparisons:

1. no-KL Stage2: step-20 minus val-before-train model1 macro `mean@3`;
2. model2-KL Stage2: step-20 minus val-before-train model1 macro `mean@3`;
3. step-20 model1 difference between model2-KL and no-KL arms;
4. each extracted model1's independent Stage3 trajectory and final result.

Interpret model1 as:

- **co-adapting** if it improves during Stage2 in both arms;
- **indirectly stabilized by model2 KL** if the model2-KL arm materially improves
  model1 relative to no-KL despite model1 KL remaining disabled;
- **sacrificed** if model1 consistently degrades while model2 improves;
- **recoverable endpoint** if model1 Stage3 reverses a Stage2 decline or extends a
  Stage2 gain;
- **inactive/unclear** if changes remain within paired sampling uncertainty.

| Observation | Supported conclusion |
| --- | --- |
| M2-KL improves Stage2 model2 and Stage3-model2, without hurting model1 | Model2 drift was a material cause; keep model2-only KL. |
| M2-KL improves Stage2 model2 but not Stage3-model2 | Stage3 objective/data erases the retained capability; revise Stage3, not Stage2 KL. |
| Stage2 model1 improves and Stage3-model1 wins | The fused objective transfers useful capability into model1; model1 is a viable endpoint. |
| Stage2 model2 improves but Stage3-model2 falls immediately | Stage3 continuation is incompatible with the Stage2 endpoint or validation/training distributions differ. |
| No-KL and M2-KL are indistinguishable within `std@3` and task-level paired evidence | Reject a practical benefit from coefficient `0.01` at this budget; do not claim KL helps. |
| Both Stage3 branches fail to retain Stage2 gains | Reject the current Stage3 handoff objective as a plateau-breakthrough mechanism. |

Scientific conclusions must compare paired task identities and raw repetitions, not
only aggregate peaks. Peak-only improvements that are smaller than observed sampling
variation are exploratory, not positive results.

## Artifacts

- matrix overlay: `recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml`
- manifest renderer: `scripts/stage123_matrix_manifest.py`
- phase adapter: `scripts/stage123_phase_adapter.py`
- queue: `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_model2_kl_split_stage3_queue.sh`
- monitor: `recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_model2_kl_split_stage3.sh`

The overlay inherits the existing Stage123 dataset, scorer, timeout, calibration,
and source-provenance descriptors while replacing only the experiment identity,
validation protocol, KL arms, and Stage3 branching topology. The frozen legacy
`stage123.yaml` remains unchanged.

## Launch Boundary

This plan and implementation do not authorize GPU use or training. Before launch:

1. focused CPU tests and dry-run rendering must pass;
2. the matrix manifest hash and source paths must be recorded;
3. launch must use tmux;
4. queue and monitor must use the same manifest and state root;
5. release-gate policy remains mandatory before registry or W&B publication.
