# Qwen3-1.7B Math Cold-Start and Stage123 Matrix

- Status: COLD-START V1/V2 INVALIDATED; WHOLE-MESSAGE LOSS MASK FIX VERIFIED; V3 NOT LAUNCHED
- Date: 2026-07-20
- Hardware target: 8 x NVIDIA L40S 46 GB

## Objective

Determine the minimum supervised format cold-start needed by Qwen3-1.7B, then
run a data-matched Stage1 -> Stage2 -> Stage3 matrix for beta `0.0` and `0.1`,
including no-KL versus model2-only-KL Stage2 and independent Stage3 continuation
from both extracted submodels.

The integrated queue may launch only after a fresh idle-GPU check. It performs
cold-start selection and the Stage123 handoff under the user's pre-authorized,
pre-registered rules without a later manual pause.

## 2026-07-20 Cold-Start Loss-Mask Incident

Cold-start V1 (`lr=2e-5`) and V2 (`lr=5e-6`) are invalid as training evidence.
The SFT parquet contains complete `<think>...</think>` reasoning, but
`MultiTurnSFTDataset` rendered every turn independently. Qwen3's context-sensitive
Thinking template removes reasoning when an assistant message is rendered alone,
so the resulting `loss_mask` supervised only the final `<answer>...</answer>` and
`<|im_end|>` tokens. Setting `data.ignore_input_ids_mismatch=True` suppressed the
warning while preserving the incorrect per-turn sequence.

The corrected cold-start path sets `data.tokenize_whole_message=True` and
`data.ignore_input_ids_mismatch=False`. The dataset now renders the complete
conversation once and derives assistant spans from the whole-message token
sequence. Regression tests and a 32-row probe using the actual Qwen3-1.7B
tokenizer verify that `<think>`, the full reasoning body, `</think>`, `<answer>`,
and `<|im_end|>` receive loss while system/user content does not.

V3 must use fresh paths and run identity. Its manifests are
`math_qwen3_1p7b_cold_start_cotmask_v3.yaml` and
`math_qwen3_1p7b_stage123_cotmask_v3.yaml`. Launch remains disabled until the
V3 manifest/code review is complete; V1/V2 checkpoints must not be selected as
Model1 or published to DB/W&B.

### Machine-Enforced Gates

This failure mode is now blocked at four layers:

1. `MultiTurnSFTDataset` has a deterministic CPU regression test for a
   context-sensitive Qwen Thinking template, including multi-turn assistant
   spans.
2. `scripts/check_sft_loss_mask_policy.py` rejects any new
   `ignore_input_ids_mismatch=True` launcher outside the frozen allowlist and
   requires the math launcher to use whole-message tokenization.
3. The tracked `.githooks/pre-push` and dedicated GitHub Actions workflow run
   the policy canary and regression tests. The repository hook is installed by
   `scripts/install_repo_hooks.sh` through `core.hooksPath=.githooks`.
4. Before any V3 optimizer step, `scripts/math_cold_start_queue.py` runs
   `scripts/validate_sft_loss_mask.py` against the actual Qwen3 tokenizer and
   generated SFT parquet, validates 32 fixed-seed samples, and writes a
   structured SHA256-bound receipt. Training cannot start if any sample omits
   reasoning/answer/EOS supervision or includes system/user text in the loss.

## Frozen Data Contract

Source: `/data-1/dataset/math/train_rl_format.parquet`, exactly 7,500 Hendrycks
MATH rows. The preparation script applies one permutation using seed `20260719`
and preserves the resulting order inside every shard.

| Shard | Rows | Use |
| --- | ---: | --- |
| `cold_start` | 1,100 | Format SFT only |
| `stage1` | 2,560 | Initial Stage1 beta `0.0` and `0.1`; exactly `40 x 64` prompts |
| `stage2` | 1,280 | Stage2 treatment and first 20 steps of matched control; exactly `20 x 64` prompts |
| `stage3` | 2,560 | Stage3 treatment and final 40 steps of matched control; exactly `40 x 64` prompts |

The four primary shards are pairwise disjoint. Their sizes are aligned to the
fixed prompt batch and phase lengths, so Stage1, Stage2, and Stage3 each consume
their shard exactly once without wraparound. `stage1_control` is the only
intentional reuse: it concatenates `stage2` then `stage3` without reshuffling so
the 60-step pure-Stage1 continuation sees exactly the same 3,840 ordered prompts
as the 20-step Stage2 plus 40-step Stage3 treatment.

The generated `dataset_receipt.json` records source/shard hashes, source row
indices, ordering, and overlap policy. Every formal queue must verify it before
training.

## Cold-Start Protocol

The queue evaluates the raw post-trained Qwen3-1.7B model at step `0`. If it
does not pass, SFT resumes in increments of five optimizer steps through a
maximum of step `40`. Cold-start format screening uses the complete Math-7
validation set with `n=1`; no subset validation is allowed. Stage1, Stage2, and
Stage3 performance validation remains `n=3`.

Validation datasets:

1. AIME-2025
2. MATH-500
3. AMC23
4. AQUA
5. GSM8K
6. MAWPS
7. SVAMP

Every checkpoint records accuracy, the response-level complete intersection,
and:

- `<think>` completeness rate;
- `<answer>` completeness rate;
- `\boxed{}` extraction success rate;
- reward-grader success rate;
- EOS rate;
- truncation rate.

Pre-registered admission gate:

| Metric | Threshold |
| --- | ---: |
| complete format contract | >= 95% |

The complete contract is the intersection of exactly one ordered `<think>`
pair, exactly one ordered `<answer>` pair, successful boxed extraction,
successful reward-grader execution, and EOS/non-truncation. Individual rates
remain diagnostics but do not replace the intersection gate.

The queue stops after the first passing checkpoint, writes all candidate
metrics, selects the earliest passing checkpoint automatically, and writes one
immutable `model1_selection.json`. It then hands Model1 to Stage123. This is
deterministic because the user pre-authorized the earliest checkpoint satisfying
the frozen 95% complete-format gate.

### Step-0 result

The raw Qwen3-1.7B post-trained/Instruct model was evaluated on the complete
Math-7 set on 2026-07-19 with `n=1`, temperature `0.2`, top-p `0.95`, and
`MAX_RESPONSE_LENGTH=4096`. The run completed all 2,798 prompts successfully.

| Metric | Math-7 macro | Response-level micro | Gate |
| --- | ---: | ---: | ---: |
| `<think>` complete | 73.0% | 89.9% | >= 95% |
| `<answer>` complete | 67.4% | 85.1% | >= 98% |
| boxed extraction | 40.7% | 50.2% | >= 95% |
| reward grader succeeds | 100.0% | 100.0% | >= 99% |
| EOS | 72.1% | 90.2% | >= 98% |
| truncation | 27.9% | 9.8% | <= 2% |

Only 44.4% of all responses simultaneously passed both tag checks, boxed
extraction, reward-grader execution, and EOS. Among non-truncated responses,
1,140 of 2,798 total prompts still failed boxed extraction, primarily because
the model returned `<answer> 81 </answer>` rather than
`<answer> \\boxed{81} </answer>`. Competition-level prompts also had severe
length failure: AIME-2025 truncation was 93.3%, AMC23 was 52.5%, and MATH-500
was 25.8%.

Decision: step `0` is not admissible. A supervised format cold start is
required, beginning with the smallest candidate at step `5` and continuing in
five-step increments until the first full Math-7 checkpoint reaches 95%
response-level complete-format success.

## Shared Training and Validation Configuration

- `MAX_PROMPT_LENGTH=500`
- `MAX_RESPONSE_LENGTH=4096`
- rollout max model length `4596`
- prompt batch `64`
- rollout `N=8`
- cold-start format validation `n=1`; Stage1/2/3 validation `n=3`
- validation temperature `0.2`, top-p `0.95`, sampling enabled
- train/validation/save interval: every `5` steps
- data seed `20260719`, `data.shuffle=False`
- fusion lambda `0.8`
- model2-only KL type `low_var_kl`, coefficient `0.01`
- initial rollout GPU utilization candidate `0.60`, subject to fresh probe

All training phases validate on the complete Math-7 list. The primary best
checkpoint metric is the unweighted macro average of the seven dataset-level
`acc/mean@3` values:

`val-core/math7_macro/acc/mean@3`

For joint Stage2 validation, checkpoint selection uses Model2:

`val-core/model2/math7_macro/acc/mean@3`

## Matrix

The matrix is fully matched for beta `0.0` and beta `0.1`.

For each beta:

1. Stage1 trains 40 steps on the `stage1` shard.
2. Stage1 control continues 60 steps on ordered `stage2 -> stage3` data,
   producing effective step 100.
3. Stage2 no-KL trains 20 steps on `stage2`.
4. Stage2 model2-KL trains the same 20 steps on `stage2`.
5. Each Stage2 arm extracts model1 and model2.
6. Four Stage3 runs train each extracted submodel for 40 steps on `stage3`.

The Stage1 control is a Stage1-like restart from the merged Stage1 weight with a
fresh optimizer and warmup, matching the optimizer reset at the Stage2 handoff;
it is not a seamless continuation of Stage1 optimizer state.

This produces 16 runs total: two Stage1 sources, two controls, four Stage2
runs, and eight Stage3 runs.

## Execution Gates

1. GPUs must be freshly verified idle immediately before launch.
2. Dataset receipt must verify with no overlap.
3. The actual-model loss-mask preflight receipt must pass before training.
4. Only the V3 whole-message-mask manifests may be admitted for a future launch;
   V1/V2 are diagnostic-only invalid runs.
5. The queue reuses the completed, identity-matched step-zero Math-7 `n=1`
   result and validates every five steps thereafter.
6. The earliest checkpoint meeting the 95% complete-format gate becomes Model1.
7. Stage123 starts automatically after cold-start training/validation releases
   the GPUs.
8. Queue and monitor run in separate tmux sessions with append-only event and
   notification ledgers.

Failed or incomplete runs remain diagnostic-only and must pass the training
result release gate before DB or W&B publication.
