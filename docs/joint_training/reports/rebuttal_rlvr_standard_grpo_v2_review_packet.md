# Human Review Packet: Rebuttal Standard GRPO v2

## Decision state

G1b v1 received `REQUEST_CHANGES` from the human reviewer. This v2 packet
applies those changes and was **HUMAN APPROVED on 2026-07-28**. This approval
closes only G1b; it does not claim that G0, G3, or G4 passed and does not itself
launch calibration, training, or Hope submission.

Review target:

```text
recipe/on_policy_wdl_sft/rebuttal_rlvr/frozen_grpo_v2.env
version: rebuttal-standard-grpo-v2
SHA-256: 8dafbac42f911445e11438009ab2cf5a5f5e2075ce0bfb68abdd3629b9e54177
```

## Human-requested changes applied

1. Freeze input order with `DATA_SHUFFLE=false`; formal RL seeds remain the
   pre-registered constants `20260727/28/29` and are never generated at runtime.
2. Use current standard verl GRPO surfaces instead of historical Project-2G
   deviations: symmetric clip `0.2/0.2`, dual-clip C `3.0`, `token-mean`,
   `grad_clip=1.0`, and no rollout IS/RS.
3. Enable the GRPO example's explicit reference KL loss:
   `use_kl_loss=true`, coefficient `0.001`, type `low_var_kl`; KL-in-reward
   remains disabled.
4. Use the strict `<think>/<answer>` scorer under `recipe/joint_training/`.
5. Enable online validation before training and every five trainer steps through
   step 115 with `n=3`,
   temperature `1.0`, top-p `0.95`, top-k `-1`, sampling enabled.
6. Retain only the online-best checkpoint and latest checkpoint. The best is
   model-only; the latest is a full resumable checkpoint.
7. Raise the H20 rollout-memory calibration grid to `0.60/0.70/0.80`; the
   selected value must still pass both arms with peak total GPU memory at most
   90%, finite loss/grad norm, and zero worker retry.

## Frozen scientific configuration

| Surface | Frozen value |
| --- | --- |
| advantage / loss | `grpo`; `vanilla`; normalize group advantage by std |
| PPO clip | base/low/high `0.2/0.2/0.2`; dual-clip C `3.0` |
| loss aggregation | `token-mean` |
| rollout correction | IS `null`; RS `null`; bypass disabled |
| optimizer | `torch.optim.AdamW`; betas `[0.9,0.999]`; eps `1e-8`; weight decay `0.1` |
| learning rate | `5e-7`; five trainer-step warmup; constant scheduler |
| gradient clip / PPO epochs | `1.0`; `1` |
| reference KL | loss enabled; coefficient `0.001`; `low_var_kl` |
| KL in reward | disabled; coefficient `0.0` |
| batch | 64 prompts; 8 rollouts/prompt; 8-prompt actor mini-batch |
| length | prompt 500; response 4096 |
| train sampling | temperature `1.0`; top-p `1.0`; top-k `-1`; sample |
| data order | data shuffle `false`; actor mini-batch shuffle `false` |
| entropy | coefficient `0.0`; diagnostic calculation enabled |
| horizon | 115 trainer steps; one PPO epoch |
| save / validation cadence | every 5 steps; validate before training |
| online validation | `n=3`, temperature `1.0`, top-p `0.95`, top-k `-1` |
| best metric | `val-core/HuggingFaceH4/MATH-500/acc/mean@3` |
| retention | best model-only plus latest full checkpoint |
| reward manager | `naive`; DAPO overlong reward shaping disabled |
| strict grader | `recipe/joint_training/custom_reward_function_latex_verify.py` |
| joint/WDL extras | joint training, submodel KL, WDL fallback all disabled |

The two arms consume this exact configuration. Their only scientific
difference is the admitted initialization checkpoint.

## Why these values are called standard GRPO

Current verl config defaults and the registered `vanilla` loss use symmetric
clip `0.2`, dual-clip C `3.0`, `token-mean`, `grad_clip=1.0`, and rollout IS
disabled. The repository's GRPO examples explicitly enable reference KL loss
with coefficient `0.001` and `low_var_kl`. This v2 therefore drops the former
historical-reconstruction choices `C=10`, `seq-mean-token-sum`,
`grad_clip=500`, token rollout IS, and DAPO reward-manager naming.

`token-mean` does not introduce a DAPO overlong reward penalty. It averages
the policy objective across valid response tokens. The strict scorer supplies
only binary correctness/format reward; the overlong buffer is disabled.

## Best/latest storage contract

Training saves immediately before validating every five steps from step 5
through step 115. Step 0 is validated before training without creating another
copy of the admitted initialization checkpoint.
After validation:

- the best step by online MATH-500 `mean@3` is protected;
- if best is not latest, its optimizer shards are removed, leaving model-only;
- the latest step remains a complete resumable checkpoint;
- every other `global_step_*` directory is removed;
- if the latest is also best, only one full checkpoint remains.

This does not conflict with `SAVE_FREQ=5`. The cadence creates candidates;
trainer-level retention reduces them to best plus latest. Because intermediate
checkpoints are deleted, offline Math-7 is required only for initialization,
retained best, and final/latest. The final step 115 remains the sole
confirmatory checkpoint; best-checkpoint results are secondary diagnostics.

## Strict reward and validation contract

The runtime scorer requires exactly one complete, ordered, non-empty
`<think>...</think>` block and exactly one `<answer>...</answer>` block. It
extracts `\\boxed{}` only from `<answer>`, requires EOS, and assigns `-1` to
format failure, missing boxed answer, truncation, or incorrect answers.

Online validation uses the same strict scorer and `n=3`. Offline confirmatory
Math-7 also uses the strict scorer with frozen per-request seeds. Online best
selection does not replace the final-step primary analysis.

## H20 calibration boundary

| Selectable system knob | Candidates |
| --- | --- |
| rollout GPU memory utilization | `0.60`, `0.70`, `0.80` |
| generation micro-batch size | `16`, `32` |
| log-prob micro-batch size | `4`, `8` |
| actor dynamic-token budget | `9192`, `18384` |

Choose the fastest common profile passing both arms. Calibration may not
change any scientific setting, reward/scorer behavior, data order, seed,
validation, or checkpoint-retention rule.

## Recorded human re-review response

```text
Verdict: APPROVE
Reviewed frozen-config SHA-256: 8dafbac42f911445e11438009ab2cf5a5f5e2075ce0bfb68abdd3629b9e54177
Standard GRPO surface: ACCEPT
KL loss true / 0.001 / low_var_kl: ACCEPT
Strict scorer: ACCEPT
Fixed data order and pre-registered seeds: ACCEPT
Online n=3 and best/latest retention: ACCEPT
H20 0.60/0.70/0.80 calibration grid: ACCEPT AS CALIBRATION CANDIDATES
Requested changes and evidence: none; live H20 selection remains G4 evidence
Reviewer / date: experiment owner / 2026-07-28
```
