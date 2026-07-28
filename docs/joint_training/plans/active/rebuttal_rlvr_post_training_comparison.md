# Rebuttal RLVR Post-Training Comparison: Offline WDL-SFT vs Ordinary SFT

- Status: DIRECT WORKER ROUTE READY AFTER MODEL PLACEMENT; G0 CONDITIONAL-CHECKPOINT ASSUMPTION ACCEPTED; G1a/G1b/G2-LOCAL PASSED; AUDITED HOPE G3/G4 EVIDENCE PENDING
- Date: 2026-07-28
- Scope: MATH first; code-task work starts only after the paired MATH H20 gate G4 passes
- Hardware target: one Meituan AFO worker with 8 x H20-141G per job
- Source request: [Feishu document revision 5](https://ocnwds5io8yp.feishu.cn/docx/Ed0xd8YmQoQJkjxl0E5cFSt7ngd)
- Reusable RLVR references: `recipe/on_policy_wdl_sft/ablation_single_model/run_2g_math_base.sh` and `run_2g_math_sft.sh`
- Meituan reference: `docs/joint_training/guides/meituan_platform.md`

## 1. Decision Summary

This experiment tests whether the paper's **offline WDL-SFT** produces a better
starting point for otherwise identical RLVR post-training than ordinary SFT.
The treatment is the initialization checkpoint; every RLVR setting after that
checkpoint must be shared between the two arms.

The executable comparison is a paired `2 x 3` fixed-checkpoint matrix:

- two initialization methods: ordinary SFT and paper offline WDL-SFT;
- one colleague-supplied checkpoint per method;
- three paired RLVR seeds;
- six RLVR jobs in total.

Because public/source provenance cannot recover initialization seed, optimizer,
training steps, data receipt, or checkpoint-selection details, this matrix does
not identify a population-level initialization-method effect. Its claim is
conditional on the exact two supplied checkpoints. The former 18-cell,
three-initialization-pair matrix remains a stronger deferred design, not a
launch requirement for this run.

The MATH train file is frozen to:

```text
/data-1/dataset/math/train_rl_format.parquet
```

The evaluation suite is the existing Math-7 set: AIME-2025, MATH-500, AMC23,
AQUA, GSM8K, MAWPS, and SVAMP.

No job is launched from this physical host. R02 is the pinned public WDL-SFT
4B snapshot. R01 is pre-registered as
`R01_ORDINARY_SFT_4B_AM1P4M`; both become executable when the colleague places
the two model directories at the declared paths. Unrecoverable initialization
metadata is accepted as an explicit assumption rather than a launch blocker.
All downstream GRPO cells use the same 7,500-row MATH file; AM-1.4M describes
the initialization SFT stage, not the RLVR training data.

## 2. Research Question and Claim Boundary

For a fixed RLVR budget `B`, the executable estimand is conditional on the two
colleague-supplied checkpoints:

$$
\Delta_B^{W,S} = \mathbb{E}_{r}\left[
Q\left(\operatorname{RLVR}_B(W; r)\right)
- Q\left(\operatorname{RLVR}_B(S; r)\right)
\right],
$$

where `W` and `S` are the exact supplied offline WDL-SFT and ordinary-SFT
checkpoints, `r` is one of three paired RLVR seeds, and `Q` is the frozen
final-checkpoint Math-7 primary metric. The comparison matches downstream RLVR
data/configuration/budget; it does not claim the initialization procedures are
fully provenance-matched.

If the six-cell matrix passes, the allowed claim is:

> Under the frozen MATH RLVR recipe and three paired RLVR seeds, the supplied
> WDL-SFT checkpoint yields better final Math-7 performance than the supplied
> ordinary-SFT checkpoint.

This must not be generalized into a population-level WDL-SFT-versus-SFT method
claim. That stronger claim still requires the deferred 18-run provenance-matched
matrix.

End-to-end efficiency is a separate secondary estimand. This design does **not**
by itself prove that WDL-SFT is end-to-end cheaper. Initialization compute and
RLVR compute must therefore be reported separately:

$$
C_{\mathrm{total}} = C_{\mathrm{init}} + C_{\mathrm{RLVR}}.
$$

Only if total training tokens, rollout tokens, GPU-hours, and completion rates
also favor WDL-SFT may the result be described as end-to-end compute-efficient.

## 3. G0: Checkpoint Identity and Provenance Gate

### 3.1 Required checkpoint pair

Each initialization pair must be a deployable single-model checkpoint pair
described by one machine-validated `paired_init_manifest`. The manifest freezes
the following invariant before any Math-7 result from this experiment exists:

| Property | Ordinary SFT arm | Offline WDL-SFT arm |
| --- | --- | --- |
| Base model | Same exact repo and revision | Same exact repo and revision |
| Architecture / size | Same | Same |
| Tokenizer | Same files and hashes | Same files and hashes |
| Initialization SFT corpus | AM-1.4M | AM-1.4M |
| Prompt/template/max length | Exact same files, hashes, and limits | Exact same files, hashes, and limits |
| Target-model supervised budget | Same target tokens, optimizer updates, batch, optimizer, and LR schedule | Same target tokens, optimizer updates, batch, optimizer, and LR schedule |
| Seed pairing | Initialization seed `i` | Exact paired initialization seed `i` |
| Checkpoint universe | Frozen step list before training | Same frozen step list |
| Selection rule | Final predeclared update; no eval-best selection | Same final predeclared update and frozen paper extraction rule |
| Output form | Standard HF single-model directory | Standard HF single-model directory selected by the frozen extraction rule |
| Post-checkpoint RL | None | None |

The only allowed initialization-method difference is the paper offline WDL-SFT
mechanism itself: its weak/strong-model signal and the extra model/rollout
compute intrinsically required to produce that signal. Those extra tokens and
GPU-hours are not forced to equal ordinary SFT, but must be recorded for the
secondary end-to-end accounting. Any other difference fails the formal-pair
validator. The WDL-SFT checkpoint must not be inferred from a name containing
`WDL`; provenance must positively show the offline training stage and the frozen
rule used to select or extract the final single model.

If historical paper checkpoints cannot prove the exact paired fields above,
they remain valid only for a fixed-checkpoint pilot. A formal method-level
matrix then requires regeneration from a reviewed paired-initialization recipe.

### 3.2 Required evidence

For every checkpoint, write an immutable receipt containing:

- absolute storage path or Hugging Face repo plus commit revision;
- `config.json`, generation config, tokenizer files, safetensors index and all
  referenced shards, with size and SHA-256;
- base-model repo and revision;
- training code commit and entry script;
- dataset receipt/hash, row count, split, and prompt/template version;
- initialization seed, optimizer, learning rate, update count, supervised token
  count, and checkpoint-selection rule;
- complete candidate-run/checkpoint universe inventory hash, pair-admission rule,
  selection timestamp, and declaration of any historical Math-7 access;
- `trainer_state.json`, or an equivalent source-backed manifest and canonical
  training log if the deployable export intentionally omits trainer state;
- for offline WDL-SFT, weak/strong model roles and the exact paper-stage output
  selected for downstream RLVR;
- an explicit classifier: `ordinary_sft` or `offline_wdl_sft`.

The pair manifest stores both receipt hashes plus equality assertions for every
matched field in Section 3.1. `validate_paired_init` must fail closed on a
missing field, unequal hash/value, an unapproved checkpoint step, or an
unrecognized classifier. Recording a mismatch is not admission.

Formal historical pairs are admissible only when the complete candidate
universe was registered before the accepted plan commit and an auditable record
shows pair admission did not use existing Math-7 scores. If more than three
eligible pairs exist, select the three lowest numeric initialization seeds; do
not choose by score. If the inventory/timestamp/no-access condition cannot be
proved, regenerate paired initializations. A historical single-pair pilot may
still run after disclosing all previously viewed evaluation results, but it
cannot support the method-level claim.

### 3.3 Hard exclusions

The gate must reject all of the following even when the files load correctly:

- any On-Policy SFT or On-Policy WDL-SFT checkpoint;
- `wdl_sft`, `wdl_sft_is`, or staged joint-training RL outputs from this branch;
- MiniRL, GRPO, PPO, group-advantage, DPO, or other RL/post-RL checkpoints;
- a joint/fused checkpoint without a predeclared deterministic single-model
  extraction rule;
- mismatched base revision, model size, tokenizer, or MATH data provenance;
- a checkpoint chosen after inspecting this experiment's Math-7 results.

Known excluded examples include:

```text
/data-1/model_weights/WDL-SFT-4B-MATH-M5-5/step_300_model2
/data-1/model_weights/WDL-SFT-Qwen3-4B-MATH-2A-SFT/step_300
/data-1/model_weights/MINIRL-Qwen3-4B-MATH-2Z-SFT/step_300
AlexGeek/WDL-GROUP-ADV-IS-*
AlexGeek/ONPOLICY-SFT-*
AlexGeek/WDL-SFT-STAGED-V1-S2-*
```

The ordinary-SFT artifact is intentionally named before delivery:
`R01_ORDINARY_SFT_4B_AM1P4M`. Its portable default path is
`${MODEL_ROOT}/R01_ORDINARY_SFT_4B_AM1P4M`, where `MODEL_ROOT` derives from
`ROOT`. R01 fails before launch until the directory, `config.json`, and a model
weight file exist.

The public candidate `chhao/Weak-Driven-Learning` was inspected at immutable
revision `1bfdcc4506656288b115b8fa1d4e446f4e344f12` on 2026-07-28. Its live
`config.json` identifies a Qwen3-4B model and the repository contains one
8,045,067,711-byte `pytorch_model.bin`. This is the frozen R02 artifact. Its
model card identifies AM-1.4M as the initialization corpus, matching the
intended R01 corpus. The download receipt proves source revision, file
size/hash, and loadability; unavailable initialization seed, optimizer, update
count, data receipt, and checkpoint-selection fields remain unavailable and
must not be fabricated. On 2026-07-28 the experiment owner accepted those
fields as an irrecoverable external-checkpoint assumption. The result therefore
compares the supplied checkpoints conditionally and cannot be reported as a
fully provenance-matched initialization-method estimate. The card also
advertises an 8B variant, but
the Hub currently exposes no 8B branch, revision, or separate public model;
`R03` therefore remains fail-closed rather than inferring weights from prose.
The verified R02 snapshot is stored under configurable `HF_MODEL_CACHE_ROOT`
at `models--chhao--Weak-Driven-Learning/snapshots/1bfdcc...`; its download-only
receipt is `/data-1/model_weights/manifests/chhao-weak-driven-learning-4b-download-20260728.json`.

### 3.4 Pre-registration and admission outcomes

Before handoff, the repository registers every locally knowable R01/R02 fact,
every expected path, and every unavailable provenance field. The colleague
supplies artifacts into those named slots; they do not reconstruct receipt CLI
arguments or infer scientific settings. Pre-registration is not evidence
fabrication: missing fields remain visible in the assumption receipt, while
only model presence/loadability is a launch-time requirement for this route.

- One R01/R02 checkpoint pair under the recorded assumption: admit the six-run
  conditional checkpoint comparison.

- Three authoritative pairs that pass `validate_paired_init` would admit the
  deferred 18-run method-level matrix.
- The current owner-approved assumption admits the six-run fixed-checkpoint
  comparison as soon as both model directories pass the direct entry's
  loadability checks. Its conclusion must say “conditional on this checkpoint
  pair”; unavailable provenance is disclosed rather than treated as matched.
- Missing model files still block launch. Missing irrecoverable training
  metadata no longer blocks this conditional route.

## 4. Frozen MATH Inputs

### 4.1 RLVR training data

```text
TRAIN_FILE=/data-1/dataset/math/train_rl_format.parquet
```

Current verified local facts: 7,500 source rows, `data_source` equal to
`ck46/hendrycks_math`, and 7,405 prompts after the current 500-token prompt
filter. With 64 prompts per trainer/rollout step and `drop_last=True`, one
epoch is 115 trainer steps: `115 x 64 = 7,360` prompts are consumed and 45
eligible prompts are dropped.

This is the downstream RLVR corpus for both arms. It is intentionally distinct
from the AM-1.4M corpus used to produce the initialization checkpoints;
fairness requires the arms to share each stage's data, not for the SFT and
RLVR stages to reuse one corpus.

Before launch, record the file SHA-256, schema, row count, and recomputed hash of
the 7,405 eligible source-row positions under the bound init tokenizer and
500-token filter. The file SHA binds prompt content and row order. The exact
per-cell sampler contract is the immutable repo sampler implementation plus
`data.shuffle=false`, `drop_last=true`, batch size 64, and the manifest RL seed;
the two arms in a paired cell must use the same seed and therefore the same
prompt order. A shared train receipt must not pretend that one sampler hash can
represent all three RL seeds.

### 4.2 Math-7 evaluation files

The seven frozen datasets are:

1. AIME-2025 (30 prompts)
2. MATH-500 (500 prompts)
3. AMC23 (40 prompts)
4. AQUA (254 prompts)
5. GSM8K (1,319 prompts)
6. MAWPS (355 prompts)
7. SVAMP (300 prompts)

The launch receipt must record the exact seven paths, hashes, row counts,
prompt/template projection, ordered `extra_info.index` projection, and grader
revision. The worker recomputes both projections from each live parquet; a
64-hex placeholder is not evidence. Validation order must not vary by arm.
These files are evaluation inputs, not checkpoint-selection inputs.

The frozen grader is the strict implementation at
`recipe/joint_training/custom_reward_function_latex_verify.py`. It requires
one complete, ordered `<think>` block and one complete `<answer>` block,
extracts `\\boxed{}` only from `<answer>`, and requires EOS. The older copy
under `recipe/on_policy_wdl_sft/` is explicitly rejected because it can award
correctness from a boxed expression anywhere in the response. Formal workers
bind the strict file path/SHA, function name, recipe submodule commit, and
image digest, then overwrite any inherited custom-reward environment value.

### 4.3 Train/evaluation disjointness

`validate_train_eval_disjointness` is a fail-closed admission gate. For every
training and Math-7 row it records the source ID, normalized ground-truth hash,
and a canonical problem hash produced by extracting the final user problem,
applying Unicode NFKC, normalizing line endings, collapsing whitespace, and
removing only the frozen system/instruction wrapper. Numbers and mathematical
symbols remain unchanged.

The validator rejects any shared source ID or canonical problem hash. It also
rejects a deterministic near-duplicate when token 5-gram Jaccard similarity is
at least `0.90` and the normalized ground-truth hashes match. The report stores
every compared row ID and hash.

An overlap does not get silently filtered. It blocks the plan and requires an
amendment that either creates a decontaminated training receipt and recomputes
the filtered row count/step budget, or removes the affected benchmark and
pre-registers a new primary macro before launch. Until this validator passes,
MATH-500 and the broader Math-7 macro are not confirmatory evidence.

## 5. Frozen RLVR Algorithm

G1b review rejected the historical Project-2G deviations and selected current
standard verl GRPO surfaces for this new paired comparison. The configuration
uses the registered `vanilla` PPO loss, symmetric clipping, token-mean loss,
standard gradient clipping, no rollout correction, and explicit reference KL
loss as used by the repository's GRPO examples. It must not inherit DAPO
overlong shaping, WDL loss, joint-model, reverse-SFT, submodel-KL, or fallback
behavior.

The human-review source of truth is
`recipe/on_policy_wdl_sft/rebuttal_rlvr/frozen_grpo_v2.env`. Its version is
`rebuttal-standard-grpo-v2`; formal manifests bind the file SHA-256. The
line-item review packet is
`docs/joint_training/reports/rebuttal_rlvr_standard_grpo_v2_review_packet.md`.

### 5.1 Enabled configuration

| Setting | Frozen value |
| --- | --- |
| `algorithm.adv_estimator` | `grpo` |
| `policy_loss.loss_mode` | `vanilla` |
| `norm_adv_by_std_in_grpo` | `True` |
| clip base / low / high / dual-clip C | `0.2 / 0.2 / 0.2 / 3.0` |
| loss aggregation | `token-mean` |
| rollout correction | IS and RS disabled (`null`) |
| optimizer | `torch.optim.AdamW`, betas `[0.9,0.999]`, eps `1e-8`, weight decay `0.1`, zero-indexed constant scheduler |
| learning rate | `5e-7` |
| warmup | 5 trainer/scheduler steps, approximately 40 AdamW updates |
| gradient clip | `1.0` |
| reference KL loss | enabled, coefficient `0.001`, type `low_var_kl`; KL-in-reward disabled |
| PPO epochs | 1 |
| prompts / step | 64 |
| rollouts / prompt | 8 |
| actor mini-batch | 8 prompts |
| total training | 115 trainer/scheduler steps / about 920 AdamW updates / one filtered-data epoch |
| max prompt / response | `500 / 4096` |
| rollout decoder | temperature `1.0`, top-p `1.0`, top-k `-1` |
| data / actor shuffle | `false / false`; input order fixed |
| entropy | coefficient `0.0`; `calculate_entropy=true` for diagnostics |
| reward | strict binary format-and-correctness verifier; truncated/no-EOS is `-1` |
| online validation | step 0 and every 5 trainer steps through step 115; `n=3`, temperature `1.0`, top-p `0.95` |
| checkpoint candidates | every 5 steps from 5 through 115; initialization is external step 0 |
| retained checkpoints | online-best model-only plus latest full resumable checkpoint |
| offline Math-7 checkpoints | initialization, retained best, and final/latest step 115 |
| checkpoint cadence | `SAVE_FREQ=TEST_FREQ=5`; final step 115 is saved and validated |

The learning rate and 115-step budget retain the strongest project-specific
evidence from the prior MATH runs. The algorithm surface no longer claims to
replay those runs: G1b explicitly chose current standard GRPO values over the
historical launcher's token IS, C `10`, token-sum aggregation, and gradient
clip `500`.

Step counters are reported separately. Each trainer/rollout step has 64
prompts and 512 sampled sequences. Mini-batch 8 prompts with one PPO epoch
produces eight logical AdamW updates per trainer step, or about 920 across the
run. The LR scheduler advances once after each actor `update_policy`, so it has
115 scheduler steps; warmup 5 spans five scheduler steps and approximately 40
AdamW updates.

### 5.2 Explicitly disabled configuration

The new common launcher must pass the resolved Hydra keys below rather than
only exporting similarly named shell variables:

```text
algorithm.rollout_correction.rollout_is=null
algorithm.rollout_correction.rollout_is_threshold=null
algorithm.rollout_correction.rollout_is_batch_normalize=false
algorithm.rollout_correction.rollout_rs=null
algorithm.rollout_correction.rollout_rs_threshold=null
algorithm.rollout_correction.bypass_mode=false
algorithm.rollout_correction.loss_type=ppo_clip
algorithm.use_kl_in_reward=false
algorithm.kl_ctrl.kl_coef=0.0
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
+actor_rollout_ref.model.joint_training=false
actor_rollout_ref.actor.track_joint_submodel_losses=false
actor_rollout_ref.actor.submodel_kl.enabled=false
+actor_rollout_ref.actor.policy_loss.all_correct_sft_fallback=false
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.clip_ratio_c=3.0
actor_rollout_ref.actor.grad_clip=1.0
reward_model.reward_manager=naive
```

The new launcher makes the reviewed scientific surface explicit, including
optimizer implementation/betas/eps/step indexing, base clip ratio, and
entropy diagnostics, and archives the fully resolved Hydra config. This
prevents later upstream-default changes from converting the experiment into a
different recipe while retaining the same run name.

### 5.3 Paired randomization

Formal RLVR seeds are frozen to `20260727`, `20260728`, and `20260729`. The new
launcher must map `RLVR_SEED` to, at minimum:

```text
data.shuffle=false
data.seed=${RLVR_SEED}
actor_rollout_ref.actor.data_loader_seed=${RLVR_SEED}
actor_rollout_ref.actor.fsdp_config.seed=${RLVR_SEED}
actor_rollout_ref.ref.fsdp_config.seed=${RLVR_SEED}
+actor_rollout_ref.rollout.seed=${RLVR_SEED}
```

The rollout seed is an additive Hydra key because the current `RolloutConfig`
does not declare it even though the async vLLM server consumes it. A dry-run
must prove that all six values appear in the resolved config; a seeded two-run
probe must prove reproducible prompt ordering and seed routing before G2 passes.

The same `(initialization_seed, RLVR_SEED)` cell must receive the same configured
seeds and prompt order in both arms. Different model policies will naturally
sample different text, and distributed kernels need not be bitwise
deterministic; pairing means identical configured randomization, not identical
trajectories.

## 6. Experiment Matrix

### 6.1 Deferred formal method-level matrix

| Initialization pair | Ordinary SFT jobs | Offline WDL-SFT jobs |
| --- | --- | --- |
| `I1` | RL seeds 20260727/28/29 | Same RL seeds |
| `I2` | RL seeds 20260727/28/29 | Same RL seeds |
| `I3` | RL seeds 20260727/28/29 | Same RL seeds |

Total: 18 RLVR jobs. This stronger matrix is not launchable from the available
external artifacts and is no longer a prerequisite for the current run.

### 6.2 Executable conditional-checkpoint matrix

The current matrix has six jobs: two arms by three paired RLVR seeds. It
validates the launch path and estimates a conditional effect,
but it must be labeled `PILOT_FIXED_PAIR` in job IDs, result tables, W&B, and
any written conclusion.

### 6.3 Retry policy

`JOB_TAG` identifies the scientific cell; `ATTEMPT_ID` identifies each platform
attempt. Every attempt keeps the same seed, data/model/profile hashes, LR,
decoder, response length, batch size, and checkpoint rule.

- Platform resume is allowed only when model, optimizer, RNG, data-loader, and
  resolved-config state all exist and their receipt hashes match. Partial
  metrics before the resume remain diagnostic.
- Otherwise an infrastructure retry starts from the admitted initialization in
  a clean attempt directory with a new `ATTEMPT_ID`. At most one automatic
  infrastructure retry is allowed.
- The confirmatory analysis uses the earliest hash-valid attempt that reaches
  step 115; every failed attempt remains in completion-rate and stability
  reporting.
- NaN, reward/gradient collapse, deterministic OOM under the frozen profile, or
  another scientific failure is not an infrastructure retry condition. That
  cell remains a failed outcome and cannot be replaced with a favorable seed.

## 7. H20 Calibration Without Algorithm Tuning

H20 work is divided into system calibration and the scientific experiment.
System calibration may change memory/throughput knobs only; it may not inspect
Math-7 accuracy to choose a profile.

### 7.1 Fixed AFO resource request

Start from the existing `platform/hope_ablation/run.hope` shape:

```text
workers=1
worker.gcoresh20-141g=8
worker.memory=1920000
worker.vcore=128
queue=root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train
docker shm=512 GiB
```

### 7.2 Calibration sequence

After G0 and image/path admission, use one checkpoint from each arm and the
same fixed five-step training slice:

1. Path-only/preflight: imports, model load, dataset load, reward import, Ray,
   vLLM FlashInfer, one rollout, and persistent output write/read.
2. Memory-share sweep: rollout GPU utilization `0.60`, `0.70`, then `0.80`,
   keeping all other knobs fixed.
3. At the highest profile safe for **both** arms, test only throughput knobs:
   generation micro-batch `16 -> 32`, log-prob micro-batch `4 -> 8`, and actor
   dynamic-token budget `9192 -> 18384`, one dimension at a time.
4. Sample `nvidia-smi` used/total memory once per second on all eight GPUs.
   Require every GPU's peak `memory.used / memory.total <= 0.90`, finite
   loss/grad norm, and no worker retry.
5. Ignore warmup steps 1-2. Among valid profiles, choose the profile with the
   lowest median step time over steps 3-5, provided the step-time coefficient
   of variation over those steps is at most 15% in both arms.

Calibration runs set `VAL_BEFORE_TRAIN=False`, `TEST_FREQ=-1`,
`SAVE_FREQ=-1`, `KEEP_BEST_CKPT=False`, and disable W&B publication. They keep
diagnostic logs but write no training checkpoint and do not enter the pilot or
formal result matrix.

If a candidate profile fails for one arm, both arms move to the same lower
profile. No arm-specific hardware fallback is allowed in the formal matrix.

These are the only four selectable profile fields. TP remains `1`, rollout
agent workers `4`, max sequences `256`, eager mode `true`, chunked prefill
`true`, rollout/log-prob token limit `4596`, actor param offload `false`, and
optimizer offload `false`. A new candidate or any additional system variable
requires a plan amendment before calibration. In particular, “token budget”
means only actor dynamic packing budget; it never changes the 4096 response
cap, sample filtering, effective batch, or optimizer-update count.

After calibration, freeze a machine-readable `h20_profile.json` with the image
digest, driver/CUDA/PyTorch/vLLM/FlashInfer versions, the complete AFO resource
projection, all memory knobs, observed peak memory, and step timing. The
projection includes queue/usergroup, H20 resource key/count, workers, worker
memory/vcore/script, SHM, retry, and failover. A separate G4 admission binds the
profile and two distinct `SUCCEEDED` arm terminal receipts, raw status/worker
evidence, and the actual staged `run.hope` bytes. An independently allowlisted
SSH Ed25519 reviewer signs that canonical attestation in the
`rebuttal-rlvr-g4` namespace. A self-hashed profile alone can never become
`formal_frozen`. The profile, both terminal receipts, and each structured
worker-evidence receipt must carry the same exact NVIDIA-driver, CUDA-driver,
CUDA-runtime, PyTorch, vLLM, and FlashInfer projection. Its canonical hash is
covered by the G4 signature, and every formal worker re-probes the live stack
before the training launcher; a missing package/probe or any version drift is
fatal.

Before G5, every admitted cell must pass a fixed-profile preflight: all six
cells for `PILOT_FIXED_PAIR`, or all eighteen for `FORMAL`. The preflight covers
manifest/render validation, init-model load, exact data/grader receipt load,
and one rollout under its own job identity. If any cell fails, that launch mode
remains blocked until one common profile passes every cell. A failure that
still appears during execution is retained as a failed cell; it is never
dropped from the matrix or replaced.

## 8. Evaluation and Statistical Contract

### 8.1 Primary metric

The sole confirmatory metric is the unweighted macro average of the seven
dataset-level final-checkpoint `acc/mean@3` values:

$$
\mathrm{mean@3}_d =
\frac{1}{3|P_d|}\sum_{p \in P_d}\sum_{j=1}^{3} z_{d,p,j},
\qquad
Q = \frac{1}{7}\sum_{d \in \mathrm{Math7}} \mathrm{mean@3}_d,
$$

where `z` is the frozen grader's binary correctness verdict for one generated
response. Thus `mean@3` is response-level sample accuracy averaged over three
draws and prompts; it is not majority vote. For comparison:

$$
\mathrm{pass@3}_d = \frac{1}{|P_d|}
\sum_{p \in P_d}\mathbf{1}\!\left[\sum_{j=1}^{3}z_{d,p,j} \ge 1\right].
$$

The current evaluator's `pass@1` from an `n=3` sample set is algebraically the
same response-success estimate as `mean@3`, so it is not presented as
independent evidence. This plan does not require a separate `mean@1` run.

Evaluation is fixed at `n=3`, temperature `0.2`, top-p `0.95`, top-k `-1`, max
response 4096, sampling enabled, `ignore_eos=false`, tokenizer-defined EOS/stop
handling, and the frozen image's vLLM/FlashInfer version. The final checkpoint
at step 115 is primary. No Math-7 result may choose another checkpoint.

Each request seed is derived as the low 31 bits of:

```text
SHA256("math7-eval-v1|<init_pair>|<rl_seed>|<dataset_sha>|<prompt_id>|<draw_index>")
```

Every current Math-7 file has a unique `extra_info.index`. Freeze:

```text
prompt_id = "<dataset_sha256>:<canonical-json(extra_info.index)>"
```

Canonical JSON is UTF-8, keys sorted, separators `(',', ':')`, and no ASCII
escaping. Missing or duplicate `extra_info.index` fails the dataset receipt.
The evaluator must retain the raw index value and computed `prompt_id`; row
position or `data_source` alone is not an identifier.

The derivation excludes `arm` and `checkpoint_step`, so paired arms and curve
steps use the same request seeds. G2 must extend or wrap the existing evaluator
to pass the per-request seed and persist `cell_id`, `arm`, `checkpoint_hash`,
`checkpoint_step`, `dataset_sha`, stable `prompt_id`, `draw_index`,
`eval_seed`, response text, extraction result, and grader verdict. A global
vLLM seed without these per-request records does not pass.

### 8.2 Required secondary reporting

- per-benchmark `mean@3` and `pass@3`;
- Math-7 macro and dataset-size-weighted micro scores;
- step-0 initialization score and step-115 gain from initialization;
- online n=3 trajectories at step 0 and every 5 steps through step 115;
- paired final deltas for every cell and each initialization-pair mean;
- extraction failure, complete `<think>/<answer>` contract, EOS, truncation,
  response length, and grader-error rate;
- reward mean, all-correct/all-incorrect/mixed group rates, policy loss, gradient
  norm, entropy, and approximate KL diagnostics;
- completion rate, retry/OOM/NaN/timeout counts, tokens, wall time, and GPU-hours;
- initialization-only, RLVR-only, and end-to-end compute accounting.

Peak checkpoint is diagnostic only. If included, report every arm under the
same rule and label it non-confirmatory.

Offline Math-7 evaluates the admitted initialization, retained online-best
checkpoint, and final/latest step 115. Storage limits intentionally prevent
retaining all five intermediate full checkpoints. The launcher instead runs
strict-scorer online validation with `n=3` at step 0 and every 5 steps through
step 115; those
trajectories are descriptive and never select the confirmatory checkpoint.

### 8.3 Statistical analysis

Report raw scores before aggregate statistics. The checked-in analysis script
and its SHA-256 are frozen before G5; its RNG seed is `20260730`. Each of 10,000
hierarchical paired-bootstrap replicates performs this exact sequence:

1. resample the three initialization-pair clusters with replacement;
2. within each selected cluster, resample its three RLVR-seed cells with
   replacement, preserving the WDL/SFT pair;
3. independently for each of the seven benchmarks, resample stable prompt IDs
   with replacement and apply the same prompt multiplicities to both arms and
   every selected cell; keep all three draw records for a prompt as one block;
4. recompute each dataset `mean@3`, the unweighted seven-dataset macro, and the
   WDL-minus-SFT paired delta.

Report the 2.5/97.5 percentiles plus all nine raw paired deltas. The three
initialization clusters provide limited frequentist resolution, so the raw
effect sizes and consistency remain mandatory even when the interval excludes
zero.

With only one initialization pair, do prompt-level and RL-seed paired bootstrap
but do not present it as method-level uncertainty; the initialization remains a
fixed condition.

## 9. What Counts as Evidence

### 9.1 Deferred method-level support

The formal result supports the claim only when all of these hold:

1. G0-G6 integrity checks pass with the approved manifests, hashes, common H20
   profile, and no unplanned arm-specific override.
2. The final Math-7 macro paired delta is positive and its hierarchical 95%
   interval has lower bound above zero.
3. Each of the three initialization-pair mean deltas is positive; the result is
   not carried by one lucky initialization.
4. All 18 cells have an admitted final checkpoint and evaluation after the
   predeclared retry policy. A missing or substituted cell prevents a formal
   pass rather than disappearing from the denominator.

Format, stability, curve AUC, and compute are mandatory diagnostics and
disclosures; they cannot rescue a failed primary test or veto a passing primary
test through an unregistered qualitative judgment. They determine how narrowly
the result may be interpreted, especially whether any efficiency claim is
allowed.

### 9.2 Current conditional-checkpoint evidence

- positive average delta with a 95% interval crossing zero;
- one initialization pair negative while the pooled mean is positive;
- a positive six-run fixed-checkpoint paired result;
- a positive observed effect with one or more missing formal cells;
- higher final score but no reliable total-compute accounting;
- a gain visible only in secondary metrics.

These outcomes can support the exact conditional claim in Section 2, not the
deferred method-level claim.

### 9.3 Does not support the hypothesis

- zero or negative final primary delta;
- a win only on one small benchmark, one seed, or one cherry-picked checkpoint;
- higher online reward or training `mean@1` without frozen final Math-7 evidence;
- a peak-checkpoint win that disappears at step 115;
- an incomplete formal matrix after the allowed infrastructure retry;
- arm-specific LR, KL, response length, decoder, retry, or H20 profile tuning;
- use of On-Policy WDL, WDL-IS, MiniRL, GRPO-warm-start, or unverified weights as
  the offline WDL-SFT initialization;
- failure to disclose the extra initialization compute consumed by WDL-SFT.

## 10. Meituan Four-Layer Launch Contract

Implementation must create a new MATH RLVR family rather than overloading the
historical ablation names:

```text
platform/hope_rebuttal_rlvr/run.hope
platform/hope_rebuttal_rlvr/jupyter.sh
platform/hope_rebuttal_rlvr/submit_manifest.py
platform/hope_rebuttal_rlvr/manifest.schema.json
platform/hope_rebuttal_rlvr/README.md
recipe/on_policy_wdl_sft/rebuttal_rlvr/meituan/env.sh
recipe/on_policy_wdl_sft/rebuttal_rlvr/meituan/jupyter.sh
recipe/on_policy_wdl_sft/rebuttal_rlvr/run_math_{sft,wdl}.sh
recipe/on_policy_wdl_sft/rebuttal_rlvr/_common_math_rlvr.sh
recipe/on_policy_wdl_sft/rebuttal_rlvr/paired_init.schema.json
recipe/on_policy_wdl_sft/rebuttal_rlvr/validate_inputs.py
recipe/on_policy_wdl_sft/rebuttal_rlvr/run_math7_eval.sh
recipe/on_policy_wdl_sft/rebuttal_rlvr/analyze_paired_math7.py
tests/on_policy_wdl_sft/test_rebuttal_rlvr_hope_submitter.py
tests/on_policy_wdl_sft/test_rebuttal_rlvr_contract.py
```

This family uses the AFO INI `run.hope` format. It must not copy the code-task
YAML-style renderer.

### 10.1 One parent root

Layer 1 exposes exactly these required AFO environment keys:

```text
afo.app.env.ROOT = /mnt/dolphinfs/.../<user>/lgx
afo.app.env.REPO_SUBPATH = <repo-under-ROOT>
afo.app.env.REPO_COMMIT = <approved-manifest commit>
```

The Layer-2 shim must fail closed and export the interface literally:

```bash
: "${ROOT:?ROOT must be set by run.hope}"
: "${REPO_SUBPATH:?REPO_SUBPATH must be set by run.hope}"
: "${REPO_COMMIT:?REPO_COMMIT must be set by run.hope}"
export ROOT REPO_SUBPATH REPO_COMMIT
export REPO_ROOT="${REPO_ROOT:-$ROOT/$REPO_SUBPATH}"
export LGX="${LGX:-$ROOT}"
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$REPO_COMMIT"
test -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
```

`LGX` is transition compatibility only; `ROOT` must never fall back to an old
account or hard-coded DolphinFS path. `REPO_SUBPATH` should point to an immutable
worktree created for `REPO_COMMIT`, not a shared checkout that later jobs update
with `git pull`. The repo receipt also freezes and verifies recursive submodule
commit/status output before Layer 3 executes. Layer 3 derives defaults for all
persistent inputs, outputs, and caches from `ROOT`:

```text
$ROOT/models/...                 init checkpoints
$ROOT/data/math/...              RL train file
$ROOT/data/math7/...             seven evaluation files
$ROOT/verl-exp/checkpoints/...   checkpoints
$ROOT/verl-exp/eval/...          evaluation outputs
$ROOT/verl-exp/logs/...          logs and receipts
$ROOT/verl-exp/wandb_runs/...    offline W&B staging
$ROOT/verl-exp/cache/hf/...      HF_HOME / HUGGINGFACE_HUB_CACHE
$ROOT/verl-exp/cache/datasets/... HF_DATASETS_CACHE
$ROOT/verl-exp/cache/xdg/...     XDG_CACHE_HOME
```

Local/manual wrappers remain individually overridable. Formal AFO jobs derive
the repo, train file, seven Math-7 files, outputs, and caches from the single
manifest `ROOT`; inherited worker/client path variables are overwritten. Ray,
vLLM, ZMQ, and other high-churn scratch use pod-local `/tmp`, never DolphinFS.
The path receipt records the exact resolved init path plus checkpoint, eval,
log, W&B, receipt, HF, datasets, XDG, Ray, vLLM, TMP, and ZMQ roots actually
passed to the worker validator. Formal init paths must be concrete directories
below `$ROOT/models/rebuttal_rlvr/init`; merely recording that parent while
launching `$ROOT/arbitrary/other-model` fails. Init models on DolphinFS must be
flat real-file directories, not symlinked HF cache layouts. A different formal
layout requires an amended path adapter and new path/data receipts; ad hoc
environment overrides fail validation.

### 10.2 Manifest-driven parallel submission

The checked-in schema owns the interface. A pilot/formal manifest may originate
outside Git, but `validate_manifest` must write an immutable approved copy to
the experiment receipt root and freeze its SHA-256 before rendering. Each row
contains at least:

```text
schema_version, arm, init_pair, rl_seed, init_model_path,
paired_init_manifest, checkpoint_receipt, train_receipt, math7_receipt,
grader_receipt, image_reference, image_digest, h20_profile_path,
h20_profile_hash, h20_calibration_receipt, root, repo_subpath, output_policy_version,
repo_commit, repo_submodule_receipt, submitter_source_hash,
algorithm_config_hash, eval_config_hash, path_override_receipt,
attempt_policy, retry_of
```

The renderer computes `CELL_HASH` as SHA-256 over sorted, compact canonical
JSON of the schema-restricted value subset and every execution-affecting
approved field: schema version, arm, init pair, RLVR
seed, resolved init path/content/paired-init receipts, train/Math-7/grader
receipts, algorithm/eval config hashes, repo commit/submodule receipt,
submitter source hash, image digest, H20 profile and signed-calibration hashes,
ROOT/repo path and any
path-override receipt, output-policy version, and attempt policy. It excludes
only `ATTEMPT_ID`, `retry_of`, status/timestamps, and paths derived from
`JOB_TAG`. The canonical cell JSON is archived beside every attempt.

The renderer then derives identity and paths without caller-provided
alternatives:

```text
JOB_TAG=<arm>-<init_pair>-r<rl_seed>-<CELL_HASH[:12]>
RUN_PREFIX=rebuttal-rlvr-<JOB_TAG>
afo.app.name=rebuttal-rlvr-<JOB_TAG>-<ATTEMPT_ID>
afo.app.env.JOB_TAG=<JOB_TAG>
afo.app.env.CELL_HASH=<CELL_HASH>
afo.app.env.ATTEMPT_ID=<ATTEMPT_ID>
BASE_CKPT_DIR=$ROOT/verl-exp/checkpoints/rebuttal_rlvr/$JOB_TAG
LOG_DIR=$ROOT/verl-exp/logs/rebuttal_rlvr/$JOB_TAG
WANDB_DIR=$ROOT/verl-exp/wandb_runs/rebuttal_rlvr/$JOB_TAG
EVAL_ROOT=$ROOT/verl-exp/eval/rebuttal_rlvr/$JOB_TAG
RECEIPT_ROOT=$ROOT/verl-exp/receipts/rebuttal_rlvr/$JOB_TAG
```

Retries keep `JOB_TAG` and these scientific roots but receive a unique
`ATTEMPT_ID`; attempt-specific submission logs and rendered inputs live under
`$RECEIPT_ROOT/attempts/$ATTEMPT_ID`.

For each row, the submitter creates a `mktemp` stage directory under an explicit
client scratch root, copies only the INI template and shim, renders values
with a strict INI parser/serializer, and invokes `hope run run.hope` with the
stage directory as cwd. Renderer-controlled values must match
`[A-Za-z0-9_./:@+-]+` and reject CR/LF, NUL, section/comment characters, and
duplicate keys. The parsed result must match a code-owned allowlist for every
section, key, and static or manifest-derived value. `worker.script` is the
sole literal exception to the no-space value whitelist and must equal exactly
`bash jupyter.sh`; no other literal exception is allowed. The stage directory
contains only `run.hope` and its matching `jupyter.sh`. Immediately before
`hope run`, the submitter rejects symlinks or a third file, recomputes both
hashes, and archives the exact submitted bytes. Before cleanup it archives the
approved manifest, exact rendered `run.hope` and `jupyter.sh` bytes, their
hashes, resolved environment, signed H20 attestation/terminal/raw evidence,
submit stdout/stderr, return code, AFO job ID, repo/submodule receipt, and
submitter source hash under the attempt receipt. Secrets and live credentials
must never enter the manifest or receipt.

The submitter never runs training locally. Submission-process concurrency
defaults to 8 and is capped at 10, but it remains disabled until G3 records the
installed Hope return semantics. The current implementation admits only
semantics where `hope run` returns after scheduler acceptance. Completion-
blocking semantics fail closed and require a reviewed implementation amendment;
they are not treated as short RPCs.

Pilot/formal submission uses waves. `MAX_ACTIVE_JOBS` defaults to 8 and may rise only
to a colleague-confirmed quota no larger than 10. A durable ledger maps
`(JOB_TAG, ATTEMPT_ID)` to manifest SHA, full app name, AFO job ID, and last
known state. A global index maps each `JOB_TAG` to its full `CELL_HASH` and
rejects an existing receipt root with a different hash. Manifest whitespace or
repackaging cannot bypass cell-level duplicate detection. Re-running the
submitter refuses to duplicate a nonterminal or already completed cell;
the next wave is admitted only when a full-line parser recognizes every row in
the all-user platform listing and proves the active count is below the cap.
The status command must emit exactly one fully matched stdout line and empty
stderr. Active-list headers and the empty marker are frozen literal lines,
disjoint from the full-line job-row regex. Partial parsing, overlapping
classifications, stderr, duplicate IDs, mixed empty markers, or unknown state
fail closed. The G3-signed semantics receipt freezes one absolute global ledger
path so concurrent submitters share the same lock and quota state.

Before trusting the batch entry on the colleague's Hope client:

1. record `hope --help` and confirm `hope run` is the installed submission verb;
2. submit one harmless path/image smoke and verify whether `hope run` returns
   after scheduler acceptance or blocks for job completion;
3. identify the installed status/list command and freeze its raw-state mapping
   to `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, or `UNKNOWN`; queued/running
   count as active and unknown blocks admission;
4. verify lookup by unique `afo.app.name` and AFO job ID, then interrupt one
   harmless client after scheduler acceptance and reconcile the job back into
   the ledger without duplicate submission;
5. verify worker environment precedence and extract a real job ID;
6. dry-render all six pilot or all eighteen formal jobs and assert all
   identities and output roots are unique;
7. only then submit a paired two-arm H20 calibration batch.

G3 is not a self-asserted JSON flag. Its attestation payload must be signed by
an SSH Ed25519 key in the clean checkout's reviewer allowlist; the key owner
must differ from the live OS submitter identity. The signature binds the Hope
semantics hash, terminal `SUCCEEDED` smoke receipt, job/app identity, exact
smoke `run.hope`/shim/image/path hashes, and an existing review-evidence file.
The reviewer-key allowlist is empty until the colleague supplies an authorized
key through review, so pilot/formal submission currently fails closed.

The following validators are blocking G2 interfaces:

- `validate_manifest`: schema, paired-init admission, immutable hashes, unique
  `JOB_TAG`/roots, repo/submodule/source revision, and every path under `ROOT`
  or an explicit hashed allowlist;
- `validate_train_eval_disjointness`: exact source/problem hashes and the
  deterministic near-duplicate rule in Section 4.3 produce zero overlap;
- `validate_receipts`: init model, train data, all Math-7 files, grader, image,
  repo/submodules, submitter source, and H20-profile hashes exactly match the
  approved manifest;
- `validate_h20_profile`: image digest, full signed G4 evidence chain, actual
  staged INI queue/H20 count/memory/vcore/SHM/retry/failover projection, and
  every calibrated memory knob match;
- `validate_render`: no `REPLACE_*`, all `afo.app.env.*` keys resolve, exact
  ROOT/repo commit/app identity/image/resource/profile match, strict unique INI
  keys, and no unapproved environment value;
- `validate_submission_receipt`: submitted-byte hashes, return code, and real
  AFO job ID exist before a cell is marked submitted.

Each validator returns nonzero on missing or unequal input, and its negative
test must first demonstrate the gate turns red. No local training queue or
`scripts/training_queue_monitor.sh` is used for the parallel AFO matrix. The
branch training-script index must be updated when the runnable wrappers and
submitter are created or first used.

## 11. Execution Gates

| Gate | Required evidence | State |
| --- | --- | --- |
| G0 checkpoint provenance | Known facts and unavailable fields registered; owner accepts irrecoverable external provenance and narrows the claim to the supplied pair | **CONDITIONAL-CHECKPOINT ASSUMPTION ACCEPTED; MODEL PATHS PENDING** |
| G1a experiment-plan acceptance | User accepts research question, metrics, claim boundary, and matrix | **Passed in intent; fresh-reader review APPROVE on 2026-07-27** |
| G1b frozen-config review | Human review of standard-GRPO v2 env SHA, strict scorer, fixed order, online n=3, best/latest retention, and H20 boundary | **PASSED 2026-07-28; SHA `8dafbac...e54177`** |
| G2 implementation | wrappers, four layers, one-command handoff, receipt pre-registration, tests/index, and automatic post-success release wiring | **LOCAL IMPLEMENTATION PASSED: 70 focused tests; offline Math-7 analysis is deferred and is not a handoff blocker** |
| G3 Hope/image/path smoke | Required for audited parallel Hope submission, not for the owner-approved direct worker route | **LOCAL PREPARATION COMPLETE / AUDITED HOPE EVIDENCE PENDING** |
| G4 paired H20 calibration | Required for a signed common profile in the audited Hope route; direct entry performs an eight-H20 live check and uses overridable system knobs | **LOCAL PROFILE PREPARED / SIGNED MEITUAN EVIDENCE PENDING** |
| G5 launch admission | PILOT: exactly 6 admitted renders; FORMAL: exactly 18; all earlier gates, hashes, quota and all-user cap pass | Not started |
| G6 result acceptance | all final checkpoints/evals complete; release gates pass | Not started |

No later gate may waive an earlier gate. In particular, a working Meituan image
does not authorize training from unverified weights.

## 12. Implementation and Review Deliverables

Source implementation may proceed after G1a. The direct conditional route may
launch after both model paths pass its loadability checks and G1b remains bound;
the audited parallel Hope route still requires its environment-owned G3/G4
evidence. G2 must
provide:

1. two thin default-local, overridable-everything training wrappers;
2. one shared standard-GRPO v2 launcher with explicit enabled/disabled assertions;
3. the complete four-layer Meituan family and one-root path adapter;
4. a manifest schema, renderer, concurrent Hope submitter, and unique-output
   collision checks;
5. the six fail-closed validators in Section 10.2 and immutable paired-init,
   dataset, grader, image, and H20-profile receipts;
6. strict online n=3 validation every five steps; offline Math-7 and paired
   result analysis run after logs/checkpoints return and are not required to
   prepare or hand off the training queue;
7. static tests, a negative test that catches rollout-IS drift away from
   `null`, strict-scorer drift, shell syntax checks, and both six-job pilot and eighteen-job formal
   dry-render tests;
8. updated `docs/joint_training/guides/training_script_index.md` and launch
   runbook;
9. background release integration triggered only by terminal
   `success_complete`: release-gate check, registry import, W&B sync, and marker
   verification; incomplete/failed attempts remain local-only and publication
   failure is reported separately from training success.

Independent review must check checkpoint identity, algorithm provenance, paired
seed/data routing, H20 symmetry, parent-root completeness, job-output
uniqueness, metric implementation, and the no-cherry-picking rule.

Before G0, the wrappers may use Qwen3-4B-Base only with the explicit pair
`RUN_MODE=smoke ALLOW_BASE_PLACEHOLDER=1`. The launcher rewrites the classifier
to `placeholder_base_smoke` and rejects that classifier in formal mode. No
metric produced by this path is experiment evidence.

## 13. Transition to Code Tasks

Code-task planning and scripts may begin immediately after both MATH arms pass
G4 on Meituan with the same image, common H20 profile, and persistent-path
contract. Code launch
still needs its own offline WDL-SFT/ordinary-SFT checkpoint provenance gate;
the currently available code WDL checkpoint
`/data-1/.cache/Qwen3-4B-Base-Code-WDL-M1/checkpoint-39` has no matching
ordinary-SFT checkpoint and therefore cannot yet support the code comparison.

## 14. Current Handoff Record (2026-07-28)

- R02 is downloaded, load-probed, and pinned to Hub revision
  `1bfdcc4506656288b115b8fa1d4e446f4e344f12`; its download receipt records the
  8,045,067,711-byte weight and SHA-256.
- R01 is the explicit unresolved slot
  `R01_ORDINARY_SFT_4B_AM1P4M`. The colleague must provide the model directory;
  a similarly named Base or RL model is not substituted.
- Both initialization arms are registered as AM-1.4M SFT artifacts. Both
  downstream RLVR arms consume the exact 7,500-row MATH file.
- G1b is human-approved. The direct worker route checks eight H20 devices live
  and may launch after model placement. Signed G3/G4 evidence remains required
  only for the audited parallel Hope route.
- Offline Math-7 result analysis is deferred until the colleague returns logs
  and checkpoints. It does not block preparing the training handoff.

The direct route's next input is only placement of R01 and R02 at their named
paths. The colleague then runs `platform/hope_rebuttal_rlvr/run_colleague.sh`
with `R01|R02` and one registered seed. The separate audited parallel route
continues to wait for Meituan-owned G3/G4 receipts.
