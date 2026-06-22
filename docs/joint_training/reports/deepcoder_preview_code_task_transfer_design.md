# DeepCoder-Preview Code-Task Transfer Design

- Status: EXPERIMENT DESIGN DOCUMENT
- Created: 2026-06-08
- Branch: `feature/on-policy-wdl-sft`
- Dataset: `agentica-org/DeepCoder-Preview-Dataset`
- Primary train file: `/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet`
- Related completed domain run: KodCode Stage1 + Stage2 code-task experiment

This is an experiment-design document, not a `/goal` execution plan. The
current executable Stage1 plan lives in
`docs/joint_training/plans/active/deepcoder_stage1_training_execution_plan.md`.

## Objective

Transfer the code-task On-Policy WDL-SFT Stage1 -> Stage2 protocol from
KodCode-Light-RL-10K to DeepCoder-Preview-Dataset, while keeping the scientific
question unchanged:

```text
Does Stage2, inserted at an early Stage1 handoff checkpoint, improve over the
same-budget Stage1 continuation on code tasks without reward/execution collapse?
```

DeepCoder is a new data-domain experiment, not a continuation of KodCode. The
first launch must therefore re-check the data/reward contract and then collect a
dense Stage1 trajectory before selecting any Stage2 handoff step.

## Fixed Data Decision

Use the processed DeepCoder train split for training:

| Split | Path | Rows | Role |
| --- | --- | ---: | --- |
| train | `/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet` | 23,287 | Stage1 training source |
| dev | `/data-1/dataset/code/verl_rl/deepcoder_preview_dev_rl_format.parquet` | 1,000 | data/reward sanity and in-domain diagnostics only |
| official test | `/data-1/dataset/code/verl_rl/deepcoder_preview_official_test_rl_format.parquet` | 687 | holdout diagnostics; never train |

Keep the current training seed unless a follow-up ablation explicitly changes it:

| Seed | Value | Meaning |
| --- | ---: | --- |
| `DATA_SEED` | `20260604` | training sampler seed, kept from KodCode for cross-domain control |
| DeepCoder split seed | `20260608` | data conversion split seed, already recorded in the DeepCoder audit |

## Data Statistics Gate

The DeepCoder audit and the 2026-06-08 mock read give the following pre-launch
statistics.

| Statistic | DeepCoder | KodCode-Light-RL-10K | Reading |
| --- | ---: | ---: | --- |
| train rows | 23,287 | 10,000 | DeepCoder is 2.33x larger |
| steps per epoch at `TRAIN_PROMPT_BSZ=64` | 364 | 157 | 150 steps sees 41.2% of DeepCoder but 96.0% of KodCode |
| prompt tokens mean / median | 585 / 556 | 289 / 221 | DeepCoder prompts are about 2x longer |
| prompt tokens p95 / max | 1,031 / 6,239 | 632 / 2,087 | DeepCoder has many more long prompts |
| prompts over 1024 tokens | 1,288 | 9 | `MAX_PROMPT_LENGTH=1024` truncation/filter behavior matters |
| difficulty proxy easy / medium / hard | 1,620 / 16,423 / 6,931 | 8,335 / 1,612 / 53 | DeepCoder is much harder by local proxy |
| train parquet size | 4.8G | 19M | DeepCoder rows carry much heavier test payloads |

DeepCoder source mix:

| Source | Total rows | Train rows in parquet mock | Notes |
| --- | ---: | ---: | --- |
| PrimeIntellect Synthetic-1 | 16,252 | 15,607 | dominant source |
| TACO verified | 7,436 | 7,100 | contest-style tasks |
| LiveCodeBench v5 | 878 | 580 | train plus test split |
| Codeforces | 408 | 0 in train | official-test holdout source |

Mock checks already completed on 2026-06-08:

- `pandas.read_parquet` can read train/dev/KodCode parquets.
- DeepCoder train/dev schema is `data_source, ability, reward_model, prompt, split, extra_info`.
- The prompt contract contains `<think>`, `<answer>`, and fenced `python`.
- `reward_model` and `extra_info` are JSON strings for DeepCoder, compatible
  with the current `rl_dataset.py` JSON-string restoration path.
- Stage1 wrapper dry-run with DeepCoder overrides exits before
  `verl.trainer.main_ppo` and prints the expected DeepCoder config.

Do not run generated code, reward probes, or eval harnesses from the repo root.
Follow `docs/joint_training/constraints/principles/workspace_artifact_hygiene.md`;
scratch work goes under `/data-1/tmp/verl_agent_scratch/...`, while durable
reports go under `/data-1/dataset/DeepCoder-Preview-Dataset/reports/`.

## Experiment Batches

### Batch 0: Data and reward mock

Purpose: prove the DeepCoder conversion, prompt contract, and reward dispatch
are still aligned after switching away from KodCode.

Status: partially complete from the 2026-06-08 mock. Before training launch,
repeat only the minimum checks needed if code changed:

```bash
python3 - <<'PY'
import pandas as pd
for path in [
    "/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet",
    "/data-1/dataset/code/verl_rl/deepcoder_preview_dev_rl_format.parquet",
]:
    df = pd.read_parquet(path, columns=["data_source", "prompt", "reward_model", "extra_info"])
    print(path, len(df), df["data_source"].value_counts().head().to_dict())
PY
```

Acceptance:

- train/dev parquet read succeeds;
- prompt contract still matches `code-think-answer-python-v1`;
- source counts match the audit within the expected train/dev split;
- no files are created outside approved scratch/report roots;
- if reward code changed, a small reference-answer probe passes before Stage1.

### Batch 1: Stage1 DeepCoder beta sweep with dense handoff retention

Run the first two real DeepCoder Stage1 experiments after explicit user
approval:

| Run | Beta | Steps | Protected handoff checkpoints | Purpose |
| --- | ---: | ---: | --- | --- |
| `DEEPCODER-S1-BETA0` | 0.0 | 150 | 30,40,...,140; latest 150 | forward-only Stage1 baseline on DeepCoder |
| `DEEPCODER-S1-BETA01` | 0.1 | 150 | 30,40,...,140; latest 150 | matched reverse-SFT Stage1 variant |

Keep fixed against KodCode unless a launch review changes it:

- `INIT_MODEL_PATH=/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/...`
- `LR=5e-7`
- `TRAIN_PROMPT_BSZ=64`
- `ROLLOUT_N=8`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=4096`
- `DATA_SEED=20260604`
- `DATA_SHUFFLE=True`
- online validation: HumanEval+ and MBPP+ only;
- online decode: `VAL_N=1`, `VAL_TEMPERATURE=0.2`, `VAL_TOP_P=0.95`;
- best metric: `val-core/HumanEval+/acc/pass@1`;
- BigCodeBench and LiveCodeBench stay offline diagnostics, not online plateau
  selectors.

Because the current shared queue was written for KodCode defaults, the first
DeepCoder launch must either use direct wrappers with all dataset env vars
inline, or add a DeepCoder-specific thin queue wrapper before launch. Do not rely
on implicit tmux environment inheritance for `CODE_TRAIN_FILE`.

Direct dry-run example already passed:

```bash
DRY_RUN=1 \
CODE_TRAIN_FILE=/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet \
TRAIN_FILE=/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet \
RUN_PREFIX=ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION \
DATA_SEED=20260604 \
TOTAL_TRAINING_STEPS=150 \
SAVE_FREQ=5 TEST_FREQ=5 \
PROTECTED_CKPT_STEPS='[30,40,50,60,70,80,90,100,110,120,130,140]' \
PROTECTED_CKPT_STRIP_OPTIMIZER=True \
MIN_FREE_GB_FOR_CKPT=300 \
bash recipe/on_policy_wdl_sft/code_task/run_s1_code_onpolicy_sft_beta_0_retention.sh
```

Non-dry-run must be launched in tmux only after approval and disk gating.

### Batch 2: Stage2 handoff intervention

Run Stage2 after the Stage1 curves identify the handoff step. The exact
handoff step is intentionally not fixed yet.

Candidate handoff set:

```text
30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140
```

Selection rule:

- pick the earliest step that has a stable local plateau or strong upward
  inflection on HumanEval+ without MBPP+ collapse;
- if beta `0.0` and beta `0.1` peak in different regions, allow different
  handoff steps but report them explicitly;
- do not use `official_test` to choose the handoff;
- compare Stage2 to both its Stage1 source checkpoint and same-budget Stage1
  continuation.

Stage2 train data:

- create a DeepCoder non-overlap shard only after the handoff step is selected;
- use `create_code_stage2_nonoverlap_shard.py --source deepcoder_preview_train...`
  with the same `DATA_SEED=20260604`;
- verify zero overlap against the Stage1 consumed sampler prefix.

Presentation target:

- final paper/table view should focus on effective steps 0-100 where possible;
- Stage1 still trains to 150 so we can see whether DeepCoder plateaus later than
  KodCode and avoid choosing a handoff from a truncated curve.

### Batch 3: Baseline

Run a DeepCoder baseline after the Stage1 data-domain transfer is known healthy.
The baseline must use the same train split, init model, online validation,
decode settings, seed, batch geometry, and checkpoint budget unless the plan is
explicitly revised.

Preferred baseline for the paper:

```text
single-model code RL / MiniRL-style baseline on DeepCoder, evaluated with the
same online pass@1 curve and offline mean@3 plus pass@3 diagnostics.
```

If implementation time is tight, keep a same-budget Stage1 continuation from the
selected handoff as the minimum internal baseline for Stage2. Do not present
Stage2 as effective unless it beats that same-budget continuation under the same
validation protocol.

## Disk Estimate and Launch Gate

Observed KodCode retention checkpoint sizes:

| Checkpoint class | Observed size |
| --- | ---: |
| protected model-weight-only handoff checkpoint | about 17G each |
| normal latest step-150 checkpoint with optimizer | about 47G |
| completed beta `0.0` retention run, protected 70-120 plus latest | 162G |
| completed beta `0.1` retention run, protected 70-120 plus latest | 146G |

DeepCoder requested retention is denser: step 30 through 140 every 10 steps,
plus latest step 150. Estimated checkpoint footprint:

| Scope | Estimate |
| --- | ---: |
| one beta, 12 protected checkpoints plus latest | about 251G |
| one beta with an extra distinct best checkpoint | about 268G |
| two betas | about 502-536G |

Current live disk check on 2026-06-08 showed only about 68G free on `/data-1`.
Therefore Batch 1 cannot be launched safely on the current disk state.

Launch gate:

- require at least 300G free for one beta;
- require at least 600G free for the two-beta Stage1 batch;
- or implement verified archive/offload semantics before dense retention;
- do not lower the checkpoint-retention set without user approval, because
  Stage2 handoff selection depends on those checkpoints.

## Control Variables

The following must remain fixed across DeepCoder Stage1, Stage2, and baseline
runs unless an explicit ablation says otherwise:

- base model and tokenizer;
- train split and data seed;
- train batch size, rollout count, learning rate, max prompt/response length;
- reward function: `official_aligned_reward.py`;
- online validation files and decode settings;
- offline eval decode: report both `mean@3` and `pass@3` when using N=3;
- W&B project and metrics path conventions;
- tmux-based launch and monitor flow.

## Expected Outcomes

DeepCoder should be harder than KodCode. Expect lower early online pass@1,
lower reward correct ratio, more reward latency, and possibly a later Stage1
plateau because 150 steps covers only 41.2% of the train split. If the method is
working, the useful signal is not merely that Stage1 improves; it is that a
short Stage2 from a selected handoff improves over same-budget Stage1
continuation and remains stable on both HumanEval+ and MBPP+.

Negative or ambiguous outcomes are still informative:

- Stage1 improves but Stage2 fails to beat same-budget continuation: code-domain
  evidence for Stage2 remains unverified.
- HumanEval+ improves while MBPP+ regresses: treat as benchmark-specific, not a
  clean method win.
- reward health degrades on DeepCoder: stop and fix the execution/reward path
  before launching Stage2 or baseline.

## Approval Checkpoints

Agents may do read-only inspection, dry-runs, and Batch 0 mock checks without
launch approval. The following require explicit user approval:

- any command that enters `verl.trainer.main_ppo`;
- any tmux training queue;
- checkpoint deletion, retention reduction, or archive cleanup;
- selecting the Stage2 handoff step;
- launching Batch 2 Stage2 or Batch 3 baseline.
