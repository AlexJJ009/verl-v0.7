# Goal: Stage 1 On-Policy SFT Beta Grid Search

- Status: IMPLEMENTED AND SMOKE-PASSED - Stage 1 wrappers/queue/monitor exist and passed a 10-step usability smoke; full beta grid still requires separate explicit authorization
- Target branch: `feature/on-policy-wdl-sft`
- Target loss mode: `wdl_sft`
- Current target stage: Stage 1 single-model On-Policy SFT only
- Deferred stage: Stage 2 joint On-Policy WDL-SFT, not part of this goal
- Target recipe family: `recipe/on_policy_wdl_sft/staged_v1/`
- Target platform family: `platform/hope_staged_v1/`
- Goal status file: `docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search_status.md`
- W&B project: `OnPolicySFT-Then-WDLSFT-StagedV1`
- Created: 2026-05-28
- Last updated: 2026-05-28

## 1. Objective

Define an executable plan for the current experimental phase:

1. Run **Stage 1 single-model On-Policy SFT** from Qwen3-4B-Base.
2. Search Stage 1 `wdl_sft_beta` over:
   ```text
   0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
   ```
3. For each beta run, select the best checkpoint by dense validation.
4. Across beta runs, select the best Stage 1 beta/checkpoint candidate.

This goal is **not** to run Stage 2. Stage 2 joint WDL-SFT is recorded only as a future phase after Stage 1 beta search has produced a healthy selected checkpoint and the user explicitly authorizes the next phase.

The plan-authoring phase is complete. The current execution goal is to implement and verify the Stage 1 beta-search path. A short Stage 1 usability smoke may be launched as validation, but the full 11-run beta grid still requires separate explicit authorization.

## 2. Decision

The immediate workflow is:

```text
finalize Stage 1 beta-search plan
-> implement/adjust Stage 1 beta wrappers and monitor
-> run an optional short Stage 1 smoke for usability validation
-> after user authorization, run Stage 1 beta grid
-> compare best checkpoints across beta values
-> stop and report Stage 1 result
-> Stage 2 remains deferred until separately authorized
```

The beta grid in this goal belongs to **Stage 1**, not Stage 2.

Rationale:

- The project now treats On-Policy WDL-SFT as a later workflow, not the first training step.
- Before introducing the two-model Stage 2 workflow, we need to understand how the simpler single-model On-Policy SFT behaves under different reverse-SFT beta values.
- If Stage 1 beta search fails or shows no healthy checkpoint, there is no reason to spend compute on Stage 2.

## 3. Code Status

The v1 loss already exists in the current branch:

- `verl/trainer/ppo/core_algos.py` registers `loss_mode="wdl_sft"`.
- `verl/trainer/ppo/ray_trainer.py` applies raw reward labels for `loss_mode in {"wdl_sft", "wdl_sft_is"}` before actor update.
- The loss is ordinary reward-filtered SFT:
  - correct rollouts receive forward SFT;
  - incorrect rollouts receive reverse SFT only when `wdl_sft_beta > 0`;
  - no `old_log_prob` IS correction;
  - no rollout IS weights;
  - no GRPO group advantage as the learning signal.

Therefore this goal should not add a new core loss. It should use the existing code path with script-level configuration.

## 4. Non-Negotiable Boundaries

### In Scope

- Prepare and verify Stage 1 beta-search scripts only.
- Keep Stage 1 as single-model on-policy SFT:
  - `loss_mode=wdl_sft`
  - `joint_training=False`
  - `rollout_is=null`
  - `rollout_rs=null`
  - no joint model
- Search Stage 1 `wdl_sft_beta` over `0.0..1.0` at `0.1` intervals.
- Use the same data seed and data policy across all beta values.
- Support local launch and Meituan AFO four-layer launch for Stage 1 beta runs.
- Use W&B offline during training, then sync/upload offline runs after each run.
- Keep dense validation/checkpoint cadence for curve visibility.
- Keep `VAL_BEFORE_TRAIN=False` by default. The same Base model should not repeatedly run step-0 validation for every beta.
- Keep latest full checkpoint plus best checkpoint only for each beta run.
- Maintain a monitor/queue that can run the Stage 1 beta grid sequentially and check disk space before every run.

### Explicitly Out Of Scope

- Do not implement or expand Stage 2 beta search in this goal.
- Do not launch Stage 2 beta grid.
- Do not use Stage 2 results as an acceptance criterion for this goal.
- Do not change the core `wdl_sft` algorithm.
- Do not add importance sampling, rollout IS, KL penalty, or GRPO group-advantage logic.
- Do not launch any training or smoke test until the user explicitly authorizes execution.

Stage 1 wrappers must explicitly disable KL-related behavior instead of relying on inherited defaults:

```text
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0
```

## 5. Two Search Axes

This workflow has two different searches. They must not be conflated.

### 5.1 Checkpoint Search Inside Each Beta Run

Every Stage 1 beta run produces multiple validation points and checkpoints. The chosen checkpoint for that beta is selected by:

```text
val-core/HuggingFaceH4/MATH-500/acc/mean@3
```

Default checkpoint/validation policy:

```text
TEST_FREQ=5
SAVE_FREQ=5
VAL_N=3
VAL_BEFORE_TRAIN=False
KEEP_BEST_CKPT=True
MAX_ACTOR_CKPTS_TO_KEEP=1
MAX_CRITIC_CKPTS_TO_KEEP=1
BEST_CKPT_METRIC_KEY=val-core/HuggingFaceH4/MATH-500/acc/mean@3
BEST_CKPT_METRIC_MODE=max
BEST_CKPT_STRIP_OPTIMIZER=True
```

Output for each beta run:

- best checkpoint path;
- best step;
- best MATH-500 `mean@3`;
- AIME sanity metric if available;
- latest checkpoint path;
- W&B offline directory and final W&B URL or sync blocker.

### 5.2 Beta Grid Search Across Stage 1 Runs

Stage 1 beta grid:

```text
0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

Comparison must use the same:

- init model;
- train file;
- `data.seed`;
- total training steps;
- validation cadence;
- prompt batch;
- rollouts per prompt;
- max prompt/response length;
- learning rate;
- W&B project;
- checkpoint selection metric.

If a later learning-rate search is needed, create a separate goal. This goal searches Stage 1 beta, not learning rate.

## 6. Stage 1 Default Configuration

Purpose: train Qwen3-4B-Base with single-model on-policy SFT while sweeping reverse-SFT beta.

| Knob | Value |
| --- | --- |
| Init model | `/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Train file | `/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet` |
| Loss | `wdl_sft` |
| Beta grid | `0.0..1.0`, step `0.1` |
| Joint model | disabled |
| Rollout correction | `rollout_is=null`, `rollout_rs=null` |
| LR | `5e-7` |
| Total steps | `150` by default for authorized beta screening; `10` only for authorized usability smoke |
| Prompt batch | `64` for beta screening; may be reduced for smoke if needed |
| Rollouts per prompt | `8` |
| Max prompt / response | `500 / 4096` |
| Validation | MATH-500 + AIME-2025 |
| Validation samples | `VAL_N=3` |
| Validation/checkpoint cadence | `TEST_FREQ=5`, `SAVE_FREQ=5` |
| Validation before train | disabled, `VAL_BEFORE_TRAIN=False` |
| Data seed | `20260528` |
| Retention | latest full checkpoint + best checkpoint only |
| W&B mode | `WANDB_MODE=offline` |

Stage 1 pass condition for a beta run:

- run reaches the configured final step, or failure is concretely diagnosed;
- W&B offline run is created;
- validation outputs are recorded;
- `best_checkpoint.json` exists or the latest checkpoint is explicitly recorded as fallback;
- checkpoint retention does not keep more than latest plus best;
- disk usage stays within the guardrail.

Stage 1 pass condition for the grid:

- every beta value either completes or has a recorded blocker;
- completed beta runs are compared in one table;
- the selected beta/checkpoint is justified by metrics and health checks;
- missing runs are clearly marked as blockers rather than silently ignored.

## 7. W&B Policy

All Stage 1 beta-search experiments should use the dedicated W&B project:

```text
WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1
```

Training policy:

1. During training, use W&B offline mode:
   ```text
   WANDB_MODE=offline
   ```
2. After each run finishes or is intentionally stopped, upload the offline W&B run directory to the W&B project with the staged v1 sync helper.
3. The status file must record:
   - local W&B offline directory;
   - W&B project;
   - sync command;
   - sync result or blocker;
   - final W&B URL if available.

Meituan workers must default to `WANDB_MODE=offline`.

If W&B sync fails because of missing login, network, proxy, or Meituan worker connectivity, record the offline run directory and exact retry command before handoff.

## 8. Training Data Policy

The EnsembleLLM train parquet has about 104,916 prompts. With `train_batch_size=64`, one full pass is about 1,639 optimizer steps.

| Steps | Prompt coverage | Use |
| ---: | ---: | --- |
| 10 | 640 prompts, about 0.6% | Usability smoke only. |
| 150 | 9,600 prompts, about 9.2% | Default Stage 1 beta screening after authorization. |
| 300 | 19,200 prompts, about 18.3% | Confirmation if the 150-step curve is still clearly improving. |
| 500 | 32,000 prompts, about 30.5% | Extended Stage 1 curve if runtime permits. |
| 1,000 | 64,000 prompts, about 61.0% | Expensive confirmation after choosing a candidate beta. |
| 1,639 | one full pass | Expensive final check, not the first beta search. |

To avoid data-subset randomness:

- keep `data.shuffle=true`, but fix `data.seed=20260528`;
- keep `data.train_max_samples=-1` by default, so the dataloader samples a deterministic shuffled prefix of the full dataset for short runs;
- do not compare beta runs that use different seeds or different `train_max_samples`;
- if explicit subsets are needed later, create named parquet shards once, record their construction script and seed, and reuse the same shard across all beta values.

## 9. Disk And Monitor Policy

Each Stage 1 run is expected to need roughly `50-65G` when retaining latest plus best checkpoints, plus W&B/log overhead. Use `80G/run` as the conservative planning number.

For the full Stage 1 beta grid:

| Count | Estimate |
| ---: | ---: |
| 1 beta run | `50-65G`, plan for `80G` |
| 11 beta runs | `550-715G`, plan for about `880G` |

Before each beta run, the monitor/queue must check:

- free space under the checkpoint/output mount;
- free space under W&B/log mount if separate;
- GPU availability;
- no conflicting training container/session is already running;
- expected output directory does not collide with an existing run unless resuming is explicit.

Default disk guard:

```text
MIN_FREE_GB_FOR_CKPT=160
```

If the host cannot retain all 11 beta runs at once, the plan must switch to an explicit archival/cleanup policy before launching the grid. Do not rely on implicit cleanup.

## 10. Required Files For The Current Stage 1 Goal

These are the files this plan expects for the current Stage 1 implementation. Existing Stage 2 files may remain in the repo, but they are not acceptance criteria for this goal.

Recipe family:

```text
recipe/on_policy_wdl_sft/staged_v1/
├── README.md
├── run_s1_base_sft.sh
├── run_s1_beta_0.sh
├── run_s1_beta_01.sh
├── run_s1_beta_02.sh
├── run_s1_beta_03.sh
├── run_s1_beta_04.sh
├── run_s1_beta_05.sh
├── run_s1_beta_06.sh
├── run_s1_beta_07.sh
├── run_s1_beta_08.sh
├── run_s1_beta_09.sh
├── run_s1_beta_10.sh
├── run_stage1_beta_search_queue.sh
├── sync_wandb_offline.sh
└── meituan/
    ├── env.sh
    └── jupyter.sh
```

Platform family:

```text
platform/hope_staged_v1/
├── README.md
├── jupyter.sh
├── run.hope
└── submit_beta_search.sh
```

Required Stage 1 Meituan experiments:

```text
s1-beta-0
s1-beta-01
s1-beta-02
s1-beta-03
s1-beta-04
s1-beta-05
s1-beta-06
s1-beta-07
s1-beta-08
s1-beta-09
s1-beta-10
```

## 11. Meituan Requirements

The Stage 1 beta-search workflow must follow the four-layer Meituan launch pattern:

| Layer | Path |
| --- | --- |
| Platform template | `platform/hope_staged_v1/` |
| Platform shim | `platform/hope_staged_v1/jupyter.sh` |
| Recipe adapter | `recipe/on_policy_wdl_sft/staged_v1/meituan/` |
| Run wrappers | `recipe/on_policy_wdl_sft/staged_v1/run_*.sh` |

All paths must be default-local and overridable. DolphinFS paths belong only in `meituan/env.sh` or `run.hope`.

## 12. Implementation Tasks

1. Update the goal-local status file to say the current goal is Stage 1 beta search only.
2. Verify the existing `wdl_sft` code path and record that no core loss change is required.
3. Ensure `run_s1_base_sft.sh` can accept a beta override without changing algorithm code.
4. Create or update Stage 1 beta wrappers for every beta value from `0.0` to `1.0`.
5. Create or update the Stage 1 beta-search queue/monitor.
6. Ensure the queue/monitor checks disk space before every run.
7. Ensure all Stage 1 wrappers use:
   - `LOSS_MODE=wdl_sft`;
   - `WDL_SFT_BETA=<grid value>`;
   - `JOINT_TRAINING=False`;
   - `ROLLOUT_IS=null`;
   - `ROLLOUT_RS=null`;
   - `algorithm.use_kl_in_reward=False`;
   - `actor_rollout_ref.actor.use_kl_loss=False`;
   - `actor_rollout_ref.actor.kl_loss_coef=0.0`;
   - `TOTAL_TRAINING_STEPS=150` default;
   - `VAL_N=3`;
   - `VAL_BEFORE_TRAIN=False`;
   - `TEST_FREQ=5`;
   - `SAVE_FREQ=5`;
   - fixed `DATA_SEED=20260528`;
   - W&B project and offline mode defaults.
8. Ensure Meituan four-layer launch supports all Stage 1 beta wrappers.
9. Ensure W&B offline sync helper usage is documented for this recipe family.
10. Update `docs/joint_training/guides/training_script_index.md`.
11. Update active plan index and bridge docs if the goal entry changes.
12. Run shell syntax checks.
13. Start one real short Stage 1 smoke under the current execution goal, unless a concrete runtime blocker is found first.
14. Stop after the smoke and report usability. Do not launch the full beta grid without user authorization.

## 13. Usability Smoke Acceptance

The plan may use one short real training smoke only after explicit user authorization.

Default smoke shape:

```bash
tmux new-session -s staged_v1_s1_smoke
TOTAL_TRAINING_STEPS=10 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=5 \
VAL_N=3 \
WANDB_MODE=offline \
WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1 \
WDL_SFT_BETA=0.0 \
bash recipe/on_policy_wdl_sft/staged_v1/run_s1_base_sft.sh
```

Acceptance:

- launched in tmux;
- real training path starts, not just shell dry-run;
- `loss_mode=wdl_sft`;
- configured `wdl_sft_beta` is visible in logs/config;
- `joint_training=False`;
- `rollout_is=null`;
- W&B offline directory is created;
- run reaches final configured step, or any failure is diagnosed and recorded;
- status file records command, tmux session, log path, checkpoint path, W&B offline dir, and result.

Smoke should disable pre-training validation. Full validation belongs to pilot/full runs, not to the 10-step usability smoke.

Smoke is not evidence that a beta value is good. It only proves the Stage 1 training path is usable.

## 14. Full Stage 1 Beta-Grid Handoff After User Authorization

After user authorization, execute:

1. Launch the Stage 1 beta queue, default `TOTAL_TRAINING_STEPS=150`.
2. Run beta values sequentially unless the user explicitly authorizes parallel execution.
3. Before each beta run, check disk space and GPU/process availability.
4. Monitor until completion or failure.
5. Sync each W&B offline run after completion.
6. Inspect `best_checkpoint.json`, validation curve, and latest checkpoint for each completed beta.
7. Compare beta values in a single Stage 1 results table.
8. Select the best Stage 1 beta/checkpoint candidate, or report that the grid did not produce a healthy candidate.
9. Stop. Do not launch Stage 2.

Stage 1 beta comparison table must include at least:

- beta value;
- W&B run URL or offline sync blocker;
- best checkpoint path and best step;
- MATH-500 `mean@3` best score;
- AIME sanity metric;
- latest checkpoint path;
- run status: complete, failed, skipped, or blocked;
- disk-space notes if relevant;
- whether the run is eligible for longer confirmation.

## 15. Deferred Stage 2

Stage 2 is intentionally deferred.

Future Stage 2 intent:

- use original Base as model1;
- use selected Stage 1 checkpoint as model2;
- run joint On-Policy WDL-SFT with `loss_mode=wdl_sft`;
- decide later whether Stage 2 needs its own beta grid.

Stage 2 must not be treated as part of this goal's implementation, monitor, smoke, or done definition. It requires a new or revised plan and explicit user authorization after Stage 1 beta search is complete.

## 16. Agent Work Split

The Main Agent owns final integration, file edits, launch decisions, status file maintenance, and user-facing conclusions. Subagents may explore or review, but they do not own final integration.

| Subtask | Owner | Reviewer gate |
| --- | --- | --- |
| Method boundary | Main or Method subagent | Method reviewer |
| Stage 1 beta scripts | Main or Script subagent | Script/Meituan reviewer |
| Stage 1 queue/monitor | Main or Ops subagent | Ops reviewer |
| W&B offline/upload flow | Main or Ops subagent | Ops/W&B reviewer |
| Optional smoke run | Main Agent | Runtime reviewer |
| Final docs/status | Main Agent | Final reviewer |

Reviewer protocol:

- Reviewer input must include this goal file, the status file, relevant diff, and command outputs.
- Reviewer verdicts are `PASS`, `WARN`, or `FAIL`.
- `FAIL` blocks dependent work unless the user explicitly accepts the failure.
- `WARN` requires either a fix or a recorded limitation in the status file.
- Main Agent must summarize reviewer feedback and response in the status file.
- The Main Agent must not move into a dependent milestone until the corresponding reviewer gate is `PASS` or has a recorded `WARN` with accepted limitation and follow-up.

## 17. Reviewer Gates

### 17.1 Method Reviewer

PASS requires:

- The current goal is clearly Stage 1 only.
- Stage 1 is clearly single-model on-policy SFT.
- The Stage 1 beta grid is explicit: `0.0..1.0` at `0.1` intervals.
- Stage 2 is clearly deferred and out of scope.
- No IS, rollout IS, KL, or GRPO group-advantage mechanism is introduced.
- Checkpoint search and beta grid search are described as separate searches.

### 17.2 Script/Meituan Reviewer

PASS requires:

- all current-scope files in Section 10 exist after implementation is authorized;
- Stage 1 beta wrappers are runnable locally;
- Stage 1 beta wrappers are runnable via Meituan `EXPERIMENT=s1-beta-*`;
- no Stage 2 wrapper is required for this goal's PASS;
- all parent paths are default-local and overridable;
- Meituan path overrides stay in `meituan/env.sh` or platform templates;
- shell syntax checks pass.

### 17.3 Ops/W&B Reviewer

PASS requires:

- `WANDB_MODE=offline` is the training default;
- `WANDB_PROJECT=OnPolicySFT-Then-WDLSFT-StagedV1` is set by the staged family;
- W&B sync helper exists and documents exact usage;
- queue/monitor checks disk space before each beta run;
- status file records W&B offline dirs and sync results;
- if W&B sync fails because of missing login, network, or Meituan worker connectivity, the offline run directory and exact retry command are recorded before handoff.

### 17.4 Runtime Reviewer

PASS requires, only if user authorized a smoke:

- 10-step Stage 1 smoke launch command is recorded;
- tmux session name is recorded;
- log path and checkpoint path are recorded;
- smoke reaches final step or the failure is concretely diagnosed;
- no full beta-grid training was launched without user authorization;
- no Stage 2 training was launched;
- smoke is not used to make checkpoint, beta, or model-quality conclusions.

## 18. Status Discipline

Maintain:

```text
docs/joint_training/plans/active/on_policy_sft_then_wdl_sft_beta_search_status.md
```

The status file must include:

- branch and latest relevant commit if any;
- current milestone;
- intentional files changed;
- scripts created or changed;
- tests/checks run and exact results;
- smoke command, tmux session, log path, checkpoint path, W&B offline dir, and result if a smoke is authorized;
- W&B sync command and result after any run is complete;
- reviewer verdicts;
- blockers or user decisions needed;
- next concrete action.

At the start of any resumed session, read the status file before continuing. Before handoff or final response, update it with the current state.

## 19. Done Definition For This Goal

This goal is complete only when:

- The plan clearly states that the current goal is Stage 1 beta search only.
- Stage 2 is explicitly deferred and excluded from acceptance.
- Stage 1 beta wrappers cover `0.0..1.0` at `0.1` intervals after implementation is authorized.
- A Stage 1 queue/monitor exists after implementation is authorized and can run the beta grid sequentially.
- The monitor checks disk space before each run.
- `TOTAL_TRAINING_STEPS=150`, `VAL_N=3`, `VAL_BEFORE_TRAIN=False`, `TEST_FREQ=5`, `SAVE_FREQ=5`, latest+best retention, and fixed `DATA_SEED=20260528` are configured.
- W&B offline project and sync helper are in place.
- Meituan four-layer launch path supports Stage 1 beta wrappers.
- Training script index is updated after runnable scripts are created or changed.
- Shell syntax checks pass after script edits.
- A real Stage 1 10-step smoke is launched only if the user authorizes it, or a concrete runtime blocker is recorded.
- No full beta grid and no Stage 2 run is launched without user authorization.
- The status file is current enough for a new agent to resume without chat history.
