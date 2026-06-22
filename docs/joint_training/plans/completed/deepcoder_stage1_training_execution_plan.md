# DeepCoder Stage1 Training Execution Plan

- Status: COMPLETED / SUPERSEDED BY NEGATIVE TRANSFER RESULT
- Created: 2026-06-09
- Superseded: 2026-06-22 by
  `docs/joint_training/reports/deepcoder_kodcode_failure_analysis.md`
- Branch: `feature/on-policy-wdl-sft`
- Scope: DeepCoder Stage1 only; no Stage2 handoff selection, no Stage2 launch,
  no baseline launch.
- Design doc: `docs/joint_training/reports/deepcoder_preview_code_task_transfer_design.md`

## Supersession Note

This plan is preserved as the execution contract for the DeepCoder Stage1 batch.
It is no longer an active `/goal` target. The resulting DeepCoder runs did not
serve as a good drop-in replacement for KodCode under the current code-task
Stage1 setup, so the main code-task line returns to KodCode. Any future
DeepCoder work should be created as a new controlled ablation plan, not resumed
from this Stage1 execution plan.

## Goal

Complete the first DeepCoder Stage1 training batch using the processed
DeepCoder-Preview train split filtered under the `verl-harness` Docker runtime
to `prompt_tokens <= 1024`:

| Run | Beta | Prefix | Final step | Protected checkpoints |
| --- | ---: | --- | ---: | --- |
| DeepCoder Stage1 beta0 | 0.0 | `ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION` | 150 | 30,40,...,140 plus latest 150 |
| DeepCoder Stage1 beta01 | 0.1 | `ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION` | 150 | 30,40,...,140 plus latest 150 |

This plan is intentionally narrower than the DeepCoder transfer design. It
exists so a `/goal` can execute Stage1 without making Stage2 or baseline
decisions.

## Non-Goals

- Do not select the Stage2 handoff step.
- Do not create a Stage2 non-overlap shard.
- Do not merge Model2 or launch Stage2.
- Do not run the DeepCoder baseline.
- Do not delete, archive, or reduce protected checkpoints without explicit user
  approval.

## Data and Fixed Configuration

| Variable | Value |
| --- | --- |
| `CODE_TRAIN_FILE` / `TRAIN_FILE` | `/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet` |
| source train manifest | `/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_manifest.json` |
| filtered train rows | `22,063` kept from `23,287`; `1,224` overlong prompts removed |
| filtered dev parquet | `/data-1/dataset/code/verl_rl/deepcoder_preview_dev_prompt1024_rl_format.parquet` |
| filtered dev rows | `952` kept from `1,000`; `48` overlong prompts removed |
| `DATA_SEED` | `20260604` |
| `DATA_SHUFFLE` | `True` |
| `TRAIN_PROMPT_BSZ` | `64` |
| `ROLLOUT_N` | `8` |
| `LR` | `5e-7` |
| `MAX_PROMPT_LENGTH` | `1024` |
| `MAX_RESPONSE_LENGTH` | `4096` |
| online validation | HumanEval+ and MBPP+ only |
| online decode | `VAL_N=1`, `VAL_TEMPERATURE=0.2`, `VAL_TOP_P=0.95` |
| best metric | `val-core/HumanEval+/acc/pass@1` |

Training uses `recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py`.
BigCodeBench and LiveCodeBench are offline diagnostics and must not be added to
online plateau selection.

## Required Gates

### G0: Workspace and data sanity

Before any launch or dry-run with side effects:

- read `docs/joint_training/constraints/principles/workspace_artifact_hygiene.md`;
- set scratch paths outside the repo root when needed:
  `SCRATCH_ROOT=/data-1/tmp/verl_agent_scratch/$USER/deepcoder_stage1_<timestamp>`;
- do not use repo root for generated-code or reward execution cwd;
- after side-effecting checks, run `git status --short --ignored` and explain
  new paths.

### G1: Wrapper and queue dry-run

Required commands:

```bash
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_0_retention.sh
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_s1_code_deepcoder_beta_01_retention.sh
DRY_RUN=1 QUEUE_DRY_RUN_VALIDATE_WRAPPERS=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue.sh
DRY_RUN=1 QUEUE_MODE=deepcoder_stage1 WXPUSHER_NOTIFY=0 timeout 2 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh || true
```

Acceptance:

- both wrappers print the DeepCoder train parquet path, not KodCode;
- both wrappers point to the `deepcoder_preview_train_prompt1024_rl_format.parquet`
  prefiltered train file;
- both wrappers print the expected DeepCoder run prefixes;
- `TOTAL_TRAINING_STEPS=150`;
- `PROTECTED_CKPT_STEPS=[30,40,50,60,70,80,90,100,110,120,130,140]`;
- queue dry-run validates both wrappers and exits before training;
- no command enters `verl.trainer.main_ppo`.

### G2: Disk and collision gate

Immediately before launch:

```bash
df -h /data-1 /data-1/checkpoints
find /data-1/checkpoints -maxdepth 1 -type d -name 'ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-*'
```

Acceptance:

- `/data-1` and `/data-1/checkpoints` have at least 600G free for the two-beta
  batch;
- no existing DeepCoder Stage1 checkpoint prefix collision is present unless
  `ALLOW_RESUME=1` is explicitly approved for a known partial run;
- queue uses `MIN_FREE_GB=600`.

### G3: Launch

Only after G1 and G2 pass, launch in tmux:

```bash
tmux new-session -d -s code_task_deepcoder_stage1_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=600 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_DEEPCODER_STAGE1_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue.sh"

tmux new-session -d -s code_task_deepcoder_stage1_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=deepcoder_stage1 POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

The queue is strict: beta `0.1` starts only after beta `0.0` reaches final
checkpoint and metrics, or after a recorded repair/resume decision.

## Completion Criteria

Stage1 is complete only when both runs have:

- final checkpoint at `global_step_150`;
- protected checkpoints at `global_step_30,40,...,140`;
- metrics JSONL under
  `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/`;
- final online HumanEval+ and MBPP+ pass@1 records;
- queue status TSV entry marked `completed`;
- monitor evidence showing no active hard error.

The final summary must report, for each beta:

- checkpoint root;
- protected checkpoint list;
- final metrics path;
- HE+ best step and pass@1;
- MBPP+ best step and pass@1;
- latest step 150 HE+/MBPP+ pass@1;
- reward health summary: correct ratio, extraction/compile/runtime/timeout
  signals if present.

## Blocked Conditions

Mark the Stage1 execution blocked, not complete, if any of these hold:

- disk drops below the configured gate before launch;
- checkpoint prefix collision exists and resume is not explicitly approved;
- wrapper dry-run points to KodCode or a wrong dataset;
- official validation dependency failure recurs;
- reward execution shows systematic dependency/extraction/timeout failure;
- any run exits before final checkpoint and metrics and cannot be repaired
  without changing the experiment.

## Historical `/goal` Prompt

```text
ARCHIVAL ONLY. Do not execute this prompt as a new goal without first creating
a new DeepCoder controlled-ablation plan. The original execution prompt was:

Follow docs/joint_training/plans/completed/deepcoder_stage1_training_execution_plan.md. Complete DeepCoder Stage1 only: run beta0 then beta0.1 on /data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet with dense retention at steps 30,40,...,140 and latest step150. First verify wrapper/queue dry-runs, disk >=600G, no prefix collisions, and workspace hygiene. Launch only the approved Stage1 queue and monitor in tmux. Do not select Stage2 handoff, create Stage2 shard, merge Model2, launch Stage2/baseline, delete checkpoints, or reduce retention. Final summary must include checkpoint roots, protected checkpoint existence, metrics paths, HE+/MBPP+ pass@1 best/latest, queue status, and any blocked evidence.
```
