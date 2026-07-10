# L40S Resource and Training Readiness Index

Audit date: 2026-07-11

Host: `ecm-b6f0-0001` (8 x NVIDIA L40S, 46 GiB each, 582 GiB system RAM)

## Storage Layout

| Logical path | Physical path | Status |
|---|---|---|
| `/data-1/model_weights` | `/data-1/model_weights` | PASS |
| `/data-1/checkpoints` | `/data-2/checkpoints` | PASS, compatibility symlink |
| `/data-1/dataset` | `/data-2/dataset` | PASS, compatibility symlink |
| `/data-1/wandb_runs` | `/data-2/wandb_runs` | PASS, compatibility symlink |
| `/data-1/experiment_registry` | `/data-2/experiment_registry` | PASS, compatibility symlink |
| `/data-1/code_eval_envs` | `/data-2/code_eval_envs` | REBUILT, compatibility symlink |

All Docker launch paths must mount both `/data-1` and `/data-2`; mounting only
`/data-1` leaves the compatibility symlinks dangling inside the container.

## Qwen3-1.7B Code-Task Assets

PASS:

- FRAC25/FRAC50 cold-start format models under
  `/data-1/model_weights/format_cold_start_fraction/`
- FRAC25 beta0.1 Stage1 checkpoint, including `global_step_40`
- FRAC50 beta0.1 Stage1 checkpoint, including `global_step_40`
- FRAC25 Stage2 merged Model2 at the queue's `step_40_s2steps60` path
- FRAC25/FRAC50 Stage2 train parquets and manifests
- HumanEval+, MBPP+, and LiveCodeBench v5 validation parquets
- BigCodeBench v0.1.4 source JSONL

Not complete:

- The exact `Qwen/Qwen3-1.7B` snapshot used by the code-task lineage is missing:
  `/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
  The available `Qwen3-1.7B-Base` cache is not treated as an equivalent
  replacement because its tokenizer/config contract differs from the existing
  cold-start and Stage1 artifacts.
- No usable Stage2 checkpoint exists. Run `1783579397` stopped before step 1.
- FRAC50 merged Model2 is not present, but can be regenerated from Stage1 step 40.
- LiveCodeBench release-v5 JSONL cache is incomplete. The migrated snapshot has
  broken blob links, and direct no-proxy access to `huggingface.co:443` timed out.

## Historical Stage2 Failure

The first full-LCB attempt used the reward manager's implicit 300-second timeout.
The later restart injected `CODE_REWARD_TIMEOUT=30` into Ray workers, but the
outer DAPO reward manager still used `reward.timeout=300`. Eight RewardLoopWorker
processes grew to roughly 83-88 GiB each, and Ray killed them at the 95 percent
host-memory threshold before training step 1.

The shared code-task wrappers now expose and pass:

- `CODE_REWARD_MANAGER_TIMEOUT` (Stage2 default: 30 seconds)
- `CODE_REWARD_NUM_WORKERS` (L40S default: 4)

The full validation dataset remains enabled; only reward concurrency is reduced
to fit the L40S host's smaller RAM capacity.

## Runtime Validation

PASS:

- NVIDIA driver `590.48.01`; all 8 L40S GPUs visible in Docker
- PyTorch `2.9.1+cu126`, CUDA available, vLLM `0.12.0`
- The existing cold-start and merged Model2 configs/tokenizers load offline in
  `verl-harness`; this does not replace the missing exact Model1 snapshot.
- Checkpoint writes through `/data-1/checkpoints` physically land on `/data-2`
- HumanEval+ and MBPP+ official reward dependency smoke tests pass
- `verl-harness:latest` contains the evaluator runtime dependencies; EvalPlus,
  BigCodeBench, and LiveCodeBench modules import with the rebuilt source paths.
- The host launcher forwards queue experiment variables while preserving the
  container virtual environment.
- Hydra resolves `reward.timeout` and `reward.num_workers` without duplicate
  append errors.

BLOCKED:

- Full-LCB reward smoke requires the missing release-v5 JSONL blobs.
- Stage2 Model1 initialization requires the missing exact `Qwen/Qwen3-1.7B`
  snapshot; do not silently substitute `Qwen3-1.7B-Base`.
- Do not launch the four-row Stage2 queue until the LCB cache is restored and a
  one-row initialization smoke passes with `reward.timeout` and worker count
  visible in the resolved Hydra configuration.

## Launcher

Install the host launcher with:

```bash
bash scripts/l40s/install_launchers.sh
```

The installed `/data-1/verl07/run_train.sh` mounts `/data-1`, `/data-2`, and the
current repository into `verl-harness:latest`.
