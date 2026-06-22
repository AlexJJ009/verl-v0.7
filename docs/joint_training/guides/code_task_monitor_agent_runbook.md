# Code Task Monitor Agent Runbook

This runbook is for the human or Codex Monitor Agent supervising code-task
On-Policy SFT queues. It is different from the shell monitor:

- `recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh` only
  polls tmux/checkpoints/metrics and sends notifications.
- The Monitor Agent uses this document to classify failures, decide whether a
  repair is mechanical, and decide whether to skip, resume, or ask the user.

Do not start, resume, or relaunch training from this runbook unless the user has
approved that experiment queue.

## Current Code-Task Queue

- Queue: `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`
- Monitor: `recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh`
- Mode: `QUEUE_MODE=full`
- Primary run prefix: `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2`
- Dataset: `/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet`
- Online validation: full HumanEval+ plus full MBPP+ only, with
  `VAL_N=1`, `VAL_TEMPERATURE=0.2`, `VAL_TOP_P=0.95`, and core `pass@1`
  metrics. BigCodeBench and LiveCodeBench are offline-only for candidate
  checkpoints and final reports.
- Retained best checkpoint is HE+-primary by `BEST_CKPT_METRIC_KEY`; candidate
  selection must still inspect HE+ best, MBPP+ best, and latest checkpoints
  before deciding which checkpoints need offline BigCodeBench/LiveCodeBench.
- Old tiny official validation parquets: `/data-1/dataset/code/verl_rl/official_*_val.parquet`
  are not valid for checkpoint selection.
- Reward: `recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py`
- Training seed: `DATA_SEED=20260604`, `DATA_SHUFFLE=True`
- Formal Stage1 target: `TOTAL_TRAINING_STEPS=150`
- Queue failure policy: strict serial scheduling with
  `QUEUE_CONTINUE_ON_FAILURE=0`; a failed item blocks the queue. Do not continue
  to the next training or eval item until the Monitor Agent has diagnosed the
  cause, applied a recorded repair if needed, and resumed or relaunched that
  same item with explicit index filters. The only pre-authorized exception is
  the Stage1 plateau `/goal`: after an item is proven unrecoverable and recorded
  as failed/skipped with evidence, the Monitor Agent may move to the next
  independent Stage1 beta item with `START_INDEX`/`END_INDEX`.
- Queue item status file:
  `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue_status.tsv`

## `/goal` Prompt

Use
`docs/joint_training/guides/code_task_stage1_plateau_monitor_goal.md` as the
single source of truth when starting a dedicated Monitor Agent for the approved
Stage1 plateau-finding queue. Keep this runbook as the detailed reference for
failure classes, command examples, and evidence collection.

## Evidence Chain

Always separate documented state from live state. For a live training check, use
this order. Routine checks are every 30 minutes; run these probes once per
routine check unless a log, notification, or idle GPU state triggers debug mode.

1. `tmux ls`
2. `docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}\t{{.Command}}'`
3. `nvidia-smi`
4. queue and monitor logs under `recipe/on_policy_wdl_sft/code_task/`
5. checkpoint directories under `/data-1/checkpoints`
6. metrics JSONL under `recipe/on_policy_wdl_sft/code_task/metrics`
7. Ray logs under the current run's Ray/session temp directory if the failure is
   not explained by the launcher logs

Do not infer "not running" from one missing artifact. A queue log state like
`checkpoint-only` or `metrics-present` means the run may be alive but not
finished.

## WxPusher Policy

The queue and monitor scripts own unattended notifications. A live Codex chat
session is not required after the tmux jobs are running.

Send WxPusher for:

- queue started or complete;
- run started or reached the final checkpoint;
- early stop, missing dependency, OOM, disk pressure, or required user decision;
- idle expensive GPU resources after a queue crash.

Do not send WxPusher for:

- local dry-runs;
- routine progress while the user is actively chatting;
- long raw stack traces or secrets.

Use this shape:

```text
# <title>
Status: <completed | failed | blocked | needs decision | skipped>

What happened: <one short factual sentence>
Evidence: <path, step, metric, error class, or tmux name>
Next action: <what the queue did or what decision is needed>
```

## Automatic Actions Allowed

These are mechanical and may be done by the Monitor Agent after collecting
evidence:

- restart only the monitor script if the training queue is healthy and the
  monitor died;
- restart only the queue supervisor if the training tmux is healthy and the
  supervisor died, because the queue can adopt an active run tmux and continue
  waiting for its final checkpoint;
- continue only after a skipped pre-launch item when the skip is explicitly
  allowed by this runbook, such as low disk before launch; failures during or
  after launch block the queue until repaired;
- fix a narrow runtime/API compatibility bug that does not change numerics,
  then run a dry-run or import check before resuming;
- resume a queue with index filters only when the previous completed/failed
  item boundary is clear and the resume uses the same config, dataset, seed, and
  run prefix.
- under the Stage1 plateau `/goal` only, after a current Stage1 item is
  diagnosed as unrecoverable, record the failure and use queue index filters to
  launch the next independent Stage1 beta item. This is a skip, not a resume,
  and the status file must preserve the failed item evidence.

## Actions Requiring User Approval

Ask the user before doing any of these:

- changing `TRAIN_PROMPT_BSZ`, `ROLLOUT_N`, `MAX_RESPONSE_LENGTH`,
  `TOTAL_TRAINING_STEPS`, optimizer, learning rate, scheduler/warmup,
  `weight_decay`, `grad_clip`, dataset, reward, or validation set;
- deleting checkpoints or merged model weights;
- resuming from a checkpoint when `ALLOW_RESUME=0` blocked the queue;
- lowering response length or rollout count after OOM;
- relaunching a run under a new `RUN_PREFIX`;
- continuing a scientifically degraded run just because it is mechanically
  alive;
- starting Stage2, merging a Stage1 checkpoint for Stage2, or launching any
  Stage2/offline handoff queue.

## Failure Classes

| Symptom | Evidence to collect | Default action |
| --- | --- | --- |
| Disk below threshold before launch | `df -h /data-1 /data-1/checkpoints`, queue log `SKIP`, status file `skipped_disk` | Skip the current independent queue item and notify. Do not delete anything automatically. |
| Disk full mid-run | queue/training log has `No space left on device`; checkpoint write failed | Mark that run failed and block the queue. Do not move to the next item until the disk issue is handled and the same item is resumed/retried or explicitly abandoned by the user. |
| CUDA OOM or vLLM OOM | training log/Ray log has `out of memory`, `CUDA OOM`, KV cache allocation failure | Do not change experiment semantics automatically. Notify with candidate mechanical knobs. |
| Missing official evaluator | reward metadata has `code_reward_dependency_error`, import failure for EvalPlus/BigCodeBench/LCB | Block the queue. Monitor Agent must install/fix official deps, rerun dependency and reference reward checks, record the fix, then resume/relaunch the failed item. |
| KodCode reward runner failure | `kodcode_exec` error, pytest/capsys/capfd failure, reference answer does not pass | Fix runner/env first, rerun the 200-sample reference probe, then resume only if config is unchanged. |
| Extraction/format collapse | high `code_reward_extraction_fail` or many missing fenced answers | Treat as quality failure, not infra failure. Stop/skip only with user approval unless prior policy already says this branch is worthless. |
| Checkpoint collision | existing `/data-1/checkpoints/<RUN_PREFIX>_*` and `ALLOW_RESUME=0` | Block and ask whether to resume, archive/delete stale checkpoint after verification, or relaunch with a new prefix. |
| Quality collapse while alive | validation drops sharply, grad norm spikes, or reward types become pathological | Notify and ask. A live process is not sufficient reason to keep spending GPU. |
| Proxy/upload TLS EOF | `200 Connection established` followed by TLS EOF or quota errors | Preserve progress. Do not retry upload loops until proxy subscription and node health are verified. |

## OOM Decision Rules

Prefer changes that reduce transient memory without changing samples:

- reduce per-GPU log-prob/token chunk limits if the failure is in log-prob
  computation;
- lower rollout GPU memory utilization if vLLM is overcommitting;
- reduce micro-batch size only when it does not change prompt batch, rollout
  count, or optimizer semantics.

Do not change these without approval because they change the experiment:

- `TRAIN_PROMPT_BSZ`;
- `ROLLOUT_N`;
- `MAX_RESPONSE_LENGTH`;
- dataset or filtering;
- validation `n`.

For this code Stage1, `MAX_RESPONSE_LENGTH=4096` is part of the experiment
contract.

## Disk Policy

Pre-launch:

- queue scripts check free checkpoint space and skip items below
  `MIN_FREE_GB`;
- notify with the free-space number and the skipped label.

Mid-run:

- if the run stops due to disk, do not automatically resume the same run;
- if the next item depends on the failed checkpoint, skip it and notify;
- if the next item is independent and the queue policy allows continuation, it
  may proceed after a fresh disk check.

Cleanup:

- never delete checkpoints only because disk is low;
- deletion requires archive verification and audit;
- `/data-1/model_weights/registry.jsonl` and
  `/data-1/model_weights/manifests/*.json` are the evidence for archived model
  weights;
- keep any `plateau_p60` checkpoint out of deletion queues unless the user
  explicitly revokes that rule.

## Resume Policy

Resume is allowed only when all are true:

- same `RUN_PREFIX`;
- same dataset path and seed;
- same model init path;
- same training hyperparameters;
- latest checkpoint step is consistent with queue state;
- no evidence of corrupt checkpoint writes;
- `ALLOW_RESUME=1` is explicitly set by the launching command or approved by the
  user.

For partial queue continuation, use the queue's index filters:

```bash
START_INDEX=<idx> END_INDEX=<idx> ALLOW_RESUME=1 ...
```

This is for resuming or relaunching a known failed item after a recorded fix.
Do not start later experiments first unless the user explicitly changes the
policy for that incident, or the active Stage1 plateau `/goal` pre-authorizes
skipping a proven unrecoverable item to the next independent Stage1 beta item.

## Code Reward Debug Path

Use these checks before blaming model quality:

```bash
python3 recipe/on_policy_wdl_sft/code_task/verify_code_eval_deps.py
python3 recipe/on_policy_wdl_sft/code_task/verify_code_reward_env.py
python3 recipe/on_policy_wdl_sft/code_task/verify_code_reward_metadata_dump.py
python3 recipe/on_policy_wdl_sft/code_task/prepare_kodcode_light_rl_dataset.py --verify-only
```

For KodCode-specific reward failures, inspect:

- `/data-1/dataset/KodCode-Light-RL-10K/reports/kodcode_light_rl_10k_reward_sample200.json`
- `recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py`
- reward metadata keys such as `code_reward_error_type`,
  `code_reward_dependency_error`, `code_reward_extraction_fail`,
  `code_reward_compile_error`, `code_reward_runtime_error`, and
  `code_reward_timeout`.

Official validation is stricter than KodCode training reward: HumanEval+/MBPP+,
BigCodeBench, and LiveCodeBench must use their official harnesses. Missing
official packages are hard failures, not local-runner fallbacks.

## Common Commands

Dry-run full queue:

```bash
DRY_RUN=1 QUEUE_DRY_RUN_VALIDATE_WRAPPERS=1 WXPUSHER_NOTIFY=0 \
  bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh
```

Launch formal Stage1 queue after approval:

Do not run this command from a monitor-only goal unless the approval is already
recorded in the current thread or runlog.

```bash
tmux new-session -d -s code_task_full_queue \
  "cd /data-1/verl07/verl && ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh"
```

Restart the full queue supervisor to adopt a currently active training tmux:

```bash
tmux kill-session -t code_task_full_queue
tmux new-session -d -s code_task_full_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=100 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh"
```

Resume/retry a repaired failed item only:

```bash
START_INDEX=<idx> END_INDEX=<idx> ALLOW_RESUME=1 ALLOW_CODE_FULL_TRAINING=1 \
  bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh
```

Launch monitor:

```bash
tmux new-session -d -s code_task_full_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=full bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

Live status:

Run this once per 30-minute routine check unless the monitor is in debug mode.

```bash
tmux ls
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}\t{{.Command}}'
nvidia-smi
tail -n 120 recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.log
tail -n 120 recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.log
find /data-1/checkpoints -maxdepth 1 -type d -name 'ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2_*' | sort | tail -5
```

Hard-failure scan:

```bash
rg -n "Traceback|Error|Exception|CUDA|out of memory|No space left|dependency|ImportError|RayTaskError" \
  recipe/on_policy_wdl_sft/code_task/*.log /tmp/ray/session_latest/logs 2>/dev/null
```

## Past Local Lessons Used Here

- Queue/monitor scripts, not the live chat session, should own WxPusher
  milestones for unattended training.
- Live training health must be checked through tmux, process/container, GPU,
  logs, checkpoints, metrics, and Ray logs, not plan docs alone.
- A vLLM/Qwen signature mismatch was a safe mechanical repair only because it
  preserved numerics and passed a targeted check before queue resume.
- A branch with clear quality collapse can be stopped and the next queued branch
  continued when the user has approved that policy.
- Disk cleanup must be archive-verified before deletion; L40S filesystem
  archive is valid only through the manifest-driven verification path.
