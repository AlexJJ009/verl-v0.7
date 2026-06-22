# Code Task Stage1 Plateau Monitor Goal

Use the short text below as the `/goal` objective for the dedicated Monitor
Agent after the formal Stage1 queue is approved and launched. The full contract
follows the short prompt.

```text
Follow docs/joint_training/guides/code_task_stage1_plateau_monitor_goal.md and docs/joint_training/guides/code_task_monitor_agent_runbook.md. Supervise the approved code-task Stage1 plateau queue for V2 KodCode runs only: beta0 prefix ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2 and beta0.1 prefix ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2. Do not start Stage2. Check every 30 minutes unless debugging. Use Docker/uv verl-harness only. Track tmux, docker, GPU, queue logs, checkpoints, metrics, and Ray logs when needed. First try narrow non-semantic repairs; if an item is unrecoverable, record failed/skipped evidence and move only to the next independent Stage1 beta item. Final summary must report HE+ best, MBPP+ best, latest/final checkpoints, selected offline candidates, and readiness for offline eval.
```

## Full Contract

```text
Objective: supervise the code-task Stage1 plateau-finding queue until both independent Stage1 runs either complete, are repaired and resumed, or are explicitly recorded as failed/skipped with evidence.

Scope:
- Queue: recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh
- Monitor script: recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh
- Runs:
  - beta=0.0, RUN_PREFIX=ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2
  - beta=0.1, RUN_PREFIX=ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2
- Stage1 only. Do not start Stage2.

Environment:
- Use Docker/uv harness only: verl-harness:latest.
- Do not use host conda for training, reward, or eval checks.
- Official evaluator PYTHONPATH must include the repo, /data-1/code_eval_envs/official_site, and /data-1/code_eval_envs/LiveCodeBench.

Experiment contract:
- TOTAL_TRAINING_STEPS=150.
- TEST_FREQ=5 and SAVE_FREQ=5.
- Checkpoint retention is latest plus best: MAX_ACTOR_CKPTS_TO_KEEP=1, KEEP_BEST_CKPT=True.
- Online validation runs only full HumanEval+ plus full MBPP+.
- Online validation parameters are VAL_N=1, VAL_TEMPERATURE=0.2, VAL_TOP_P=0.95.
- Core code validation metric is pass@1.
- BigCodeBench and LiveCodeBench are not online-val benchmarks; run them only for candidate checkpoints and final reports through offline official eval.

Check cadence:
- Routine status check every 30 minutes. Do not poll more frequently unless a queue/log/notification indicates a problem.
- At each check, read concise evidence only; avoid tailing huge logs unless diagnosing a failure.
- At each TEST_FREQ/SAVE_FREQ boundary, confirm checkpoint step and latest validation metrics.
- Enter immediate debug mode only if tmux exits unexpectedly, GPU becomes idle while the queue should be running, metrics stop advancing, or logs show a hard error.

Evidence chain for every status check:
1. tmux ls
2. docker ps
3. nvidia-smi
4. queue log and status TSV latest lines
5. checkpoint directory and latest_checkpointed_iteration.txt
6. metrics JSONL latest step and key metrics
7. Ray logs only if launcher logs do not explain the failure

Metrics to inspect:
- val-core/HumanEval+/acc/pass@1
- val-core/MBPP+/acc/pass@1
- response_length/clip_ratio and response/aborted_ratio
- wdl_sft/correct_ratio or equivalent training correctness signal
- actor/grad_norm
- code_reward_extraction_fail, code_reward_compile_error, code_reward_runtime_error, code_reward_timeout, code_reward_dependency_error

Allowed autonomous actions:
- Restart only the shell monitor if the training queue is healthy and the monitor died.
- Restart only the queue supervisor if an active training tmux is healthy; the queue can adopt the active run.
- Fix narrow runtime/API/env issues that do not change experiment semantics.
- After a fix, run focused checks in verl-harness:latest and record the command plus result summary.
- If the current Stage1 item is unrecoverable and the next item is independent, record failed/skipped evidence and use START_INDEX/END_INDEX to move to the next beta item.

Actions requiring user approval:
- Changing TRAIN_PROMPT_BSZ, ROLLOUT_N, MAX_RESPONSE_LENGTH, dataset, reward, validation set, optimizer/lr, total steps, or RUN_PREFIX.
- Deleting checkpoints or model weights.
- Relaunching under a new prefix.
- Continuing a scientifically degraded run just because the process is alive.
- Starting Stage2.

Failure policy:
- First diagnose and try a narrow repair for the current run.
- If repair is impossible, record the failed item with error class, latest checkpoint, latest metrics, and why it is unrecoverable.
- Only then skip to the next independent Stage1 beta item.
- Skipping is not success; preserve it in the queue status/runlog.

Completion criteria:
- Both beta=0.0 and beta=0.1 have final step 150 checkpoints and metrics, or each incomplete item has an explicit failed/skipped record with evidence.
- The final monitor summary reports a candidate table with HE+ best step/path, MBPP+ best step/path, latest step/path, selected offline candidates, and whether offline candidate eval is ready.
- The trainer's retained best checkpoint is HE+-primary by `BEST_CKPT_METRIC_KEY`; do not treat it as the only Stage1 candidate if MBPP+ peaks at a different step.
```

Related runbook: `docs/joint_training/guides/code_task_monitor_agent_runbook.md`.
