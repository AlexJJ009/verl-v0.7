# Code Task Training Queue Runlog

## 2026-06-24 DeepCoder Instruct2507 R8K official offline eval

Formal queue:
`recipe/on_policy_wdl_sft/code_task/run_code_deepcoder_instruct2507_r8k_offline_n3_queue.sh`.

Failure recorded for case
`deepcoder_i2507_r8k_beta0_step120/bigcodebench`.

- Symptom: the queue failed during BigCodeBench official local scoring, after
  generation/conversion had already produced reusable eval artifacts. This was
  not a CUDA OOM diagnosis.
- Diagnostic evidence:
  - BigCodeBench official local evaluator failed under
    `CODE_OFFICIAL_EVAL_PARALLEL=8`.
  - The scorer error pattern was `BrokenProcessPool` with child process
    SIGTERM.
  - A selected 33-task window, `BigCodeBench/340-350`, reproduced the failure.
  - Re-running with BigCodeBench scorer parallelism `1` passed the original
    `1037/3420` stopping point, then failed again around `BigCodeBench/348`.
    That task asks the model to stop processes by name; the generated samples
    used `os.kill` / `pkill`, which can terminate the local scorer process
    tree if not blocked before official eval.
- Interpretation: this is a BigCodeBench official scorer parallelism/process
  pool stability issue, not evidence that the checkpoint, vLLM generation, code
  extraction, or source benchmark JSONL is invalid.
- Avoidance rule for future official offline eval queues:
  - sanitize copied BigCodeBench official samples before scoring: block
    generated process-control code (`os.kill`, `os.killpg`, `pkill`,
    `killall`) by replacing it with a safe failing stub and record
    `bigcodebench_unsafe_samples_report.json`. Sanitized samples must remain
    failed samples, not pass credits;
  - after sanitizer is in place, BigCodeBench may use the normal official
    scorer parallelism, e.g. `BIGCODEBENCH_OFFICIAL_EVAL_PARALLEL=8` inherited
    from `CODE_OFFICIAL_EVAL_PARALLEL=8`. Use `1` or `2` only as a deliberate
    fallback when diagnosing a fresh scorer failure;
  - keep `SKIP_COMPLETED=1` for resumed queue runs, so already completed
    HumanEval+/MBPP+ cases are not regenerated or rescored while recovering
    BigCodeBench;
  - keep BigCodeBench parallelism configurable through
    `BIGCODEBENCH_OFFICIAL_EVAL_PARALLEL`, because its scorer can fail for
    different reasons than EvalPlus or LiveCodeBench.
- Recovery rule after this failure:
  - do not regenerate if the case already has intact `raw_generations` and
    converted official sample files;
  - resume/retry from the BigCodeBench scoring stage using those artifacts, so
    the queue can continue from BigCodeBench rather than repeating
    merge/model2 extraction/vLLM generation.
- Key evidence paths:
  - Queue script:
    `recipe/on_policy_wdl_sft/code_task/run_code_deepcoder_instruct2507_r8k_offline_n3_queue.sh`.
  - Queue logs:
    `recipe/on_policy_wdl_sft/code_task/eval_logs/run_code_deepcoder_instruct2507_r8k_offline_n3_queue.log`,
    `recipe/on_policy_wdl_sft/code_task/eval_logs/run_code_deepcoder_instruct2507_r8k_offline_n3_resume_queue.log`,
    and
    `recipe/on_policy_wdl_sft/code_task/eval_logs/run_code_deepcoder_instruct2507_r8k_offline_n3_resume2_queue.log`.
  - Status TSVs:
    `recipe/on_policy_wdl_sft/code_task/eval_logs/run_code_deepcoder_instruct2507_r8k_offline_n3_status.tsv`,
    `recipe/on_policy_wdl_sft/code_task/eval_logs/run_code_deepcoder_instruct2507_r8k_offline_n3_resume_status.tsv`,
    and
    `recipe/on_policy_wdl_sft/code_task/eval_logs/run_code_deepcoder_instruct2507_r8k_offline_n3_resume2_status.tsv`.
  - Failed/scored case log:
    `recipe/on_policy_wdl_sft/code_task/eval_logs/deepcoder_i2507_r8k_beta0_step120_bigcodebench_n3.log`.
  - Reusable artifacts:
    `/data-1/eval_outputs/code_task/deepcoder_instruct2507_r8k_unified_n3/deepcoder_i2507_r8k_beta0_step120/humaneval/official_summary.json`,
    `/data-1/eval_outputs/code_task/deepcoder_instruct2507_r8k_unified_n3/deepcoder_i2507_r8k_beta0_step120/mbpp/official_summary.json`,
    `/data-1/eval_outputs/code_task/deepcoder_instruct2507_r8k_unified_n3/deepcoder_i2507_r8k_beta0_step120/bigcodebench/raw_generations_n3.jsonl`,
    and
    `/data-1/eval_outputs/code_task/deepcoder_instruct2507_r8k_unified_n3/deepcoder_i2507_r8k_beta0_step120/bigcodebench/bigcodebench_samples_n3.jsonl`.
- Acceptance standard for declaring the recovery complete:
  - the resumed queue log shows sanitizer-enabled BigCodeBench scoring. Normal
    recovered runs may use `bigcodebench_parallel=8`; `1` or `2` should be
    treated as a fallback/debug setting rather than the default;
  - the BigCodeBench case directory contains `official_summary.json` plus
    `bigcodebench_unsafe_samples_report.json`;
  - the unsafe-sample report records any blocked `os.kill` / `pkill` /
    `killall` samples, and those rows are counted as safe failures;
  - the status TSV records completed HumanEval+/MBPP+ cases as
    `skipped-completed` when resuming with `SKIP_COMPLETED=1`;
  - final summary files are written under
    `/data-1/eval_outputs/code_task/deepcoder_instruct2507_r8k_unified_n3/`
    only after all selected benchmark cases have their own
    `official_summary.json`.

## 2026-06-09 DeepCoder Stage1 queue

Started by Codex at local time `2026-06-09 10:00`.

Design and execution docs:

- Experiment design:
  `docs/joint_training/reports/deepcoder_preview_code_task_transfer_design.md`.
- Stage1 execution plan:
  `docs/joint_training/plans/active/deepcoder_stage1_training_execution_plan.md`.

Launch command:

```bash
tmux new-session -d -s code_task_deepcoder_stage1_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=600 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_DEEPCODER_STAGE1_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue.sh"

tmux new-session -d -s code_task_deepcoder_stage1_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=deepcoder_stage1 POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

Pre-launch gates:

- Wrapper dry-runs passed for beta `0.0` and beta `0.1`.
- Queue dry-run passed with `QUEUE_DRY_RUN_VALIDATE_WRAPPERS=1`.
- Meituan dispatch dry-runs selected the DeepCoder wrappers for both beta values.
- `/data-1` and `/data-1/checkpoints` had about `769G` free, above the `600G`
  two-beta dense-retention gate.
- No existing DeepCoder Stage1 checkpoint prefix collision was found.

Run items:

| Index | Label | Prefix | Final step | Status |
| ---: | --- | --- | ---: | --- |
| 0 | `deepcoder-s1-beta0` | `ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION` | 150 | launched as `ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780970440` |
| 1 | `deepcoder-s1-beta01` | `ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION` | 150 | pending behind beta `0.0` |

Initial live evidence:

- tmux sessions: `code_task_deepcoder_stage1_queue`,
  `code_task_deepcoder_stage1_monitor`, and `code_task_s1_deepcoder_beta0`.
- Docker container: `8a399d7fbdc0` (`verl-harness:latest`, `nifty_borg`).
- Active checkpoint root:
  `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780970440`.
- Queue log:
  `recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue.log`.
- Queue status file:
  `recipe/on_policy_wdl_sft/code_task/run_code_task_deepcoder_stage1_queue_status.tsv`.
- Run log:
  `recipe/on_policy_wdl_sft/code_task/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780970440.log`.
- The launched Hydra command used
  `data.train_files=/data-1/dataset/code/verl_rl/deepcoder_preview_train_rl_format.parquet`,
  `DATA_SEED=20260604`, `TRAIN_PROMPT_BSZ=64`, `ROLLOUT_N=8`,
  `TOTAL_TRAINING_STEPS=150`, and
  `trainer.protected_ckpt_steps=[30,40,50,60,70,80,90,100,110,120,130,140]`.
- Ray initialized successfully by `10:01` local time. Initial validation and
  step-1 metrics were still pending at the first post-launch check.
- Outcome at `2026-06-09 10:18`: beta `0.0` failed at step `0` after
  successful initial HumanEval+/MBPP+ validation. The first training rollout
  crashed in `verl/experimental/agent_loop/agent_loop.py` while concatenating
  prompt tensors with mixed widths, e.g. expected `1024` but got `1379`.
- Diagnosis: the DeepCoder parquet stores `prompt` as a JSON string. Runtime
  `RLHFDataset.__getitem__` restores that string into chat messages, but the
  overlong-prompt filter had evaluated the JSON string directly, so `1,224`
  train prompts longer than `1024` tokens were not removed.
- Repair: generated Docker-runtime-filtered files:
  `/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet`
  (`22,063/23,287` rows kept) and
  `/data-1/dataset/code/verl_rl/deepcoder_preview_dev_prompt1024_rl_format.parquet`
  (`952/1,000` rows kept). Manifests live beside the parquets as
  `*_prompt1024_manifest.json`.

## 2026-06-05 formal KodCode Stage1 queue

Started by Monitor Agent at local time `2026-06-05 00:18`.

Launch command:

```bash
tmux new-session -d -s code_task_full_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=100 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh"

tmux new-session -d -s code_task_full_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=full POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

Queue policy:

- Run formal Stage1 `beta=0.0` first, then `beta=0.1`.
- Active policy as of `2026-06-05 05:25`: the queue is strict serial dispatch
  with `QUEUE_CONTINUE_ON_FAILURE=1`.
- If an independent item fails, the queue records the item as failed and moves
  to the next experiment. The failed item is not considered recovered.
- After failure, Monitor Agent diagnoses the cause, applies a narrow recorded
  repair if allowed by `code_task_monitor_agent_runbook.md`, and resumes or
  relaunches only that failed item with explicit index filters.

Run items:

| Index | Label | Prefix | Final step | Status |
| ---: | --- | --- | ---: | --- |
| 0 | `s1-full-beta0` | `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1` | 150 | launched as `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780589927` |
| 1 | `s1-full-beta01` | `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1` | 150 | pending |

Initial live evidence:

- tmux sessions: `code_task_full_queue`, `code_task_full_monitor`,
  `code_task_s1_kodcode_beta0`.
- Docker: one `verl-harness` container, name `determined_wing`.
- Checkpoint root:
  `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780589927`.
- At first check, the run was initializing Ray and filtering the 10,000-row
  KodCode train split.
- `/data-1` free space before launch was about `155G`.

Recorded repair after launch:

- `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` now defaults
  `QUEUE_CONTINUE_ON_FAILURE=0`.
- Reason at that time: the formal experiment queue was temporarily configured
  to stop after a failed first experiment, pending user clarification.
- This is an orchestration-only change. It does not change training data,
  reward, model, sampling, optimizer, or validation numerics for the running
  beta `0.0` job.
- Superseded by the policy update below.

Recorded policy update after user clarification:

- `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` now defaults
  `QUEUE_CONTINUE_ON_FAILURE=1`.
- `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` now lets a
  checkpoint-collided item advance to the next item when
  `QUEUE_CONTINUE_ON_FAILURE=1`, instead of hard-exiting the queue.
- Reason: the formal queue should strictly dispatch independent experiments in
  sequence. If one experiment fails, the queue records the failure and moves to
  the next experiment; Monitor Agent debug/repair happens separately, and a
  failed item can resume/relaunch only after the repair is recorded.
- This is an orchestration-only change. It does not change training data,
  reward, model, sampling, optimizer, or validation numerics.
- Superseded by the active `2026-06-05 05:25` policy update: failed independent
  items are recorded and the queue moves on; repair/resume happens separately.

Current launch command after policy update:

```bash
tmux new-session -d -s code_task_full_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=100 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh"
```

Behavior probe at local time `2026-06-05 00:29`:

- Command used `QUEUE_CONTINUE_ON_FAILURE=1 START_INDEX=0 END_INDEX=1
  MIN_FREE_GB=0` with a short timeout to verify checkpoint-collision behavior.
- Observed behavior: existing failed `beta=0.0` checkpoint
  `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780589927`
  was skipped as blocked, and the queue advanced to `beta=0.1`.
- Side effect: the probe briefly launched
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780590601`.
- Cleanup: stopped tmux session `code_task_s1_kodcode_beta01` and Docker
  container `fbc434f6fd3d`; no active training container remained afterward.
- The probe-created checkpoint directory was not deleted. It must be treated as
  stale/probe data unless the user explicitly approves cleanup or resume.

Active policy restoration at local time `2026-06-05`:

- `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` restored to
  default `QUEUE_CONTINUE_ON_FAILURE=0`.
- `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` restored to
  hard-block checkpoint collisions when `ALLOW_RESUME=0`.
- Reason: the active Monitor Agent goal requires diagnosing and fixing the
  failed current experiment, then resuming/relaunching that experiment before
  later experiments run.
- This is an orchestration-only change. It does not change training data,
  reward, model, sampling, optimizer, or validation numerics.
- Superseded by the active `2026-06-05 05:25` policy update: strict dispatch now
  continues to the next independent experiment after recording the failure;
  Monitor Agent debug/repair is required before resuming the failed item.

Recorded repair and relaunch at local time `2026-06-05 00:33-00:42`:

- Failure fixed: previous beta `0.0` run failed in `val_before_train=True`
  with `ModuleNotFoundError: No module named 'evalplus'`.
- Root cause: official evaluator packages existed under
  `/data-1/code_eval_envs/official_site` and
  `/data-1/code_eval_envs/LiveCodeBench`, but local code-task launchers did not
  export those paths into `PYTHONPATH` for the `verl-harness` training runtime.
- Code repair:
  - `recipe/on_policy_wdl_sft/code_task/run_s1_code_base.sh` now exports
    `CODE_EVAL_OFFICIAL_SITE=/data-1/code_eval_envs/official_site`,
    `LCB_REPO_DIR=/data-1/code_eval_envs/LiveCodeBench`, and prepends both to
    `PYTHONPATH`.
  - `recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh`
    received the same export so Stage2 uses the same official-evaluator
    dependency boundary.
- Verification in the same Docker image used by training:
  - `evalplus` import resolved from
    `/data-1/code_eval_envs/official_site/evalplus`, version `0.3.1`.
  - `bigcodebench` import resolved from
    `/data-1/code_eval_envs/official_site/bigcodebench`, version `0.2.5`.
  - `lcb_runner` import resolved from
    `/data-1/code_eval_envs/LiveCodeBench/lcb_runner`.
  - `recipe/on_policy_wdl_sft/code_task/verify_code_eval_deps.py` returned
    `"ok": true`.
- Stale/probe checkpoint handling:
  - Empty failed beta `0.0` directory
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780589927`
    and empty probe beta `0.1` directory
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780590601`
    were moved, not deleted, into
    `/data-1/checkpoints/quarantine_code_task_20260605/`.
  - Both were 4KB directories with no `global_step_*` checkpoint and no valid
    `latest_checkpointed_iteration.txt`.
- Relaunch command:

```bash
tmux new-session -d -s code_task_full_queue \
  "cd /data-1/verl07/verl && MIN_FREE_GB=100 QUEUE_CONTINUE_ON_FAILURE=0 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh"

tmux new-session -d -s code_task_full_monitor \
  "cd /data-1/verl07/verl && QUEUE_MODE=full POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh"
```

- Active run after relaunch:
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881`.
- Initial health evidence:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `11767eebdec3` (`verl-harness`,
    `dazzling_bhaskara`).
  - GPU memory after model/rollout startup: about `42GB / 80GB` on each of 8
    GPUs.
  - Initial validation completed and dumped generations to
    `recipe/on_policy_wdl_sft/code_task/validation/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881/0.jsonl`.
  - Official validation methods appeared in logs: `evalplus`,
    `bigcodebench`, and LiveCodeBench metrics with
    `official_aligned/mean@3 = 1.0`.
  - `code_reward_dependency_error/mean@3 = 0.0` for the validation metrics,
    confirming the prior missing-dependency failure is fixed.
- Queue policy remains failure-blocking:
  `QUEUE_CONTINUE_ON_FAILURE=0`; beta `0.1` must not start until beta `0.0`
  reaches final checkpoint and metrics, or Monitor Agent records a repair and
  resumes beta `0.0`.

Step-1 health evidence at local time `2026-06-05 00:43`:

- Metrics file:
  `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881.jsonl`.
- `training/global_step=1`, `training/epoch=0`.
- Reward/training health:
  - `wdl_sft/n_correct=94`, `wdl_sft/n_incorrect=0`,
    `wdl_sft/correct_ratio=0.18359375`.
  - `actor/pg_loss=116.1280791060999`.
  - `actor/grad_norm=46.389522552490234`.
  - `response_length/clip_ratio=0.173828125`,
    `response/aborted_ratio=0.0`.
- Performance/memory:
  - `perf/max_memory_allocated_gb=28.828469276428223`,
    `perf/max_memory_reserved_gb=34.33203125`.
  - `perf/mfu/actor=0.3596955956362855`.
  - `perf/throughput=903.6744835699659`.
  - GPU live check showed about `42GB / 80GB` used on each of 8 GPUs.
- No `ModuleNotFoundError`, `EvalPlus official evaluator is unavailable`,
  CUDA OOM, or disk-full error was present in the active run log at this check.

Routine Monitor Agent check at local time `2026-06-05 00:45`:

- tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
  `code_task_s1_kodcode_beta0`.
- Docker container alive: `11767eebdec3` (`verl-harness`,
  `dazzling_bhaskara`), up about 9 minutes.
- GPU live check: all 8 GPUs active, about `39.4GB / 80GB` used per GPU,
  utilization about `70-91%`.
- Active checkpoint root:
  `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881`.
- Active metrics file had 2 records: initial validation (`step=0`) and
  training `step=1`.
- Latest metrics remained:
  - `training/global_step=1`
  - `wdl_sft/correct_ratio=0.18359375`
  - `actor/grad_norm=46.389522552490234`
  - `response_length/clip_ratio=0.173828125`
  - `response/aborted_ratio=0.0`
  - `perf/throughput=903.6744835699659`
  - `perf/max_memory_allocated_gb=28.828469276428223`
  - `perf/mfu/actor=0.3596955956362855`
- Active run log scan found no current hard errors matching traceback,
  `RayTaskError`, `ModuleNotFoundError`, EvalPlus unavailable, OOM, disk-full,
  killed/aborted, or NCCL error patterns.
- Historical `evalplus` failure still exists in `run_code_task_full_queue.log`
  for old run `..._1780589927`; it is not part of the active run
  `..._1780590881`.

Routine Monitor Agent check at local time `2026-06-05 00:45:41`:

- tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
  `code_task_s1_kodcode_beta0`.
- Docker container alive: `11767eebdec3` (`verl-harness`,
  `dazzling_bhaskara`), up about 11 minutes.
- GPU live check: all 8 GPUs active, about `39.4GB / 80GB` used per GPU,
  utilization about `68-74%`.
- Active run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881`.
- Active metrics file had 3 records:
  - initial validation `step=0`
  - training `global_step=1`
  - training `global_step=2`
- Latest training metrics at `global_step=2`:
  - `wdl_sft/n_correct=72`
  - `wdl_sft/correct_ratio=0.140625`
  - `actor/grad_norm=48.2675666809082`
  - `response_length/clip_ratio=0.185546875`
  - `response/aborted_ratio=0.0`
  - `perf/throughput=959.3049576325881`
  - `perf/mfu/actor=0.3713271934409833`
  - `perf/max_memory_allocated_gb=34.55926561355591`
- No active-run hard errors matched traceback, RayTaskError,
  `ModuleNotFoundError`, EvalPlus unavailable, OOM, disk-full, killed/aborted,
  or NCCL error patterns.
- No checkpoint has been written yet, as expected before the first
  `SAVE_FREQ=5` boundary.
- No repair or parameter change was made in this check.

Supplemental Monitor Agent check at local time `2026-06-05 00:46:45`:

- tmux sessions still alive: `code_task_full_queue`, `code_task_full_monitor`,
  `code_task_s1_kodcode_beta0`.
- Docker container still alive: `11767eebdec3`, up about 12 minutes.
- GPU live check: all 8 GPUs active, about `39.4GB / 80GB` used per GPU,
  utilization about `66-92%`.
- Active metrics file still had 3 records, latest at `global_step=2`.
- Latest metrics unchanged from the previous check:
  - `wdl_sft/n_correct=72`
  - `wdl_sft/correct_ratio=0.140625`
  - `actor/grad_norm=48.2675666809082`
  - `response_length/clip_ratio=0.185546875`
  - `response/aborted_ratio=0.0`
  - `perf/throughput=959.3049576325881`
  - `perf/max_memory_allocated_gb=34.55926561355591`
- Active run log scan again found no current hard errors.
- Checkpoint directory is still 4KB with no `global_step_*`, which is expected
  before the first `SAVE_FREQ=5` checkpoint.
- No repair or parameter change was made in this check. Next checks should stay
  at the requested half-hour cadence unless the monitor reports a failure.

Routine Monitor Agent check at local time `2026-06-05 00:47:38`:

- tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
  `code_task_s1_kodcode_beta0`.
- Docker container alive: `11767eebdec3`, up about 12 minutes.
- GPU live check: all 8 GPUs active, about `49.8-58.1GB / 80GB` used per GPU,
  utilization about `46-92%`. No OOM observed.
- Active metrics file had 4 records, latest at `global_step=3`.
- Latest training metrics at `global_step=3`:
  - `wdl_sft/n_correct=81`
  - `wdl_sft/correct_ratio=0.158203125`
  - `actor/grad_norm=53.76076889038086`
  - `response_length/clip_ratio=0.19921875`
  - `response/aborted_ratio=0.0`
  - `perf/throughput=959.5099051857504`
  - `perf/mfu/actor=0.3665640284587791`
  - `perf/max_memory_allocated_gb=34.55926561355591`
  - `actor/lr=2e-07`
- Active run log scan found no current hard errors.
- Checkpoint directory is still 4KB with no `global_step_*`, which is expected
  before the first `SAVE_FREQ=5` checkpoint.
- No repair or parameter change was made in this check.

Routine Monitor Agent check and repair at local time `2026-06-05 01:17:30`:

- Full half-hour check found the formal code-task queue stopped:
  - `code_task_full_queue`, `code_task_full_monitor`, and
    `code_task_s1_kodcode_beta0` tmux sessions were absent.
  - `docker ps` showed no active `verl-harness` training container.
  - All 8 GPUs were idle: about `1MB / 81920MB` used and `0%` utilization.
  - `/data-1` had about `155G` free, above the formal queue `MIN_FREE_GB=100`
    gate; no disk-full condition was observed.
- Active failed run:
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881`.
- Metrics evidence before failure:
  - metrics file:
    `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881.jsonl`
  - 5 records total: initial validation plus training `global_step=1..4`.
  - Latest training record at `global_step=4`:
    - `wdl_sft/n_correct=76`
    - `wdl_sft/correct_ratio=0.1484375`
    - `actor/grad_norm=42.14192199707031`
    - `response_length/clip_ratio=0.18359375`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=925.25644375499`
    - `perf/max_memory_allocated_gb=34.55926561355591`
    - `actor/lr=3e-07`
- Checkpoint evidence:
  - checkpoint dir existed but was only `4.0K` and had no
    `latest_checkpointed_iteration.txt` or `global_step_*` directory.
  - Because `SAVE_FREQ=5`, the run stopped before the first checkpoint could be
    written; there is no valid training state to resume from.
- Failure diagnosis:
  - Not CUDA OOM, not disk-full, not missing official evaluator.
  - Training crashed in the KodCode reward path when `ast.parse(code)` raised
    `MemoryError: Parser stack overflowed - Python source too complex to parse`.
  - Exact stack path:
    `RewardLoopWorker.compute_score` -> `official_aligned_reward.py` ->
    `score_kodcode_exec` -> `_run_subprocess` -> `ast.parse(code)`.
  - This is a reward robustness bug: a single pathological generated Python
    sample should receive a failed score, not terminate the Ray reward worker.
- Mechanical repair applied:
  - File changed: `recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py`.
  - `_run_subprocess` now catches `MemoryError` and `RecursionError` from
    `ast.parse(code)` and returns `score=0.0`, `code_reward_status=compile_error`,
    and `code_reward_compile_error=1` with stderr excerpt
    `python parser failed on generated code`.
  - `_code_defines_entry_point` now also catches `MemoryError` and
    `RecursionError`, returning `False` rather than propagating the parser
    failure.
  - No training hyperparameters, dataset, seed, validation set, rollout count,
    response length, or model init path were changed.
- Verification after repair:
  - Host: `python3 -m py_compile recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py` passed.
  - Host monkeypatch test: forced `ast.parse` to raise `MemoryError`; reward
    returned `score=0.0`, `code_reward_status=compile_error`,
    `code_reward_compile_error=1`, and `_code_defines_entry_point(...) == False`.
  - Docker/`verl-harness` monkeypatch test passed with
    `parser_memoryerror_guard=ok`.
  - Docker official dependency check passed:
    `verify_code_eval_deps.py` reported EvalPlus `0.3.1`, BigCodeBench `0.2.5`,
    and LiveCodeBench/lcb_runner official imports available.
  - Docker reward env check passed: `verify_code_reward_env.py` returned
    `ok=true`, reference pass rate `1.0`, malformed extraction fail rate `1.0`,
    and wrong-output fail rate `1.0`.
  - `verify_code_dataset.py --verify-only` passed for the legacy code dataset
    path; this is not the formal KodCode file but confirms shared prompt tooling.
  - Direct KodCode training parquet check passed:
    `/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format.parquet`
    exists, has `10000` rows, expected columns, and sampled rows carry the
    `code-think-answer-python-v1` prompt contract.
  - Existing KodCode reports remain healthy:
    `kodcode_light_rl_10k_validation.json` has `ok=true`, `rows=10000`,
    `contract_ok=10000`; `kodcode_light_rl_10k_reward_sample200.json` has
    `ok=true`, `sample_size=200`, `failure_count=0`, `total_tests=1161`.
  - WxPusher repair notification was sent successfully at this repair point.
- Quarantine action:
  - The empty failed checkpoint directory was moved, not deleted, to
    `/data-1/checkpoints/quarantine_code_task_20260605_012048/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780590881`.
  - The failed metrics JSONL and run logs remain in place for audit.
- Resume/relaunch decision:
  - Because no checkpoint exists before `SAVE_FREQ=5`, the same beta0 item will
    be relaunched from the same `RUN_PREFIX`, model init, dataset, seed, and
    hyperparameters rather than resumed from a nonexistent checkpoint.
  - Relaunch will use queue index filters `START_INDEX=0 END_INDEX=0` so beta0.1
    cannot start before beta0 completes.

Relaunch confirmation at local time `2026-06-05 01:21:21`:

- Relaunched formal queue with strict beta0-only filters:
  `MIN_FREE_GB=100 QUEUE_CONTINUE_ON_FAILURE=0 START_INDEX=0 END_INDEX=0 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`.
- Relaunched monitor with half-hour polling:
  `QUEUE_MODE=full POLL_SEC=1800 bash recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh`.
- WxPusher relaunch notification was sent successfully.
- Short post-launch confirmation:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`).
  - New beta0 run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
  - New checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
  - Confirmed launch command still uses the intended config: `data.seed=20260604`,
    `data.train_batch_size=64`, `actor_rollout_ref.rollout.n=8`,
    `actor_rollout_ref.rollout.response_length=4096`, validation `n=3`,
    `trainer.total_training_steps=150`, official validation parquets, and
    `official_aligned_reward.py`.
  - No beta0.1 checkpoint directory was created; the queue did not advance to
    experiment 2.
- Next Monitor Agent full check should occur on the half-hour cadence unless a
  queue/monitor notification reports an earlier failure.

Routine Monitor Agent check at local time `2026-06-05 01:51:30`:

- Active beta0 run:
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about 30 minutes.
  - GPU live check: all 8 GPUs active, about `39.4GB / 80GB` used per GPU,
    utilization about `72-85%`.
- Metrics status:
  - metrics file exists:
    `recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683.jsonl`.
  - 11 records total; latest training/validation record at `global_step=10`.
  - Latest metrics at `global_step=10`:
    - `val-core/HumanEval+/acc/mean@3=0.16666666666666666`
    - `val-core/MBPP+/acc/mean@3=0.5`
    - `val-core/BigCodeBench/acc/mean@3=0.0`
    - `val-core/LiveCodeBench/acc/mean@3=0.0`
    - `wdl_sft/n_correct=96`
    - `wdl_sft/correct_ratio=0.1875`
    - `actor/grad_norm=56.18273162841797`
    - `response_length/clip_ratio=0.1796875`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=850.3211676883856`
    - `perf/max_memory_allocated_gb=33.951030254364014`
    - `actor/lr=5e-07`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `10`.
  - checkpoint dir size is about `47G`.
  - `best_checkpoint.json` points to `global_step_10` with metric
    `val-core/HumanEval+/acc/mean@3=0.16666666666666666`.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
  - A broad `Traceback` scan matched generated-code unit-test failure stderr
    inside reward output; this is expected scoring metadata, not a framework
    crash.
- Disk pressure:
  - `/data-1` is about `97%` used with about `108G` available.
  - Ray emitted warnings that its session directory is over 95% full and object
    creation may fail if spilling is required.
  - This is above the configured `MIN_FREE_GB=100` launch gate, so no automatic
    stop/skip was performed, but it is close enough to require attention.
  - WxPusher disk-pressure notification was sent.
  - No checkpoint or data was deleted by the Monitor Agent.
- No repair or parameter change was made in this check.

Routine Monitor Agent check at local time `2026-06-05 02:21:30`:

- Active beta0 run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about an hour.
  - GPU live check: all 8 GPUs active, about `39.4GB / 80GB` used per GPU,
    utilization about `68-77%`.
- Metrics status:
  - metrics file exists and has 25 records.
  - Latest metric record available at check time was `global_step=24`; queue log
    shows training progress `24/150`.
  - Latest metrics at `global_step=24`:
    - `wdl_sft/n_correct=124`
    - `wdl_sft/correct_ratio=0.2421875`
    - `actor/grad_norm=38.89573287963867`
    - `response_length/clip_ratio=0.12109375`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=837.0900533899521`
    - `perf/max_memory_allocated_gb=34.54342174530029`
    - `actor/lr=5e-07`
  - Latest validation record at `global_step=20`:
    - `val-core/HumanEval+/acc/mean@3=0.0`
    - `val-core/MBPP+/acc/mean@3=0.3333333333333333`
    - `val-core/BigCodeBench/acc/mean@3=0.0`
    - `val-core/LiveCodeBench/acc/mean@3=0.0`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `20`.
  - checkpoint dir size is about `63G`.
  - checkpoint subdir sizes: `global_step_10` about `17G`, `global_step_20`
    about `47G`.
  - `best_checkpoint.json` still points to `global_step_10` with
    `val-core/HumanEval+/acc/mean@3=0.16666666666666666`.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- Disk pressure:
  - `/data-1` is about `97%` used with about `92G` available.
  - This is now below the formal queue launch gate `MIN_FREE_GB=100`.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - The current already-running beta0 item was not stopped by the Monitor Agent,
    and no checkpoint/data deletion was performed automatically.
  - WxPusher low-disk notification was sent.
- No repair or parameter change was made in this check.

Routine Monitor Agent check at local time `2026-06-05 02:51:30`:

- Active beta0 run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about 2 hours.
  - GPU live check: all 8 GPUs active, about `45.9-47.0GB / 80GB` used per GPU,
    utilization about `97-99%`.
- Metrics status:
  - metrics file exists and has 39 records.
  - Latest metric record available at check time was `global_step=38`; queue log
    shows training progress `38/150`.
  - Latest metrics at `global_step=38`:
    - `wdl_sft/n_correct=173`
    - `wdl_sft/correct_ratio=0.337890625`
    - `actor/grad_norm=35.14952850341797`
    - `response_length/clip_ratio=0.02734375`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=537.4617627223524`
    - `perf/max_memory_allocated_gb=34.54342174530029`
    - `actor/lr=5e-07`
  - Latest validation record at `global_step=35`:
    - `val-core/HumanEval+/acc/mean@3=0.16666666666666666`
    - `val-core/MBPP+/acc/mean@3=0.6666666666666666`
    - `val-core/BigCodeBench/acc/mean@3=0.3333333333333333`
    - `val-core/LiveCodeBench/acc/mean@3=0.0`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `35`.
  - checkpoint dir size is about `63G`.
  - checkpoint subdir sizes: `global_step_10` about `17G`, `global_step_35`
    about `47G`.
  - `best_checkpoint.json` still points to `global_step_10` with
    `val-core/HumanEval+/acc/mean@3=0.16666666666666666`.
  - Retention appears to be working as intended: only best plus latest checkpoint
    directories are present.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- Disk pressure:
  - `/data-1` remains about `97%` used with about `92G` available, below the
    formal queue launch gate `MIN_FREE_GB=100`.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - The already-running beta0 item was not stopped by the Monitor Agent, and no
    checkpoint/data deletion was performed automatically.
- No repair or parameter change was made in this check.

Routine Monitor Agent check at local time `2026-06-05 03:21:30`:

- Active beta0 run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about 2 hours.
  - GPU live check: all 8 GPUs hold about `39.0GB / 80GB`; instant utilization
    was mixed (`0-54%`) because the check landed during checkpoint/validation
    transition. Queue log and metrics show training is still progressing, so this
    was not treated as idle-GPU failure.
- Metrics status:
  - metrics file exists and has 55 records.
  - Latest metric record available at check time was `global_step=54`; queue log
    then showed checkpoint/validation for `global_steps=55`.
  - Latest metrics at `global_step=54`:
    - `wdl_sft/n_correct=236`
    - `wdl_sft/correct_ratio=0.4609375`
    - `actor/grad_norm=30.328628540039062`
    - `response_length/clip_ratio=0.0078125`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=409.65178670121395`
    - `perf/max_memory_allocated_gb=34.729820251464844`
    - `actor/lr=5e-07`
  - Latest complete validation record at `global_step=50`:
    - `val-core/HumanEval+/acc/mean@3=0.3333333333333333`
    - `val-core/MBPP+/acc/mean@3=1.0`
    - `val-core/BigCodeBench/acc/mean@3=0.3333333333333333`
    - `val-core/LiveCodeBench/acc/mean@3=0.3333333333333333`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `55`.
  - checkpoint dir size remains about `63G`.
  - checkpoint subdir sizes: `global_step_50` about `17G`, `global_step_55`
    about `47G`.
  - `best_checkpoint.json` now points to `global_step_50` with
    `val-core/HumanEval+/acc/mean@3=0.3333333333333333`.
  - Queue log confirms best-checkpoint optimizer stripping removed 8 optimizer
    shards from `global_step_50`.
  - Retention continues to work as intended: only best plus latest checkpoint
    directories are present.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- Disk pressure:
  - `/data-1` remains about `97%` used with about `92G` available, below the
    formal queue launch gate `MIN_FREE_GB=100`.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - No checkpoint/data deletion was performed automatically.
- No repair or parameter change was made in this check.

Routine Monitor Agent check at local time `2026-06-05 03:51:30`:

- Active beta0 run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about 3 hours.
  - GPU live check: all 8 GPUs active, about `45.8GB / 80GB` used per GPU,
    utilization about `96-99%`.
- Metrics status:
  - metrics file exists and has 71 records.
  - Latest metric record available at check time was `global_step=70`; queue log
    shows training progress `70/150`.
  - Latest metrics at `global_step=70`:
    - `val-core/HumanEval+/acc/mean@3=0.6666666666666666`
    - `val-core/MBPP+/acc/mean@3=1.0`
    - `val-core/BigCodeBench/acc/mean@3=0.6666666666666666`
    - `val-core/LiveCodeBench/acc/mean@3=0.0`
    - `wdl_sft/n_correct=265`
    - `wdl_sft/correct_ratio=0.517578125`
    - `actor/grad_norm=18.16840171813965`
    - `response_length/clip_ratio=0.0078125`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=348.5299107193846`
    - `perf/max_memory_allocated_gb=34.729820251464844`
    - `actor/lr=5e-07`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `70`.
  - checkpoint dir size is about `47G`.
  - `best_checkpoint.json` points to `global_step_70` with
    `val-core/HumanEval+/acc/mean@3=0.6666666666666666`.
  - Only `global_step_70` is present; best and latest are currently the same
    checkpoint.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- Disk pressure:
  - `/data-1` remains about `97%` used but available space has recovered to about
    `108G`, just above the formal queue launch gate `MIN_FREE_GB=100`.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - No checkpoint/data deletion was performed automatically.
- No repair or parameter change was made in this check.

Routine Monitor Agent check at local time `2026-06-05 04:21:30`:

- Active beta0 run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about 3 hours.
  - GPU memory held on all 8 GPUs at about `39.4GB / 80GB`; instant utilization
    was mixed (`0-55%`) because the check landed during validation/checkpoint
    transition. Queue log and metrics show active progress, so this was not
    treated as idle-GPU failure.
- Metrics status:
  - metrics file exists and has 91 records.
  - Latest metric record available at check time was `global_step=90`; queue log
    shows training progress at checkpoint `90/150`.
  - Latest metrics at `global_step=90`:
    - `val-core/HumanEval+/acc/mean@3=0.3333333333333333`
    - `val-core/MBPP+/acc/mean@3=1.0`
    - `val-core/BigCodeBench/acc/mean@3=1.0`
    - `val-core/LiveCodeBench/acc/mean@3=0.0`
    - `wdl_sft/n_correct=264`
    - `wdl_sft/correct_ratio=0.515625`
    - `actor/grad_norm=14.69464111328125`
    - `response_length/clip_ratio=0.0`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=677.8519061577699`
    - `perf/max_memory_allocated_gb=34.729820251464844`
    - `actor/lr=5e-07`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `90`.
  - checkpoint dir size is about `63G`.
  - checkpoint subdir sizes: `global_step_70` about `17G`, `global_step_90`
    about `47G`.
  - `best_checkpoint.json` still points to `global_step_70` with
    `val-core/HumanEval+/acc/mean@3=0.6666666666666666`.
  - Retention continues to work as intended: best plus latest checkpoint
    directories are present.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- Disk pressure:
  - `/data-1` is about `97%` used with about `92G` available, below the formal
    queue launch gate `MIN_FREE_GB=100`.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - No checkpoint/data deletion was performed automatically.
- No repair or parameter change was made in this check.

Routine Monitor Agent check at local time `2026-06-05 04:51:30`:

- Active beta0 run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- Runtime status:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`),
    up about 4 hours.
  - GPU memory held on all 8 GPUs at about `39.4GB / 80GB`; instant utilization
    was mixed (`0-53%`) because the check landed during a transition. Queue log
    and metrics show active progress, so this was not treated as idle-GPU
    failure.
- Metrics status:
  - metrics file exists and has 112 records.
  - Latest metric record available at check time was `global_step=111`; queue log
    shows training progress `111/150`.
  - Latest metrics at `global_step=111`:
    - `wdl_sft/n_correct=290`
    - `wdl_sft/correct_ratio=0.56640625`
    - `actor/grad_norm=12.497139930725098`
    - `response_length/clip_ratio=0.001953125`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=378.71776506669454`
    - `perf/max_memory_allocated_gb=34.729820251464844`
    - `actor/lr=5e-07`
  - Latest validation record at `global_step=110`:
    - `val-core/HumanEval+/acc/mean@3=0.5`
    - `val-core/MBPP+/acc/mean@3=1.0`
    - `val-core/BigCodeBench/acc/mean@3=0.3333333333333333`
    - `val-core/LiveCodeBench/acc/mean@3=0.0`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt` reports `110`.
  - checkpoint dir size is about `63G`.
  - checkpoint subdir sizes: `global_step_70` about `17G`, `global_step_110`
    about `47G`.
  - `best_checkpoint.json` still points to `global_step_70` with
    `val-core/HumanEval+/acc/mean@3=0.6666666666666666`.
  - Retention continues to work as intended: best plus latest checkpoint
    directories are present.
  - No beta0.1 checkpoint directory exists; queue has not advanced to experiment 2.
- Error status:
  - Active run system-level hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- Disk pressure:
  - `/data-1` is about `97%` used with about `91G` available, below the formal
    queue launch gate `MIN_FREE_GB=100`.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - No checkpoint/data deletion was performed automatically.
- No repair or parameter change was made in this check.

Policy update at local time `2026-06-05 05:25-05:28`:

- User updated the queue behavior requirement:
  - strict serial dispatch;
  - if an item fails, record it and move to the next independent experiment;
  - after failure, Monitor Agent must debug and repair before resuming/retrying
    that failed item.
- Script changes applied:
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` now
    defaults `QUEUE_CONTINUE_ON_FAILURE=1` and writes item outcomes to
    `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue_status.tsv`.
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` can adopt
    an already-active training tmux when the queue supervisor is restarted.
  - Failed, skipped, completed, and already-complete items are recorded in the
    status file with timestamp, index, label, prefix, status, and evidence.
  - Existing partial checkpoints without an active tmux are recorded as failed
    when `ALLOW_RESUME=0`; the queue then advances if
    `QUEUE_CONTINUE_ON_FAILURE=1`.
- Documentation changes applied:
  - `docs/joint_training/guides/code_task_monitor_agent_runbook.md` now states
    the active policy and the filtered resume/retry path.
  - `docs/joint_training/guides/training_script_index.md` now records the full
    queue as failure-continuing and adopt-capable.
- Verification:
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`
    passed.
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh`
    passed.
- Live beta0 state before supervisor restart:
  - training tmux `code_task_s1_kodcode_beta0` remains active.
  - queue tmux `code_task_full_queue` was still running with the old command
    containing `QUEUE_CONTINUE_ON_FAILURE=0 START_INDEX=0 END_INDEX=0`.
  - metrics for active run
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683` reached
    `global_step=137`, with latest validation at `global_step=135`:
    HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`, BigCodeBench mean@3
    `0.6666666666666666`, LiveCodeBench mean@3 `0.3333333333333333`.
  - `/data-1` free space is about `91G`; Ray still warns that `/data-1/ray_tmp`
    is over 95% full.
- Next action: restart only the queue supervisor in tmux `code_task_full_queue`
  with the new default policy so it adopts the already-running beta0 training
  tmux. Do not restart or kill `code_task_s1_kodcode_beta0`.

Supervisor restart confirmation at local time `2026-06-05 05:26`:

- Restarted only tmux session `code_task_full_queue` with:
  `MIN_FREE_GB=100 ALLOW_CODE_FULL_TRAINING=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`.
- The training tmux `code_task_s1_kodcode_beta0` was not killed or restarted.
- New supervisor command no longer contains `QUEUE_CONTINUE_ON_FAILURE=0` or
  `START_INDEX=0 END_INDEX=0`; it uses the script default
  `QUEUE_CONTINUE_ON_FAILURE=1` and covers indices `0-1`.
- New queue log confirms adoption of the active run:
  `adopting active code_task_s1_kodcode_beta0`.
- Live training evidence at restart: training progress around `138/150`; active
  checkpoint dir
  `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
- WxPusher queue-start notification from the restarted supervisor succeeded.

Policy correction at local time `2026-06-05 05:27-05:30`:

- Active goal context clarified the intended Monitor Agent policy:
  - check every half hour, not too frequently;
  - if OOM or a runbook-known issue happens, diagnose and repair, then resume
    the current failed experiment;
  - do not directly switch to the next training/eval item after experiment 1
    fails.
- Reverted the formal full queue default to failure-blocking:
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` now defaults
    `QUEUE_CONTINUE_ON_FAILURE=0` again.
  - The adopt-active-run behavior remains, so restarting the supervisor can keep
    watching an already-running training tmux without restarting training.
- Updated documentation:
  - `docs/joint_training/guides/code_task_monitor_agent_runbook.md` now states
    that failures during or after launch block the queue until the same item is
    diagnosed, repaired, and resumed/retried.
  - `docs/joint_training/guides/training_script_index.md` now records the full
    queue as failure-blocking with active-run adoption.
- Live evidence before supervisor restart:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Active beta0 run remains
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
  - Metrics reached `global_step=139`; latest validation remains at
    `global_step=135` with HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - GPU memory is active on all 8 GPUs, about `66-68GB / 80GB`.
  - `/data-1` free space is about `91G`; Ray still warns that `/data-1/ray_tmp`
    is over 95% full.
- Next action: restart only queue supervisor `code_task_full_queue` with the new
  failure-blocking default. Do not kill or restart training tmux
  `code_task_s1_kodcode_beta0`.

Supervisor restart confirmation after policy correction at local time `2026-06-05 05:28-05:30`:

- Syntax checks passed:
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh`
- Restarted only queue supervisor tmux `code_task_full_queue`; training tmux
  `code_task_s1_kodcode_beta0` remained the original session from `01:21`.
- New queue log explicitly confirms active policy:
  `QUEUE_CONTINUE_ON_FAILURE=0`.
- New queue log confirms active-run adoption:
  `adopting active code_task_s1_kodcode_beta0`.
- Runtime state after restart:
  - tmux sessions alive: `code_task_full_queue`, `code_task_full_monitor`,
    `code_task_s1_kodcode_beta0`.
  - Docker container alive: `aee03043ecaf` (`verl-harness`, `eager_mclean`), up
    about 4 hours.
  - Active metrics reached `global_step=140` with `wdl_sft/correct_ratio=0.52734375`,
    `wdl_sft/n_correct=270`, `actor/grad_norm=13.839275360107422`,
    `response_length/clip_ratio=0.0`, `response/aborted_ratio=0.0`.
  - Latest validation at `global_step=140`: HumanEval+ mean@3
    `0.6666666666666666`, MBPP+ mean@3 `1.0`, BigCodeBench mean@3 `1.0`,
    LiveCodeBench mean@3 `0.0`.
  - Checkpoint retention: `latest_checkpointed_iteration.txt=140`; checkpoint
    dirs are `global_step_70` and `global_step_140`; `best_checkpoint.json`
    still points to `global_step_70` with HumanEval+ mean@3
    `0.6666666666666666`.
  - Checkpoint dir size about `63G`.
- Hard-error scan found no `Error executing job`, `RayTaskError`, missing
  official evaluator, CUDA OOM, disk-full, killed/aborted, NCCL error, or parser
  `MemoryError` recurrence.
- Disk remains tight: `/data-1` has about `91G` free and Ray continues warning
  `/data-1/ray_tmp` is over 95% full. No deletion or parameter change was made.
- Next routine Monitor Agent check should be around `2026-06-05 05:58-06:00`
  unless the queue/monitor reports a failure earlier.

Routine Monitor Agent check at local time `2026-06-05 05:58-06:00`:

- Queue state:
  - beta0 completed normally; this was not a failure skip.
  - Queue advanced to beta0.1 after beta0 final checkpoint and metrics were
    present, which is consistent with the failure-blocking policy.
  - Active run is now
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
- Runtime status:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`, `ecstatic_shaw`), up
    about 20 minutes.
  - GPU memory active on all 8 GPUs, about `63.2-63.9GB / 80GB`; utilization at
    check time ranged from `31%` to `100%`.
- Disk status:
  - `/data-1` and `/data-1/checkpoints` have about `61G` free, `98%` used.
  - Ray is warning that `/data-1/ray_tmp` is over 95% full and object creation
    may fail if spilling is required.
  - This is the main active risk. No deletion was performed automatically.
  - WxPusher disk-pressure notification was sent successfully at this check.
- beta0 completion evidence:
  - Metrics file exists with `151` records.
  - Final beta0 record at `global_step=150`:
    - HumanEval+ mean@3 `0.8333333333333333`
    - MBPP+ mean@3 `1.0`
    - BigCodeBench mean@3 `1.0`
    - LiveCodeBench mean@3 `0.0`
    - `wdl_sft/correct_ratio=0.458984375`
    - `wdl_sft/n_correct=235`
    - `actor/grad_norm=13.170166969299316`
    - `response_length/clip_ratio=0.0`
    - `response/aborted_ratio=0.0`
    - `perf/max_memory_allocated_gb=34.729820251464844`
    - `actor/lr=5e-07`
  - Checkpoint evidence:
    - `latest_checkpointed_iteration.txt=150`
    - only `global_step_150` remains under the beta0 checkpoint dir;
      `best_checkpoint.json` also points to `global_step_150`.
    - beta0 checkpoint dir size is about `47G`.
  - Queue status file records beta0 as completed at `2026-06-05 05:38:24` with
    final checkpoint and metrics paths.
- beta0.1 active evidence:
  - Checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - Metrics file exists and currently has `6` records.
  - Current beta0.1 checkpoint state:
    - `latest_checkpointed_iteration.txt=5`
    - `global_step_5` exists
    - `best_checkpoint.json` points to `global_step_5`
    - checkpoint dir size is about `47G`
  - Latest beta0.1 metrics at `global_step=5`:
    - HumanEval+ mean@3 `0.0`
    - MBPP+ mean@3 `0.8333333333333333`
    - BigCodeBench mean@3 `0.3333333333333333`
    - LiveCodeBench mean@3 `0.0`
    - `wdl_sft/correct_ratio=0.177734375`
    - `wdl_sft/n_correct=91`
    - `actor/grad_norm=52.96256637573242`
    - `response_length/clip_ratio=0.2109375`
    - `response/aborted_ratio=0.0`
    - `perf/max_memory_allocated_gb=34.359960079193115`
    - `actor/lr=4e-07`
  - Val-before-train beta0.1 record at step `0` was HumanEval+ mean@3
    `0.3333333333333333`, MBPP+ mean@3 `0.8333333333333333`, BigCodeBench
    mean@3 `0.0`, LiveCodeBench mean@3 `0.0`.
- Error status:
  - A broad scan of `run_code_task_full_queue.log` still matches historical
    fixed failures from earlier beta0 attempts (`evalplus` missing and parser
    `MemoryError`). These are old log entries, not current beta0.1 failures.
  - A beta0.1-specific hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- No repair or parameter change was made in this check.
- Next routine Monitor Agent check should be around `2026-06-05 06:28-06:30`,
  unless the queue/monitor reports a failure earlier.

Routine Monitor Agent check at local time `2026-06-05 06:28-06:30`:

- Active run remains
  `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
- Runtime status:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`, `ecstatic_shaw`), up
    about 50 minutes.
  - GPU memory active on all 8 GPUs, about `39.4-39.5GB / 80GB`; utilization at
    check time ranged from `67%` to `81%`.
- Disk status:
  - `/data-1` and `/data-1/checkpoints` have about `45G` free, `99%` used.
  - Ray continues warning that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - This is now critical operational risk. No deletion was performed
    automatically.
  - WxPusher critical disk-pressure notification was sent successfully at this
    check.
- Metrics status:
  - Metrics file exists and currently has `20` records.
  - Latest metric record at `global_step=19`:
    - `wdl_sft/correct_ratio=0.17578125`
    - `wdl_sft/n_correct=90`
    - `actor/grad_norm=55.29714584350586`
    - `response_length/clip_ratio=0.154296875`
    - `response/aborted_ratio=0.0`
    - `perf/throughput=873.1198958693458`
    - `perf/max_memory_allocated_gb=34.41904067993164`
    - `actor/lr=5e-07`
  - Latest validation record at `global_step=15`:
    - HumanEval+ mean@3 `0.0`
    - MBPP+ mean@3 `0.3333333333333333`
    - BigCodeBench mean@3 `0.6666666666666666`
    - LiveCodeBench mean@3 `0.0`
- Checkpoint status:
  - `latest_checkpointed_iteration.txt=15`.
  - checkpoint dirs currently present: `global_step_5` and `global_step_15`.
  - `best_checkpoint.json` points to `global_step_5` with HumanEval+ mean@3
    `0.0`.
  - checkpoint dir size about `63G`.
  - Retention is still best+latest; because best is step 5 and latest is step 15,
    two checkpoint directories are expected.
- Error status:
  - beta0.1-specific hard-error scan found no `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
  - SyntaxWarning lines from generated code continue to appear in reward workers;
    these are not treated as training failures.
- No repair or parameter change was made in this check.
- Next routine Monitor Agent check should be around `2026-06-05 06:58-07:00`,
  unless the queue/monitor reports a failure earlier.

Queue policy update at local time `2026-06-05 07:00`:

- User updated the queue behavior requirement:
  - strict serial scheduling: only one training item runs at a time;
  - if an item fails, record the failure and move to the next independent
    experiment instead of blocking the whole queue;
  - the failed item must not be resumed automatically; Monitor Agent must debug
    it, record the repair, then resume/retry that same item with explicit index
    filters.
- Script changes applied:
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` now
    defaults `QUEUE_CONTINUE_ON_FAILURE=1`.
  - Existing active-run adoption remains in
    `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh`.
- Documentation changes applied:
  - `docs/joint_training/guides/code_task_monitor_agent_runbook.md` now states
    the failure-continuing queue policy and the repaired-item retry boundary.
  - `docs/joint_training/guides/training_script_index.md` now records the full
    queue as failure-continuing and adopt-capable.
- Live state before supervisor restart:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Current training tmux `code_task_s1_kodcode_beta01` remains active and must
    not be killed or restarted.
- Next action: run syntax checks, then restart only queue supervisor
  `code_task_full_queue` if it is still using the old failure-blocking default.

Supervisor restart confirmation at local time `2026-06-05 07:01`:

- Verification before restart:
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`
    passed.
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh`
    passed.
  - `bash -n recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh`
    passed.
  - `DRY_RUN=1 WXPUSHER_NOTIFY=0 bash
    recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` printed
    `QUEUE_CONTINUE_ON_FAILURE=1`.
- Restarted only queue supervisor tmux `code_task_full_queue` with:
  `MIN_FREE_GB=100 ALLOW_CODE_FULL_TRAINING=1 bash
  recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`.
- Training tmux `code_task_s1_kodcode_beta01` was not killed or restarted.
- New supervisor evidence:
  - queue log printed `QUEUE_CONTINUE_ON_FAILURE=1`;
  - beta0 was detected as already complete at `step=150`;
  - current beta0.1 training was adopted:
    `adopting active code_task_s1_kodcode_beta01`;
  - supervisor is waiting on beta0.1 to reach `final=150`.
- Light health sample after restart:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - active beta0.1 metrics file has `35` records and reached
    `global_step=34`.
  - latest step 34 metrics:
    `wdl_sft/correct_ratio=0.419921875`, `wdl_sft/n_correct=215`,
    `actor/grad_norm=41.130226135253906`,
    `response_length/clip_ratio=0.05859375`, `response/aborted_ratio=0.0`,
    `perf/throughput=630.4183639923842`,
    `perf/max_memory_allocated_gb=34.44479513168335`.
  - latest validation remains step `30`: HumanEval+ mean@3
    `0.3333333333333333`, MBPP+ mean@3 `0.8333333333333333`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - checkpoint `latest_checkpointed_iteration.txt=30`; `best_checkpoint.json`
    points to `global_step_30`; checkpoint dir size is about `47G`.
  - GPU memory is active on all 8 GPUs, about `39.4-39.5GB / 80GB`.
  - `/data-1` and `/data-1/checkpoints` have about `61G` free, `98%` used.
  - beta0.1-specific hard-error scan found no current `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- No training parameter, dataset, reward, checkpoint retention, or checkpoint
  deletion change was made.

Policy correction at local time `2026-06-05 07:02-07:05`:

- Active goal context clarified that the Monitor Agent policy is
  failure-blocking:
  - check roughly every half hour;
  - if OOM or a runbook-known issue happens, diagnose and repair first;
  - resume/retry the same failed experiment before moving to the next training
    or eval item;
  - every repair must be fully recorded for later audit.
- Script changes applied:
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` now
    defaults `QUEUE_CONTINUE_ON_FAILURE=0`.
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` now sets
    the default after resolving `QUEUE_MODE`: `full` defaults to
    `QUEUE_CONTINUE_ON_FAILURE=0`, while smoke/pilot keep
    `QUEUE_CONTINUE_ON_FAILURE=1` unless explicitly overridden.
- Documentation changes applied:
  - `docs/joint_training/guides/code_task_monitor_agent_runbook.md` restored
    the formal full-queue policy: failed items block the queue until the same
    item is diagnosed, repaired, and resumed/retried.
  - `docs/joint_training/guides/training_script_index.md` restored the full
    queue note to failure-blocking and adopt-capable.
- Verification:
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`
    passed.
  - `bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh`
    passed.
  - `bash -n recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh`
    passed.
  - `DRY_RUN=1 WXPUSHER_NOTIFY=0 bash
    recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh` printed
    `QUEUE_CONTINUE_ON_FAILURE=0`.
  - `DRY_RUN=1 WXPUSHER_NOTIFY=0 bash
    recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` printed
    `QUEUE_CONTINUE_ON_FAILURE=1`, preserving smoke behavior.
- Restarted only queue supervisor tmux `code_task_full_queue` with:
  `MIN_FREE_GB=100 ALLOW_CODE_FULL_TRAINING=1 bash
  recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.sh`.
- Training tmux `code_task_s1_kodcode_beta01` was not killed or restarted.
- New supervisor evidence:
  - queue log printed `QUEUE_CONTINUE_ON_FAILURE=0`;
  - beta0 was detected as already complete at `step=150`;
  - current beta0.1 training was adopted:
    `adopting active code_task_s1_kodcode_beta01`;
  - supervisor is waiting on beta0.1 to reach `final=150`.
- Health sample after restart:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`,
    `ecstatic_shaw`).
  - active beta0.1 metrics file has `36` records and reached
    `global_step=35`.
  - latest step 35 metrics:
    `wdl_sft/correct_ratio=0.365234375`, `wdl_sft/n_correct=187`,
    `actor/grad_norm=43.7203483581543`,
    `response_length/clip_ratio=0.046875`, `response/aborted_ratio=0.0`,
    `perf/throughput=493.08753710345053`,
    `perf/max_memory_allocated_gb=34.44479513168335`.
  - latest validation at step `35`: HumanEval+ mean@3 `0.0`, MBPP+ mean@3
    `0.6666666666666666`, BigCodeBench mean@3 `0.3333333333333333`,
    LiveCodeBench mean@3 `0.3333333333333333`.
  - checkpoint `latest_checkpointed_iteration.txt=35`; current checkpoint dirs
    are `global_step_30` and `global_step_35`; `best_checkpoint.json` points to
    `global_step_30`; checkpoint dir size is about `63G`.
  - GPU memory is active on all 8 GPUs, about `39.0GB / 80GB`.
  - `/data-1` and `/data-1/checkpoints` have about `45G` free, `99%` used.
  - Ray still warns that `/data-1/ray_tmp` is over 95% full and object creation
    may fail if spilling is required.
  - beta0.1-specific hard-error scan found no current `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- WxPusher disk-risk notification was sent successfully because free space is
  about `45G` and future checkpoint writes are at risk.
- No training parameter, dataset, reward, checkpoint retention, or checkpoint
  deletion change was made.

Health check at local time `2026-06-05 07:35 CST`:

- Low-frequency monitor policy was respected: this was the next full check
  after the `07:02-07:05` repair/check window.
- Runtime state:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`,
    `ecstatic_shaw`, up about 2 hours).
  - GPU memory active on all 8 GPUs, about `39.4GB / 80GB` each, GPU
    utilization about `57-60%`.
- Active experiment:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file exists with `53` records.
  - progress advanced from the prior full check at `global_step=35` to latest
    metrics `global_step=52`.
  - queue supervisor log shows it is still waiting on the same beta0.1 item,
    not skipping to another experiment.
- Latest train metrics:
  - step `52`: `wdl_sft/correct_ratio=0.478515625`,
    `wdl_sft/n_correct=245`, `actor/grad_norm=24.301881790161133`,
    `response_length/clip_ratio=0.025390625`,
    `response/aborted_ratio=0.0`, `perf/throughput=431.08684994724047`,
    `perf/max_memory_allocated_gb=34.57762145996094`,
    `actor/lr=5e-07`.
  - recent train metrics from steps `39-52` show nonzero correct ratios
    roughly `0.35-0.56`, no aborted responses, and clip ratio below about
    `0.05`.
- Latest validation metrics:
  - step `40`: HumanEval+ mean@3 `0.3333333333333333`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - step `45`: HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3 `0.0`.
  - step `50`: HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3 `0.0`.
- Checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=50`.
  - checkpoint dirs present: `global_step_45`, `global_step_50`.
  - `best_checkpoint.json` points to `global_step_45` with
    `val-core/HumanEval+/acc/mean@3=0.5`.
  - checkpoint dir size is about `63G`.
- Disk state:
  - `/data-1` and `/data-1/checkpoints` still have about `45G` free and are
    `99%` used.
  - Ray continues to warn that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required.
  - This is a persistent serious risk, but no disk-full error was observed in
    the active beta0.1 log at this check.
- Hard-error scan on the beta0.1-specific active log found no current
  `Error executing job`, `RayTaskError`, missing official evaluator, CUDA OOM,
  disk-full, killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- WxPusher decision:
  - a dry-run notification payload was checked for the continuing disk-risk
    message;
  - no duplicate live notification was sent at this check because the same
    `~45G` disk-risk condition had already been pushed at `07:02-07:05` and no
    new failure, repair, relaunch, or user decision occurred.
- No repair was needed.
- No training parameter, dataset, reward, queue policy, checkpoint retention,
  checkpoint deletion, or disk cleanup change was made.

Health check at local time `2026-06-05 09:05 CST`:

- Low-frequency monitor policy was respected: this check was about 30 minutes
  after the `08:36 CST` full check.
- Runtime state:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`,
    `ecstatic_shaw`, up about 3 hours).
  - GPU memory active on all 8 GPUs, about `39.4-39.5GB / 80GB` each.
  - GPU utilization sample showed GPU `0` at `55%` and the other GPUs at
    `0%`. Queue/training log showed the run had just advanced from step `105`
    to step `107`; this sample was interpreted as a transient rollout/training
    phase rather than a stall.
- Active experiment:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file exists with `108` records.
  - progress advanced from the prior full check at `global_step=88` to latest
    metrics `global_step=107`.
  - queue supervisor log still shows it is waiting on the same beta0.1 item,
    not skipping to another experiment.
- Latest train metrics:
  - step `107`: `wdl_sft/correct_ratio=0.494140625`,
    `wdl_sft/n_correct=253`, `actor/grad_norm=14.923698425292969`,
    `response_length/clip_ratio=0.001953125`,
    `response/aborted_ratio=0.0`, `perf/throughput=377.0145524701325`,
    `perf/max_memory_allocated_gb=34.57762145996094`,
    `actor/lr=5e-07`.
  - recent train metrics from steps `90-107` show nonzero correct ratios
    roughly `0.43-0.60`, no aborted responses, and very low clip ratios.
- Latest validation metrics:
  - step `90`: HumanEval+ mean@3 `0.3333333333333333`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3
    `0.6666666666666666`.
  - step `95`: HumanEval+ mean@3 `0.16666666666666666`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - step `100`: HumanEval+ mean@3 `0.6666666666666666`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3 `0.0`.
  - step `105`: HumanEval+ mean@3 `0.6666666666666666`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3
    `0.3333333333333333`.
- Checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=105`.
  - checkpoint dirs present: `global_step_100`, `global_step_105`.
  - `best_checkpoint.json` now points to `global_step_100` with
    `val-core/HumanEval+/acc/mean@3=0.6666666666666666`.
  - checkpoint dir size remains about `63G`.
- Disk state:
  - `/data-1` and `/data-1/checkpoints` now have about `44G` free and are
    `99%` used.
  - Ray continues to warn that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required; sample free space in Ray warning
    is about `43.96GB`.
  - Free space is lower than the prior checks, so disk remains the main
    operational risk, but no disk-full error was observed in the active beta0.1
    log at this check.
- Hard-error scan on the beta0.1-specific active log found no current
  `Error executing job`, `RayTaskError`, missing official evaluator, CUDA OOM,
  disk-full, killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- WxPusher decision:
  - no duplicate live notification was sent at this check because there was no
    new failure, repair, relaunch, completion, or user decision. The persistent
    disk risk is already known and was previously pushed.
- No repair was needed.
- No training parameter, dataset, reward, queue policy, checkpoint retention,
  checkpoint deletion, or disk cleanup change was made.

Health check at local time `2026-06-05 08:36 CST`:

- Low-frequency monitor policy was respected: this check was about 30 minutes
  after the `08:05 CST` full check.
- Runtime state:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`,
    `ecstatic_shaw`, up about 3 hours).
  - GPU memory active on all 8 GPUs, about `39.4-39.5GB / 80GB` each.
  - GPU utilization sample was mixed: GPUs `2`, `3`, and `7` at about
    `53-57%`, others at `0%`. Queue/training log showed the run had just
    advanced through step `88`; this sample was interpreted as a transient
    phase rather than a stall.
- Active experiment:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file exists with `89` records.
  - progress advanced from the prior full check at `global_step=69` to latest
    metrics `global_step=88`.
  - queue supervisor log still shows it is waiting on the same beta0.1 item,
    not skipping to another experiment.
- Latest train metrics:
  - step `88`: `wdl_sft/correct_ratio=0.46484375`,
    `wdl_sft/n_correct=238`, `actor/grad_norm=17.36363983154297`,
    `response_length/clip_ratio=0.0`, `response/aborted_ratio=0.0`,
    `perf/throughput=453.12705092121183`,
    `perf/max_memory_allocated_gb=34.57762145996094`,
    `actor/lr=5e-07`.
  - recent train metrics from steps `73-88` show nonzero correct ratios
    roughly `0.43-0.61`, no aborted responses, and very low clip ratios.
  - one transient high `actor/grad_norm=90.26691436767578` appeared at step
    `84`; subsequent steps returned to about `15-17`, so no intervention was
    taken.
- Latest validation metrics:
  - step `75`: HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - step `80`: HumanEval+ mean@3 `0.3333333333333333`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.3333333333333333`, LiveCodeBench mean@3 `0.0`.
  - step `85`: HumanEval+ mean@3 `0.3333333333333333`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.3333333333333333`, LiveCodeBench mean@3
    `0.3333333333333333`.
- Checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=85`.
  - checkpoint dirs present: `global_step_45`, `global_step_85`.
  - `best_checkpoint.json` still points to `global_step_45` with
    `val-core/HumanEval+/acc/mean@3=0.5`.
  - checkpoint dir size remains about `63G`.
- Disk state:
  - `/data-1` and `/data-1/checkpoints` still have about `45G` free and are
    `99%` used.
  - Ray continues to warn that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required; sample free space in Ray warning
    is about `44.19GB`.
  - This remains the main operational risk, but no disk-full error was observed
    in the active beta0.1 log at this check.
- Hard-error scan on the beta0.1-specific active log found no current
  `Error executing job`, `RayTaskError`, missing official evaluator, CUDA OOM,
  disk-full, killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- WxPusher decision:
  - no duplicate live notification was sent at this check because the disk
    risk is materially the same as the already-pushed `07:02-07:05` risk and
    no new failure, repair, relaunch, completion, or user decision occurred.
- No repair was needed.
- No training parameter, dataset, reward, queue policy, checkpoint retention,
  checkpoint deletion, or disk cleanup change was made.

Final completion check at local time `2026-06-05 10:05 CST`:

- Queue completion:
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue_status.tsv`
    records beta0 as completed at `2026-06-05 05:38:24`.
  - The same status file records beta0.1 as completed at
    `2026-06-05 09:59:45`, with `step=150`.
  - Queue log records
    `completed code_task_s1_kodcode_beta01 ... step=150 final=150` at
    `2026-06-05 09:59:45`.
  - Queue log records `code-task full queue complete` at
    `2026-06-05 09:59:45`.
  - `code_task_full_queue` and `code_task_s1_kodcode_beta01` tmux sessions are
    no longer present; only `code_task_full_monitor` remains.
- Runtime state:
  - no active Docker training container was present in `docker ps`.
  - GPU memory was essentially idle on all 8 GPUs, about `1MB / 80GB`, with
    `0%` utilization.
  - `/data-1` and `/data-1/checkpoints` had about `45G` free and remained
    `99%` used.
- Final beta0.1 metrics:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file exists with `151` records.
  - final recorded `global_step=150`.
  - final train metrics at step `150`:
    `wdl_sft/correct_ratio=0.458984375`, `wdl_sft/n_correct=235`,
    `actor/grad_norm=13.085588455200195`,
    `response_length/clip_ratio=0.0`, `response/aborted_ratio=0.0`,
    `perf/throughput=761.6565793966381`,
    `perf/max_memory_allocated_gb=34.57762145996094`.
  - final validation at step `150`: HumanEval+ mean@3
    `0.8333333333333333`, MBPP+ mean@3 `1.0`, BigCodeBench mean@3
    `0.3333333333333333`, LiveCodeBench mean@3 `0.3333333333333333`.
- Final beta0.1 checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=150`.
  - checkpoint dirs present: `global_step_115`, `global_step_150`.
  - `best_checkpoint.json` points to `global_step_115` with
    `val-core/HumanEval+/acc/mean@3=0.8333333333333333`.
  - checkpoint dir size is about `63G`.
- Hard-error interpretation:
  - beta0.1-specific active log scan found no current `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
  - Queue log contains `BrokenPipeError: [Errno 32] Broken pipe` from wandb
    teardown and `resource_tracker` warnings after the training progress had
    reached `100%|...| 150/150`; these were interpreted as exit-cleanup
    warnings, not training failure evidence.
- WxPusher:
  - queue log records successful WxPusher API responses for beta0.1 completion
    and full-queue completion at `2026-06-05 09:59:45`, both with
    `"status": "创建发送任务成功"` and `"success": true`.
  - a dry-run completion notification payload was also checked by the Monitor
    Agent at `10:05 CST`; no extra live duplicate notification was sent because
    the queue script had already sent the completion notifications
    successfully.
- Repairs and changes:
  - no repair was needed at final check.
  - no training parameter, dataset, reward, queue policy, checkpoint retention,
    checkpoint deletion, or disk cleanup change was made.
  - the formal full queue is complete: beta0 and beta0.1 both reached
    `150/150` and the queue supervisor reported full completion.

Monitor cleanup at local time `2026-06-05 10:05 CST`:

- After final completion was verified, the only remaining code-task tmux
  session was `code_task_full_monitor`.
- That monitor session was a completed-queue residual and was stopped with
  `tmux kill-session -t code_task_full_monitor`.
- Post-cleanup `tmux ls | rg code_task` showed no remaining code-task tmux
  sessions.

Health check at local time `2026-06-05 09:35 CST`:

- Runlog ordering note:
  - recent health-check blocks in this file are not strictly chronological
    because several entries were inserted near repeated matching text instead
    of appended to EOF;
  - this `09:35 CST` check is appended at EOF and future monitor entries should
    continue appending at EOF.
- Low-frequency monitor policy was respected: this check was about 30 minutes
  after the `09:05 CST` full check.
- Runtime state:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`,
    `ecstatic_shaw`, up about 4 hours).
  - GPU memory active on all 8 GPUs, about `45.6-46.9GB / 80GB` each.
  - GPU utilization was high at sample time, about `96-97%` on all GPUs.
- Active experiment:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file exists with `131` records.
  - progress advanced from the prior full check at `global_step=107` to latest
    metrics `global_step=130`.
  - queue supervisor log still shows it is waiting on the same beta0.1 item,
    not skipping to another experiment.
- Latest train metrics:
  - step `130`: `wdl_sft/correct_ratio=0.53515625`,
    `wdl_sft/n_correct=274`, `actor/grad_norm=12.600939750671387`,
    `response_length/clip_ratio=0.0`, `response/aborted_ratio=0.0`,
    `perf/throughput=447.18645715923213`,
    `perf/max_memory_allocated_gb=34.57762145996094`,
    `actor/lr=5e-07`.
  - recent train metrics from steps `111-130` show nonzero correct ratios
    roughly `0.43-0.63`, no aborted responses, and near-zero clip ratios.
- Latest validation metrics:
  - step `115`: HumanEval+ mean@3 `0.8333333333333333`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.3333333333333333`, LiveCodeBench mean@3 `0.0`.
  - step `120`: HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3
    `0.6666666666666666`.
  - step `125`: HumanEval+ mean@3 `0.6666666666666666`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - step `130`: HumanEval+ mean@3 `0.6666666666666666`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3
    `0.3333333333333333`.
- Checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=130`.
  - checkpoint dirs present: `global_step_115`, `global_step_130`.
  - `best_checkpoint.json` now points to `global_step_115` with
    `val-core/HumanEval+/acc/mean@3=0.8333333333333333`.
  - checkpoint dir size remains about `63G`.
- Disk state:
  - `/data-1` and `/data-1/checkpoints` still have about `44G` free and are
    `99%` used.
  - Ray continues to warn that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required; sample free space in Ray warning
    is about `43.91GB`.
  - Free space remains the main operational risk, but no disk-full error was
    observed in the active beta0.1 log at this check.
- Hard-error scan on the beta0.1-specific active log found no current
  `Error executing job`, `RayTaskError`, missing official evaluator, CUDA OOM,
  disk-full, killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- WxPusher decision:
  - no duplicate live notification was sent at this check because there was no
    new failure, repair, relaunch, completion, or user decision. The persistent
    disk risk is already known and was previously pushed.
- No repair was needed.
- No training parameter, dataset, reward, queue policy, checkpoint retention,
  checkpoint deletion, or disk cleanup change was made.

Health check at local time `2026-06-05 08:05 CST`:

- Low-frequency monitor policy was respected: this check was about 30 minutes
  after the `07:35 CST` full check.
- Runtime state:
  - tmux sessions alive: `code_task_full_monitor`, `code_task_full_queue`,
    `code_task_s1_kodcode_beta01`.
  - Docker container alive: `efbfbb4575b2` (`verl-harness`,
    `ecstatic_shaw`, up about 2 hours).
  - GPU memory active on all 8 GPUs, about `67.4-68.6GB / 80GB` each, GPU
    utilization `99%` on all GPUs at sample time.
- Active experiment:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file exists with `70` records.
  - progress advanced from the prior full check at `global_step=52` to latest
    metrics `global_step=69`.
  - queue supervisor log still shows it is waiting on the same beta0.1 item,
    not skipping to another experiment.
- Latest train metrics:
  - step `69`: `wdl_sft/correct_ratio=0.560546875`,
    `wdl_sft/n_correct=287`, `actor/grad_norm=16.767873764038086`,
    `response_length/clip_ratio=0.001953125`,
    `response/aborted_ratio=0.0`, `perf/throughput=404.8996779611553`,
    `perf/max_memory_allocated_gb=34.57762145996094`,
    `actor/lr=5e-07`.
  - recent train metrics from steps `54-69` show nonzero correct ratios
    roughly `0.42-0.61`, no aborted responses, and low clip ratios below about
    `0.014`.
- Latest validation metrics:
  - step `55`: HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.0`, LiveCodeBench mean@3 `0.0`.
  - step `60`: HumanEval+ mean@3 `0.3333333333333333`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3
    `0.3333333333333333`.
  - step `65`: HumanEval+ mean@3 `0.5`, MBPP+ mean@3 `1.0`,
    BigCodeBench mean@3 `0.6666666666666666`, LiveCodeBench mean@3 `0.0`.
- Checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=65`.
  - checkpoint dirs present: `global_step_45`, `global_step_65`.
  - `best_checkpoint.json` still points to `global_step_45` with
    `val-core/HumanEval+/acc/mean@3=0.5`.
  - checkpoint dir size remains about `63G`.
- Disk state:
  - `/data-1` and `/data-1/checkpoints` still have about `45G` free and are
    `99%` used.
  - Ray continues to warn that `/data-1/ray_tmp` is over 95% full and object
    creation may fail if spilling is required; sample free space in Ray warning
    is about `44.23GB`.
  - This remains the main operational risk, but no disk-full error was observed
    in the active beta0.1 log at this check.
- Hard-error scan on the beta0.1-specific active log found no current
  `Error executing job`, `RayTaskError`, missing official evaluator, CUDA OOM,
  disk-full, killed/aborted, NCCL error, or parser `MemoryError` recurrence.
- WxPusher decision:
  - no duplicate live notification was sent at this check because the disk
    risk is materially the same as the already-pushed `07:02-07:05` risk and
    no new failure, repair, relaunch, completion, or user decision occurred.
- No repair was needed.
- No training parameter, dataset, reward, queue policy, checkpoint retention,
  checkpoint deletion, or disk cleanup change was made.

Final completion check at local time `2026-06-05 10:05 CST`:

- Queue completion:
  - `recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue_status.tsv`
    records beta0 as completed at `2026-06-05 05:38:24`.
  - The same status file records beta0.1 as completed at
    `2026-06-05 09:59:45`, with `step=150`.
  - Queue log records
    `completed code_task_s1_kodcode_beta01 ... step=150 final=150` at
    `2026-06-05 09:59:45`.
  - Queue log records `code-task full queue complete` at
    `2026-06-05 09:59:45`.
  - `code_task_full_queue` and `code_task_s1_kodcode_beta01` tmux sessions are
    no longer present; only `code_task_full_monitor` remains.
- Runtime state:
  - no active Docker training container was present in `docker ps`.
  - GPU memory was essentially idle on all 8 GPUs, about `1MB / 80GB`, with
    `0%` utilization.
  - `/data-1` and `/data-1/checkpoints` had about `45G` free and remained
    `99%` used.
- Final beta0 metrics:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`.
  - metrics file has `151` records and final `global_step=150`.
  - final validation: HumanEval+ mean@3 `0.8333333333333333`, MBPP+ mean@3
    `1.0`, BigCodeBench mean@3 `1.0`, LiveCodeBench mean@3 `0.0`.
  - `latest_checkpointed_iteration.txt=150`.
  - `best_checkpoint.json` points to `global_step_150` with
    `val-core/HumanEval+/acc/mean@3=0.8333333333333333`.
- Final beta0.1 metrics:
  - run id:
    `ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - metrics file has `151` records and final `global_step=150`.
  - final train metrics at step `150`:
    `wdl_sft/correct_ratio=0.458984375`, `wdl_sft/n_correct=235`,
    `actor/grad_norm=13.085588455200195`,
    `response_length/clip_ratio=0.0`, `response/aborted_ratio=0.0`,
    `perf/throughput=761.6565793966381`,
    `perf/max_memory_allocated_gb=34.57762145996094`.
  - final validation at step `150`: HumanEval+ mean@3
    `0.8333333333333333`, MBPP+ mean@3 `1.0`, BigCodeBench mean@3
    `0.3333333333333333`, LiveCodeBench mean@3 `0.3333333333333333`.
- Final beta0.1 checkpoint state:
  - checkpoint dir:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`.
  - `latest_checkpointed_iteration.txt=150`.
  - checkpoint dirs present: `global_step_115`, `global_step_150`.
  - `best_checkpoint.json` points to `global_step_115` with
    `val-core/HumanEval+/acc/mean@3=0.8333333333333333`.
  - checkpoint dir size is about `63G`.
- Hard-error interpretation:
  - beta0.1-specific active log scan found no current `Error executing job`,
    `RayTaskError`, missing official evaluator, CUDA OOM, disk-full,
    killed/aborted, NCCL error, or parser `MemoryError` recurrence.
  - Queue log contains `BrokenPipeError: [Errno 32] Broken pipe` from wandb
    teardown and `resource_tracker` warnings after the training progress had
    reached `100%|...| 150/150`; these were interpreted as exit-cleanup
    warnings, not training failure evidence.
- WxPusher:
  - queue log records successful WxPusher API responses for beta0.1 completion
    and full-queue completion at `2026-06-05 09:59:45`, both with
    `"status": "创建发送任务成功"` and `"success": true`.
  - a dry-run completion notification payload was also checked by the Monitor
    Agent at `10:05 CST`; no extra live duplicate notification was sent because
    the queue script had already sent the completion notifications
    successfully.
- Repairs and changes:
  - no repair was needed at final check.
  - no training parameter, dataset, reward, queue policy, checkpoint retention,
    checkpoint deletion, or disk cleanup change was made.
  - the formal full queue is complete: beta0 and beta0.1 both reached
    `150/150` and the queue supervisor reported full completion.

Monitor cleanup at local time `2026-06-05 10:05 CST`:

- After final completion was verified, the only remaining code-task tmux
  session was `code_task_full_monitor`.
- That monitor session was a completed-queue residual and was stopped with
  `tmux kill-session -t code_task_full_monitor`.
- Post-cleanup `tmux ls | rg code_task` showed no remaining code-task tmux
  sessions.

Post-run validation correction at local time `2026-06-05 13:50 CST`:

- The formal Stage1 queue's online validation metrics from the completed beta0 and beta0.1 runs are not reliable for checkpoint selection or effect comparison.
- Root cause: the default online validation parquet set used only tiny files:
  - `/data-1/dataset/code/verl_rl/official_humaneval_plus_val.parquet`: 2 prompts.
  - `/data-1/dataset/code/verl_rl/official_mbpp_plus_val.parquet`: 2 prompts.
  - `/data-1/dataset/code/verl_rl/official_bigcodebench_val.parquet`: 1 prompt.
  - `/data-1/dataset/code/verl_rl/official_livecodebench_val.parquet`: 1 prompt.
- Script correction:
  - generated full HumanEval+ online validation parquet at `/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet` with 164 prompts.
  - changed Stage1 default `CODE_VAL_FILES` to that single full HumanEval+ parquet; `VAL_N=3` now means 164 prompts x 3 = 492 online validation outputs.
  - updated Meituan env to default to the same full HumanEval+ validation path under `$CODE_DATA_ROOT/online_full_humaneval_plus/`.
- Offline eval implementation correction:
  - replaced metadata-only `eval_code_vllm.py` with real vLLM generation.
  - added `run_code_offline_eval_case.sh` and `run_code_offline_eval_queue.sh` to run merge -> vLLM generation -> shared extraction -> official EvalPlus scoring inside `verl-harness`.
  - mounted host EvalPlus cache into Docker at `/root/.cache/evalplus` to avoid network downloads during official scoring.
  - separated merge/generation PYTHONPATH from official evaluator PYTHONPATH to avoid `official_site` ABI pollution such as `orjson` import errors.
  - moved WxPusher notification to the host queue layer because `/root/agent-core` is not mounted in the container.
- Full HumanEval+ official offline eval on step-150 checkpoints, `N_SAMPLES=3`, `temperature=1.0`, `top_p=0.95`, `max_tokens=4096`, `tensor_parallel=4`, official EvalPlus `parallel=8`:
  - beta0 step150 (`ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V1_1780593683`): 164 tasks, 492 outputs, extraction `492/492 ok`, EvalPlus base pass rate `0.7520325203252033`, HumanEval+ plus pass rate `0.693089430894309`.
  - beta0.1 step150 (`ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V1_1780609106`): 164 tasks, 492 outputs, extraction `492/492 ok`, EvalPlus base pass rate `0.7621951219512195`, HumanEval+ plus pass rate `0.6951219512195121`.
- Result paths:
  - `/data-1/eval_outputs/code_task/full_official/beta0_step150/humaneval/official_summary.json`.
  - `/data-1/eval_outputs/code_task/full_official/beta01_step150/humaneval/official_summary.json`.
- Disk cleanup after eval:
  - deleted merged eval copies under `/data-1/model_weights/code_task/offline_eval/*/actor_step150`.
  - retained raw generations, converted official samples, conversion reports, and official EvalPlus result JSON under `/data-1/eval_outputs/code_task/full_official/`.
  - final eval output footprint: about `7.6M`; `/data-1` returned to about `45G` free.
- Runtime cleanup:
  - no remaining `code_offline*` or `code_task*` tmux sessions.
  - no active eval Docker containers.
  - all 8 GPUs returned to idle.

Official cache migration at local time `2026-06-05 15:xx CST`:

- Runtime cache policy was tightened after the user clarified that official
  benchmark data must not come from another user's cache directory.
- Project-owned roots:
  - Hugging Face cache: `/data-1/.cache/huggingface`
  - EvalPlus cache via `XDG_CACHE_HOME`: `/data-1/.cache/evalplus`
  - official raw source root: `/data-1/dataset/code/official_sources`
  - BigCodeBench official full JSONL:
    `/data-1/dataset/code/official_sources/bigcodebench/BigCodeBench-v0.1.4.jsonl`
  - BigCodeBench official hard JSONL:
    `/data-1/dataset/code/official_sources/bigcodebench/BigCodeBench-Hard-v0.1.4.jsonl`
  - LiveCodeBench `release_v1` HF arrow remains in the project HF cache and
    is recorded in
    `/data-1/dataset/code/official_sources/official_cache_manifest.json`;
    it was not duplicated because the file is about `1.25GB`.
- Script changes:
  - `prepare_project_official_cache.py` materializes BCB JSONL files under the
    official source root and writes the manifest.
  - `prepare_official_only_validation.py` now defaults to
    `PROJECT_CACHE_ROOT=/data-1/.cache`, `HF_HOME=/data-1/.cache/huggingface`,
    `XDG_CACHE_HOME=/data-1/.cache`, and project BCB override paths, and records
    these paths in its manifest.
  - `run_code_offline_eval_queue.sh` now passes project HF/cache env vars,
    offline HF flags, `CODE_OFFICIAL_SOURCE_ROOT`, and
    `BIGCODEBENCH_OVERRIDE_PATH` into Docker. It mounts EvalPlus cache at
    `/data-1/.cache/evalplus`, not `/root/.cache/evalplus`.
  - `run_code_offline_eval_case.sh` and `eval_code_official.py` now use the
    same project cache env and offline HF flags for EvalPlus, BigCodeBench, and
    LiveCodeBench official scorers.
  - `meituan/env.sh` now exposes `HF_DATASETS_CACHE`,
    `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`,
    `CODE_OFFICIAL_SOURCE_ROOT`, and `BIGCODEBENCH_OVERRIDE_PATH`.
- Historical note: the HumanEval+ offline results above were produced before
  this cache-policy correction. They remain the historical EvalPlus scoring
  result, but future official eval reruns should use only the project-owned
  cache paths listed here.
