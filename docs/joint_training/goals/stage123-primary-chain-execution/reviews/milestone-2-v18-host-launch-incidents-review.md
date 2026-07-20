# Milestone 2 V18 Host Launch Incidents Review

## Review Identity

- Reviewer: independent GPT-5.5 medium Reviewer.
- Review type: scoped implementation/runtime review of F-EX-LAUNCH-02, F-EX-LAUNCH-03, and F-EX-LAUNCH-04.
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`.
- Frozen Plan version: 18.
- Reviewed commit: `425f844734607b6e02bcd83a1de702d6e3239a30`.
- Scope exclusions: no implementation changes, no production-code edits, no launch action, no edits to `runtime.jsonl` or `findings.jsonl`.

## Overall Verdict

PASS.

The current live Stage123 run is using the exact tmux-propagated `STAGE123_EXECUTION_STATE_ROOT`, `RAY_ADDRESS=local` is present in tmux and the running child environment, the formal state/checkpoint identities are fresh with `resume_from_checkpoint=false`, prior failed pretraining attempts are preserved under the configured scratch archive, local Ray/vLLM/train processes are alive, initial validation completed, metrics show training steps 1-3, and protected-asset comparison passes.

Nonterminal vLLM HTTP bind retry tracebacks were observed during startup, but they did not terminate the run: the log proceeds to validation metrics and training progress afterward, and no fatal OOM/EADDRINUSE/training-failed pattern appears after training starts.

## Per-Finding Verdict Table

| Finding | Verdict | Evidence |
| --- | --- | --- |
| F-EX-LAUNCH-02 | PASS | `tmux show-environment -g` reports `STAGE123_EXECUTION_STATE_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state`; `/proc/3055075/environ` contains the same value; batch runner PID `3054977` uses `--state-root` with that exact path. |
| F-EX-LAUNCH-03 | PASS | Failed pretraining archive contains `20260716T1202-control-empty-checkpoint`, `..._1784203722-stale-ray-address`, and `..._1784204184-vllm-port-collision`; current formal state root is `primary-v18-20260716T123423Z-portfix/state`; current control state has `attempt=1`, `resume_from_checkpoint=false`, `status=running`, `child_id=3055075`; current formal checkpoint root is `..._1784205674`. |
| F-EX-LAUNCH-04 | PASS | `RAY_ADDRESS=local` is present in tmux and child env; live `main_ppo`, Ray raylet, vLLM workers, EngineCore workers, and GPU compute processes are active; log shows initial validation metrics and training progress through step 3 without terminal failure. |

## Commands And Evidence

### Runtime And Identity

```bash
git rev-parse HEAD
git rev-parse 425f844734607b6e02bcd83a1de702d6e3239a30
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Evidence:
- Both `git rev-parse` commands returned `425f844734607b6e02bcd83a1de702d6e3239a30`.
- `validate-runtime` exited 0 with `plan_status=READY`, `plan_version=18`, `goal_status=ACTIVE`, `pending_user_decisions=[]`.

### Tmux And Environment Propagation

```bash
tmux list-sessions
tmux show-environment -g | sort | rg 'STAGE123|RAY_ADDRESS|ALLOW_QWEN|EXPERIMENT_BATCH|REPO_HOST'
tmux list-panes -a -F '#S:#I.#P active=#{pane_active} pid=#{pane_pid} cmd=#{pane_current_command} title=#{pane_title}'
tr '\0' '\n' < /proc/3055075/environ | sort | rg 'STAGE123_EXECUTION_STATE_ROOT|RAY_ADDRESS|STAGE123_BATCH|ALLOW_QWEN|REPO_HOST'
```

Evidence:
- Session `stage123_primary_chain` exists; pane command is `python3`, PID `3054977`.
- Tmux global env:
  - `RAY_ADDRESS=local`.
  - `STAGE123_EXECUTION_STATE_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state`.
- Batch runner command:
  - `python3 /data-1/code/verl/scripts/experiment_execution_core.py batch-run --manifest /data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state --repo-root /data-1/code/verl`.
- Child environment includes:
  - `RAY_ADDRESS=local`.
  - `STAGE123_EXECUTION_STATE_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state`.
  - `STAGE123_BATCH_MANIFEST_SHA256=f129078f00e6e5924f1ec4cc27812543e374b4639fc6c037c00eaed13a79198c`.
  - `STAGE123_BATCH_ADMISSION_BUNDLE_SHA256=b0f232241479adcd3ad8fde8e99eb2e3c06f2cc0ffa96f11b4645a885898e6b7`.

### State, Retry, And Failed Attempt Preservation

```bash
STATE_ROOT=/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state
find "$STATE_ROOT" -maxdepth 2 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
jq . "$STATE_ROOT/stage123-primary-chain-frac25-p40-v1.json"
jq . "$STATE_ROOT/frac25-stage1-control.json"
tail -n 30 "$STATE_ROOT/events.jsonl"
find /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/failed-pretraining-attempts -maxdepth 5 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %s %p\n' | sort
ls -ld /data-1/checkpoints/CODE-S1-CONTROL-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16_1784205674
```

Evidence:
- Formal state files exist only under the new state root:
  - `stage123-primary-chain-frac25-p40-v1.json`.
  - `frac25-stage1-control.json`.
  - `events.jsonl`.
- Batch state: `status=running`, `current_item_id=stage123-primary`, `current_run_id=frac25-stage1-control`, `items=[]`, `phases=[]`, `batch_manifest_sha256=f129078f00e6e5924f1ec4cc27812543e374b4639fc6c037c00eaed13a79198c`.
- Control state: `attempt=1`, `resume_from_checkpoint=false`, `status=running`, `failure=null`, `cleanup=null`, `child_id=3055075`.
- Events show `item_started`, `phase_started` for `frac25-stage1-control`, and atomic state `status=running`.
- Failed attempt archive contains:
  - `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/failed-pretraining-attempts/20260716T1202-control-empty-checkpoint`.
  - `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/failed-pretraining-attempts/CODE-S1-CONTROL-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16_1784203722-stale-ray-address`.
  - `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/failed-pretraining-attempts/CODE-S1-CONTROL-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16_1784204184-vllm-port-collision`.
- Current formal checkpoint identity is fresh and separate:
  - `/data-1/checkpoints/CODE-S1-CONTROL-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16_1784205674`.

### Live Ray, Training, And GPU Evidence

```bash
ps -eo pid,ppid,pgid,stat,lstart,cmd --sort=pid | rg -i 'stage123|run_code_task_qwen3_1p7b|stage123_phase_adapter|main_ppo|ray|vllm|tmux|RAY_ADDRESS=local|STAGE123_EXECUTION_STATE_ROOT|wandb'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Evidence:
- Live process chain:
  - `3054977`: `experiment_execution_core.py batch-run`.
  - `3055075`: `docker run ... --env RAY_ADDRESS --env STAGE123_EXECUTION_STATE_ROOT ... stage123_phase_adapter.py --run-id frac25-stage1-control`.
  - `3055123`: `stage123_phase_adapter.py`.
  - `3055331`: `python3 -m verl.trainer.main_ppo ... trainer.experiment_name=CODE-S1-CONTROL-..._1784205674 ... trainer.resume_mode=auto`.
  - `3056215`: live Ray raylet.
  - `3067833` through `3067840`: `ray::WorkerDict.actor_rollout_update_weights`.
  - `3070016` through `3070024`: `ray::vLLMHttpServer`.
  - `3071104` through `3071170`: `VLLM::EngineCore`.
- GPU compute query shows eight actor workers and eight vLLM workers using GPUs; example rows:
  - `3067833, ray::WorkerDict.actor_rollout_update_weights, 11708 MiB`.
  - `3071453, VLLM::Worker, 10936 MiB`.
- GPU summary shows all 8 GPUs in use with nonzero utilization and memory usage, e.g. GPU 0 `19897 MiB / 46068 MiB, 48%`, GPU 7 `24159 MiB / 46068 MiB, 40%`.

### Validation And Training Progress

```bash
LOG=/data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/runtime/frac25-stage1-control/logs/CODE-S1-CONTROL-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16_1784205674.log
rg -n 'Initial validation metrics|Training Progress|rollout_mode\(eval_only=False\)|global_step|step:' "$LOG" | tail -80
rg -n 'CUDA out of memory|OutOfMemory|EADDRINUSE|Address already in use|RuntimeError:.*EADDRINUSE|Training failed|SIGKILL|NCCL.*error|Aborted' "$LOG" || true
awk 'NR>=2808 && /Traceback|ERROR:|CUDA out of memory|EADDRINUSE|Address already in use|RayTaskError|WorkerCrashed|Training failed|SIGKILL|Aborted/{print NR ":" $0}' "$LOG" | tail -40
tail -n 10 /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/runtime/frac25-stage1-control/logs/metrics/OnPolicyWDLSFT-CodeTask/CODE-S1-CONTROL-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16_1784205674.jsonl
```

Evidence:
- Initial validation completed:
  - Log line 2762: `Initial validation metrics`.
  - Metrics line `step=0` includes `val-core/HumanEval+/acc/pass@1=0.4634146341463415`, `val-core/MBPP+/acc/pass@1=0.47354497354497355`, `val-core/LiveCodeBench/acc/pass@1=0.07765830346475508`.
- Training started and advanced:
  - Log line 2808: `Training Progress: 0%| | 0/60`.
  - Log line 2815: `Training Progress: 2%| | 1/60`.
  - Log line 2818: `Training Progress: 3%| | 2/60`.
  - Metrics JSONL contains `step=1`, `training/global_step=1`; `step=2`, `training/global_step=2`; `step=3`, `training/global_step=3`.
- Fatal-pattern scan returned no exact fatal matches for OOM, terminal EADDRINUSE, training failure, SIGKILL, NCCL error, or abort.
- Traceback scan after training start line 2808 returned no rows.
- Nonterminal startup observation: log lines 816-920 contain vLLM HTTP server bind retry tracebacks and `address already in use`, followed by line 930 `_validate() entered`, line 933 validation meta info, line 2762 initial validation metrics, and line 2808+ training progress. These startup tracebacks were not terminal for the current run.

### Protected Assets

```bash
python3 docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
git diff --cached --name-status
git status --short docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh /data-1/dataset/code/verl_rl
```

Evidence:
- Protected compare output: `{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}`.
- `git diff --cached --name-status` returned no rows.
- Protected-ish status command returned no rows for the checked baseline, manifest, profile, and dataset paths.

## Blocking Findings

None.

## Deferred Suggestions

None for the scoped findings.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

This is a live point-in-time review. It verifies that the current run passed the specific historical launch incidents and has reached training step 3, but it does not certify future completion of the 60-step control run or later Stage2/Stage3 phases.
