# WDL Group-Advantage IS Status

- Goal file: `docs/joint_training/plans/active/wdl_group_advantage_is_goal.md`
- Current branch: `feature/on-policy-wdl-sft`
- Status: IN PROGRESS - implementation, CPU/static tests, and GPU smoke validated; parent commit and real training handoff pending
- Last updated: 2026-05-20

## Current Task

Implement `wdl_group_adv_is` according to the goal contract.

Current implementation choice: trainer stores the effective coefficient
`G_i = A_i + F_i` in `advantages` before actor update. The loss receives
already-augmented advantages and does not infer all-correct fallback from
zero-valued advantages.

User decisions now incorporated:

- Treat the goal as the implementation contract for the next session.
- Complete all Meituan four-layer launch files in the implementation pass.
- Keep `algorithm.norm_adv_by_std_in_grpo=false`.
- Preserve positive SFT signal for all-correct groups through an explicit
  fallback term, so the method is not pure group advantage.

## Completed Milestones

- Downloaded and indexed the GFT arXiv reference under
  `docs/joint_training/references/external/`.
- Drafted the active goal file.
- Ran independent reviewer checks for goal strictness and method/code risk.
- Incorporated reviewer feedback into the goal:
  - added hard guards for `rollout_is_weights` and non-`seq-mean-token-sum`;
  - added exact-value and gradient-level IS tests;
  - added Meituan-compatible script family requirements;
  - added reviewer input protocol and WARN/FAIL handling;
  - added status/commit discipline and done definition.
- Incorporated final user decisions:
  - changed the goal status to implementation contract;
  - added all-correct positive-SFT fallback `G_i = A_i + F_i`;
  - made `norm_adv_by_std_in_grpo=false` a hard first-run default;
  - made complete Meituan four-layer launch files a blocking done item;
  - added fallback metrics, tests, and reviewer gates.
- Added post-completion training handoff requirement:
  - after the implementation contract fully passes, start one real training run
    in tmux;
  - supervise 30 completed training steps;
  - if the 30-step window is clean, leave training running and end the session;
  - if bug/OOM/hang/non-finite metrics/method-contract violation appears,
    debug, fix, rerun needed validation, relaunch, and keep supervising until
    30 clean steps complete.
- Added intermediate GPU probe requirement:
  - during implementation, runnable milestones may use 1-3 step real GPU
    probes before final acceptance;
  - probe evidence must be recorded in this status file;
  - any probe failure is treated as an implementation failure and must be
    debugged, fixed, and rerun before dependent milestones continue.
- Strengthened timely-commit requirement:
  - after a coherent milestone passes its required tests/reviewer gate, commit
    before moving to the next dependent milestone;
  - do not accumulate more than one coherent milestone of uncommitted
    intentional changes;
  - if a commit is blocked by failing tests or user-owned dirty files, record
    the exact blocker and working-tree state here.
- Implemented core `wdl_group_adv_is` policy loss in
  `verl/trainer/ppo/core_algos.py`:
  - registered loss mode `wdl_group_adv_is`;
  - multiplies detached `rho = exp(log_prob - old_log_prob)` into token loss;
  - uses detached binary trust-region mask by effective coefficient sign;
  - requires `seq-mean-token-sum`;
  - fails fast on non-`None` `rollout_is_weights`;
  - emits sign-split ratio and clip metrics without beta metrics.
- Implemented trainer-side fallback routing in
  `verl/trainer/ppo/ray_trainer.py`:
  - `wdl_group_adv_is` is excluded from WDL raw reward-label override;
  - all-correct positive-SFT fallback is computed from true prompt `uid`
    groups and raw rewards;
  - group-level zero/mixed/all-correct fallback metrics are logged.
- Added complete launch family:
  - `recipe/on_policy_wdl_sft/group_advantage_is/README.md`
  - `recipe/on_policy_wdl_sft/group_advantage_is/_common_group_adv_is.sh`
  - `recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh`
  - `recipe/on_policy_wdl_sft/group_advantage_is/meituan/env.sh`
  - `recipe/on_policy_wdl_sft/group_advantage_is/meituan/jupyter.sh`
  - `platform/hope_group_advantage_is/README.md`
  - `platform/hope_group_advantage_is/jupyter.sh`
  - `platform/hope_group_advantage_is/run.hope`
- Added tests:
  - `tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py`
  - `tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py`
  - `tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py`
- Fixed GPU-smoke runtime issues found during validation:
  - added `all_correct_sft_fallback` and `pos_sft_fallback_coef` to
    `PolicyLossConfig` in `verl/workers/config/actor.py`;
  - made `_common_group_adv_is.sh` clamp the default overlong buffer length to
    `MAX_RESPONSE_LENGTH` for short smoke runs, while failing fast for explicit
    invalid `OVERLONG_BUFFER_LEN`;
  - exposed `LR_WARMUP_STEPS` with default `5` so same-batch ratio-path smoke
    can set `LR_WARMUP_STEPS=0` without changing the real experiment default.
- Updated method note with the implemented group-advantage IS formula and
  the difference from standard GRPO.

## Intended Files Changed So Far

- `docs/joint_training/plans/active/wdl_group_advantage_is_goal.md`
- `docs/joint_training/plans/active/wdl_group_advantage_is_status.md`
- `docs/joint_training/plans/active/README.md`
- `CLAUDE.md`
- `AGENTS.md`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `recipe/on_policy_wdl_sft/group_advantage_is/`
- `platform/hope_group_advantage_is/`
- `tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py`
- `tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py`
- `tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py`
- `docs/joint_training/courses/method_on_policy_wdl_sft.tex`

Related reference files from the prior paper-download step:

- `docs/joint_training/references/external/gft_arxiv_2604_14258.md`
- `docs/joint_training/references/external/gft_arxiv_2604_14258.pdf`
- `docs/joint_training/references/external/gft_arxiv_2604_14258_source/`

## Tests And Validation

Run on 2026-05-20:

```text
env PYTHONPATH=/root/buaa/pip_temp/pytest-target:/root/buaa/local_data1/verl07/verl \
  python3 -m pytest -q \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py
```

Result before GPU smoke fixes:

```text
21 passed, 4 warnings in 5.32s
```

Notes:

- The base host lacked `pytest`; installed `pytest`, `pytest-asyncio`,
  `pytest-rerunfailures`, `omegaconf`, `hydra-core`, `codetiming`, and
  `peft --no-deps` into `/root/buaa/pip_temp/pytest-target` only.
- An initial `uv run pytest ...` failed because the sandbox could not write to
  `/root/.cache/uv`; rerunning with `UV_CACHE_DIR=/root/buaa/pip_temp/uv-cache`
  reached dependency resolution but failed on optional `pyext`.
- A plain `uv run --no-project --with pytest ...` did not see the system
  torch/numpy environment, so the final validation used current `python3` with
  the temporary pytest target on `PYTHONPATH`.
- Static shell syntax checks are also covered by
  `test_wdl_group_advantage_is_scripts.py` via `bash -n`.

Additional validation:

After adding `PolicyLossConfig` fields and smoke-script fixes:

```text
env PYTHONPATH=/root/buaa/pip_temp/pytest-target:/root/buaa/local_data1/verl07/verl \
  python3 -m pytest -q \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py \
  tests/trainer/config/test_algo_config_on_cpu.py
```

Result:

```text
31 passed, 5 warnings in 6.00s
```

After Test reviewer WARN, added direct exact-surrogate and all-same policy-loss
tests, then reran:

```text
env PYTHONPATH=/root/buaa/pip_temp/pytest-target:/root/buaa/local_data1/verl07/verl \
  python3 -m pytest -q \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py \
  tests/trainer/config/test_algo_config_on_cpu.py
```

Result:

```text
33 passed, 5 warnings in 5.86s
```

Script-only rerun after exposing `LR_WARMUP_STEPS`:

```text
env PYTHONPATH=/root/buaa/pip_temp/pytest-target:/root/buaa/local_data1/verl07/verl \
  python3 -m pytest -q tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py
```

Result:

```text
8 passed in 0.07s
```

```text
python3 -m py_compile \
  verl/trainer/ppo/core_algos.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/workers/config/actor.py
```

Result: passed.

```text
git diff --check -- verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py \
  docs/joint_training/courses/method_on_policy_wdl_sft.tex \
  docs/joint_training/plans/active/wdl_group_advantage_is_status.md \
  platform/hope_group_advantage_is \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py
```

Result: passed.

```text
git -C recipe diff --check -- on_policy_wdl_sft/group_advantage_is
```

Result: passed.

Rerun after handoff on 2026-05-20:

```text
env PYTHONPATH=/root/buaa/pip_temp/pytest-target:/root/buaa/local_data1/verl07/verl \
  python3 -m pytest -q \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py \
  tests/trainer/config/test_algo_config_on_cpu.py
```

Result:

```text
33 passed, 5 warnings in 5.82s
```

Additional reruns:

```text
python3 -m py_compile \
  verl/trainer/ppo/core_algos.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/workers/config/actor.py
git diff --check -- verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py \
  docs/joint_training/courses/method_on_policy_wdl_sft.tex \
  docs/joint_training/plans/active/wdl_group_advantage_is_status.md \
  platform/hope_group_advantage_is \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_loss.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_trainer.py \
  tests/on_policy_wdl_sft/test_wdl_group_advantage_is_scripts.py
git -C recipe diff --check -- on_policy_wdl_sft/group_advantage_is
```

Result: all passed.

Codex pre-commit hook rerun after handoff on 2026-05-20:

```text
docker run --rm --gpus all --ipc=host \
  -v /data-1/verl07/verl:/workspace/verl \
  -v /data-1:/data-1 \
  verl-harness \
  bash -lc 'cd /workspace/verl && pytest tests/joint_training/ -q --tb=short'
```

Result:

```text
12 failed, 193 passed, 3 skipped, 7 warnings in 51.94s
```

Observed failures are in the existing `tests/joint_training/` suite, not in
the new `tests/on_policy_wdl_sft/` target:

- missing local tokenizer path `/data-1/.cache/huggingface/QwenJoint-1.7B`;
- stale joint-training recipe script expectation for
  `ROLLOUT_GPU_MEMORY_UTILIZATION`;
- cosine-similarity precision assertion returning `1.0000001192092896`;
- CPU tests entering FlashAttn/Triton `logprobs_from_logits` path;
- vLLM API compatibility drift around `extract_layer_index`,
  legacy loader modules, and `compute_logits` signature.

Because `.codex/config.toml` installs `.codex/hooks/pre-commit-tests.sh` as a
Codex `PreToolUse` hook for `git commit`, ordinary parent-repo commits are
blocked until this historical suite is repaired or the hook policy is changed.

Environment probe before GPU smoke:

```text
nvidia-smi
```

Result: all 8 A800 80GB GPUs idle, no running GPU processes.

```text
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
```

Result: no running Docker containers.

```text
tmux ls
```

Result: existing sessions `13`, `cx`, `cx-remote`, `eval_dispatch`, `med_main`;
none identified as an active verl training run.

Attempted direct Python dependency import probe for `flashinfer`, `vllm`,
`flash_attn`, `ray`, etc. The probe hung before printing module status and was
terminated with:

```text
kill -TERM 1944537 1944546 1944547
```

No GPU smoke launched yet.

GPU smoke attempt 1:

- tmux session: `wdl_group_adv_is_smoke`
- result: failed before actor update.
- blocker:

```text
PolicyLossConfig.__init__() got an unexpected keyword argument 'all_correct_sft_fallback'
```

Response: added the two policy-loss config fields to
`verl/workers/config/actor.py`, then reran CPU/static tests.

GPU smoke attempt 2:

- tmux session: `wdl_group_adv_is_smoke2`
- log:
  `/data-1/tmp/wdl_group_adv_is_smoke2/logs/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE2_1779293544.log`
- result: failed during reward-loop initialization.
- blocker:

```text
AssertionError: max_resp_len must be larger than overlong_buffer.len
```

Cause: smoke used `MAX_RESPONSE_LENGTH=512` while the script defaulted
`OVERLONG_BUFFER_LEN=1024`; DAPO validates the length even when the overlong
buffer is disabled.

Response: `_common_group_adv_is.sh` now clamps the default overlong buffer
length to `MAX_RESPONSE_LENGTH` for short runs and raises on explicit invalid
override. Static script tests were updated and rerun.

GPU smoke attempt 3, minimal infrastructure smoke:

- tmux session: `wdl_group_adv_is_smoke3`
- container: `hopeful_turing`
- run id:
  `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE3_1779293908`
- log:
  `/data-1/tmp/wdl_group_adv_is_smoke3/logs/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE3_1779293908.log`
- metrics:
  `/data-1/tmp/wdl_group_adv_is_smoke3/logs/metrics/OnPolicyWDLSFT/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE3_1779293908.jsonl`
- key launch overrides:
  `TOTAL_TRAINING_STEPS=1`, `TRAIN_PROMPT_BSZ=2`,
  `TRAIN_PROMPT_MINI_BSZ=1`, `PPO_EPOCHS=1`, `MAX_RESPONSE_LENGTH=512`,
  `ROLLOUT_IS=null`, `LOSS_MODE=wdl_group_adv_is`,
  `LOSS_AGG_MODE=seq-mean-token-sum`, `VAL_BEFORE_TRAIN=False`,
  `TEST_FREQ=-1`, `SAVE_FREQ=1`, `ROLLOUT_AGENT_NUM_WORKERS=1`.
- launch command:

```text
tmux new-session -d -s wdl_group_adv_is_smoke3 \
  "docker run --gpus all --rm --shm-size=64g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data-1:/data-1 -w /data-1/verl07/verl verl-harness:latest bash -lc \
  'export RUN_PREFIX=WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE3; \
   export LR=5e-7; export LOSS_MODE=wdl_group_adv_is; \
   export MIN_FREE_GB_FOR_CKPT=1; export MAX_RESPONSE_LENGTH=512; \
   export TRAIN_PROMPT_BSZ=2; export TRAIN_PROMPT_MINI_BSZ=1; \
   export TOTAL_TRAINING_STEPS=1; export VAL_BEFORE_TRAIN=False; \
   export TEST_FREQ=-1; export SAVE_FREQ=1; \
   export RAY_TMPDIR=/data-1/tmp/wdl_group_adv_is_smoke3/ray; \
   export TMPDIR=/data-1/tmp/wdl_group_adv_is_smoke3/tmp; \
   export LOG_DIR=/data-1/tmp/wdl_group_adv_is_smoke3/logs; \
   export WANDB_DIR=/data-1/tmp/wdl_group_adv_is_smoke3/wandb; \
   export BASE_CKPT_DIR=/data-1/tmp/wdl_group_adv_is_smoke3/checkpoints; \
   export VALIDATION_OUTPUT_DIR=/data-1/tmp/wdl_group_adv_is_smoke3/validation; \
   export ROLLOUT_MAX_MODEL_LEN=1012; \
   export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=1012; \
   export ROLLOUT_MAX_NUM_BATCHED_TOKENS=1012; \
   export ROLLOUT_GPU_MEMORY_UTILIZATION=0.3; \
   export ROLLOUT_AGENT_NUM_WORKERS=1; \
   export ROLLOUT_ENABLE_SLEEP_MODE=False; \
   export ROLLOUT_FREE_CACHE_ENGINE=False; \
   bash recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh'"
```

- result: PASS, one actor update completed.
- evidence:
  - final Hydra config had `actor.policy_loss.loss_mode=wdl_group_adv_is`,
    `algorithm.rollout_correction.rollout_is=null`,
    `algorithm.norm_adv_by_std_in_grpo=false`,
    `actor.use_kl_loss=False`, `algorithm.use_kl_in_reward=False`,
    `actor.loss_agg_mode=seq-mean-token-sum`,
    `reward_kwargs.overlong_buffer_cfg.len=512`;
  - vLLM launched and loaded joint fused weights with
    `_use_model2_only=False`;
  - actor training used `attn_implementation=flash_attention_2`;
  - GPUs reached about 28.8 GiB used with active utilization during the run;
  - metrics included `training/global_step=1`,
    `timing_s/update_actor=3.0634`, `actor/grad_norm=386.3623`,
    `actor/pg_loss=-131.7472`, `actor/kl_loss=0.0`,
    `wdl_group_adv_is/zero_adv_group_fraction=0.5`,
    `wdl_group_adv_is/mixed_group_fraction=0.5`,
    `wdl_group_adv_is/all_correct_fallback_group_fraction=0.0`,
    `wdl_group_adv_is/all_incorrect_group_fraction=0.5`,
    `wdl_group_adv_is/ratio_mean=1.0`,
    `wdl_group_adv_is/ratio_max=1.0`,
    `wdl_group_adv_is/clipfrac_positive=0.0`,
    `wdl_group_adv_is/clipfrac_negative=0.0`;
  - no `actor/wdl_sft_beta` metric was emitted.

GPU smoke attempt 4, required same-batch ratio-path smoke:

- tmux session: `wdl_group_adv_is_smoke4`
- container: `strange_beaver`
- run id:
  `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE4-RATIO_1779294264`
- log:
  `/data-1/tmp/wdl_group_adv_is_smoke4/logs/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE4-RATIO_1779294264.log`
- metrics:
  `/data-1/tmp/wdl_group_adv_is_smoke4/logs/metrics/OnPolicyWDLSFT/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE4-RATIO_1779294264.jsonl`
- key added overrides relative to smoke3:
  `PPO_EPOCHS=2`, `LR_WARMUP_STEPS=0`.
- launch command:

```text
tmux new-session -d -s wdl_group_adv_is_smoke4 \
  "docker run --gpus all --rm --shm-size=64g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data-1:/data-1 -w /data-1/verl07/verl verl-harness:latest bash -lc \
  'export RUN_PREFIX=WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A-SMOKE4-RATIO; \
   export LR=5e-7; export LR_WARMUP_STEPS=0; \
   export LOSS_MODE=wdl_group_adv_is; \
   export MIN_FREE_GB_FOR_CKPT=1; export MAX_RESPONSE_LENGTH=512; \
   export TRAIN_PROMPT_BSZ=2; export TRAIN_PROMPT_MINI_BSZ=1; \
   export TOTAL_TRAINING_STEPS=1; export VAL_BEFORE_TRAIN=False; \
   export TEST_FREQ=-1; export SAVE_FREQ=1; export PPO_EPOCHS=2; \
   export RAY_TMPDIR=/data-1/tmp/wdl_group_adv_is_smoke4/ray; \
   export TMPDIR=/data-1/tmp/wdl_group_adv_is_smoke4/tmp; \
   export LOG_DIR=/data-1/tmp/wdl_group_adv_is_smoke4/logs; \
   export WANDB_DIR=/data-1/tmp/wdl_group_adv_is_smoke4/wandb; \
   export BASE_CKPT_DIR=/data-1/tmp/wdl_group_adv_is_smoke4/checkpoints; \
   export VALIDATION_OUTPUT_DIR=/data-1/tmp/wdl_group_adv_is_smoke4/validation; \
   export ROLLOUT_MAX_MODEL_LEN=1012; \
   export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=1012; \
   export ROLLOUT_MAX_NUM_BATCHED_TOKENS=1012; \
   export ROLLOUT_GPU_MEMORY_UTILIZATION=0.3; \
   export ROLLOUT_AGENT_NUM_WORKERS=1; \
   export ROLLOUT_ENABLE_SLEEP_MODE=False; \
   export ROLLOUT_FREE_CACHE_ENGINE=False; \
   bash recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh'"
```

- result: PASS, one actor update completed and same-batch ratio path became
  nontrivial.
- evidence:
  - log confirmed `Total steps: 1, num_warmup_steps: 0`;
  - metrics included `training/global_step=1`,
    `timing_s/update_actor=5.0594`, `actor/lr=5e-7`,
    `actor/grad_norm=335.6140`, `actor/pg_loss=-262.5516`,
    `actor/kl_loss=0.0`,
    `wdl_group_adv_is/ratio_mean=1.0010691992938519`,
    `wdl_group_adv_is/ratio_max=1.5991053581237793`,
    `wdl_group_adv_is/ratio_mean_pos_adv=0.06265177577733994`,
    `wdl_group_adv_is/ratio_max_pos_adv=1.22573983669281`,
    `wdl_group_adv_is/ratio_mean_neg_adv=0.4376425929367542`,
    `wdl_group_adv_is/ratio_max_neg_adv=1.5991053581237793`,
    `wdl_group_adv_is/clipfrac_positive=0.0`,
    `wdl_group_adv_is/clipfrac_negative=0.0006103515625`.

Runtime notes:

- Both successful smoke runs end with non-blocking `wandb`/DataLoader atexit
  warnings after metrics and checkpoint have already been written. Docker and
  tmux sessions exited, and `docker ps` returned no running containers.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` is forced by the common launcher and was
  in the Docker environment, but the vLLM runtime log did not explicitly echo
  the backend name. Runtime reviewer accepted this as WARN rather than FAIL.
- The short-smoke response clip ratio is high because
  `MAX_RESPONSE_LENGTH=512`; this is expected for runtime smoke and is not a
  method-quality claim for the real 4096-token experiment.

Attempted bridge sync previously:

```text
python3 /data-1/agent-tools/sync_agent_context.py sync . --direction bidirectional
```

Result: blocked by read-only filesystem error on `.codex/config.toml`.

Rerun on 2026-05-20 after implementation:

```text
python3 /data-1/agent-tools/sync_agent_context.py sync . --direction bidirectional
```

Result:

```text
error: [Errno 30] Read-only file system: '/root/buaa/local_data1/verl07/verl/.codex/config.toml'
```

## Reviewer Verdicts

### Goal Strictness Reviewer

Verdict: WARN.

Main findings:

- Add a final done definition.
- Create and maintain the goal-local status file.
- Add reviewer input protocol.
- Add WARN/FAIL handling.
- Strengthen commit/status discipline.
- Add doc-sync/bridge-sync requirement.
- Add tmux requirement for smoke/full training.

Response: addressed in the goal file and this status file.

### Method/Code Risk Reviewer

Verdict: WARN.

Main findings:

- IS must be multiplicative and detached, not mask-only.
- `rollout_is_weights` can be auto-passed by actor code; new loss must hard
  guard against it.
- `seq-mean-token-sum` cannot rely on function defaults; new loss must fail
  fast for other aggregation modes.
- KL and beta must be disabled in both config/script and loss metrics.
- Proximal old/current IS should not be described as full rollout-behavior IS.
- All-correct/all-incorrect zero-update behavior needs monitoring metrics.

Response: addressed in the goal file as blocking acceptance criteria and tests.
The later user decision changes the all-correct behavior: all-correct groups
now retain positive SFT through a fallback term; all-incorrect groups remain
zero-loss.

### Section 12 Implementation Reviewer Gates

- Method/Formulation Reviewer: PASS.
  - Verified mixed-policy old/current IS, no model2-vs-joint IS, no beta/KL,
    group advantages, all-correct fallback, all-incorrect zero loss,
    `norm_adv_by_std_in_grpo=false`, `seq-mean-token-sum`, and not standard
    GRPO.
- Core Loss Reviewer: PASS.
  - Verified registration, detached multiplicative ratio, detached binary mask,
    `agg_loss(..., **config.global_batch_info)`, fail-fast guards for
    `rollout_is_weights` and non-`seq-mean-token-sum`, and no beta metric.
- Trainer Pipeline Reviewer: PASS.
  - Verified `compute_advantage` precedes actor update, `wdl_group_adv_is` is
    excluded from WDL raw-label override, fallback uses true `uid` groups and
    raw rewards, old log-probs come from actor compute-log-prob, and metrics
    distinguish group vs response fractions.
- Config/Script/Meituan Reviewer: PASS.
  - Verified algorithm defaults, no `WDL_SFT_BETA`, all four Meituan layers,
    path override rules, `SMOKE=1` propagation, `PPO_EPOCHS` mapping, and
    shell syntax.
- Test Reviewer: PASS after follow-up fixes.
  - Initial WARN asked for direct non-neutral exact-surrogate and all-same
    policy-loss tests. Added both and reran the suite:
    `33 passed, 5 warnings in 5.85s` in the reviewer's rerun.
- Runtime Reviewer: WARN accepted.
  - Verified two successful GPU smokes, actor update, checkpoint, sign-split
    ratio diagnostics, `actor/grad_norm`, disabled rollout IS/KL/beta, and
    required `PPO_EPOCHS=2` nontrivial ratio smoke.
  - WARN reason: runtime log does not explicitly print FlashInfer backend even
    though the launcher forces `VLLM_ATTENTION_BACKEND=FLASHINFER`; status now
    records the exact smoke commands and this limitation.

## Open Blockers / Pending Work

- Bridge sync is blocked by `.codex/config.toml` read-only filesystem behavior.
- The `recipe` submodule milestone commit exists:
  `c3d84c2 Add WDL group advantage IS recipe`.
- Parent-repo milestone commit is blocked by the Codex pre-commit hook because
  the historical Docker suite `pytest tests/joint_training/ -q --tb=short`
  currently has 12 unrelated failures listed above.
- The current working tree also includes pre-existing unrelated unstaged files
  (`.codex/config.toml`, `AGENTS.md`, and `.claude/skills/experiment-registry`)
  that must not be staged blindly.
- Post-completion real training handoff has not started. It is only allowed
  after Section 14 is satisfied, then must be supervised to
  `training/global_step >= 30`.

There are no remaining method decisions needed for the next validation step.

## Next Concrete Action

Next concrete action:

1. Repair or explicitly change the historical pre-commit hook policy so the
   parent-repo milestone commit can be made without bypassing it.
2. Start the real `wdl_group_adv_is` training run in tmux.
3. Supervise until metrics prove `training/global_step >= 30`.
4. Update this status file with the real run id, command, log path,
   checkpoint path, and observed 30-step metrics.
