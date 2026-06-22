# Code Task Training and Eval Scripts Implementation Plan

- Status: ACTIVE IMPLEMENTATION PLAN, NOT LAUNCHED
- Created: 2026-06-04
- Branch: `feature/on-policy-wdl-sft`
- Target runtime: current `verl-harness` image
- Parent research plan: `code_task_extension_on_policy_wdl_sft.md`
- Intended script family: `recipe/on_policy_wdl_sft/code_task/`

## Objective

Implement the scripts needed to train and evaluate code-task Stage1 -> Stage2
On-Policy WDL-SFT in the current project and current `verl-harness` image.

The implementation must produce:

1. code data preparation scripts;
2. reproducible code-eval dependency installation inside `verl-harness`;
3. code reward and verifier smoke scripts;
4. Stage1 and Stage2 training wrappers;
5. a sequential queue script;
6. a thin monitor script;
7. offline code eval scripts;
8. Meituan/AFO launch entry points matching the math-task layered style;
9. documentation and script-index entries.

No real code training should launch until the verifier, data conversion, and
evaluation smoke gates pass.

## Execution Gates

Use these gates to avoid mixing implementation readiness with training evidence:

| Gate | What may run | Requires explicit user approval? | Unlocks |
| --- | --- | --- | --- |
| `G0-plan` | Document edits and read-only inspection. | No | implementation work. |
| `G1-build-smoke` | dependency install/verify, data conversion verify, reward verifier, shell syntax checks, wrapper dry-runs, Meituan dispatch dry-runs, tiny offline eval subsets that do not train. | No | script implementation acceptance. |
| `G2-training-smoke` | 3-5 step Stage1/Stage2 training smoke in tmux. | Yes | pilot launch decision. |
| `G3-pilot` | 20-40 step Stage1 pilot and 10-20 step Stage2 pilot. | Yes | result interpretation and next matrix. |

This plan's implementation can be marked ready at `G1-build-smoke`. Any command
that updates model weights through `verl.trainer.main_ppo` is `G2` or higher and
must not run without explicit user approval.

## Design Decisions

### Code tasks do not use boxed-answer prompts

Math tasks require `\boxed{}` because the reward function extracts a final math
answer. Code tasks should not use boxed-answer prompts.

The code prompt contract should be:

````text
<think>...</think>
<answer>
```python
...
```
</answer>
````

The implementation should require executable Python code in `<answer>`, but it
must not ask the model to put the final answer in `\boxed{}`.

### Keep code-specific scripts separate from math `staged_v1`

Put new code-task scripts under:

```text
recipe/on_policy_wdl_sft/code_task/
```

Reuse shared math infrastructure where appropriate:

- `recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh`;
- `recipe/on_policy_wdl_sft/staged_v1/_resolve_stage1_model2.sh`;
- `recipe/on_policy_wdl_sft/staged_v1/merge_stage1_model2_fixed.sh`;
- `verl.model_merger merge`;
- existing joint-model preparation.

Do not put code-specific prompt, data, eval, or queue scripts into
`staged_v1/` unless they are genuinely shared with math.

### Stage1 and Stage2 are both required

Code-task training must implement both stages:

- Stage1: single-model on-policy SFT with code execution reward.
- Stage2: fixed Model2 handoff, Model2-only rollout, joint fused-loss training.

The exact Stage2 intervention step is not fixed in this implementation plan.
The scripts must support a configurable fixed handoff step, and the concrete
step should be chosen later from code Stage1 smoke/pilot curves and reward
latency evidence.

Stage2 wrappers are not optional. The first accepted implementation must include
working Stage2 smoke and pilot launch paths, even if the first real training
run starts with Stage1 smoke.

Stage2 source provenance must be explicit. All Stage2 wrappers must require:

| Variable | Meaning | Required behavior |
| --- | --- | --- |
| `STAGE1_SOURCE_CKPT` or `STAGE1_CKPT_DIR` | Stage1 checkpoint root used as Model2 source. | Dry-run fails if neither is set or resolvable. |
| `STAGE2_HANDOFF_STEP` / `STAGE1_STEP` | Fixed Stage1 step used for handoff. | Configurable by env; never silently resolved from best checkpoint unless `ALLOW_BEST_HANDOFF=1`. |
| `MERGED_MODEL2_DIR` | HF Model2 output directory for Stage2. | Must contain or write provenance before Stage2 training. |
| `MERGED_MODEL2_PROVENANCE_FILE` | JSON provenance file. | Must include source checkpoint, handoff step, actor dir, target dir, prompt template version, and code data manifest path. |

For Stage2 dry-run before any new Stage1 smoke exists, the caller may use an
existing code checkpoint only by explicitly setting `MODEL2_PATH` and
`ALLOW_EXTERNAL_MODEL2_FOR_DRY_RUN=1`. That path is valid for launch plumbing
only and is not algorithmic Stage2 evidence.

### First runnable matrix is smoke-first, beta=0.0 only

The first implementation should prepare only:

| Run | Purpose | Default budget |
| --- | --- | --- |
| `CODE-S1-SMOKE-BETA0` | Verifies Stage1 code reward and logging in `verl-harness`. | 3-5 steps |
| `CODE-S1-PILOT-BETA0` | Short Stage1 pilot after smoke passes. | 20-40 steps |
| `CODE-S2-SMOKE-BETA0-BETA0` | Verifies fixed Model2 handoff and Stage2 code reward. | 3-5 steps |
| `CODE-S2-PILOT-BETA0-BETA0` | Short matched-beta Stage2 pilot. | 10-20 steps |

Do not implement or launch `beta=0.1` wrappers until `beta=0.0` reward and
offline eval are reviewed. This keeps reward correctness separate from reverse
term behavior.

## Target File Layout

### Environment and dependency scripts

| File | Purpose |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/requirements-code-eval.txt` | Pinned or bounded Python dependencies for code reward/eval inside `verl-harness`. |
| `recipe/on_policy_wdl_sft/code_task/install_code_eval_deps.sh` | Installs code-eval dependencies into the active `verl-harness` Python environment using the repo's package manager. |
| `recipe/on_policy_wdl_sft/code_task/verify_code_eval_deps.py` | Verifies importability, package versions, CLI availability, dataset cache visibility, and sandbox/test execution prerequisites. |
| `recipe/on_policy_wdl_sft/code_task/code_harness_registry.json` | Records which benchmark uses official harness, local runner, or diagnostic-only runner. |

Initial dependency targets:

| Benchmark / function | Preferred source | Required handling |
| --- | --- | --- |
| HumanEval | EvalPlus official package / API | Use official EvalPlus data and scorer for pass@1. |
| HumanEval+ | EvalPlus official package / API | Do not use local simplified `test_plus.jsonl` as official plus. |
| MBPP | EvalPlus official package / API where applicable | Use official EvalPlus-compatible path for comparable scoring. |
| MBPP+ | EvalPlus official package / API | Do not use local simplified `test_plus.jsonl` as official plus. |
| BigCodeBench | Official `bigcodebench` package or official GitHub install | Prefer official evaluator if installable in `verl-harness`; otherwise label local runner as `local-runner`. |
| LiveCodeBench | Official LiveCodeBench repository / package path | Prefer official code-generation evaluator if installable in `verl-harness`; otherwise label local public-test runner as `local-runner`. |
| Training reward execution | Sandbox Fusion first, `prime_code` local fallback for smoke only | Must run generated code with timeout, memory, and dependency accounting. |

Dependency installation acceptance:

- `verify_code_eval_deps.py` runs inside `verl-harness`;
- install uses a fixed command recorded in the report, preferably
  `python3 -m pip install -r requirements-code-eval.txt` or the repo-approved
  `uv pip` equivalent;
- package cache/wheelhouse location is explicit and overrideable, with a
  dolphinfs-compatible cache path for Meituan/AFO;
- imports succeed for `evalplus` and every installed official harness;
- BigCodeBench and LiveCodeBench official harness status is recorded as one of
  `official-installed`, `official-unavailable`, or `local-runner`;
- any `official-unavailable` status records attempted command, source URL or
  package name, version constraint, exit code, and stderr excerpt;
- EvalPlus data loading works without using the simplified local plus JSONLs;
- local dataset paths under `/data-1/dataset/EnsembleLLM-data-processed/` are
  visible inside the container;
- sandbox/test execution can run a trivial passing Python program, a failing
  assertion, an import failure, and a timeout case;
- output is written to a machine-readable environment report.

### Data and verifier scripts

| File | Purpose |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/prepare_code_rl_dataset.py` | Convert `/data-1/dataset/code/code-train.jsonl` into verl RL parquet plus manifests. |
| `recipe/on_policy_wdl_sft/code_task/create_code_stage2_nonoverlap_shard.py` | Create Stage2 train shard with zero overlap against Stage1 consumed rows. |
| `recipe/on_policy_wdl_sft/code_task/verify_code_dataset.py` | Verify parquet schema, prompt format, JSON test cases, source counts, and token-length summary. |
| `recipe/on_policy_wdl_sft/code_task/verify_code_reward_env.py` | Run reference/wrong-answer reward checks inside `verl-harness`; record failure taxonomy. |
| `recipe/on_policy_wdl_sft/code_task/code_extraction.py` | Shared extraction utility for `<answer>`, fenced Python, raw Python fallback, and malformed-output classification. |
| `recipe/on_policy_wdl_sft/custom_reward_function_code.py` | Custom reward wrapper loaded by trainer via `CUSTOM_REWARD_FN_PATH`. |

Required reward metadata schema:

| Key | Type | Meaning |
| --- | --- | --- |
| `score` | float | Final scalar reward. |
| `acc` | float | 1.0 if all selected tests pass, else 0.0. |
| `code_reward_status` | string | One of `pass`, `wrong_answer`, `extraction_fail`, `compile_error`, `runtime_error`, `timeout`, `dependency_error`, `sandbox_error`. |
| `code_reward_extraction_fail` | int | 1 iff no executable code is extracted. |
| `code_reward_compile_error` | int | 1 iff Python compile/import before execution fails. |
| `code_reward_runtime_error` | int | 1 iff execution raises non-timeout runtime error. |
| `code_reward_timeout` | int | 1 iff execution times out. |
| `code_reward_dependency_error` | int | 1 iff a missing package/import causes failure. |
| `code_reward_num_tests` | int | Number of tests attempted. |
| `code_reward_num_passed` | int | Number of tests passed. |
| `pred` | string | Extracted code or a short sentinel such as `[NO_CODE]`. |
| `verification_method` | string | `sandbox_fusion`, `prime_code`, `evalplus`, or `local_exec`. |

Training and validation logging must prove these fields survive the reward
manager path. The implementation must set or verify `reward_extra_keys` so the
keys appear in metrics JSONL and validation dumps, not only in the immediate
return value of `custom_reward_function_code.py`.

Expected generated data paths:

| Artifact | Default path |
| --- | --- |
| Stage1 train parquet | `/data-1/dataset/code/verl_rl/code_train_rl_format.parquet` |
| Stage1 manifest | `/data-1/dataset/code/verl_rl/code_train_rl_format.manifest.json` |
| Stage2 train parquet | `/data-1/dataset/code/verl_rl/code_stage2_after_s1_seed20260528.parquet` |
| Stage2 manifest | `/data-1/dataset/code/verl_rl/code_stage2_after_s1_seed20260528.manifest.json` |
| smoke val parquet | `/data-1/dataset/code/verl_rl/code_val_smoke.parquet` |
| HumanEval smoke parquet | `/data-1/dataset/code/verl_rl/humaneval_val_smoke.parquet` |

### Training wrappers

| File | Purpose |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/run_s1_code_base.sh` | Common Stage1 code launcher. Sets code data, code reward, code validation, conservative rollout settings, and sources the existing Stage1 launcher. |
| `recipe/on_policy_wdl_sft/code_task/run_s1_code_smoke_beta_0.sh` | Thin Stage1 smoke wrapper, `beta=0.0`, 3-5 steps. |
| `recipe/on_policy_wdl_sft/code_task/run_s1_code_pilot_beta_0.sh` | Thin Stage1 pilot wrapper, `beta=0.0`, 20-40 steps. |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh` | Common Stage2 code launcher. Reuses fixed Model2 handoff and sets code reward/data/eval knobs. |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_smoke_beta0_beta0.sh` | Thin Stage2 smoke wrapper from fixed Stage1 Model2, `beta=0.0`. |
| `recipe/on_policy_wdl_sft/code_task/run_s2_code_pilot_beta0_beta0.sh` | Thin Stage2 pilot wrapper from fixed Stage1 Model2, `beta=0.0`. |

Default training settings:

| Variable | Smoke default | Pilot default |
| --- | ---: | ---: |
| `WDL_SFT_BETA` | `0.0` | `0.0` |
| `LR` | `5e-7` | `5e-7` |
| `TRAIN_PROMPT_BSZ` | `4` or `8` | `16` first, `32` only after reward latency proof |
| `ROLLOUT_N` | `2` or `4` | `4` |
| `TRAIN_PROMPT_MINI_BSZ` | `TRAIN_PROMPT_BSZ * ROLLOUT_N` | same |
| `MAX_PROMPT_LENGTH` | `1024` initial | adjust from token-length report |
| `MAX_RESPONSE_LENGTH` | `4096` | `4096` initially |
| `TEST_FREQ` | `1` or `5` | sparse after smoke |
| `SAVE_FREQ` | `1` or `5` | aligned with handoff checkpoints |
| `VAL_N` | `1` for smoke | `1` online, offline eval handles pass@k |
| `BEST_CKPT_METRIC_KEY` | code val pass metric | code val pass metric |

Every wrapper must follow default-local, overridable-everything:

```bash
export CODE_TRAIN_FILE=${CODE_TRAIN_FILE:-"/data-1/dataset/code/verl_rl/code_train_rl_format.parquet"}
```

Do not unconditionally assign local paths.

Training wrappers are not accepted until they are reachable through both:

- direct local launch in `verl-harness`;
- Meituan/AFO `EXPERIMENT=...` dispatch through the code-task platform and
  recipe adapters.

### Queue and monitor

| File | Purpose |
| --- | --- |
| `scripts/training_queue_monitor.sh` | Generic sequential tmux queue monitor, imported or adapted from `verl-dual-rollout`; one shared implementation. |
| `recipe/on_policy_wdl_sft/code_task/run_code_task_smoke_queue.sh` | Host-side queue for Stage1 smoke -> fixed Model2 merge -> Stage2 smoke. |
| `recipe/on_policy_wdl_sft/code_task/run_code_task_pilot_queue.sh` | Host-side queue for Stage1 pilot -> fixed Model2 merge -> Stage2 pilot. |
| `recipe/on_policy_wdl_sft/code_task/monitor_code_task_queue_notify.sh` | Thin monitor: declares run prefixes/tmux names/code health checks and calls generic monitor logic. |
| `docs/joint_training/guides/code_task_monitor_agent_runbook.md` | Monitor Agent runbook for failure classification, WxPusher boundaries, disk/OOM/resume policy, and user-approval gates. |

Queue requirements:

- run only in tmux;
- support `START_INDEX` / `END_INDEX` for partial resume;
- check disk space for checkpoints, W&B, generated model2 dirs, and eval output;
- check GPU utilization before launch;
- refuse incomplete checkpoint collisions unless `ALLOW_RESUME=1`;
- refuse stale merged Model2 dirs unless `ALLOW_OVERWRITE_MERGED_MODEL2=1`;
- wait for final checkpoint and final online code metrics before marking a run
  complete;
- send WxPusher notifications for queue start, run start, run completion,
  failure, and queue completion when the notifier exists.
- keep automatic shell behavior limited to detection, skip, and notification;
  repair/relaunch decisions follow the Monitor Agent runbook.

Monitor requirements:

- do not copy the full queue/monitor driver logic into every experiment;
- define only code-task arrays and health gates, then delegate to
  `training_queue_monitor_main`;
- report code-specific health:
  - latest step;
  - reward `acc`;
  - extraction-fail rate;
  - compile-error rate;
  - runtime-error rate;
  - timeout rate;
  - dependency/import-error rate;
  - response length / overlong rate;
  - checkpoint and metrics paths.

Queue/monitor boundary:

- queue scripts may contain code-specific orchestration: Stage1 launch, fixed
  Model2 merge, provenance checks, Stage2 launch, and resource gates;
- monitor scripts must stay thin and delegate shared polling/logging to
  `training_queue_monitor_main`;
- completion checks must use code metrics, not math keys.

Initial online code metric keys:

| Metric key | Use |
| --- | --- |
| `val-core/code_val_smoke/acc/pass@1` or equivalent data-source-specific key | default `BEST_CKPT_METRIC_KEY` for smoke. |
| `val-aux/*/code_reward_extraction_fail/mean@1` | format/extraction health. |
| `val-aux/*/code_reward_compile_error/mean@1` | compile health. |
| `val-aux/*/code_reward_runtime_error/mean@1` | runtime health. |
| `val-aux/*/code_reward_timeout/mean@1` | timeout health. |
| `val-aux/*/code_reward_dependency_error/mean@1` | dependency health. |

If the exact metric prefix differs after implementation, the wrapper README and
monitor must record the actual observed keys from a `G1` tiny validation or
`G2` smoke.

### Offline eval scripts

| File | Purpose |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py` | Shared vLLM generation and code extraction for code benchmarks. |
| `recipe/on_policy_wdl_sft/code_task/eval_code_evalplus.py` | EvalPlus-backed HumanEval/MBPP pass@1 wrapper, ported from old `qwen` scripts where useful. |
| `recipe/on_policy_wdl_sft/code_task/eval_code_local_exec.py` | Local-runner BigCodeBench/LiveCodeBench diagnostic eval with explicit local-runner label. |
| `recipe/on_policy_wdl_sft/code_task/run_code_eval_model2_common.sh` | Container-side common runner: merge/extract Model2 if needed, verify tokenizer files, run selected evals. |
| `recipe/on_policy_wdl_sft/code_task/run_code_eval_queue.sh` | Host-side sequential eval queue for baseline, Stage1 source, Stage2 best, Stage2 final. |

Eval output requirements:

- write one machine-readable summary per model and benchmark;
- include generation parameters;
- include harness name and path;
- include environment info;
- include extraction-fail, compile-error, runtime-error, timeout, and
  dependency/import-error counts;
- label BigCodeBench/LiveCodeBench as `local-runner` unless official packages
  are installed and verified;
- record whether each benchmark used `official-installed`,
  `official-data-local-runner`, or `local-diagnostic`;
- verify extracted model dirs include `chat_template.jinja`,
  `tokenizer_config.json`, and safetensors weights before vLLM launch.

Primary eval mode:

```text
pass@1, temperature=0.0, top_p=1.0, n=1, thinking enabled
```

Secondary diagnostic mode:

```text
pass@k only, temperature=1.0, top_p=0.95, n=3
```

Do not report simplified `test_plus.jsonl` as official EvalPlus plus scores.

### Meituan compatibility

Meituan/AFO compatibility is part of implementation done, not a follow-up.
Code-task training wrappers must support the same layered launch style as math
tasks.

Required Meituan-ready files:

| File | Requirement |
| --- | --- |
| `recipe/on_policy_wdl_sft/code_task/meituan/env.sh` | Override all local paths: datasets, model paths, checkpoints, W&B, generated model2 dirs, eval outputs, Sandbox Fusion URL, and concurrency knobs. |
| `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh` | Resolve `EXPERIMENT` to `run_${EXPERIMENT//-/_}.sh`; fail fast when data/model/sandbox vars are missing. |
| `platform/hope_code_task/run.hope` | Template platform config with `EXPERIMENT=...` and documented smoke/pilot examples. |
| `platform/hope_code_task/jupyter.sh` | Thin platform shim that locates the repo and execs `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh`. |
| `platform/hope_code_task/README.md` | Meituan launch notes, supported `EXPERIMENT` values, and required path/sandbox variables. |

Initial supported `EXPERIMENT` values:

| `EXPERIMENT` | Target wrapper |
| --- | --- |
| `s1-code-smoke-beta-0` | `run_s1_code_smoke_beta_0.sh` |
| `s1-code-pilot-beta-0` | `run_s1_code_pilot_beta_0.sh` |
| `s2-code-smoke-beta0-beta0` | `run_s2_code_smoke_beta0_beta0.sh` |
| `s2-code-pilot-beta0-beta0` | `run_s2_code_pilot_beta0_beta0.sh` |

Meituan adapter requirements:

- export every path consumed by code wrappers and shared common launchers:
  `DATA_ROOT`, `TRAIN_FILE`, `TEST_FILES`, `CODE_TRAIN_FILE`,
  `CODE_VAL_FILES`, `BASE_MODEL_PATH`, `MODEL2_PATH`, `MODEL_PATH`,
  `BASE_CKPT_DIR`, `WANDB_DIR`, `HF_HOME`, `RAY_TMPDIR`, `TMPDIR`,
  `VERL_FILE_LOGGER_ROOT`, `VALIDATION_DATA_DIR`, `MERGED_MODEL2_DIR`,
  `STAGE1_MERGED_MODEL_ROOT`, eval output roots, and generated dataset roots;
- export reward execution knobs:
  `CUSTOM_REWARD_FN_PATH`, `CUSTOM_REWARD_FN_NAME`, `REWARD_MANAGER`,
  `SANDBOX_FUSION_URL`, `SANDBOX_FUSION_MAX_CONCURRENT`,
  `SANDBOX_FUSION_MEMORY_LIMIT_MB`, and `CODE_REWARD_TIMEOUT`;
- fail fast when required code train parquet, validation parquet, init model,
  reward function, or sandbox URL is missing for non-smoke runs;
- keep dolphinfs paths only in `meituan/env.sh`, never in per-experiment
  `run_*.sh` wrappers;
- support `SMOKE=1` to force tiny budgets and local-fallback verifier settings
  when appropriate.
- run an env-var audit against wrappers and shared launchers:
  `rg '\$\{[^}]+:-/data-1' recipe/on_policy_wdl_sft/code_task recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh`;
  every local default path must have a Meituan override in `meituan/env.sh` or a
  documented non-path reason;
- set `WANDB_MODE=offline` by default;
- pre-create persistent output directories;
- route high-churn temp dirs such as `TMPDIR` and `RAY_TMPDIR` to
  container-local storage when running on AFO, while checkpoints, W&B, model2
  dirs, eval outputs, and generated data go to dolphinfs;
- require flat model-weight directories for init/model2 paths, not unresolved
  HF symlink-cache entries.

## Implementation Order

### Step 1: Add reproducible dependency installation and dependency verifier

Files:

- `requirements-code-eval.txt`;
- `install_code_eval_deps.sh`;
- `verify_code_eval_deps.py`;
- `code_harness_registry.json`.

Acceptance:

- install script runs inside the current `verl-harness` image;
- EvalPlus is installed and importable;
- official BigCodeBench install is attempted or explicitly marked unavailable
  with error text;
- official LiveCodeBench install is attempted or explicitly marked unavailable
  with error text;
- Python execution smoke covers pass/fail/import-error/timeout;
- environment report records package versions, harness status, dataset paths,
  Python executable, and whether Sandbox Fusion URL is configured.

Reviewer checks:

- rerun `verify_code_eval_deps.py` inside `verl-harness`;
- confirm the report distinguishes official harness from local runner;
- confirm the simplified local plus JSONLs cannot satisfy official EvalPlus
  checks;
- confirm failed official installs are documented rather than silently falling
  back.

### Step 2: Add shared code extraction and reward env verifier

Files:

- `code_extraction.py`;
- `custom_reward_function_code.py`;
- `verify_code_reward_env.py`.

Acceptance:

- runs inside `verl-harness`;
- depends on the environment report from Step 1;
- reference answers from at least 20 sampled train rows are evaluated;
- known wrong answers fail;
- malformed outputs produce `extraction_fail`, not crashes;
- failure taxonomy is saved as JSON;
- local fallback is explicitly marked `smoke_only` unless Sandbox Fusion is
  configured.

Reviewer checks:

- run the verifier in `verl-harness`, not host Python;
- inspect at least 5 failure examples;
- confirm no broad false-negative pattern from missing imports or extractor
  mistakes;
- confirm the reward returns structured metadata keys required by this plan.

### Step 3: Add data conversion and verification

Files:

- `prepare_code_rl_dataset.py`;
- `create_code_stage2_nonoverlap_shard.py`;
- `verify_code_dataset.py`.

Acceptance:

- Stage1 parquet has 19,457 rows or a documented exclusion file;
- `test_case` JSON parse succeeds for all included rows;
- source counts match the raw JSONL;
- prompt has no boxed-answer instruction;
- prompt has exactly one system message and one user message;
- Stage2 shard has zero overlap with Stage1 consumed prompt ids;
- manifests include source path, output path, row count, SHA256, prompt
  template version, and test-case schema counts.

Reviewer checks:

- verify row counts from raw JSONL and parquet independently;
- run `verify_code_dataset.py --verify-only` inside `verl-harness`;
- inspect 10 prompt examples and confirm no `\boxed{}` instruction appears;
- inspect dict-style and list-style `test_case` examples.

### Step 4: Add Meituan adapter skeleton

Files:

- `recipe/on_policy_wdl_sft/code_task/meituan/env.sh`;
- `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh`;
- `platform/hope_code_task/run.hope`;
- `platform/hope_code_task/jupyter.sh`;
- `platform/hope_code_task/README.md`.

Acceptance:

- all initial `EXPERIMENT` values resolve to intended wrapper names, even before
  wrappers are fully implemented;
- dispatch dry-run can run without launching training;
- missing required data/model/sandbox variables fail fast with clear messages;
- env-var audit has no unexplained local `/data-1` path defaults.

Reviewer checks:

- run Meituan dispatch dry-runs for all initial `EXPERIMENT` values;
- inspect `env.sh` for full path overrides and AFO temp-dir policy;
- confirm platform shim is thin and contains no experiment-specific logic.

### Step 5: Add Stage1 and Stage2 wrappers

Files:

- `run_s1_code_base.sh`;
- `run_s1_code_smoke_beta_0.sh`;
- `run_s1_code_pilot_beta_0.sh`;
- `run_s2_code_model2_rollout_common.sh`;
- `run_s2_code_smoke_beta0_beta0.sh`;
- `run_s2_code_pilot_beta0_beta0.sh`.

Acceptance:

- each wrapper supports `DRY_RUN=1` or equivalent config-print mode;
- every path is overrideable by env;
- wrappers use code reward function and code train/val files by default;
- Stage1 uses single-model On-Policy SFT;
- Stage2 uses fixed Model2 handoff and `JOINT_TRAINING_ROLLOUT_SOURCE=model2`;
- Stage2 handoff step is configurable through env, not hard-coded to a
  research decision in the wrapper;
- Stage2 requires merged Model2 provenance;
- `BEST_CKPT_METRIC_KEY` points to a code validation metric, not MATH-500;
- smoke defaults are bounded: `TRAIN_PROMPT_BSZ<=8`, `ROLLOUT_N<=4`,
  `TOTAL_TRAINING_STEPS<=5`, and `VAL_N=1`.

Reviewer checks:

- run dry runs in `verl-harness`;
- confirm printed config contains code data paths and code reward path;
- confirm no math boxed dataset or math reward path remains in code wrappers;
- confirm Stage2 refuses missing/stale Model2 provenance;
- confirm each wrapper can be resolved through
  `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh` by setting the
  matching `EXPERIMENT` value.

### Step 6: Add generic queue monitor and code queue scripts

Files:

- `scripts/training_queue_monitor.sh`;
- `run_code_task_smoke_queue.sh`;
- `run_code_task_pilot_queue.sh`;
- `monitor_code_task_queue_notify.sh`.

Acceptance:

- queue can launch through `/data-1/verl07/run_train.sh` or direct
  `docker run --rm --gpus all ... verl-harness` fallback;
- queue refuses incomplete checkpoint collisions unless `ALLOW_RESUME=1`;
- queue refuses stale merged Model2 dirs unless
  `ALLOW_OVERWRITE_MERGED_MODEL2=1`;
- queue handles Stage1 -> fixed Model2 merge -> Stage2 orchestration itself;
- monitor is thin and calls `training_queue_monitor_main`;
- queue/monitor logs include exact run prefixes, script paths, tmux names,
  checkpoint dirs, and metric files;
- no real run is launched during implementation acceptance unless the user
  explicitly approves.

Reviewer checks:

- run shell syntax checks;
- run queue dry-run mode if implemented;
- inspect that the monitor did not fork a full copy of old P50/P60 monitor
  logic;
- confirm `START_INDEX` / `END_INDEX` partial-run support.

### Step 7: Add offline code eval scripts

Files:

- `eval_code_vllm.py`;
- `eval_code_evalplus.py`;
- `eval_code_local_exec.py`;
- `run_code_eval_model2_common.sh`;
- `run_code_eval_queue.sh`.

Acceptance:

- Code SFT checkpoint-38 can be evaluated in pass@1 mode;
- HumanEval/MBPP official EvalPlus path is separate from simplified local
  plus files;
- BigCodeBench/LiveCodeBench results use official harnesses when installed and
  verified, otherwise are labeled `local-runner`;
- eval summary includes harness, env, generation params, pass rate, extraction
  fail, compile/runtime/timeout/dependency errors;
- model merge/extract verifies `chat_template.jinja`;
- eval queue can evaluate baseline, Stage1 source, Stage2 best, and Stage2
  final.

Reviewer checks:

- run one tiny eval subset in `verl-harness`;
- inspect raw completions and extracted code for at least 5 examples;
- confirm plus-score reporting cannot accidentally use simplified JSONL as
  official plus;
- confirm summary JSON schema contains the required registry-import fields;
- confirm summary JSON includes required fields: `model_id`, `checkpoint_path`,
  `benchmark`, `harness_status`, `harness_path`, `generation_params`,
  `pass_at_1`, `num_tasks`, `extraction_fail_rate`, `compile_error_rate`,
  `runtime_error_rate`, `timeout_rate`, and `dependency_error_rate`.

### Step 8: Documentation and script index

Files:

- `recipe/on_policy_wdl_sft/code_task/README.md`;
- `docs/joint_training/guides/training_script_index.md`;
- this plan;
- optionally `CLAUDE.md` / `AGENTS.md` if active focus changes.

Acceptance:

- every runnable script is listed in the training script index;
- Meituan recipe adapter and platform shim are listed or linked from the index
  as launch-support files;
- README includes launch commands, dry-run commands, verifier commands, queue
  commands, eval commands, and Meituan `EXPERIMENT=...` examples;
- README says code tasks do not use boxed-answer prompts;
- README says full training is blocked until reward/data/eval gates pass.

Training script index required entries:

- `run_s1_code_base.sh`;
- `run_s1_code_smoke_beta_0.sh`;
- `run_s1_code_pilot_beta_0.sh`;
- `run_s2_code_model2_rollout_common.sh`;
- `run_s2_code_smoke_beta0_beta0.sh`;
- `run_s2_code_pilot_beta0_beta0.sh`;
- `run_code_task_smoke_queue.sh`;
- `run_code_task_pilot_queue.sh`;
- `monitor_code_task_queue_notify.sh`;
- `scripts/training_queue_monitor.sh`;
- `run_code_eval_model2_common.sh`;
- `run_code_eval_queue.sh`;
- `recipe/on_policy_wdl_sft/code_task/meituan/env.sh`;
- `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh`;
- `platform/hope_code_task/run.hope`;
- `platform/hope_code_task/jupyter.sh`.

Data prep, dependency install, verifier, and pure Python eval helpers should be
documented in the code-task README. Add them to the training script index only
if they become launch-support gates used by a real run.

Reviewer checks:

- compare script index against `rg --files recipe/on_policy_wdl_sft/code_task`;
- confirm no runnable script is undocumented;
- run Meituan dispatch dry-runs for all initial `EXPERIMENT` values, without
  launching real training;
- confirm `platform/hope_code_task/jupyter.sh` is a thin shim and does not
  duplicate experiment-specific logic;
- confirm docs distinguish smoke, pilot, and real experiment launch.

## Shared Main-Agent and Reviewer Checklist

Use this same checklist for the implementing main agent and reviewer subagent.
The reviewer should not invent a separate rubric.

### A. Runtime boundary

- All data/reward/eval smoke commands run inside current `verl-harness`.
- Host `qwen` env is used only as reference material.
- Any host-only result is labeled host-only and cannot satisfy launch
  acceptance.
- Code-eval dependencies are installed or verified inside `verl-harness`, not
  only in the old `qwen` conda env.
- Official harness status is recorded for EvalPlus, BigCodeBench, and
  LiveCodeBench before training scripts are accepted.

### B. Prompt boundary

- Code prompts do not contain `\boxed{}` or math final-answer wording.
- Code prompts require executable Python inside `<answer>`.
- Train, validation, and offline eval use the same extraction contract.

### C. Reward correctness

- Reference code passes on sampled train rows.
- Known wrong code fails.
- Missing code block returns structured extraction failure.
- Compile/runtime/timeout/import errors are counted separately.
- One bad sample cannot hang the reward batch.
- Required `code_reward_*` metadata keys appear in reward output, metrics JSONL,
  and validation dumps.

### D. Data correctness

- Raw and parquet row counts match or exclusions are documented.
- `test_case` schema variants are counted and handled.
- Stage2 shard overlap with Stage1 consumed rows is zero.
- Manifest files are written and checked.

### E. Training wrappers

- No math train files, math validation files, or latex reward path remains in
  code wrappers.
- All paths are default-local and overridable by env.
- Every runnable training wrapper is reachable through the Meituan/AFO
  `EXPERIMENT` dispatch path.
- Both Stage1 and Stage2 wrappers exist and pass dry-run checks.
- Stage2 handoff step is configurable and provenance-checked.
- Stage2 has fixed Model2 provenance and refuses stale merges.
- Smoke settings are small: `TRAIN_PROMPT_BSZ<=8`, `ROLLOUT_N<=4`.
- Pilot settings remain bounded until reward latency is measured:
  `TRAIN_PROMPT_BSZ<=32`, `ROLLOUT_N<=4`, unless the verifier report justifies
  larger values.

### F. Queue and monitor

- Long-running work is tmux-owned.
- Queue supports direct docker fallback when `/data-1/verl07/run_train.sh` is
  missing.
- Monitor delegates shared logic to `scripts/training_queue_monitor.sh`.
- Notifications are sent by queue/monitor scripts, not by a live Codex session.

### F2. Meituan launch path

- `platform/hope_code_task/jupyter.sh` exists and is a thin platform shim.
- `recipe/on_policy_wdl_sft/code_task/meituan/env.sh` overrides all local paths
  used by wrappers and shared common launchers.
- `recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh` maps every supported
  `EXPERIMENT` to an existing wrapper.
- Meituan dry-run fails fast with clear messages if required data/model/sandbox
  variables are missing.
- No per-experiment wrapper contains dolphinfs paths.
- Env-var audit covers wrappers and shared launchers, and every local default
  path is overridden or justified.
- AFO runtime uses offline W&B, persistent output dirs, container-local temp
  dirs, and flat model-weight paths.

### G. Offline eval

- pass@1 greedy eval is primary.
- pass@3 sampling is secondary; code-task reporting does not use mean@3.
- EvalPlus official scores do not use simplified local plus files.
- BigCodeBench/LiveCodeBench official harnesses are preferred when installed;
  local-runner scores are labeled as local-runner.
- Raw outputs and extracted code are saved for inspection.
- Summary JSON follows the documented schema and records harness status.

### H. Documentation

- Runnable scripts are indexed.
- README includes exact dry-run and smoke commands.
- README separates `G1-build-smoke` commands from user-approved
  `G2-training-smoke` commands.
- The plan status remains "not launched" until the user approves a run.

## Stop Conditions

Stop and report instead of continuing if any of these happens:

- reference-answer pass rate on sampled verifier rows is below 90%, unless
  every failure is manually classified as invalid source data;
- missing dependencies dominate failures;
- code extraction fails on any required case: fenced Python inside `<answer>`,
  fenced Python without `<answer>`, raw Python fallback, missing-code sentinel,
  and repeated malformed output;
- Sandbox Fusion is required but no usable URL/environment exists;
- code-eval dependencies cannot be installed or verified inside `verl-harness`;
- official harness fallback status is not recorded for EvalPlus, BigCodeBench,
  and LiveCodeBench;
- Stage1 or Stage2 dry run still points to math reward/data;
- any runnable training wrapper cannot be reached through the Meituan
  `EXPERIMENT` dispatch path;
- Meituan env leaves a local `/data-1` path active for required train/eval
  data, model, checkpoint, W&B, or sandbox variables;
- Stage2 can reuse stale Model2 weights without provenance failure;
- any `G2` or `G3` command would launch a training run without explicit user
  approval;
- offline eval cannot distinguish official EvalPlus from local simplified
  plus files.

## Definition of Done

This implementation plan is done only when:

1. all target scripts exist;
2. code-eval dependency installation and verifier pass in `verl-harness`;
3. Stage1 and Stage2 dry-runs both pass;
4. `G1-build-smoke` commands pass in `verl-harness`;
5. Meituan/AFO adapters and platform entry points exist and pass dispatch
   dry-runs for all initial training wrappers;
6. reviewer subagent applies the shared checklist and reports no blocking
   issues;
7. training script index is updated for every runnable script and launch-support
   entry point;
8. no real full training run has launched without explicit approval;
9. the next action is a user-approved `G2-training-smoke` queue, not an
   ambiguous "ready".
