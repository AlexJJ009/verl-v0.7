# Code Task Extension for On-Policy WDL-SFT

- Status: ACTIVE RESEARCH PLAN, NOT IMPLEMENTATION READY
- Created: 2026-06-03
- Branch: `feature/on-policy-wdl-sft`
- Recipe family: `recipe/on_policy_wdl_sft/staged_v1/`
- Related math evidence: `plateau_handoff_stage1_stage2.md`
- Historical code workspace: `/root/buaa/czh_rl_icml_2026/EnsembleLLM`

## Objective

Extend the Stage1 -> Stage2 On-Policy WDL-SFT workflow from math tasks to code
generation tasks, using executable test-case reward instead of boxed-answer
verification.

The intended claim is:

```text
If code reward execution is reliable and prompt/eval formats are aligned, then
the Stage1 -> Stage2 intervention should be testable on code tasks with the
same scientific question as math: whether a short Stage2 improves over its
Stage1 source without introducing format or execution collapse.
```

This plan is intentionally gated. Code training should not start until the
execution reward path is validated, because missing dependencies, brittle code
extraction, or incomplete sandboxing can turn correct completions into false
negatives.

## Non-Goals

- Do not launch full code Stage1/Stage2 training from this plan alone.
- Do not report HumanEval+ / MBPP+ scores from the simplified local
  `test_plus.jsonl` files as official EvalPlus scores.
- Do not treat local BigCodeBench or LiveCodeBench subprocess results as
  official benchmark scores unless the official harness is installed and
  validated.
- Do not reuse historical buggy code-eval results without marking the bug
  boundary.
- Do not change the WDL-SFT loss before the reward and eval plumbing have a
  passing smoke.

## Research Snapshot

### Local `qwen` environment

The old code-task workflow is centered around the lowercase conda environment:

```text
/root/buaa/conda_envs/qwen
```

Observed key packages:

| Package | Version / status |
| --- | --- |
| Python | `3.11.14` |
| PyTorch | `2.8.0+cu128` |
| vLLM | `0.11.0` |
| Transformers | `4.57.1` |
| datasets | `3.5.0` |
| evalplus | `0.3.1` |
| evaluate | missing |
| bigcodebench Python package | missing |
| livecodebench Python package | missing |

Environment risks:

- `requests` emits dependency-version warnings.
- BigCodeBench and LiveCodeBench were historically evaluated through local
  subprocess runners, not official Python packages.
- Training in this repo should still run in `verl-harness`; the `qwen` env is
  a reference for dependencies and old scripts, not the target runtime.

### Historical code training

The old project used three main families:

| Family | Evidence path | Notes |
| --- | --- | --- |
| WDL three-stage code training | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_wdl_code_v3_qwen3_4b_lr1e5.sh` | Stage1 trains `m1`, Stage2 trains `m2`, Stage3 uses `llmboost_train.py` with logit fusion weights `0.3,0.7`. |
| Code baselines | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_code_baselines_v3_qwen3_8b.sh` | NEFTune / Gaussian-noise baseline scripts. Typical settings include `BATCH=1`, `GRAD_ACCUM=32`, `MAX_SEQ=4096`, lr around `1e-5` or `2e-7`. |
| SSB code | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/scripts/run_ssb_v3_qwen3_4b_code.sh` | Generates rollouts, runs tests to split correct/incorrect code, then trains sparse top-K KD and merges LoRA. |

Historical model paths to consider as baselines or initialization references:

| Model | Path |
| --- | --- |
| Code SFT | `/data-1/.cache/Qwen3-4B-Base-Code-SFT/checkpoint-38` |
| Qwen3-4B code WDL M1 | `/data-1/.cache/Qwen3-4B-Base-Code-WDL-M1/checkpoint-39` |
| Qwen3-8B code WDL M1 | `/data-1/.cache/Qwen3-8B-Base-Code-WDL-M1/checkpoint-39` |

Important historical pitfall:

- After extracting a submodel, `chat_template.jinja` must be copied into the
  extracted model directory. A previous code eval failed because vLLM could not
  find `tokenizer.chat_template`. Model merge/extract scripts for this plan
  must check this file explicitly.

### Local datasets

| Dataset | Path | Rows | Current role |
| --- | --- | ---: | --- |
| Code train | `/data-1/dataset/code/code-train.jsonl` | 19,457 | Primary training source after conversion to verl RL parquet. |
| HumanEval | `/data-1/dataset/EnsembleLLM-data-processed/HumanEval/test.jsonl` | 164 | Eval; can use local runner, but official scoring should use EvalPlus API/data. |
| HumanEval simplified plus | `/data-1/dataset/EnsembleLLM-data-processed/HumanEval/test_plus.jsonl` | 164 | Local diagnostic only; not official EvalPlus. |
| MBPP | `/data-1/dataset/EnsembleLLM-data-processed/MBPP/test.jsonl` | 500 | Eval; can use local runner, but official scoring should use EvalPlus API/data. |
| MBPP simplified plus | `/data-1/dataset/EnsembleLLM-data-processed/MBPP/test_plus.jsonl` | 378 | Local diagnostic only; not official EvalPlus. |
| BigCodeBench | `/data-1/dataset/EnsembleLLM-data-processed/BigCodeBench/test.jsonl` | 1,140 | Eval with local runner unless official harness is added. |
| LiveCodeBench | `/data-1/dataset/EnsembleLLM-data-processed/LiveCodeBench/test.jsonl` | 400 | Eval with local runner; public tests are easy, private tests need decode/validation. |

`code-train.jsonl` fields:

- `prompt`: chat messages, currently user-only;
- `reference_answer`: reference Python code;
- `chosen`: assistant response with reasoning/code;
- `test_case`: JSON string, either dict-like `inputs` / `outputs` or list-like
  cases depending on source;
- `source`: mostly `codeio`, plus `OpenCoder`, `OpenCoderStage2`, and `prime`.

No standard APPS or CodeContests local dataset was found. The current train set
should therefore be treated as the primary code training source unless a new
dataset import is explicitly planned.

### Historical eval harness

Known old eval entry points:

| Harness | Path | Scope |
| --- | --- | --- |
| EvalPlus wrapper | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/eval_code_evalplus.py` | HumanEval / MBPP, greedy generation, `enable_thinking=True`, EvalPlus scoring. |
| Local code runner | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/eval_vllm_code.py` | BigCodeBench-style JSONL with generated code plus `test_code` in subprocess. |
| Local LiveCodeBench runner | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/eval_vllm_livecodebench.py` | LCB public tests, functional/stdin handling. |
| Shared prompts | `/root/buaa/czh_rl_icml_2026/EnsembleLLM/utils/prompts.py` | Same `<think>...</think><answer>...</answer>` family as current work. |

Known reliability boundaries:

- Old HumanEval results before the `check()` fix are buggy.
- Old BigCodeBench results before the unittest stderr fix are buggy.
- BigCodeBench local runner results are comparable to the same local runner,
  but not necessarily to official BigCodeBench reporting.
- LiveCodeBench local runner currently uses public tests by default, so it is
  useful for diagnostic pass/fail direction but not a complete private-test
  score.

## Current `verl` Integration Points

The current repo already has most of the reward wiring needed for code tasks:

- `recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh` parameterizes
  `TRAIN_FILE`, `TEST_FILES`, `CUSTOM_REWARD_FN_PATH`,
  `CUSTOM_REWARD_FN_NAME`, `REWARD_MANAGER`, and `BEST_CKPT_METRIC_KEY`.
- `verl/trainer/ppo/reward.py` supports dynamically loaded custom reward
  functions.
- `verl/utils/reward_score/__init__.py` routes code-like data sources
  `codecontests`, `apps`, `codeforces`, and `taco` to Sandbox Fusion if a URL
  is configured, otherwise to `prime_code`.
- `verl/utils/reward_score/prime_code/__init__.py` can run `inputs` / `outputs`
  test cases locally.
- `verl/utils/reward_score/sandbox_fusion/__init__.py` supports Sandbox Fusion
  execution and continuous mode.
- `docs/examples/sandbox_fusion_example.rst` documents the expected config:
  `reward_model.reward_manager=prime`,
  `reward_model.sandbox_fusion.url`,
  `reward_model.sandbox_fusion.max_concurrent`, and
  `reward_model.sandbox_fusion.memory_limit_mb`.

The missing pieces are not the loss or trainer loop. The missing pieces are:

1. code data conversion;
2. a robust code reward wrapper;
3. code-specific launch wrappers;
4. Sandbox Fusion or local execution environment validation;
5. offline code eval scripts adapted to the current checkpoint merge/extract
   flow;
6. Meituan-compatible path/env overrides.

## Proposed Training/Eval Design

### Prompt contract

Use the same high-level reasoning wrapper as the old code project:

```text
<think>...</think>
<answer>...</answer>
```

Inside `<answer>`, require a Python code block:

````text
```python
...
```
````

Reason:

- the current code reward path is most reliable when it can extract executable
  Python from a fence;
- the old eval scripts already handle thinking-enabled Qwen3 chat prompts;
- training reward, validation reward, and offline eval should share one code
  extraction function to avoid format-driven false negatives.

### Primary train data

Convert:

```text
/data-1/dataset/code/code-train.jsonl
```

to verl RL parquet with columns:

| Column | Required content |
| --- | --- |
| `data_source` | Use a code-specific value that routes to the intended reward path, or use a custom reward wrapper that ignores the built-in router. |
| `prompt` | Chat prompt with system message plus user task. |
| `ability` | `code` |
| `reward_model` | Dict containing `style=code_execution` and `ground_truth=<test_case_json>`. |
| `split` | `train` |
| `extra_info` | Source row index, original `source`, test-case schema, and optional reference-answer metadata. |

Acceptance requires 19,457 output rows unless exclusions are documented.

### Validation data during training

For the first training smoke, do not run the full offline code suite every
validation interval. Use a small converted validation shard with cheap,
representative executable tests.

Recommended staged validation:

1. `code_train_holdout_smoke`: 100-200 held-out train-format tasks, same
   `test_case` schema as training.
2. `humaneval_smoke`: 20-40 HumanEval tasks through the same code extraction
   and execution wrapper.
3. Full offline eval only after checkpoints are merged/extracted.

This keeps reward latency measurable and avoids making each training validation
interval a full benchmark run.

### Offline eval matrix

Run after Stage1/Stage2 checkpoints are merged and Model2 is extracted:

| Model | Purpose |
| --- | --- |
| `/data-1/.cache/Qwen3-4B-Base-Code-SFT/checkpoint-38` | Code SFT baseline. |
| `/data-1/.cache/Qwen3-4B-Base-Code-WDL-M1/checkpoint-39` | Historical WDL code baseline, same eval harness only. |
| Stage1 handoff checkpoint Model2 | Source model for Stage2 comparison. |
| Stage2 best checkpoint Model2 | Peak intervention evidence. |
| Stage2 final checkpoint Model2 | Stability evidence. |

Benchmarks:

- HumanEval;
- HumanEval+ via official EvalPlus data/API only;
- MBPP;
- MBPP+ via official EvalPlus data/API only;
- BigCodeBench local runner first, official harness later if installed;
- LiveCodeBench public-test local runner first, private-test/official path only
  after decode and dependency validation.

Eval settings:

| Mode | Parameters | Purpose |
| --- | --- | --- |
| Comparable pass@1 | `temperature=0.0`, `top_p=1.0`, `n=1` | Match standard code-eval reporting and old EvalPlus wrapper. |
| Diagnostic pass@3 / mean@3 | `temperature=1.0`, `top_p=0.95`, `n=3` | Optional consistency with the math offline-eval style and old DPO mean@3 records. |

Pass@1 is the primary code metric. Sampling metrics can be secondary but should
not replace pass@1.

## Reward and Sandbox Plan

### Reward wrapper

Add a custom reward function later, not in this plan file:

```text
recipe/on_policy_wdl_sft/custom_reward_function_code.py
```

Required signature:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ...
```

Required behavior:

- extract code from `<answer>`, fenced Python blocks, and raw Python fallback;
- reject missing or malformed code with a clear metadata reason;
- run executable tests through Sandbox Fusion if configured;
- fallback to local `prime_code` only for smoke and debugging unless its Python
  compatibility is fully validated in `verl-harness`;
- return a dict with at least:
  - `score`;
  - `acc`;
  - `pred`;
  - `verification_method`;
  - `error_type`;
  - `num_tests`;
  - `num_passed`;
  - `timeout_count`;
  - `extraction_fail`.

### Sandbox policy

Training should prefer Sandbox Fusion over host-local subprocess execution.

Reasons:

- code reward executes untrusted model output;
- code reward is much slower than math reward;
- current staged math defaults would generate
  `TRAIN_PROMPT_BSZ * ROLLOUT_N = 64 * 8 = 512` completions per step, which is
  too aggressive before reward throughput is measured;
- missing Python packages in the execution environment can create false
  negatives.

Minimum Sandbox Fusion config variables for future wrappers:

```bash
SANDBOX_FUSION_URL=...
SANDBOX_FUSION_MAX_CONCURRENT=...
SANDBOX_FUSION_MEMORY_LIMIT_MB=1024
CODE_REWARD_TIMEOUT=...
```

The local fallback path is allowed only for Phase 0/1 smoke runs and must record
that the result is not a production-throughput measurement.

## Parameter Standards

### Training parameters

Start small because each rollout needs executable verification.

| Parameter | Smoke standard | Main pilot standard | Reason |
| --- | ---: | ---: | --- |
| `ROLLOUT_N` | 2-4 | 4 first, 8 only after throughput proof | Test execution cost scales linearly. |
| `TRAIN_PROMPT_BSZ` | 4-8 | 16-32 | Math default 64 is too aggressive for first code run. |
| `MAX_RESPONSE_LENGTH` | 4096 | 4096, then test 6144/8192 if truncation appears | Code tasks can need longer outputs, but longer responses increase eval cost. |
| `LR` | `5e-7` | `5e-7` first | Keep method comparison close to math unless code smoke underfits badly. |
| `beta` | `0.0` first | add `0.1` only after reward stability | Reduce moving parts while validating reward. |
| `TEST_FREQ` | sparse | sparse until reward latency known | Full code validation is expensive. |
| `SAVE_FREQ` | frequent in smoke | aligned with validation | Needed for recovery and merge/extract tests. |

Stage design:

1. Phase 1 smoke: 3-5 steps, `TRAIN_PROMPT_BSZ<=8`, `ROLLOUT_N<=4`.
2. Phase 2 pilot Stage1: short handoff search, for example 20-40 steps, after
   reward latency is measured.
3. Phase 3 pilot Stage2: short matched-beta continuation from a fixed Stage1
   Model2 checkpoint.
4. Phase 4 full experiment: only after offline eval agrees with training
   reward direction.

### Evaluation parameters

Primary:

- `pass@1`;
- `temperature=0.0`;
- `top_p=1.0`;
- `n=1`;
- `max_tokens=4096` initially;
- thinking enabled for Qwen3.

Secondary diagnostic:

- `pass@3` / `mean@3`;
- `temperature=1.0`;
- `top_p=0.95`;
- `n=3`;
- same extraction function as pass@1.

Reporting must include:

- pass rate;
- extraction-fail rate;
- compile-error rate;
- runtime-error rate;
- timeout rate;
- dependency/import-error rate;
- average generated token length;
- number of tasks evaluated;
- harness name and version/path.

## Implementation Phases

### Phase 0: Verifier and environment proof

Do this before any training.

Tasks:

1. Check which code-execution dependencies exist in `verl-harness`.
2. Decide whether Sandbox Fusion is available locally or remotely.
3. Smoke `prime_code.compute_score` on `code-train.jsonl` references and known
   wrong answers.
4. Smoke old EvalPlus wrapper in `qwen` env and then reproduce the dependency
   set in `verl-harness` or a dedicated eval env.
5. Create a small failure taxonomy for 20-50 samples:
   - accepted reference;
   - wrong answer rejected;
   - compile error;
   - runtime error;
   - timeout;
   - missing dependency;
   - extraction fail.

Exit criteria:

- reference answers pass at high rate;
- intentionally wrong answers fail;
- no timeout storm;
- no common missing dependency creates broad false negatives;
- the exact runtime environment for reward and eval is documented.

### Phase 1: Data conversion

Tasks:

1. Convert `code-train.jsonl` to verl RL parquet.
2. Create a non-overlap Stage2 train shard policy, reusing the staged-v1 math
   principle.
3. Create small validation parquets for reward smoke.
4. Save manifest files with source paths, row counts, prompt format, and
   test-case schema counts.

Exit criteria:

- converted train parquet has 19,457 rows or documented exclusions;
- all `test_case` values are JSON parseable or excluded with reasons;
- prompt rows contain exactly one system message and one user task;
- token-length distribution is measured;
- Stage1/Stage2 shard overlap is zero.

### Phase 2: Reward wrapper and smoke

Tasks:

1. Add `custom_reward_function_code.py`.
2. Add shared code extraction utilities if existing local extractor is not
   imported directly.
3. Add unit/smoke tests for extraction and reward execution.
4. Run a 3-5 step staged-v1 training smoke.

Exit criteria:

- reward returns structured metadata;
- reward batch cannot hang on one bad completion;
- training logs show nontrivial `acc`, not all zero or all one;
- checkpoint save/merge/extract works;
- extracted Model2 contains `chat_template.jinja`.

### Phase 3: Offline eval harness

Tasks:

1. Port or wrap EvalPlus for HumanEval/MBPP official pass@1.
2. Port local BigCodeBench and LiveCodeBench runners with explicit
   "local-runner" labels.
3. Share code extraction with the training reward path.
4. Add summary JSON outputs that can be imported into the local experiment
   registry.

Exit criteria:

- Code SFT checkpoint-38 reproduces historical results within an explainable
  range under the same harness;
- old WDL-M1 checkpoint-39 can be evaluated under the same harness;
- each eval summary records harness path, environment, generation parameters,
  extraction-fail rate, and execution error taxonomy.

### Phase 4: Stage1 -> Stage2 pilot

Tasks:

1. Run Stage1 code pilot with conservative rollout/test settings.
2. Pick a fixed Stage1 handoff checkpoint by step, not by best-only search.
3. Merge/extract Model2 and record provenance.
4. Run short matched Stage2.
5. Evaluate Stage1 source, Stage2 best, and Stage2 final.

Exit criteria:

- Stage2 has a fixed, inspectable Stage1 Model2 source;
- Stage2 uses a non-overlap train shard;
- best and final checkpoints are both evaluated;
- final checkpoint does not show format or execution collapse;
- results are imported into the local database only after schema validation.

## Acceptance Criteria

### Plan acceptance

This plan is accepted when it clearly identifies:

- local code datasets and their row counts;
- old `qwen` environment and missing packages;
- old training/eval scripts and their reliability boundaries;
- the current `verl` reward integration points;
- the exact reward/verifier gates before training;
- training and eval parameter standards;
- implementation phases and stop conditions.

### Environment acceptance

Before launching code training:

- `verl-harness` or the chosen reward sandbox has the required Python packages;
- EvalPlus is available for official HumanEval/MBPP plus scoring;
- BigCodeBench and LiveCodeBench runner status is labeled as local or official;
- Sandbox Fusion URL/concurrency/memory settings are configured, or local
  fallback is explicitly marked smoke-only;
- dependency/import errors are measured on representative tests.

### Reward acceptance

Before launching code training:

- at least 20 train samples are checked with reference answers and known wrong
  answers;
- reference false-negative cases are inspected and categorized;
- extraction works for `<answer>`, fenced Python, raw Python, and repeated
  degenerate output;
- timeout and runtime-error handling cannot block the whole batch;
- reward metadata is logged and visible in validation summaries.

### Data acceptance

Before launching code training:

- train parquet row count matches source count or documented exclusions;
- all test cases parse or exclusions are recorded;
- prompt format is aligned between train, validation, and offline eval;
- Stage2 train shard has zero overlap with Stage1 shard;
- data manifests are written next to generated parquets.

### Training smoke acceptance

Before full pilot:

- 3-5 steps run without OOM, deadlock, timeout storm, or all-zero rewards;
- reward latency per step is measured;
- checkpoints save correctly;
- merge/extract preserves tokenizer files, especially `chat_template.jinja`;
- raw completions show executable code in the expected answer format.

### Result acceptance

The code extension is considered a promising positive result only if:

- Stage2 best improves over Stage1 source on the primary code eval average;
- Stage2 final remains close enough to best to indicate no short-run collapse;
- improvements are not explained by extraction or dependency artifacts;
- pass@1 improves or remains stable on HumanEval/MBPP while at least one harder
  local benchmark improves;
- extraction-fail, timeout, compile-error, runtime-error, and dependency-error
  rates remain healthy and are reported.

The code extension is a useful negative result if:

- reward verification is reliable but Stage2 does not improve over Stage1; or
- Stage2 improves training reward but offline pass@1 regresses broadly; or
- code generation collapses into missing code blocks, long repetition, or
  timeout-dominated outputs.

It is not a valid algorithmic result if the reward or eval harness has broad
false negatives or unmeasured dependency failures.

## Open Decisions

Resolve these after Phase 0:

- Whether to use Sandbox Fusion as mandatory for all training, or allow local
  `prime_code` for a small pilot.
- Whether LiveCodeBench private tests should be decoded and used, or whether
  public-test local LCB remains a diagnostic-only metric.
- Whether BigCodeBench should stay on the historical local runner for
  comparability, or move to the official package if available.
- Whether the first algorithmic pilot should use only `beta=0.0`, or run matched
  `beta=0.0` and `beta=0.1` after reward smoke.
- Whether the initial code model pair should be:
  - base + Code-SFT checkpoint; or
  - Code-SFT + historical WDL-M1; or
  - a new code Stage1 source trained fully inside this repo.

Default recommendation: start with the smallest valid algorithmic pilot,
`beta=0.0`, Code-SFT as Model2 initialization, conservative reward throughput,
and full offline pass@1 before adding `beta=0.1`.
