---
configs:
- config_name: math_rlvr_train
  data_files:
  - split: train
    path: data/math/train_rl_format.parquet
- config_name: math_eval_public
  data_files:
  - split: test
    path: data/math7/**/*.parquet
- config_name: code_eval_public
  data_files:
  - split: test
    path: data/code/**/*.parquet
task_categories:
- text-generation
language:
- en
license: other
pretty_name: Rebuttal RLVR public training and evaluation assets
---

# Rebuttal RLVR public training and evaluation assets

This repository is the public, checksum-pinned subset of the data used by the
WDL rebuttal RLVR math and code workflows. It intentionally does **not** mirror
every local benchmark asset. Competition problems, hidden/public test bundles,
and derived evaluator caches are excluded when an explicit data redistribution
license was not established.

## Included files

| Relative path | Role | Rows | SHA-256 | Upstream |
|---|---|---:|---|---|
| `data/math/train_rl_format.parquet` | MATH RLVR train | 7,500 | `86531549f6825f6737ce58f0f6bfd8e0df5b0298b35cb18192e40f460ba3cb58` | `ck46/hendrycks_math@0e71a2aaa3c196023c96b67f2960fca36631ae2b` |
| `data/math7/MATH-500/math500-test_with_system_prompt.parquet` | math evaluation | 500 | `9ee8e81d86df4dbaa125432ffef38b2e88317fdf56d85ef147e9d18c063577be` | `HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| `data/math7/AQUA/aqua-test_with_system_prompt.parquet` | math evaluation | 254 | `854f5cbe7b88a065c99a5d619c9e5c76e3df783cbe3e6e559379d342f3e71cd2` | `deepmind/aqua_rat@33301c6a050c96af81f63cad5562cb5363e88971` |
| `data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet` | math evaluation | 1,319 | `a7b4521427780e8b7d28f5abd17428b103af267977ae7a9f4b73085d4c0900cb` | `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` |
| `data/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet` | code evaluation | 164 | `e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab` | `evalplus/evalplus@26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` |

`metadata/publication_inventory.json` records byte sizes, source revisions,
transformations, licenses, evaluator source pins, and the assets that were
deliberately excluded. `metadata/checksums.sha256` verifies every published
file except itself.

## Deliberately excluded

The following are needed by parts of the full local workflow but are not
redistributed here:

- AIME-2025 and AMC23: no MAA redistribution authorization was found.
- MAWPS and the MAWPS/ASDiv-derived part of SVAMP: the full data-license chain
  was not established.
- KodCode-Light-RL-10K: CC-BY-NC-4.0 plus mixed Codeforces, LeetCode, TACO and
  other sources; prompt/test redistribution was not cleared.
- MBPP+, BigCodeBench and LiveCodeBench data, tests, converted parquets, and
  the LiveCodeBench SQLite evaluator cache: source-code licenses do not by
  themselves license all benchmark data.

These omissions are fail-closed publication decisions, not claims that the
upstream datasets are unusable. Obtain the assets from their owners or through
an authorized private handoff, then place them at the expected relative paths
listed in `metadata/publication_inventory.json`. A complete Math-7 formal run
still requires all seven approved local evaluation files.

## Download and verify

Pin the repository revision supplied by the experiment manifest rather than
downloading a floating `main`:

```bash
DATASET_ROOT=/absolute/path/to/huggingface/dataset/EnsembleLLM-data
hf download beichenhang/EnsembleLLM-data \
  --repo-type dataset \
  --revision REPLACE_WITH_VERIFIED_COMMIT \
  --local-dir "$DATASET_ROOT"

cd "$DATASET_ROOT"
sha256sum -c metadata/checksums.sha256
```

The Meituan/Hope launcher treats this checkout as `DATASET_ROOT`. Models and
runtime state belong in separate `MODEL_ROOT` and `STATE_ROOT` directories; no
symlink back to another user's storage is required.

## Format and modifications

The parquet files use verl's prompt/reward schema. The MATH train conversion
adds the project system prompt, boxed-answer extraction, metadata, and two
documented fixes for empty boxed answers. Evaluation conversions add the same
prompt contract used by training. HumanEval+ is converted to the project's
`code-think-answer-python-v1` validation schema.

The transformation script for the 7,500-row MATH train file is included under
`processing/`. Evaluator implementations are not vendored here; the training
image pins EvalPlus, BigCodeBench, and LiveCodeBench source commits separately.

## Licenses and attribution

This is a multi-license collection:

- Hendrycks MATH and the MATH-derived files: MIT, Dan Hendrycks.
- AQuA: Apache-2.0, Google Inc.
- GSM8K: MIT, OpenAI.
- EvalPlus additions: Apache-2.0; HumanEval content: MIT, OpenAI.

The exact upstream license files are under `LICENSES/`. Changes made by this
project are described above and in the publication inventory. The repository
is provided for research reproducibility; each downstream user remains
responsible for complying with the applicable upstream licenses.
