# Archived proposed public subset (not the live RLdataset repository)

> **Archival design only.** This five-payload subset was never established as
> the contents of `AlexGeek/RLdataset`. The live `AlexGeek/RLdataset` repository
> currently contains the complete 13-payload private handoff bundle. Do not use
> this document as a download manifest or as approval to make that repository
> public.

This document proposed a public, checksum-pinned subset of the data used by the
WDL rebuttal RLVR math and code workflows. It was not published at the current
target. The design intentionally omitted local benchmark assets when an
explicit data redistribution license was not established.

## Included files

| Relative path | Role | Rows | SHA-256 | Upstream |
|---|---|---:|---|---|
| `data/math/train_rl_format.parquet` | MATH RLVR train | 7,500 | `86531549f6825f6737ce58f0f6bfd8e0df5b0298b35cb18192e40f460ba3cb58` | `ck46/hendrycks_math@0e71a2aaa3c196023c96b67f2960fca36631ae2b` |
| `data/math7/MATH-500/math500-test_with_system_prompt.parquet` | math evaluation | 500 | `9ee8e81d86df4dbaa125432ffef38b2e88317fdf56d85ef147e9d18c063577be` | `HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| `data/math7/AQUA/aqua-test_with_system_prompt.parquet` | math evaluation | 254 | `854f5cbe7b88a065c99a5d619c9e5c76e3df783cbe3e6e559379d342f3e71cd2` | `deepmind/aqua_rat@33301c6a050c96af81f63cad5562cb5363e88971` |
| `data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet` | math evaluation | 1,319 | `a7b4521427780e8b7d28f5abd17428b103af267977ae7a9f4b73085d4c0900cb` | `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` |
| `data/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet` | code evaluation | 164 | `e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab` | `evalplus/evalplus@26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` |

The proposed `metadata/publication_inventory.json` records byte sizes, source
revisions, transformations, licenses, evaluator source pins, and the assets
that would be excluded. Its proposed `metadata/checksums.sha256` verifies every
candidate file except itself. Neither file is a live download manifest.

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

## Publication status

No live Hugging Face target is assigned to this subset. For the complete
private handoff, exact revision, proxy admission, download, and checksum
instructions, use
`docs/joint_training/guides/rebuttal_rlvr_hf_dataset_handoff.md`.

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
