---
configs:
- config_name: math_train
  data_files:
  - split: train
    path: data/math/train_rl_format.parquet
- config_name: math_eval_aime_2025
  data_files:
  - split: test
    path: data/math7/AIME-2025/aime-2025_with_system_prompt.parquet
- config_name: math_eval_math_500
  data_files:
  - split: test
    path: data/math7/MATH-500/math500-test_with_system_prompt.parquet
- config_name: math_eval_amc23
  data_files:
  - split: test
    path: data/math7/AMC23/amc23-test_with_system_prompt.parquet
- config_name: math_eval_aqua
  data_files:
  - split: test
    path: data/math7/AQUA/aqua-test_with_system_prompt.parquet
- config_name: math_eval_gsm8k
  data_files:
  - split: test
    path: data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet
- config_name: math_eval_mawps
  data_files:
  - split: test
    path: data/math7/MAWPS/mawps-test_with_system_prompt.parquet
- config_name: math_eval_svamp
  data_files:
  - split: test
    path: data/math7/SVAMP/svamp-test_with_system_prompt.parquet
- config_name: code_train_kodcode
  data_files:
  - split: train
    path: data/code/verl_rl/kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet
- config_name: code_eval_humaneval_plus
  data_files:
  - split: test
    path: data/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet
- config_name: code_eval_mbpp_plus
  data_files:
  - split: test
    path: data/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet
- config_name: code_eval_livecodebench_v5
  data_files:
  - split: test
    path: data/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet
- config_name: code_eval_bigcodebench
  data_files:
  - split: test
    path: data/code/verl_rl/online_full_bigcodebench/official_bigcodebench_val.parquet
task_categories:
- text-generation
language:
- en
license: other
pretty_name: WDL rebuttal RLVR math and code data
---

# WDL rebuttal RLVR math and code data

This repository packages the training and evaluation Parquet files used by the
WDL rebuttal RLVR workflows. It preserves the relative `data/...` layout used
by the launch scripts:

- one 7,500-row mathematical RLVR training file;
- Math-7: AIME-2025, MATH-500, AMC23, AQuA, GSM8K, MAWPS, and SVAMP;
- one 10,000-row KodCode training file;
- Code-4: HumanEval+, MBPP+, LiveCodeBench release_v5, and BigCodeBench.

The repository contains 13 Parquet payloads and 22,860 rows in total. There is
no Code-7 contract: EvalPlus reports several metrics from the same HumanEval+
and MBPP+ files, but those metrics are not additional datasets.

## Download and validate

The download destination is not part of the dataset contract. For the first
download, choose any new destination path and set `DATASET_ROOT` to it. The
complete downloaded directory may later be moved; after moving it, set the
launcher's `DATASET_ROOT` override to the new directory. Do not rearrange files
inside the directory.

Use the immutable `DATASET_REVISION` supplied in the handoff instead of a
floating `main`. The repository is public and does not require a Hugging Face
token. In any Python 3.10+ environment:

```bash
python -m pip install \
  'huggingface_hub==1.8.0' \
  'pyarrow==23.0.1'

export DATASET_REVISION=<verified-40-character-revision>
export DATASET_ROOT=/any/path/RLdataset
test ! -e "$DATASET_ROOT"

hf download AlexGeek/RLdataset \
  --repo-type dataset \
  --revision "$DATASET_REVISION" \
  --local-dir "$DATASET_ROOT"

python "$DATASET_ROOT/validate_dataset.py" \
  --dataset-root "$DATASET_ROOT"
```

The validator checks the exact file tree and SHA-256 values, all 13 Parquet row
counts and schemas, and every row's prompt/reward fields. It must exit with
status 0 and print a JSON object containing:

```json
{
  "ok": true,
  "file_count": 18,
  "payload_count": 13,
  "payload_rows": 22860
}
```

The README does not prescribe a code, model, checkpoint, or log layout. Exact
payload paths, hashes, evaluator revisions, and conversion notes are recorded in
`metadata/publication_inventory.json` and `metadata/checksums.sha256`.

## Upstream sources, licenses, and citations

This is a multi-source collection, so `license: other` is intentional. Each
payload keeps the license or source terms declared by its upstream dataset;
there is no new blanket license for the complete collection. The exact source
records and modification notices are also embedded per payload in
`metadata/publication_inventory.json`.

### Mathematics

| Payload | Upstream/version | Upstream license or terms | Citation |
|---|---|---|---|
| MATH RLVR train | [`ck46/hendrycks_math@0e71a2a`](https://huggingface.co/datasets/ck46/hendrycks_math/tree/0e71a2aaa3c196023c96b67f2960fca36631ae2b) | MIT in the [canonical MATH repository](https://github.com/hendrycks/math) | [Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874) |
| AIME-2025 | [`MathArena/aime_2025@c94da77`](https://huggingface.co/datasets/MathArena/aime_2025/tree/c94da77eb22bbd6439e62a323bec18493a421302) | Mirror declares [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); MAA source terms also apply | [MAA AMC/AIME](https://maa.org/student-programs/amc/) |
| MATH-500 | [`HuggingFaceH4/MATH-500@6e4ed1a`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) | MIT in the canonical MATH and [PRM800K](https://github.com/openai/prm800k) repositories | [Lightman et al., 2023](https://arxiv.org/abs/2305.20050) |
| AMC23 | [`zwhe99/amc23@f9810c0`](https://huggingface.co/datasets/zwhe99/amc23/tree/f9810c0439cd3c670ec885d328a2f06a87f3694a) | Mirror states no dataset license; MAA and linked AoPS source terms apply | [MAA AMC](https://maa.org/student-programs/amc/) |
| AQuA | [`deepmind/aqua_rat@33301c6`](https://huggingface.co/datasets/deepmind/aqua_rat/tree/33301c6a050c96af81f63cad5562cb5363e88971) | [Apache-2.0](https://github.com/google-deepmind/AQuA/blob/master/LICENSE) | [Ling et al., 2017](https://aclanthology.org/P17-1015/) |
| GSM8K | [`openai/gsm8k@740312a`](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) | [MIT](https://github.com/openai/grade-school-math/blob/master/LICENSE) | [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168) |
| MAWPS | [`mwpt5/MAWPS@5769dc9`](https://huggingface.co/datasets/mwpt5/MAWPS/tree/5769dc9a31ea36eaeaab1f1e0aa1a54b6c08d804) | No dataset license stated by the mirror or [canonical repository](https://github.com/sroy9/mawps) | [Koncel-Kedziorski et al., 2016](https://aclanthology.org/N16-1136/) |
| SVAMP | [`ChilleD/SVAMP@5e0bf1e`](https://huggingface.co/datasets/ChilleD/SVAMP/tree/5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef) | [MIT](https://github.com/arkilpatel/SVAMP/blob/main/LICENSE); its stated provenance includes ASDiv-A (CC BY-NC 4.0) and MAWPS | [Patel et al., 2021](https://aclanthology.org/2021.naacl-main.168/) |

### Code

| Payload | Upstream/version | Upstream license or terms | Citation |
|---|---|---|---|
| KodCode-Light-RL-10K | [`KodCode/KodCode-Light-RL-10K@dcf78a8`](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K/tree/dcf78a8bbba9a613b596ce993c4921a38687dfcc) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); non-commercial restriction applies | [Xu et al., 2025](https://arxiv.org/abs/2503.02951) |
| HumanEval+ | [HumanEvalPlus v0.1.10](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10) | Apache-2.0 for EvalPlus additions; [MIT](https://github.com/openai/human-eval/blob/6d43fb980f9fee3c892a914eda09951f772ad10d/LICENSE) for original HumanEval | [EvalPlus](https://openreview.net/forum?id=1qvx610Cu7), [HumanEval](https://arxiv.org/abs/2107.03374) |
| MBPP+ | [MbppPlus v0.2.0](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0) | Apache-2.0 | [EvalPlus](https://openreview.net/forum?id=1qvx610Cu7), [MBPP](https://arxiv.org/abs/2108.07732) |
| LiveCodeBench release_v5 | [`livecodebench/code_generation_lite@0fe84c3`](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505) | Upstream card states only `cc` without a specific CC variant; source-platform terms also apply | [LiveCodeBench](https://arxiv.org/abs/2403.07974) |
| BigCodeBench | [BigCodeBench full v0.1.4](https://huggingface.co/datasets/bigcode/bigcodebench) | Apache-2.0 declared by the upstream dataset card | [BigCodeBench](https://arxiv.org/abs/2406.15877) |

The MATH training conversion adds the verl prompt/reward fields and two
documented empty-box answer fixes. Math evaluation conversions add the project
system prompt. KodCode uses the author-signature-v2 prompt template. Code
evaluation conversions package the upstream tasks into the project's validation
schema; the exact embedded content boundary is recorded in the inventory.

On 2026-07-29 the repository owner approved all 13 payloads for a public
release under this per-source attribution and upstream-terms policy. Approval
alone does not establish public availability: consumers must wait for a pinned
commit that has passed the credential-free download and validator gate above.
Downstream users remain responsible for the terms applicable to their use.

This README and its expected `18 / 13 / 22,860` validation summary describe the
approved full publication set. Any later payload change must regenerate and
reverify the README, inventory, checksum manifest, validator-bound exact file
tree, expected counts, and immutable revision together.
