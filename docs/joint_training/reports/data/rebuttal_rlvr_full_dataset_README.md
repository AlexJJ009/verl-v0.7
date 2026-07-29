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

> **Release safety.** This README cannot embed the SHA of the commit that
> contains itself. Do not use floating `main` and do not treat this page alone
> as a release receipt. A valid handoff supplies a 40-character
> `DATASET_REVISION` that has passed a credential-free download and this
> repository's validator. If the repository is still private, anonymous
> consumers cannot use it.

## Choose the download location

The dataset does not require a machine-specific mount point. Choose any
directory that is persistent and readable by the training job, then pass that
directory as `DATASET_ROOT`.

A sibling layout is convenient but optional:

```text
<WORK_ROOT>/
  verl-rebuttal-rlvr/   # code checkout
  RLdataset/            # DATASET_ROOT
  state/                # checkpoints, logs, and receipts
```

For example:

```bash
export WORK_ROOT=/path/chosen/by/the/operator
export CODE_ROOT="$WORK_ROOT/verl-rebuttal-rlvr"
export DATASET_ROOT="$WORK_ROOT/RLdataset"
export STATE_ROOT="$WORK_ROOT/state"

: "${WORK_ROOT:?choose WORK_ROOT first}"
: "${DATASET_ROOT:?choose DATASET_ROOT first}"
: "${STATE_ROOT:?choose STATE_ROOT first}"
```

`DATASET_ROOT` may instead live above, below, or outside `CODE_ROOT`. Do not
create a symlink merely to imitate another machine's directory layout.

## Download from public Hugging Face

Use the standard public Hugging Face endpoint. No repository token, local
proxy, or project-specific network node is part of this consumer contract.

Pin the immutable revision supplied with the handoff instead of downloading a
floating `main`:

```bash
: "${WORK_ROOT:?choose WORK_ROOT first}"
: "${DATASET_ROOT:?choose DATASET_ROOT first}"
: "${STATE_ROOT:?choose STATE_ROOT first}"
: "${DATASET_REVISION:?copy the verified 40-character commit from the handoff}"
test "${#DATASET_REVISION}" -eq 40
case "$DATASET_REVISION" in
  *[!0-9a-f]*) echo "DATASET_REVISION must be lowercase hexadecimal" >&2; exit 2 ;;
esac
test ! -e "$DATASET_ROOT"

mkdir -p "$WORK_ROOT/.venvs"
python3 -m venv "$WORK_ROOT/.venvs/rlvr-dataset"
. "$WORK_ROOT/.venvs/rlvr-dataset/bin/activate"
python -m pip install \
  'huggingface_hub==1.8.0' \
  'pyarrow==23.0.1'

hf download AlexGeek/RLdataset \
  --repo-type dataset \
  --revision "$DATASET_REVISION" \
  --local-dir "$DATASET_ROOT"
```

The operator is responsible for any proxy, bandwidth, or egress configuration
needed on their own host.

Prerequisites are Python 3.10 or newer, `venv`/`pip`, write access to the chosen
directories, and at least 100 MB of free space for the approximately 30.3 MB
logical repository plus client metadata and validation receipts. Git is needed
only for the separate code checkout.

## Validate before use

The repository ships `validate_dataset.py`. It verifies:

- the exact allowlisted file tree and absence of dataset symlinks;
- every SHA-256 digest in `metadata/checksums.sha256`;
- inventory/file agreement and byte counts;
- all 13 Parquet row counts and required Arrow schema;
- every row's non-empty prompt, reward ground truth, style, data source, and
  ability fields.

Run it from any working directory. A receipt must be written outside the exact
dataset tree:

```bash
: "${DATASET_ROOT:?choose DATASET_ROOT first}"
: "${STATE_ROOT:?choose STATE_ROOT first}"
mkdir -p "$STATE_ROOT/receipts"
python3 "$DATASET_ROOT/validate_dataset.py" \
  --dataset-root "$DATASET_ROOT" \
  --receipt "$STATE_ROOT/receipts/RLdataset-validation.json"
```

Expected release summary:

```json
{
  "ok": true,
  "file_count": 18,
  "payload_count": 13,
  "payload_rows": 22860
}
```

`hf download` may create `.cache/huggingface/`, and `git clone` may create
`.git/`; the validator ignores only those client-owned metadata directories.
Every other unexpected file is a failure.

## Dataset paths

### Mathematical training and Math-7

| Dataset | Rows | Relative path | SHA-256 |
|---|---:|---|---|
| MATH RLVR train | 7,500 | `data/math/train_rl_format.parquet` | `86531549f6825f6737ce58f0f6bfd8e0df5b0298b35cb18192e40f460ba3cb58` |
| AIME-2025 | 30 | `data/math7/AIME-2025/aime-2025_with_system_prompt.parquet` | `38f6034e6d28fedb71d29edafad068e6ea500cb570a1f1a8f53505ba5fda7ddf` |
| MATH-500 | 500 | `data/math7/MATH-500/math500-test_with_system_prompt.parquet` | `9ee8e81d86df4dbaa125432ffef38b2e88317fdf56d85ef147e9d18c063577be` |
| AMC23 | 40 | `data/math7/AMC23/amc23-test_with_system_prompt.parquet` | `fd87b30a3edd7f152cb7fe5170892ed1e0d3672daa2e195d8d40943938ee52fa` |
| AQuA | 254 | `data/math7/AQUA/aqua-test_with_system_prompt.parquet` | `854f5cbe7b88a065c99a5d619c9e5c76e3df783cbe3e6e559379d342f3e71cd2` |
| GSM8K | 1,319 | `data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet` | `a7b4521427780e8b7d28f5abd17428b103af267977ae7a9f4b73085d4c0900cb` |
| MAWPS | 355 | `data/math7/MAWPS/mawps-test_with_system_prompt.parquet` | `8d69bdff471dfa6da9996125a22cc5cffc3768fecab5d891442a84508d54fddd` |
| SVAMP | 300 | `data/math7/SVAMP/svamp-test_with_system_prompt.parquet` | `5c87092e34e85488b8fa835e11befc35f944b7ffec8c5e6fd571969004a0e34f` |

### Code training and Code-4

| Dataset | Rows | Relative path | SHA-256 |
|---|---:|---|---|
| KodCode-Light-RL-10K author-signature-v2 | 10,000 | `data/code/verl_rl/kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet` | `80467821362328e370a738ed51b7311f596496ae42572ad2f6793ed8cd51c47d` |
| HumanEval+ | 164 | `data/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet` | `e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab` |
| MBPP+ | 378 | `data/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet` | `3221e7f53c88bfbd91d788fb7bcb37168fb088fa504fddf12b9126c2147312d2` |
| LiveCodeBench release_v5 | 880 | `data/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet` | `fe7d2bfe2779bcf106492347ca173e30b9220c15c1b8783949d35edcd93a43d1` |
| BigCodeBench | 1,140 | `data/code/verl_rl/online_full_bigcodebench/official_bigcodebench_val.parquet` | `84c5ebedf6a445f86107427394e7f5e84d60d744d731a332e4d8b0e65338d962` |

The training launcher resolves the mathematical training file as
`$DATASET_ROOT/data/math/train_rl_format.parquet` and Math-7 below
`$DATASET_ROOT/data/math7/`. Code payloads remain below
`$DATASET_ROOT/data/code/verl_rl/`.

## Format

Every Parquet payload follows the project's verl prompt/reward contract. These
are the byte-identical files consumed by the current training and evaluation
launchers; building this Hugging Face release does not perform another format
conversion. The
required top-level columns are:

```text
data_source, ability, reward_model, prompt, extra_info
```

Most files also contain a top-level `split` column. GSM8K stores its split in
`extra_info`, so consumers must not require a single identical top-level schema
across all 13 files.

`prompt` is a list of `{role, content}` messages. `reward_model` contains
`ground_truth` and `style`. Mathematical prompts request a boxed final answer;
code prompts use the project's `<think>...<answer>...` executable-Python
contract.

## Evaluator boundary

This repository contains dataset Parquets, not evaluator installations or
large evaluator caches. In particular, the LiveCodeBench release_v5 SQLite
input/output index is not included. Evaluator provisioning is a separate
runtime concern.

The evaluator source-code revisions used by this project are recorded in
`metadata/publication_inventory.json`:

- EvalPlus: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
- BigCodeBench: `09dd993f46c3fbf3a799465bb96d524edcb0b199`
- LiveCodeBench: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`

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
