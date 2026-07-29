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

Every Parquet payload follows the project's verl prompt/reward contract. The
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

The source revisions used by this project are recorded in
`metadata/publication_inventory.json`:

- EvalPlus: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
- BigCodeBench: `09dd993f46c3fbf3a799465bb96d524edcb0b199`
- LiveCodeBench: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`

## Licenses and redistribution review

This is a multi-source collection, so `license: other` is intentional.

Public redistribution is documented locally for the transformed MATH train,
MATH-500, AQuA, GSM8K, and HumanEval+ payloads. The complete redistribution
chain has not yet been established for AIME-2025, AMC23, MAWPS, SVAMP,
KodCode, MBPP+, LiveCodeBench, and BigCodeBench. Source-code licenses for an
evaluator do not automatically license embedded prompts, reference answers, or
tests.

This notice records the current review boundary; it is not legal advice. The
repository owner must explicitly approve the publication set before changing
the repository from private to public, and downstream users remain responsible
for the upstream terms that apply to their use.

This README and its expected `18 / 13 / 22,860` validation summary describe the
full publication set only. If the owner instead chooses the five-payload public
subset, the README, inventory, checksum manifest, validator-bound exact file
tree, immutable revision, and expected counts must all be regenerated and
reverified. Deleting eight Parquets from this bundle is not a valid subset
release.
