# RLVR public dataset consumer handoff

## Release status

Target repository:
[`AlexGeek/RLdataset`](https://huggingface.co/datasets/AlexGeek/RLdataset).

The live audit on 2026-07-29 found:

```text
visibility: private
gated: false
main: da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c
anonymous access: HTTP 401
```

That revision is not a public release and must not be sent to an anonymous
consumer. Replace the status and revision in this guide only after a new bundle
has been uploaded privately, byte-verified, changed to public, and downloaded
successfully without credentials.

The full bundle has 13 Parquet payloads. Five have a documented public
redistribution basis; eight still need an explicit publication decision from
the repository owner. See
`docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_README.md` for the
exact boundary.

## Consumer network contract

Consumers use the standard public Hugging Face endpoint. This handoff does not
prescribe a proxy, traffic subscription, region, or project-specific network
route. Each operator owns the network configuration on their host.

Image construction is outside this handoff. This guide covers only code/data
placement and dataset validation.

## Code checkout

```text
main repository:   https://github.com/AlexJJ009/verl-v0.7.git
recipe repository: https://github.com/AlexJJ009/verl-recipe.git
branch:            codex/rebuttal-rlvr (both repositories)
```

Clone the superproject and initialize its pinned recipe submodule:

```bash
git clone --branch codex/rebuttal-rlvr \
  https://github.com/AlexJJ009/verl-v0.7.git \
  verl-rebuttal-rlvr
git -C verl-rebuttal-rlvr submodule update --init --recursive
```

The final handoff message must provide the delivered superproject commit and
recipe gitlink. Do not replace the submodule with a separately floating recipe
checkout.

## Choose paths

The colleague chooses the persistent root. Code, dataset, models, and mutable
state do not need to reproduce a path from this machine.

Recommended sibling layout:

```text
<WORK_ROOT>/
  verl-rebuttal-rlvr/
  RLdataset/
  models/
  state/
```

Example variables:

```bash
export WORK_ROOT=/path/chosen/by/the/operator
export CODE_ROOT="$WORK_ROOT/verl-rebuttal-rlvr"
export DATASET_ROOT="$WORK_ROOT/RLdataset"
export MODEL_ROOT="$WORK_ROOT/models"
export STATE_ROOT="$WORK_ROOT/state"
```

This is a recommendation, not a constraint. `DATASET_ROOT`, `MODEL_ROOT`, and
`STATE_ROOT` may be elsewhere as long as they are absolute, persistent, and
passed to the launcher. The Meituan adapter already consumes an explicit
`DATASET_ROOT`; it maps data below `$DATASET_ROOT/data/...`.

## Public download and validation

Run only after this guide records a verified public revision:

```bash
: "${WORK_ROOT:?choose WORK_ROOT first}"
: "${DATASET_ROOT:?choose DATASET_ROOT first}"
: "${STATE_ROOT:?choose STATE_ROOT first}"
test ! -e "$DATASET_ROOT"
mkdir -p "$STATE_ROOT/receipts"
mkdir -p "$WORK_ROOT/.venvs"

python3 -m venv "$WORK_ROOT/.venvs/rlvr-dataset"
. "$WORK_ROOT/.venvs/rlvr-dataset/bin/activate"
python -m pip install \
  'huggingface_hub==1.8.0' \
  'pyarrow==23.0.1'

hf download AlexGeek/RLdataset \
  --repo-type dataset \
  --revision REPLACE_WITH_VERIFIED_PUBLIC_COMMIT \
  --local-dir "$DATASET_ROOT"

python3 "$DATASET_ROOT/validate_dataset.py" \
  --dataset-root "$DATASET_ROOT" \
  --receipt "$STATE_ROOT/receipts/RLdataset-validation.json"
```

The validator is independent of the current working directory and does not
assume that the dataset is a child of the code checkout. A successful full
release reports:

```text
ok=true
file_count=18
payload_count=13
payload_rows=22860
```

Python 3.10 or newer, `venv`/`pip`, write access to the chosen directories,
and at least 100 MB of free space are required. The logical repository is
approximately 30.3 MB; the remaining allowance covers client metadata and
receipts.

## Relative data contract

The repository preserves these launcher-facing roots:

```text
$DATASET_ROOT/data/math/train_rl_format.parquet
$DATASET_ROOT/data/math7/<benchmark>/*.parquet
$DATASET_ROOT/data/code/verl_rl/<dataset>/*.parquet
```

Math-7 contains exactly AIME-2025, MATH-500, AMC23, AQuA, GSM8K, MAWPS, and
SVAMP. Code-4 contains exactly HumanEval+, MBPP+, LiveCodeBench release_v5, and
BigCodeBench. Evaluator source checkouts and the LiveCodeBench SQLite evaluator
index are not dataset payloads.

## Release gate for the handoff message

Do not tell a colleague that the dataset is ready until all of these are true:

1. the Hugging Face API reports `private=false` and `gated=false`;
2. the exact public commit is recorded here and in the final handoff message;
3. an empty, credential-free `hf download` succeeds at that commit;
4. the bundled validator passes exact tree, SHA-256, rows, schema, and semantic
   checks on the anonymous download;
5. the superproject branch and recipe gitlink have both been pushed and
   verified from a fresh clone.

The release operator proves item 3 with a new credential home and all ambient
token variables removed. A normal download from an already logged-in machine
does not prove public access:

```bash
: "${WORK_ROOT:?choose WORK_ROOT and create the pinned validation venv above}"
: "${VERIFIED_PUBLIC_COMMIT:?set the exact 40-character public commit}"
test "${#VERIFIED_PUBLIC_COMMIT}" -eq 40
case "$VERIFIED_PUBLIC_COMMIT" in
  *[!0-9a-f]*) echo "VERIFIED_PUBLIC_COMMIT must be lowercase hexadecimal" >&2; exit 2 ;;
esac

ANON_SCRATCH=$(mktemp -d)
install -d -m 700 "$ANON_SCRATCH/hf-home"
HF_CLI="$WORK_ROOT/.venvs/rlvr-dataset/bin/hf"
VALIDATION_PYTHON="$WORK_ROOT/.venvs/rlvr-dataset/bin/python"
test -x "$HF_CLI"
test -x "$VALIDATION_PYTHON"

env \
  -u HF_TOKEN \
  -u HUGGING_FACE_HUB_TOKEN \
  -u HUGGINGFACEHUB_API_TOKEN \
  -u HF_TOKEN_PATH \
  HF_HOME="$ANON_SCRATCH/hf-home" \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  HF_ENDPOINT=https://huggingface.co \
  "$HF_CLI" download AlexGeek/RLdataset \
    --repo-type dataset \
    --revision "$VERIFIED_PUBLIC_COMMIT" \
    --local-dir "$ANON_SCRATCH/RLdataset"

"$VALIDATION_PYTHON" "$ANON_SCRATCH/RLdataset/validate_dataset.py" \
  --dataset-root "$ANON_SCRATCH/RLdataset" \
  --receipt "$ANON_SCRATCH/validation-receipt.json"
```

The full `18 / 13 / 22,860` contract and the current README describe one exact
publication set. If the owner chooses the five-payload subset, rebuild the
README, inventory, checksums, validator-bound allowlist, expected counts, and
immutable revision together. Removing eight files from the full candidate is
not a valid release procedure.
