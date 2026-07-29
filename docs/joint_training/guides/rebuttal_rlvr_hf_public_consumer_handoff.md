# RLVR public dataset consumer handoff

## Release status

Target repository:
[`AlexGeek/RLdataset`](https://huggingface.co/datasets/AlexGeek/RLdataset).

The public release gate completed on 2026-07-30 (Asia/Tokyo):

```text
visibility: public
gated: false
main: b1c264a92ace36dace52babdda651e415d9e9f82
preserved parent: da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c
anonymous download and validator: PASS
```

Use the immutable public revision
`b1c264a92ace36dace52babdda651e415d9e9f82`. A credential-free download of
all 18 files passed the bundled validator at that revision. The prior private
revision remains reachable as the direct parent, so the repository history was
preserved rather than rewritten.

The repository owner approved public release of the full 13-payload collection
on 2026-07-29. The dataset card cites each upstream source and records the
license or terms declared by that source. See
`docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_README.md` for the
source table.

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

The final handoff supplies `CODE_REVISION` and `RECIPE_GITLINK`. Clone the
superproject, detach at that exact commit, and initialize the recipe gitlink:

```bash
export CODE_REVISION=REPLACE_WITH_DELIVERED_SUPERPROJECT_COMMIT
export RECIPE_GITLINK=REPLACE_WITH_DELIVERED_RECIPE_GITLINK

git clone --no-checkout \
  https://github.com/AlexJJ009/verl-v0.7.git \
  verl-rebuttal-rlvr
git -C verl-rebuttal-rlvr fetch origin "$CODE_REVISION"
git -C verl-rebuttal-rlvr checkout --detach "$CODE_REVISION"
git -C verl-rebuttal-rlvr submodule update --init --recursive

test "$(git -C verl-rebuttal-rlvr rev-parse HEAD)" = "$CODE_REVISION"
test "$(git -C verl-rebuttal-rlvr ls-tree HEAD recipe | awk '{print $3}')" = "$RECIPE_GITLINK"
test "$(git -C verl-rebuttal-rlvr/recipe rev-parse HEAD)" = "$RECIPE_GITLINK"
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
  --revision b1c264a92ace36dace52babdda651e415d9e9f82 \
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

These are the actual Parquet files consumed by the launchers. The Hugging Face
publication copies them byte-for-byte and does not introduce a second format
conversion.

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

The full `18 / 13 / 22,860` contract and the current README describe the exact
approved publication set. The immutable commit supplied in the handoff must
match that complete contract.
