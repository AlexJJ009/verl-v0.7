# RLVR public dataset consumer handoff

## Verified release

- Dataset: [`AlexGeek/RLdataset`](https://huggingface.co/datasets/AlexGeek/RLdataset)
- Public, ungated revision: `5c3ce2d6a3b5ca61c60febccf202e7ee9d2615f8`
- Contents: 18 files, 13 Parquet payloads, 22,860 rows
- Anonymous download and bundled validator: PASS

The repository contains one math training set, Math-7, one code training set,
and Code-4. The Parquet payloads are byte-identical to the files consumed by the
training and evaluation launchers. Upstream sources and terms are recorded in
the dataset card and `metadata/publication_inventory.json`.

## Code checkout

```text
main repository:   https://github.com/AlexJJ009/verl-v0.7.git
recipe repository: https://github.com/AlexJJ009/verl-recipe.git
branch:            codex/rebuttal-rlvr (both repositories)
training code:     f7c88df7f830da3acbbd7b6eda8d1a1c55544239
recipe gitlink:    a65a8acc25cb02dd4abf8e3b871d126d31a5ccbc
```

Use these immutable revisions rather than the floating branch. Do not replace
the recipe submodule with a separately floating checkout.

## Download and validate

The destination path is not part of the dataset contract. For the first
download, choose any new destination path. The complete directory may later be
moved; after moving it, set the launcher's `DATASET_ROOT` override to the new
directory. Do not rearrange files inside the downloaded directory.

The repository is public, so no Hugging Face token is required. In any Python
3.10+ environment:

```bash
python -m pip install \
  'huggingface_hub==1.8.0' \
  'pyarrow==23.0.1'

export DATASET_ROOT=/any/path/RLdataset
test ! -e "$DATASET_ROOT"

hf download AlexGeek/RLdataset \
  --repo-type dataset \
  --revision 5c3ce2d6a3b5ca61c60febccf202e7ee9d2615f8 \
  --local-dir "$DATASET_ROOT"

python "$DATASET_ROOT/validate_dataset.py" \
  --dataset-root "$DATASET_ROOT"
```

The validator must exit with status 0 and print a JSON object containing:

```json
{
  "ok": true,
  "file_count": 18,
  "payload_count": 13,
  "payload_rows": 22860
}
```

For training, set `DATASET_ROOT` to the dataset directory. The model/checkpoint
override is experiment-arm-specific and remains in that experiment's manifest
or launcher. The code, dataset, model, checkpoint, and log locations do not
need a common parent directory.

The sanitized release receipt is
`docs/joint_training/reports/data/rebuttal_rlvr_hf_public_receipt_20260730.json`.
