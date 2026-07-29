# Rebuttal RLVR Hugging Face dataset handoff

## Decision

Reuse `beichenhang/EnsembleLLM-data`. The 2026-07-29 engineering public-data
gate has passed for exactly the five allowlisted assets below and blocks the ten
restricted assets; the builder and publisher enforce that exact split. Replace
the default-branch HEAD only after write access is granted and the live parent
commit is rechecked. Do not create another dataset repo unless the owner cannot
grant write access or old history must be removed.

Use four independent paths under one colleague-owned storage boundary:

```text
STORAGE_ROOT  security and ownership boundary
REPO_ROOT     immutable clean Git checkout
DATASET_ROOT  downloaded HF dataset plus authorized private overlay
MODEL_ROOT    flat model exports
STATE_ROOT    checkpoints, eval, logs, offline W&B, receipts, caches, registry
```

`ROOT` in the Hope manifest remains the `STORAGE_ROOT` containment boundary.
`DATASET_ROOT`, `MODEL_ROOT`, and `STATE_ROOT` are separately manifest-bound and
must be strict descendants of it. This keeps the checkout clean and avoids
forcing a colleague's existing model/data layout into this project's state
directory.

## Historical repository audit

Read-only HF API audit on 2026-07-29, re-sampled immediately before this
handoff, found:

| Field | Observed value |
|---|---|
| dataset repo | `beichenhang/EnsembleLLM-data` |
| HEAD | `fabf5cb2d41103c4b6f004a7506e8505b87a8aa5` |
| visibility | public already (`private=false`) |
| access gate | `gated=auto` |
| files at HEAD | 74 |
| last modified | `2026-04-01T03:40:59Z` |
| notable legacy file | `am_deepseek_r1_filtered_ad.jsonl`, 2,303,118,015 bytes |
| current local HF identity | `AlexGeek` |
| current local write admission | denied |

The old project upload script independently names the same repository and
defaults to private creation:
`/data-1/code/czh_rl_icml_2026/dataset/upload_to_hf.py`.

The repo therefore does not need a private-to-public transition. It needs:

1. owner/write access;
2. one atomic HEAD replacement;
3. remote byte verification;
4. only then `gated=false` for anonymous download.

Deleting files from the new HEAD does **not** erase them from Git/LFS history.
If the purpose is legal erasure rather than a clean default checkout, reuse is
the wrong operation: the owner must purge history with Hugging Face support or
create a new repo. Do not claim that `delete_patterns="*"` purges old content.

## Public-data gate

The authoritative machine-readable inventory is
`docs/joint_training/reports/data/rebuttal_rlvr_dataset_publication_inventory_20260729.json`.
Its reviewed SHA-256 is
`bf9b56dbd7cbda1f88c56faa2eb0c6142312401522278fc90b56981f9d74bb9b`.
The locally rebuilt `v2` candidate passed exact-file and byte verification; its
`metadata/checksums.sha256` SHA-256 is
`fa5858bf9268756984dbe3f02acc130c902208c5f50ea9b565bcf11e19ebe8e9`.
These hashes admit the bundle contents; they do not grant HF repository write
permission and do not turn this engineering review into legal advice.

### Approved for the public bundle

| Asset | Rows | License basis |
|---|---:|---|
| Hendrycks MATH RLVR train parquet | 7,500 | MATH MIT |
| MATH-500 parquet | 500 | subset/format conversion of MIT MATH; retain MATH attribution and modification notice |
| AQuA parquet | 254 | Apache-2.0 |
| GSM8K parquet | 1,319 | MIT |
| HumanEval+ parquet | 164 | EvalPlus Apache-2.0 plus HumanEval MIT |

### Not approved for a public mirror

| Asset | Why it stays out of this public repo |
|---|---|
| AIME-2025, AMC23 | no MAA redistribution authorization found |
| MAWPS | no data-level redistribution license found |
| SVAMP full records | its MIT contribution is mixed with MAWPS/ASDiv-A records whose full rights chain was not established |
| KodCode-Light-RL-10K train parquet | CC-BY-NC plus mixed Codeforces/LeetCode/TACO/etc. prompts and tests; full redistribution chain not established |
| MBPP+ | EvalPlus license does not establish rights for the underlying MBPP records |
| BigCodeBench JSONL/parquet | source code is Apache-2.0, but a separate license for prompts/reference implementations/tests was not established |
| LiveCodeBench JSONL/parquet/SQLite cache | source code is MIT, but problems/tests come from third-party contest platforms |

This is an engineering publication gate, not a legal opinion. A later written
authorization can move an asset from restricted to public only through a new
inventory revision and review.

Evaluator **source code** stays in the image at fixed commits; it does not need
to be copied into the dataset repo. BigCodeBench full evaluation remains in a
separate Python 3.10 evaluator image. See
`docs/joint_training/guides/meituan_rlvr_image_build.md` for the image pins.

## Public repository layout

The public bundle matches the relative paths consumed by the multi-root
Meituan adapter:

```text
README.md
LICENSES/
  AQuA-Apache-2.0.txt
  EvalPlus-Apache-2.0.txt
  GSM8K-MIT.txt
  HumanEval-MIT.txt
  MATH-MIT.txt
data/
  math/
    train_rl_format.parquet
  math7/
    AQUA/aqua-test_with_system_prompt.parquet
    MATH-500/math500-test_with_system_prompt.parquet
    gsm8k/gsm8k-test_with_system_prompt.parquet
  code/verl_rl/online_full_humaneval_plus/
    official_humaneval_plus_val.parquet
metadata/
  publication_inventory.json
  checksums.sha256
processing/
  prepare_train_rl_format.py
```

The public repo is intentionally not a complete formal Math-7 or code-task
bundle. Authorized private assets must be materialized at the missing relative
paths and bound by the training receipts before a formal job is admitted.

## Build the public bundle locally

Use a new scratch directory; never stage generated data in the Git checkout:

```bash
cd /data-1/code/worktrees/verl-rebuttal-rlvr

/data-1/miniconda3/bin/python \
  scripts/prepare_rebuttal_rlvr_public_dataset.py \
  --verify-only

PUBLIC_BUNDLE=/data-1/tmp/verl_agent_scratch/rebuttal_rlvr_hf_dataset_20260729_v2
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
/data-1/miniconda3/bin/python \
  scripts/prepare_rebuttal_rlvr_public_dataset.py \
  --output "$PUBLIC_BUNDLE"

cd "$PUBLIC_BUNDLE"
sha256sum -c metadata/checksums.sha256
```

The builder copies only the explicit public allowlist, rejects symlinks, checks
size/SHA-256/parquet row count, downloads pinned license bytes, and emits a
sanitized public inventory. It never copies a restricted asset.

## Owner/write-access prerequisite

The current `AlexGeek` credential (stored under the author-side
`HF_HOME=/data-1/.cache/huggingface`) can read but cannot write
`beichenhang/EnsembleLLM-data`. A shell that does not select that `HF_HOME` has
no usable local token, which is also insufficient for publication.
Before publication, the owner must either:

- grant `AlexGeek` write access to that dataset repo; or
- run the checked-in publication command under a `beichenhang` credential.

Do not paste a token into Git, a shell command, a tmux log, or a handoff file.
Use `hf auth login` or the platform secret mechanism, then verify with
`hf auth whoami` and the script's read-only preflight.

## Atomic replacement and remote verification

Keep the repository gated while replacing the old HEAD. Run the upload in
tmux, using the live HEAD as `EXPECTED_PARENT`. The example below uses the
2026-07-29 audit SHA; rerun the read-only audit if HEAD changes.

```bash
cd /data-1/code/worktrees/verl-rebuttal-rlvr

PUBLIC_BUNDLE=/data-1/tmp/verl_agent_scratch/rebuttal_rlvr_hf_dataset_20260729_v2
HF_AUTH_HOME=${HF_AUTH_HOME:-/data-1/.cache/huggingface}
EXPECTED_PARENT=fabf5cb2d41103c4b6f004a7506e8505b87a8aa5
PUBLISH_RECEIPT_DIR=/absolute/persistent/path/hf_dataset_publish/attempt-20260729-v2

# Read-only permission, HEAD, and bundle preflight.
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
HF_HOME="$HF_AUTH_HOME" \
/data-1/miniconda3/bin/python \
  scripts/publish_rebuttal_rlvr_public_dataset.py \
  --bundle "$PUBLIC_BUNDLE" \
  --repo-id beichenhang/EnsembleLLM-data \
  --expected-parent "$EXPECTED_PARENT" \
  --receipt-dir "$PUBLISH_RECEIPT_DIR"

# Mutating operation: only after the preflight says write access is allowed.
tmux new-session -d -s hf-rlvr-dataset-publish \
  "cd /data-1/code/worktrees/verl-rebuttal-rlvr && \
   HTTP_PROXY=http://127.0.0.1:7890 \
   HTTPS_PROXY=http://127.0.0.1:7890 \
   HF_HOME='$HF_AUTH_HOME' \
   /data-1/miniconda3/bin/python \
     scripts/publish_rebuttal_rlvr_public_dataset.py \
     --bundle '$PUBLIC_BUNDLE' \
     --repo-id beichenhang/EnsembleLLM-data \
     --expected-parent '$EXPECTED_PARENT' \
     --receipt-dir '$PUBLISH_RECEIPT_DIR' \
     --confirm-repo-id beichenhang/EnsembleLLM-data \
     --make-ungated-public \
     --apply 2>&1 | tee '$PUBLISH_RECEIPT_DIR.tmx.log'; exec bash"
```

The publisher performs one Hugging Face commit with `delete_patterns="*"`
plus the new folder upload and binds it to `parent_commit`. It then downloads
every file from the new revision, verifies SHA-256, confirms the exact remote
file set, and only then changes settings to `private=false, gated=false`.
`publish_receipt.json` is the release receipt.

## Colleague storage mapping

Use the colleague's existing ownership boundary without reusing another
project's state directory:

```bash
export STORAGE_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chenzehao07
export ROOT="$STORAGE_ROOT"

export REPO_SUBPATH=wdl/WDL-SFT/verl-rebuttal-rlvr
export REPO_ROOT="$ROOT/$REPO_SUBPATH"
export DATASET_ROOT="$ROOT/huggingface/dataset/EnsembleLLM-data"
export MODEL_ROOT="$ROOT/huggingface.co"
export STATE_ROOT="$ROOT/wdl/WDL-SFT/state/rebuttal_rlvr"
```

`STATE_ROOT` is a sibling of the clean checkout, not a directory inside it.
It owns:

```text
checkpoints/rebuttal_rlvr/
eval/rebuttal_rlvr/
logs/rebuttal_rlvr/
wandb_runs/rebuttal_rlvr/
receipts/rebuttal_rlvr/
cache/
experiment_registry/
```

The formal manifest must bind these four roots and all input receipts. It still
rejects paths outside `STORAGE_ROOT`; separating roots is not permission to use
another user's DolphinFS directory.

## Colleague download and materialization

After publication, replace `VERIFIED_DATASET_COMMIT` with the commit in
`publish_receipt.json`:

```bash
mkdir -p "$(dirname "$DATASET_ROOT")"
test ! -e "$DATASET_ROOT"

hf download beichenhang/EnsembleLLM-data \
  --repo-type dataset \
  --revision VERIFIED_DATASET_COMMIT \
  --local-dir "$DATASET_ROOT"

cd "$DATASET_ROOT"
sha256sum -c metadata/checksums.sha256
```

Then add the non-public Math-7/code assets only through an authorized private
transfer, as ordinary files at the exact relative paths in
`metadata/publication_inventory.json`. Recompute the formal train/Math-7/code
receipts after the overlay; do not reuse the public-only checksums as proof of
a complete formal bundle.

## Repository source

```text
main repo:   https://github.com/AlexJJ009/verl-v0.7.git
recipe repo: https://github.com/AlexJJ009/verl-recipe.git
branch:      codex/rebuttal-rlvr (both repositories)
```

Clone both via the superproject submodule workflow documented in
`docs/joint_training/guides/meituan_rlvr_image_build.md`; pin the delivered main
commit and recipe gitlink instead of using a floating branch at job runtime.
