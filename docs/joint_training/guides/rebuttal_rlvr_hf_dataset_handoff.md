# Rebuttal RLVR Hugging Face dataset handoff

> **Internal publisher/migration record.** The proxy, Mihomo, credential, and
> history-rewrite details in this file describe this machine's operator-side
> controls. They are not requirements for a colleague downloading a public
> repository. The colleague-facing contract is
> `docs/joint_training/guides/rebuttal_rlvr_hf_public_consumer_handoff.md` and
> uses the standard public Hugging Face endpoint with a colleague-selected
> `DATASET_ROOT`. Image construction is outside that consumer handoff.

## Current decision

Use the owner-controlled dataset repository
`https://huggingface.co/datasets/AlexGeek/RLdataset`. The upload contract is the
full runnable data set requested for this branch:

- one math RLVR training file;
- Math-7: AIME-2025, MATH-500, AMC23, AQuA, GSM8K, MAWPS, and SVAMP;
- one code RLVR training file;
- Code-4: HumanEval+, MBPP+, LiveCodeBench release_v5, and BigCodeBench.

There is no executable Code-7 contract in this project. Do not turn smoke
subsets, evaluator caches, or multiple metrics over one benchmark file into
extra datasets.

The earlier private handoff used two stages:

1. atomically replace HEAD while the repository is private, then verify every
   uploaded byte;
2. remove the pre-recreate legacy dataset history, verify the new root commit,
   and classify any Hugging Face-retained post-recreate bootstrap/upload SHA by
   its exact content;

Both historical stages are complete. On 2026-07-29 the owner then approved a
new full-13 public release with a different policy: preserve the repository's
current history, append the v4 bundle to the reviewed private HEAD using
`parent_commit`, verify the appended commit while private, and only then change
visibility. The public-release path must not delete/recreate the repository or
call `super_squash_history`. The full data publication completed on 2026-07-30
(Asia/Tokyo) at `b1c264a92ace36dace52babdda651e415d9e9f82`.
A later README-only commit simplified the consumer instructions at
`3d4d0e5f1be6dad9de2613d6caf88f197ec78044`; it preserved every payload byte
and both earlier revisions. Reader testing then corrected the validator output
description and `DATASET_ROOT` wording at
`5c3ce2d6a3b5ca61c60febccf202e7ee9d2615f8`, again without changing payloads.

The 13 payloads in v4 are the exact Parquet files consumed by the project's
training and evaluation workflows. The publication builder copies their bytes
and SHA-256 values; it does not create a separate publication format.

## Verified public release

The authenticated and credential-free audits after publication observed:

| Field | Value |
|---|---|
| repository | `AlexGeek/RLdataset` |
| public HEAD | `5c3ce2d6a3b5ca61c60febccf202e7ee9d2615f8` |
| reader-fix parent | `3d4d0e5f1be6dad9de2613d6caf88f197ec78044` |
| README-update parent | `b1c264a92ace36dace52babdda651e415d9e9f82` |
| original preserved parent | `da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c` |
| visibility | public |
| gated | false |
| files | 18 |
| payloads / rows | 13 / 22,860 |
| commits on `main` | exactly `[5c3ce2d6..., 3d4d0e5f..., b1c264a..., da622cf...]` |
| anonymous download and validator | PASS |
| automatic convert ref | `refs/convert/parquet` at one root commit `70f194360a421b036709efe81d4288363f7bb30d` |
| other branches/tags/PR refs | none |
| write admission | verified for the isolated upload credential |

Do not copy a token into Git, shell history, tmux logs, receipts, or this guide.
The publisher uses the credential already stored in the selected `HF_HOME`.

An ordinary `delete_patterns="*"` commit removes old files from the new HEAD
but does not remove old revisions. The earlier private migration used the
delete/recreate fallback while private, followed by a byte-verified upload and
`HfApi.super_squash_history(branch="main")`. This is historical context only;
the approved v4 publication preserves the current repository and history.

The four pre-recreate legacy revisions are now unreachable through both the Hub
API and `/resolve/`:

```text
04668b0284dbef3f5aad51bf570a46416d09287d
df19d512f4306aee8c3abce0387f968b09a5b234
bf278c1214db9ea8d3b26e41fb33dca1520a9b07
989d5c978c46fb2b2c5c7b242f0b53e95e91ddd2
```

Hugging Face still resolves two post-recreate implementation revisions. They
contain no legacy dataset bytes and are not part of `main` history:

| Revision | Verified content |
|---|---|
| `685c0fcb81f225158f83dacd8ae99c647010bed0` | standard bootstrap only: `.gitattributes` |
| `e0d4f9ea24081e654c33d522ba6b4eed1a82c5a3` | exact same 17-file bundle as current HEAD |

Do not describe this outcome as “every old SHA is 404.” The required invariant
is narrower and stronger: every pre-recreate legacy SHA is 404; any retained
post-recreate SHA is accepted only after its complete tree is proven to be the
safe bootstrap or the exact reviewed bundle.

Hugging Face may automatically create one dataset-view conversion ref after the
parquet upload. The read-only verifier admits it only when all four facts hold:

- its name is exactly `parquet`;
- its full ref is exactly `refs/convert/parquet`;
- `list_repo_commits(revision="refs/convert/parquet")` returns exactly its one
  target commit, with no reachable parent history;
- every generated parquet LFS object is a byte-identical member of the reviewed
  13-payload bundle, `.gitattributes` is exact, and no other file appears.

The current conversion ref contains 12 reviewed payload objects plus
`.gitattributes`; Hugging Face omitted AIME-2025 from the generated viewer ref.
It contains no unknown or legacy object.

Any other convert ref, additional branch, tag, or PR ref remains a hard failure.

## Reviewed full bundle

The materialized candidate is:

```text
/data-1/tmp/verl_agent_scratch/rlvr_full_upload_candidate_20260729_v3
```

The read-only verifier and downloader pin both metadata files:

| File | SHA-256 |
|---|---|
| `metadata/publication_inventory.json` | `b5b646a28b2e6bf8a6f531f986d921fbc20e5dc7c454453c3c7ce12a2674aa5a` |
| `metadata/checksums.sha256` | `5e35eab998946be30857425525641b72bc7a1937376f7b797b45d47e71a73a59` |

The inventory's
`observed_parent_commit=df19d512f4306aee8c3abce0387f968b09a5b234` is an immutable build-time
observation from before delete/recreate. It is not a current parent or download
revision. Use `da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c` only for authorized
private-v3 verification; public consumers must use the later v4 commit that
passes anonymous verification.

The candidate contains exactly 13 payloads, plus `.gitattributes`, `README.md`,
and the two metadata files. `metadata/checksums.sha256` covers every file except
itself; the publisher pins and verifies the manifest too, so the remote
exact-file contract contains 17 files.

```text
README.md
.gitattributes
metadata/
  publication_inventory.json
  checksums.sha256
data/
  math/train_rl_format.parquet
  math7/
    AIME-2025/aime-2025_with_system_prompt.parquet
    MATH-500/math500-test_with_system_prompt.parquet
    AMC23/amc23-test_with_system_prompt.parquet
    AQUA/aqua-test_with_system_prompt.parquet
    gsm8k/gsm8k-test_with_system_prompt.parquet
    MAWPS/mawps-test_with_system_prompt.parquet
    SVAMP/svamp-test_with_system_prompt.parquet
  code/verl_rl/
    kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet
    online_full_humaneval_plus/official_humaneval_plus_val.parquet
    online_full_mbpp_plus/official_mbpp_plus_val.parquet
    online_full_livecodebench_v5/official_livecodebench_val.parquet
    online_full_bigcodebench/official_bigcodebench_val.parquet
```

The large LiveCodeBench SQLite evaluator index and evaluator source checkouts
are not dataset payloads. Evaluator source stays pinned in the training/evaluator
images; full BigCodeBench evaluation remains in its Python 3.10 evaluator image.

## Internal publisher/private-migration network route

On this publishing host only, every Hub API, upload, resolve probe, and download
must use:

```text
HF_ENDPOINT=https://huggingface.co
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```

The live route check resolves Hugging Face through
`大流量 -> [BW] 香港非家宽具体 leaf`. It rejects every node whose name contains
`家宽`, `住宅`, `residential`, or `home broadband`. Merely exporting proxy
variables is not enough because port 7890 is shared. Before and after every Hub
request, both guarded scripts verify the live Mihomo controller, ordered
Hugging Face/Xet/LFS rules, the exact `大流量` selector, membership in the
runtime `大流量` selector group and its controller projection, and one concrete
runtime proxy entry with non-empty
server/port/type. The stable admission fingerprint covers the full runtime proxy
identity, including server and port, rather than only hashing the displayed
node name. Missing controller/selector-projection evidence, an unknown leaf, a
non-Hong-Kong leaf, or any residential marker fails before data is accepted.

Both scripts reject a non-official `HF_ENDPOINT`, overwrite upper- and
lower-case proxy variables, narrow `NO_PROXY` to localhost, and disable Xet
and HF-transfer subprocess transports so downloads stay in the audited HTTP
client. Hub metadata calls are bounded at 30 seconds and download reads at 120
seconds. `huggingface_hub 1.8.x` uses an audited `httpx.Client` factory and
`0.36.x` uses an audited `requests.Session`; other backend versions fail closed.
Each actual request, including redirect/CDN hops, is admitted before send and
again from the response request URL. A failed Hub call still runs the outer
post-operation admission. The shell
commands below set the same variables explicitly.

The shared `127.0.0.1:7890` selector still has one disclosed TOCTOU limit: a
different process could switch the selector during a single request and switch
it back before the post-request sample. Run transfers in a controlled tmux
session. If every connection must be provably pinned with no shared-selector
race, use a dedicated Mihomo instance/port bound to one audited Hong Kong
non-residential leaf.

## Stage 0: completed-state read-only preflight

The checked-in command is now structurally read-only. It pins the completed
private revision and has no upload, delete, history-rewrite, reconcile, or
visibility-mutation option.

```bash
set -euo pipefail
cd /data-1/code/worktrees/verl-rebuttal-rlvr

export HF_PROXY=http://127.0.0.1:7890
export HF_AUTH_HOME=/data-1/.cache/huggingface-upload-dataset
export PRIVATE_BUNDLE=/data-1/tmp/verl_agent_scratch/rlvr_full_upload_candidate_20260729_v3
export COMPLETED_REVISION=da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c

env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACEHUB_API_TOKEN -u HF_TOKEN_PATH \
  HF_ENDPOINT=https://huggingface.co \
  HTTP_PROXY="$HF_PROXY" HTTPS_PROXY="$HF_PROXY" \
  http_proxy="$HF_PROXY" https_proxy="$HF_PROXY" \
  HF_HOME="$HF_AUTH_HOME" \
  /data-1/miniconda3/bin/python \
    scripts/publish_rebuttal_rlvr_public_dataset.py \
    --bundle "$PRIVATE_BUNDLE" \
    --repo-id AlexGeek/RLdataset \
    --revision "$COMPLETED_REVISION"
```

Run this full remote-byte audit inside tmux. Before constructing a Hub client it
admits the Hong Kong non-residential `大流量` route. It then checks the exact
private HEAD, full bundle hashes, one-root `main`, the narrowly admitted parquet
conversion content, all 17 remote bytes, the four purged legacy revisions, and
the two classified safe retained revisions. If HEAD changes, audit the new state
instead of substituting a floating `main`.

## Stages 1 and 2: private upload, verification, and history purge

The current upload and history cleanup are already complete. **Do not rerun the
upload command for the same bundle.** The immutable commit for authorized
private-v3 verification is:

```text
da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c
```

All 17 remote files at this revision were downloaded through the admitted route
and checked against `metadata/checksums.sha256`; the manifest and inventory
pins match the values above. The final repository remains private and `main`
has exactly one root commit. `publish_receipt.json` does not exist because the
public transition has not run. The sanitized, checked-in private receipt is
`docs/joint_training/reports/data/rebuttal_rlvr_hf_private_receipt_20260729.json`.

## Private-v3 completed-state mutation is disabled

The upload and history-rewrite implementation used for the completed migration
has been removed from the current operational script. Its CLI rejects
`--apply`, `--reconcile`, `--expected-parent`, and repository-visibility
options. Git history preserves the old recovery implementation for forensic
reference, but it must not be run against the completed `da622cf...` state.
The separate v4 publisher described below is the only approved mutation path.

## Full-13 public publication: completed v4 bundle

The v3 `private_handoff_only` guarded publisher remains private-only. The
separate v4 publisher was the only approved path used for the public
transition.

The new executable v4 publisher is
`scripts/publish_rebuttal_rlvr_full_dataset_v4.py`; its anonymous verifier is
`scripts/verify_rebuttal_rlvr_public_release.py`. They are bound to the v4
bundle hashes below and implement the preserve-history state machine. They have
passed local static and fake-state tests and completed the remote publication
and anonymous verification.

The repository owner approved publication of all 13 payloads on 2026-07-29.
The final pre-upload v4 bundle is:

```text
/data-1/tmp/verl_agent_scratch/rlvr_full_public_release_20260730_v4r3
publication_status=owner_approved_for_public_release
files=18
payloads=13
rows=22860
metadata/publication_inventory.json=fe90ad41b1abbf08c3bbd17f9638954ba9b15b0dcf916b3edcfa62d24b95d130
metadata/checksums.sha256=26cc2d7395e3aceb1f71ea44e150e3a458d285591766c9ad688c44efa604d394
```

It replaces the private-only README with the public consumer README, adds
`validate_dataset.py`, removes host paths from the published inventory, and
passes exact tree, SHA-256, row-count, Arrow
schema, and all-row semantic validation. The README cites every upstream
dataset and records its declared license or terms. The exact bundle was uploaded
at `b1c264a92ace36dace52babdda651e415d9e9f82`. Do not rebuild or republish it
for the existing release; regenerate all pins and use a new commit if any
README, validator, decision receipt, source record, or builder input changes.

The owner subsequently requested that the README remove suggested
`WORK_ROOT`/sibling directory planning and explain only arbitrary download,
launcher path overrides, and validation. The append-only README update changed
exactly `README.md` and `metadata/checksums.sha256`; all 13 Parquet SHA-256
values and `metadata/publication_inventory.json` remained unchanged. Its
first verified revision was `3d4d0e5f1be6dad9de2613d6caf88f197ec78044`.
Reader testing then aligned the documented validator output with its real JSON
output and removed ambiguity around the launcher override. The final verified
release is:

```text
revision=5c3ce2d6a3b5ca61c60febccf202e7ee9d2615f8
metadata/publication_inventory.json=fe90ad41b1abbf08c3bbd17f9638954ba9b15b0dcf916b3edcfa62d24b95d130
metadata/checksums.sha256=62bfaed9b1530af3f504e846ef84454cf771ad9673598a9e1bbf6e8e8c8b64cd
changed_paths=README.md,metadata/checksums.sha256
parquet_payloads_changed=false
anonymous_validation=PASS
```

The v4 release state machine is:

1. verify the exact current private HEAD, its 17 files, ordered history, and
   refs;
2. call `create_commit(revision="main", parent_commit=da622cf...)` with the
   exact 18-file v4 allowlist and no delete operation;
3. verify the new private HEAD, every byte, and history
   `[new_public_commit, da622cf...]`;
4. set `private=false, gated=false` without changing HEAD;
5. verify the same commit authenticated and then download it anonymously into
   an empty credential-free directory;
6. run the bundled validator and write the immutable public receipt.

All six phases passed. The checked-in sanitized receipt is
`docs/joint_training/reports/data/rebuttal_rlvr_hf_public_receipt_20260730.json`.
The operator receipt remains outside Git because it contains local paths and
publisher-side network-routing evidence that public consumers do not need.

Any failure before visibility changes leaves the repository private. A failure
in authenticated or anonymous public verification triggers a return to
`private=true` followed by the same full tree/history verification. The
publisher contains no `delete_repo`, `create_repo`, `super_squash_history`, or
force-history path.

## Historical private staging-host mapping (do not send to consumers)

The commands below preserve the completed private migration record for this
machine. They hard-code one operator's DolphinFS layout, token home, and local
Mihomo route, so they must not be copied into the public colleague handoff.
Public consumers choose their own paths and use
`rebuttal_rlvr_hf_public_consumer_handoff.md`.

```bash
set -euo pipefail
export STORAGE_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chenzehao07
export ROOT="$STORAGE_ROOT"
export REPO_ROOT="$ROOT/wdl/WDL-SFT/verl-rebuttal-rlvr"
export DATASET_ROOT="$ROOT/huggingface/dataset/RLdataset"
export EVALUATOR_ASSET_ROOT="$ROOT/huggingface/evaluator_assets/rebuttal_rlvr"
export MODEL_ROOT="$ROOT/huggingface.co"
export STATE_ROOT="$ROOT/wdl/WDL-SFT/state/rebuttal_rlvr"
export MIHOMO_RUNTIME_CONFIG=/root/clashctl/resources/runtime.yaml
export HF_TRANSFER_PYTHON=${HF_TRANSFER_PYTHON:-python3}
HF_TRANSFER_PYTHON=$(command -v "$HF_TRANSFER_PYTHON")
export HF_TRANSFER_PYTHON
test -x "$HF_TRANSFER_PYTHON"
"$HF_TRANSFER_PYTHON" -c 'import huggingface_hub, yaml'

: "${STAGING_SECRET_ROOT:?set an operator-only absolute path outside ROOT}"
STAGING_SECRET_ROOT=$(realpath -m "$STAGING_SECRET_ROOT")
case "$STAGING_SECRET_ROOT" in
  "$ROOT"|"$ROOT"/*) echo "STAGING_SECRET_ROOT must be outside ROOT" >&2; exit 2 ;;
esac
export HF_AUTH_HOME="$STAGING_SECRET_ROOT/huggingface-auth"
```

`HF_AUTH_HOME` is a staging-host secret path. It must not be placed under
`STATE_ROOT`, copied to DolphinFS, included in receipts, or mounted into an
offline worker. Provision the token file locally without making an unguarded
`hf auth login` API call and without putting the token on the command line or in
shell history:

```bash
install -d -m 700 "$STAGING_SECRET_ROOT" "$HF_AUTH_HOME"
HF_TOKEN_PATH="$HF_AUTH_HOME/token" "$HF_TRANSFER_PYTHON" - <<'PY'
import getpass
import os
from pathlib import Path

path = Path(os.environ["HF_TOKEN_PATH"])
if path.exists() or path.is_symlink():
    raise SystemExit(f"refusing to replace existing credential: {path}")
token = getpass.getpass("Hugging Face read token: ").strip()
if not token:
    raise SystemExit("empty token refused")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(fd, token.encode("utf-8"))
finally:
    os.close(fd)
PY
chmod 700 "$HF_AUTH_HOME"
chmod 600 "$HF_AUTH_HOME/token"
```

The downloader rejects ambient token variables, a token path outside this
`HF_HOME`, wrong ownership, group/world-readable permissions, and an empty
token. `EVALUATOR_ASSET_ROOT` is separate from `DATASET_ROOT`: the Hub
download owns the latter as an exact 17-file tree, while evaluator caches and
source receipts are provisioned independently under the former.

The downloaded repository preserves the handoff `data/...` relative layout.
The math training file is
`$DATASET_ROOT/data/math/train_rl_format.parquet`; code files stay under
`$DATASET_ROOT/data/code/verl_rl/`.

An authorized consumer can use the guarded wrapper now while the repository is
private. Do not use a bare `hf download`. The wrapper admits
`大流量 -> [BW] 香港非家宽 leaf` before and after every Hub API/file request,
rejects identity drift, downloads into a sibling partial directory, removes the
Hub-local cache, verifies exactly 17 files, and only then renames it to
`DATASET_ROOT`.

```bash
set -euo pipefail
: "${REPO_ROOT:?}"
: "${DATASET_ROOT:?}"
: "${STATE_ROOT:?}"
: "${HF_AUTH_HOME:?}"
: "${HF_TRANSFER_PYTHON:?}"
test -d "$REPO_ROOT"
test ! -e "$DATASET_ROOT"
test -d "$HF_AUTH_HOME"
test -f "$HF_AUTH_HOME/token"
mkdir -p "$STATE_ROOT/hf_receipts"

export DOWNLOAD_STATUS="$STATE_ROOT/hf_receipts/RLdataset-da622cf077ca.status"
export DOWNLOAD_RECEIPT="$STATE_ROOT/hf_receipts/RLdataset-da622cf077ca.json"
export DOWNLOAD_LOG="$STATE_ROOT/hf_receipts/RLdataset-da622cf077ca.log"
test ! -e "$DOWNLOAD_STATUS"
test ! -e "$DOWNLOAD_RECEIPT"
test ! -e "$DOWNLOAD_LOG"
! tmux has-session -t hf-rlvr-download 2>/dev/null

tmux new-session -d -s hf-rlvr-download \
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACEHUB_API_TOKEN -u HF_TOKEN_PATH \
      HF_ENDPOINT=https://huggingface.co \
      HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
      http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
      HF_HOME="$HF_AUTH_HOME" MIHOMO_RUNTIME_CONFIG="$MIHOMO_RUNTIME_CONFIG" \
      HF_TRANSFER_PYTHON="$HF_TRANSFER_PYTHON" REPO_ROOT="$REPO_ROOT" \
      DATASET_ROOT="$DATASET_ROOT" DOWNLOAD_STATUS="$DOWNLOAD_STATUS" \
      DOWNLOAD_RECEIPT="$DOWNLOAD_RECEIPT" DOWNLOAD_LOG="$DOWNLOAD_LOG" \
  bash -lc '
    set -euo pipefail
    record_status() {
      status=$?
      trap - EXIT
      status_tmp="${DOWNLOAD_STATUS}.tmp.$$"
      printf "%s\n" "$status" >"$status_tmp"
      mv "$status_tmp" "$DOWNLOAD_STATUS"
      exit "$status"
    }
    trap record_status EXIT
    cd "$REPO_ROOT"
    "$HF_TRANSFER_PYTHON" scripts/download_rebuttal_rlvr_hf_dataset.py \
      --repo-id AlexGeek/RLdataset \
      --revision da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c \
      --local-dir "$DATASET_ROOT" \
      --receipt "$DOWNLOAD_RECEIPT" \
      2>&1 | tee "$DOWNLOAD_LOG"
  '
```

Detached tmux creation is not completion. After the session exits, require the
status file and receipt before handing the data to a worker:

```bash
while tmux has-session -t hf-rlvr-download 2>/dev/null; do sleep 5; done
test -f "$DOWNLOAD_STATUS"
test "$(cat "$DOWNLOAD_STATUS")" = 0
jq -e '
  .revision == "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c"
  and .remote_private == true
  and .remote_gated == false
  and .file_count == 17
  and .authentication_mode == "explicit_operator_hf_home_token_private_only"
  and .route_admission.schema_version == 2
  and .route_admission.route_group == "大流量"
  and .route_admission.route_group_type == "Selector"
  and .route_admission.runtime_group_type == "select"
  and .route_admission.selected_namespace == "BW"
  and .route_admission.selected_region == "Hong Kong"
  and .route_admission.selected_residential == false
  and .route_admission.selector_projection_verified == true
  and .route_admission.runtime_group_identity_verified == true
  and .route_admission.runtime_proxy_identity_verified == true
  and (.route_admission.connection_hosts_verified == .observed_connection_hosts)
' "$DOWNLOAD_RECEIPT"
test "$(find "$DATASET_ROOT" -type f | wc -l)" = 17
```

Bind the immutable Hub commit plus the download receipt in the Meituan
experiment manifest. Run this only on the networked staging host that owns the
audited Mihomo configuration, never inside an offline Meituan worker. If the
colleague's host does not expose the same controller contract, transfer the
already verified flat directory and receipt through the approved internal
channel; do not disable the route gate.

The current v3 guarded downloader has no anonymous mode and always rejects
public visibility; it remains private-v3-only. Public v4 consumers use the
standard credential-free `hf download` command and then run the bundled
`validate_dataset.py`, as documented in
`rebuttal_rlvr_hf_public_consumer_handoff.md`.

## Repository source

```text
main repo:   https://github.com/AlexJJ009/verl-v0.7.git
recipe repo: https://github.com/AlexJJ009/verl-recipe.git
branch:      codex/rebuttal-rlvr (both repositories)
```

Clone through the superproject submodule workflow in
`docs/joint_training/guides/meituan_rlvr_image_build.md`. Pin the delivered main
commit and recipe gitlink rather than using a floating branch at job runtime.
