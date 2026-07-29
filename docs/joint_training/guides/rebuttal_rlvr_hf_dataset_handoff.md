# Rebuttal RLVR Hugging Face dataset handoff

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

The current private handoff used two stages:

1. atomically replace HEAD while the repository is private, then verify every
   uploaded byte;
2. remove the pre-recreate legacy dataset history, verify the new root commit,
   and classify any Hugging Face-retained post-recreate bootstrap/upload SHA by
   its exact content;

Both stages are complete. The resulting revision is permanently treated as a
`private_handoff_only` artifact. Its README and inventory explicitly prohibit
public redistribution, and the publisher exposes no public-mutation API or
`--public-transition` option. A successful private upload is not permission to
make the repository public.

Making this repository public later is a new publication, not a visibility
toggle on the current commit. It requires a v4 public-reviewed bundle with new
README/inventory/checksum pins, removal of every private-only payload, private
upload and byte verification, and proof that every revision/ref containing the
current restricted bundle is unreachable before repository visibility changes.

## Live target audit

The post-rebuild read-only audit on 2026-07-29 observed:

| Field | Value |
|---|---|
| repository | `AlexGeek/RLdataset` |
| audited HEAD | `da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c` |
| visibility | private |
| gated | false |
| files | 17 |
| logical bytes | 30,282,521 |
| payloads / rows | 13 / 22,860 |
| commits on `main` | one root commit |
| automatic convert ref | `refs/convert/parquet` at one root commit `70f194360a421b036709efe81d4288363f7bb30d` |
| other branches/tags/PR refs | none |
| write admission | verified for the isolated upload credential |

Do not copy a token into Git, shell history, tmux logs, receipts, or this guide.
The publisher uses the credential already stored in the selected `HF_HOME`.

An ordinary `delete_patterns="*"` commit removes old files from the new HEAD
but does not remove old revisions. This repository therefore used the
delete/recreate fallback while private, followed by a byte-verified upload and
`HfApi.super_squash_history(branch="main")`.

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
revision; use `da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c` for consumers and the
private receipt for current state.

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

## Mandatory network route

Every Hub API, upload, resolve probe, and download must use:

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
upload command for the same bundle.** The immutable data commit for consumers
is:

```text
da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c
```

All 17 remote files at this revision were downloaded through the admitted route
and checked against `metadata/checksums.sha256`; the manifest and inventory
pins match the values above. The final repository remains private and `main`
has exactly one root commit. `publish_receipt.json` does not exist because the
public transition has not run. The sanitized, checked-in private receipt is
`docs/joint_training/reports/data/rebuttal_rlvr_hf_private_receipt_20260729.json`.

## Completed-state mutation is disabled

The upload and history-rewrite implementation used for the completed migration
has been removed from the current operational script. Its CLI rejects
`--apply`, `--reconcile`, `--expected-parent`, and repository-visibility
options. Git history preserves the old recovery implementation for forensic
reference, but it must not be run against the completed `da622cf...` state.
Any future mutation requires a new reviewed change with a new immutable bundle
and a fresh destructive-action review.

## Future public publication: rebuild v4 first

Do not change repository visibility for the current revision. The checked-in
publisher has no `--public-transition` argument and no code path that can set
`private=False` for this v3 `private_handoff_only` bundle.

A future public release in `AlexGeek/RLdataset` must be implemented as a new v4
publication change with all of the following evidence:

1. a new README, inventory, and checksum manifest marked `public_reviewed`, with
   a documented redistribution basis for every included payload;
2. a new bundle that contains no pending-rights payload and a new set of pinned
   metadata hashes in the publisher/downloader;
3. a destructive-action review before removing/recreating or purging the
   existing private repository history;
4. private upload of v4 through the admitted Hong Kong non-residential
   `大流量` route, followed by exact byte, ref, and history verification;
5. API and `/resolve/` proof that `da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c`,
   `e0d4f9ea24081e654c33d522ba6b4eed1a82c5a3`, its conversion ref, and every
   other revision containing private-only bytes are no longer reachable;
6. a new, separately reviewed public-transition implementation bound to the v4
   commit and inventory; only that implementation may call
   `update_repo_settings(private=False, gated=False)`;
7. a fresh anonymous download into an empty directory, with all v4 hashes
   verified through the same route gate.

Reusing `da622cf...`, its README/inventory, or its private receipt for a public
transition is a hard failure. The archival five-payload public-subset design is
not itself a live v4 bundle or publication approval.

## Colleague storage mapping

Keep the colleague's ownership boundary and separate immutable inputs from
experiment state:

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

The current v3 downloader has no anonymous mode and always rejects public
visibility. A future public v4 must ship a separate downloader with a new
commit, inventory, checksum pins, and anonymous-only credential contract; do
not add a boolean anonymous branch back into the v3 entrypoint.

## Repository source

```text
main repo:   https://github.com/AlexJJ009/verl-v0.7.git
recipe repo: https://github.com/AlexJJ009/verl-recipe.git
branch:      codex/rebuttal-rlvr (both repositories)
```

Clone through the superproject submodule workflow in
`docs/joint_training/guides/meituan_rlvr_image_build.md`. Pin the delivered main
commit and recipe gitlink rather than using a floating branch at job runtime.
