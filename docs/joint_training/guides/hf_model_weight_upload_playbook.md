# Hugging Face Model Weight Upload Playbook

Status: mid-goal snapshot, captured 2026-05-31 20:10 Asia/Shanghai while the 16-item checkpoint migration queue was still running.

This guide records the working procedure for converting completed verl checkpoints into Hugging Face model weights, uploading them, registering proof, and deleting only verified local checkpoint actors. It intentionally does not store subscription tokens or Hugging Face credentials.

## Current Proven Defaults

- Run long transfers in tmux. Use one tmux session per queue shard and one cleanup watcher session.
- Use the existing migration script instead of rewriting the upload flow:
  `/data-1/model_weights/scripts/run_first_second_migration_20260528.sh`.
- Use proxy mode for Hugging Face uploads:
  `HF_NETWORK_MODE=proxy` with HTTP proxy `http://127.0.0.1:7890`.
- Keep `HF_HUB_DISABLE_XET=1`. The upload path observed here uses Hugging Face LFS/S3 endpoints, not Xet.
- Use two queue shards for parallelism when disk has enough room:
  `/data-1/model_weights/migration_queue_third_20260531_part0.json`
  and `/data-1/model_weights/migration_queue_third_20260531_part1.json`.
- Use the verified-only cleanup helper after remote verification:
  `/data-1/model_weights/scripts/delete_verified_checkpoints.py`.
- Cleanup must wait until the migration script records `cleanup_merged ok`; deleting earlier can break the script's own `audit-delete` phase.

## Proxy Configuration

The large-traffic subscription is configured outside this repo under `/root/clashctl`. Do not commit subscription URLs or tokens into project docs.

Important files:

- `/root/clashctl/.env`
- `/root/clashctl/scripts/merge-subs.sh`
- `/root/clashctl/resources/mixin.yaml`
- `/root/clashctl/resources/runtime.yaml`

Validated facts from this run:

- Current mihomo is `Mihomo Meta v1.19.17`.
- The new large-traffic subscription is Clash YAML and includes both `anytls` and `vless` nodes.
- `mihomo -t` accepts the generated config, so the current local mihomo supports the subscription's `anytls` nodes.
- Hugging Face large model upload traffic goes to `hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`.
- HF/GitHub large-transfer rules must appear before broad subscription rules such as `DomainKeyword(amazon)`, otherwise HF LFS can be routed to the wrong group.
- Keep Tailscale/private routing direct:
  `IP-CIDR,100.64.0.0/10,TAILSCALE-DIRECT,no-resolve`
  and `IP-CIDR,100.100.0.0/10,TAILSCALE-DIRECT,no-resolve`.

During this run, `大流量` was changed from `url-test` to `select` and fixed to `[BW] 🇭🇰 香港_03 | 家宽` because URLTest selected `[BW] 🇰🇷 韩国-首尔`, which had poor HF LFS throughput. For future large HF transfers, prefer a known-good large-traffic node over automatic latency-only selection.

Validate routing before and during upload:

```bash
curl -x http://127.0.0.1:7890 -I https://huggingface.co
curl -x http://127.0.0.1:7890 -I https://github.com

tmp=$(mktemp)
curl -sS --max-time 8 http://127.0.0.1:9090/connections \
  -H 'Authorization: Bearer gcgliR' -o "$tmp"
python3 - "$tmp" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read())
print("totals upload/download", d.get("uploadTotal"), d.get("downloadTotal"), "nconn", len(d.get("connections", [])))
for c in d.get("connections", []):
    meta = c.get("metadata") or {}
    host = meta.get("host") or ""
    if "hf-hub-lfs" in host or "huggingface" in host:
        print(meta.get("sourcePort"), host, c.get("chains"), "up", c.get("upload"), "down", c.get("download"))
PY
rm -f "$tmp"
```

Expected chain for HF LFS:

```text
hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com -> [..., 大流量]
```

## Launch Pattern

Start the two queue shards in tmux. Use separate `CUDA_VISIBLE_DEVICES` only for the merge step; uploads are network-bound.

```bash
tmux new-session -d -s hf_migration_third_p0 \
  "QUEUE=/data-1/model_weights/migration_queue_third_20260531_part0.json \
   STATUS_JSONL=/data-1/model_weights/logs/third_migration_20260531_p0.status.jsonl \
   RUN_LOG=/data-1/model_weights/logs/third_migration_20260531_p0.log \
   CUDA_VISIBLE_DEVICES=0 \
   HF_NETWORK_MODE=proxy \
   bash /data-1/model_weights/scripts/run_first_second_migration_20260528.sh"

tmux new-session -d -s hf_migration_third_p1 \
  "QUEUE=/data-1/model_weights/migration_queue_third_20260531_part1.json \
   STATUS_JSONL=/data-1/model_weights/logs/third_migration_20260531_p1.status.jsonl \
   RUN_LOG=/data-1/model_weights/logs/third_migration_20260531_p1.log \
   CUDA_VISIBLE_DEVICES=1 \
   HF_NETWORK_MODE=proxy \
   bash /data-1/model_weights/scripts/run_first_second_migration_20260528.sh"
```

Run the cleanup watcher in a third tmux session:

```bash
tmux new-session -d -s hf_cleanup_verified_watch \
  "python3 /data-1/model_weights/scripts/watch_delete_after_cleanup.py \
    --status-jsonl /data-1/model_weights/logs/third_migration_20260531_p0.status.jsonl \
    --status-jsonl /data-1/model_weights/logs/third_migration_20260531_p1.status.jsonl \
    --interval 120 2>&1 | tee -a /data-1/model_weights/logs/delete_after_cleanup_20260531.log"
```

## Monitoring

Summarize queue status:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for p in [Path('/data-1/model_weights/logs/third_migration_20260531_p0.status.jsonl'), Path('/data-1/model_weights/logs/third_migration_20260531_p1.status.jsonl')]:
    print('---', p.name)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    by = {}
    for r in rows:
        by.setdefault(r.get('artifact_id', '?'), []).append(r)
    for aid, rs in by.items():
        last = rs[-1]
        upload = [r for r in rs if r.get('phase') == 'upload' and r.get('result') == 'ok']
        verify = [r for r in rs if r.get('phase') == 'verify_upload' and r.get('result') == 'ok']
        cleanup = [r for r in rs if r.get('phase') == 'cleanup_merged' and r.get('result') == 'ok']
        fail = [r for r in rs if r.get('result') == 'fail']
        print(aid, 'last', last.get('phase'), last.get('result'), 'upload_ok', len(upload), 'verify_ok', len(verify), 'cleanup', len(cleanup), 'fail', len(fail))
        if upload:
            print('  commit', upload[-1].get('details'))
        if fail:
            print('  last_fail', fail[-1].get('phase'), fail[-1].get('details'))
PY
```

Check sessions and disk:

```bash
tmux list-sessions | rg 'hf_migration|hf_cleanup' || true
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | rg 'hf upload|model_merger|run_first_second_migration|watch_delete_after_cleanup' | rg -v rg || true
df -h /data-1
du -sh /data-1/checkpoints /data-1/model_weights/merged /data-1/model_weights 2>/dev/null
```

## Retry And Node Selection

Transient failures observed in this run:

- `SSL: UNEXPECTED_EOF_WHILE_READING`
- `RuntimeError: Cannot send a request, as the client has been closed`
- `httpx.RemoteProtocolError: peer closed connection without sending complete message body`
- `RuntimeError: Error while uploading 'model-00001-of-00004.safetensors' to the Hub`

The migration script retries `hf upload` up to `HF_UPLOAD_RETRIES` attempts. Do not call an artifact failed unless the status JSONL records `upload fail` or the tmux session exits.

If a single `hf upload` attempt stalls with no LFS byte growth:

1. Confirm it is still in `upload start`.
2. Sample controller connections twice, about 30 seconds apart.
3. If the same LFS connections have unchanged `upload` bytes, terminate only the `hf upload` child process.
4. Let the parent migration script enter its retry sleep and restart the upload attempt.

Do not kill the parent queue session unless the queue script has actually failed or needs a deliberate parameter change.

If URLTest picks a slow node, fix `大流量` to a known-good selector node. During the mid-goal run, `[BW] 🇭🇰 香港_03 | 家宽` was faster and more stable than `[BW] 🇰🇷 韩国-首尔` for HF LFS.

## Deletion Safety

Never delete checkpoints directly from queue state alone. Deletion requires all of:

- manifest exists under `/data-1/model_weights/manifests`;
- manifest has `upload.verified == true`;
- migration status has `cleanup_merged ok` for that artifact;
- `model_weight_manager.py audit-delete` passes;
- deletion target resolves under `/data-1/checkpoints`;
- target is the intended `actor` directory, unless a broader scope was explicitly approved.

Manual deletion command when needed:

```bash
python3 /data-1/model_weights/scripts/delete_verified_checkpoints.py --dry-run /data-1/model_weights/manifests/<artifact_id>.json
python3 /data-1/model_weights/scripts/delete_verified_checkpoints.py /data-1/model_weights/manifests/<artifact_id>.json
```

## Mid-Goal Results So Far

At the time this snapshot was written:

- `GRPO-Qwen3-4B-MATH-2G-MATHDATA-SFT-E1-step115` uploaded, verified, registered, added to collection, merged temp cleaned, actor deleted.
- `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-2H-MATHDATA-BASE-E1-step115` uploaded, verified, registered, added to collection, merged temp cleaned, actor deleted.
- `WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1D-MATHDATA-E1-step115` uploaded, verified, registered, added to collection, merged temp cleaned, actor deleted.
- `WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS-step115` uploaded, verified, registered, added to collection, merged temp cleaned, actor deleted.
- `/data-1/checkpoints` dropped from about `743G` to about `465G`.
- `/data-1` free space rose from about `143G` before this cleanup effort to about `405G`.

## Final Update Placeholder

After the current 16-item queue completes, update this guide with:

- final successful artifact list and commit links;
- final reclaimed checkpoint space;
- final proxy node choice and whether it remained stable;
- any script changes needed beyond this mid-goal snapshot;
- recommended default queue parallelism for the next batch.
