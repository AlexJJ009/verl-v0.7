---
name: l40s-resource-relay
description: Transfer Docker image tarballs, mounted-disk contents, checkpoints, model weights, datasets, and other large files between L40S servers through a private-network relay using rsync. Use for L40S-to-L40S environment migration, resource replication, resumable two-hop staging, checksum/inventory verification, or verified relay cleanup. Do not use for non-L40S endpoints, public-network transfer, raw block devices, or direct relay-to-target push.
---

# L40S Resource Relay

Use the bundled script as the only execution path. It enforces L40S endpoints,
private IPs, `rsync --partial --append-verify`, strict SSH host-key checking,
manifest verification, and target-initiated pull.

## Hard boundaries

- Transfer only between two endpoint machines whose local `nvidia-smi` inventory
  contains exclusively NVIDIA L40S GPUs. The relay itself need not be an L40S.
- Use a numeric RFC1918, CGNAT, or IPv6 ULA relay address. Reject public IPs and
  hostnames before opening a connection.
- Move every file, including manifests, with rsync. Use SSH only as rsync's
  transport and for control-plane checks, exact directory creation, capacity
  queries, and guarded staging cleanup.
- Never copy an endpoint private key to the relay. Source L40S pushes to relay;
  target L40S pulls from relay.
- Never accept `/dev/*`, block devices, sockets, FIFOs, or other special files.
  Export a Docker image to a tar file first; transfer mounted filesystem data,
  not a raw disk device.
- Transfer only a quiescent source tree. Stop writers or snapshot the mounted
  filesystem before `push`; the post-rsync checksum gates fail if content moves.
- Never delete the source. Delete one relay transfer only after a target-side
  verified receipt matches the relay manifest and the operator repeats the
  exact transfer ID.
- Run `push` and `pull` inside tmux. Do not treat an rsync exit code alone as
  end-to-end verification.

## One-time private setup

Copy `assets/l40s-resource-relay.env.example` to the default private location:

```bash
install -d -m 700 ~/.config/verl
install -d -m 700 /data-1/model_weights/logs/l40s-resource-relay
install -m 600 \
  .claude/skills/l40s-resource-relay/assets/l40s-resource-relay.env.example \
  ~/.config/verl/l40s-resource-relay.env
```

Replace every placeholder locally. Do not place a populated config, IP address,
private key, or generated receipt in Git. A dedicated key should be restricted
on the relay to the staging account/root needed by this workflow.

## Two-hop workflow

Set the helper path once in the current shell:

```bash
RELAY_TOOL=.claude/skills/l40s-resource-relay/scripts/l40s_resource_relay.py
python3 "$RELAY_TOOL" preflight
```

On the source L40S, generate a stable transfer ID and push one regular file or
directory. Use the same ID for retries. Record the printed `manifest_sha256`
through a control channel other than the relay:

```bash
tmux new-session -d -s relay-push-TRANSFER_ID \
  "umask 077; python3 '$RELAY_TOOL' push --transfer-id TRANSFER_ID --source /absolute/path 2>&1 | tee /data-1/model_weights/logs/l40s-resource-relay/push-TRANSFER_ID.log"
```

On the target L40S, install a private config/key that can read the same relay
staging root, then pull actively:

```bash
tmux new-session -d -s relay-pull-TRANSFER_ID \
  "umask 077; python3 '$RELAY_TOOL' pull --transfer-id TRANSFER_ID --expected-manifest-sha256 SOURCE_DIGEST --destination /absolute/destination 2>&1 | tee /data-1/model_weights/logs/l40s-resource-relay/pull-TRANSFER_ID.log"
```

The restored object is `/absolute/destination/<source-basename>`. An interrupted
pull may resume only when its private in-progress receipt binds the same transfer
ID, destination, and manifest digest. Inspect the final target receipt, then
remove only this verified staging directory:

```bash
python3 "$RELAY_TOOL" cleanup \
  --transfer-id TRANSFER_ID \
  --confirm-transfer-id TRANSFER_ID
```

Use `--config /absolute/private.env` to select another private profile. Environment
variables override values loaded from the file. Use `status --transfer-id ID` to
check whether staging and its completed manifest exist without transferring data.

## Completion evidence

Report all of the following before calling a migration complete:

1. source manifest digest, file count, and total bytes;
2. source-to-relay checksum dry-run with zero itemized changes;
3. target-side full manifest verification and receipt path;
4. relay cleanup result, if cleanup was requested;
5. tmux session/log location for long-running work.

If any gate fails, preserve staging and receipts for diagnosis. Do not widen the
skill to a public address or a non-L40S server to work around the failure.
Never obtain `SOURCE_DIGEST` from the relay or the payload being verified; that
would collapse the end-to-end integrity check back into relay trust.
