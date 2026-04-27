---
name: HF Hub upload workflow
description: How to upload large model checkpoints to Hugging Face Hub from this server (CLI, proxy, excludes, verification)
type: feedback
originSessionId: 93526ecd-e1cf-4c61-a0de-e9615b6058d6
---
When uploading model checkpoints to Hugging Face Hub from this server, use this workflow.

**Why:** First attempt on 2026-04-21 hit two non-obvious issues that wasted a full upload cycle — (1) httpx raising `ImportError: Using SOCKS proxy` even though the system only has `http://` proxies set, blocking `hf` CLI entirely during repo creation; (2) large upload stalling at 98% with one `CLOSE-WAIT` socket and the process hanging on retry-redraws instead of failing cleanly. Second attempt with the workflow below succeeded; xet dedup meant only the un-uploaded 335 MB was retransmitted on retry.

**How to apply:** Any time the user asks to upload a model/checkpoint to HF Hub on this machine.

### CLI
Use the new unified `hf` CLI at `/data-1/miniconda3/bin/hf` (NOT `huggingface-cli`, which isn't installed). `hf auth whoami` confirms login — user is `AlexGeek`, member of `Beihang` org. No `~/.cache/huggingface/token` file; auth is cached internally.

### Proxy workaround (mandatory)
The shell has `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:7890` set. `httpx` (used by `huggingface_hub`) misclassifies this as SOCKS and raises `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed` at `default_client_factory` — breaks BOTH repo creation and upload. `huggingface.co` is directly reachable without proxy (tested: curl --noproxy = 200 in 0.5s). Unset proxies inline in the command:

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
```

Do this inside the tmux command string so it applies to the `hf` subprocess only.

### Exclude training artifacts
Verl/FSDP checkpoint dirs mix inference-ready HF weights with ~45GB of DeepSpeed training state. Only the top-level HF-format files are needed for re-deployment. Exclude:

```
--exclude 'global_step218/*'      # DeepSpeed state (~45GB)
--exclude 'rng_state_*.pth'
--exclude 'scheduler.pt'
--exclude 'training_args.bin'
--exclude 'trainer_state.json'
--exclude 'latest'
--exclude 'zero_to_fp32.py'
```

Keep: safetensors shards + index, config.json, generation_config.json, tokenizer.*, vocab.json, merges.txt, added_tokens.json, chat_template.jinja, special_tokens_map.json.

### Run in tmux (from existing tmux rule)
```bash
tmux new-session -d -s hf-upload "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; hf upload <namespace>/<repo> <local_dir> . --no-private <excludes> --commit-message '...' 2>&1 | tee /tmp/hf-upload.log; echo EXIT=\$?; exec bash"
```

### Verifying upload (commit-on-completion semantics)
`hf upload` uploads all blobs first and creates the commit ONLY after all files finish. Until then the repo shows only `.gitattributes` (~1.5 KB) — users who pull mid-upload see an "empty" repo and think the upload failed. This is expected, not a bug.

CLI has no `hf repo-files list` subcommand. Use Python API:

```python
from huggingface_hub import HfApi
info = HfApi().repo_info('<namespace>/<repo>', files_metadata=True)
for f in info.siblings:
    print(f.rfilename, f.size, 'LFS' if f.lfs else '')
```

### Detecting a stuck upload
Signs the upload is hung even though the process is alive:
- Log mtime is current but the last N lines show the same %-progress repeating (tqdm redraw with no network activity).
- `ss -tanp | grep pid=<PID>` shows only `CLOSE-WAIT` sockets, no `ESTAB`.
- `New Data Upload` speed has collapsed from MB/s to sub-500 kB/s and isn't recovering.

When hung: `tmux kill-session -t hf-upload && pkill -9 -f "hf upload <repo>"`, then relaunch. xet dedup identifies already-uploaded chunks on the server side — retry resumes near the stall point (in one case, jumped to 100% on processing after re-hashing and only retransmitted 335 MB of 8.06 GB).
