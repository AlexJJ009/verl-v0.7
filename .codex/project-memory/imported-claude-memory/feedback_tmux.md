---
name: Tmux for long operations
description: All training runs, monitoring scripts, checkpoint transfers, and large file operations must be launched inside tmux sessions
type: feedback
originSessionId: 504dc4c8-b3f1-4ad0-bf9e-6e337aecb68d
---
Always launch long-running operations inside tmux sessions. This includes training scripts, monitoring/tail scripts, checkpoint transfers to secondary mounts, and large file downloads.

**Why:** SSH disconnections and terminal closures have interrupted training jobs and file transfers. Tmux provides session persistence regardless of connection state.

**How to apply:** Before running any training script, `rsync`/`cp` of checkpoints, or monitoring loop, first create or attach to a tmux session. When writing launch instructions for the user or in scripts, always include the tmux wrapper.
