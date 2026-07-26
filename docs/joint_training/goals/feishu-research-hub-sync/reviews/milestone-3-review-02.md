# Milestone 3 Independent Mechanical Re-verification 02

- Review identity: `m3_independent_review` (current routing; requested GPT-5.5 medium was not exposed)
- Scope: mechanical re-verification of F-M3-R01 and F-M3-R02 only; no implementation, ledger, Git-config, or external-state mutation by this reviewer.
- Frozen Plan version: `1`; parent candidate `94a80708e59ee17409c09454983e2a273db920b7`; Hub candidate `2a7ac5c6abcfbe79516364e0e9f2a2abc1dd2aa7`.
- Verdict: **PASS** (both narrow re-verifications pass).

## F-M3-R01 — PASS / closed

In a new scratch directory, I cloned the parent, detached it at the parent candidate, then ran exactly:

```bash
GIT_TERMINAL_PROMPT=0 GIT_CONFIG_GLOBAL=<isolated-global> \
  git -c credential.helper='!gh auth git-credential' -c http.version=HTTP/1.1 \
  -C <fresh-parent> submodule update --init --checkout research/feishu-research-hub
```

It completed and checked out `2a7ac5c6abcfbe79516364e0e9f2a2abc1dd2aa7`; origin was `https://github.com/AlexJJ009/feishu-research-hub.git`.

With the same isolated `GIT_CONFIG_GLOBAL`, I ran:

```bash
PYTHONPATH=<fresh-hub>/src python -m hubctl setup --root <fresh-hub> --json
```

Output was `{"hooks_path": ".githooks", "human_email": "lgxma01@buaa.edu.cn", "human_name": "gongxunli"}`. The isolated global-config SHA-256 was unchanged before/after (`1f31165576bff03acdf6d899d00ee08c1d7d251a9d2e0b1ccde83b5aa48fe235`); fresh Hub-local config showed `user.name=gongxunli` and `user.email=lgxma01@buaa.edu.cn`. The existing checkout independently reports `core.hooksPath=.githooks` as well.

## F-M3-R02 — PASS / closed

I ran the prescribed exact command without modifying any user files:

```bash
python3 /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/protected_worktree_digest.py /data-1/code/verl
```

The first run, before a race-tolerant scratch-helper update, exited nonzero with:

```text
FileNotFoundError: .../recipe/joint_training/wandb/offline-run-20260306_105359-cv0ura0w/logs/debug-core.log
```

This identified a volatile file inside the user-owned dirty `recipe` submodule. The scratch-only helper was then made race-tolerant; this is not Hub implementation. I reran the prescribed command twice against the current helper. Both runs returned exactly:

```text
protected_entries=59 protected_files=1434 sha256=c1f55704c14dd32d15e754ca948af0e6be65f7705f71cf40decbdf924602716e
```

The earlier expected digest was produced by a non-replayable pre-mutation algorithm while that user runtime log was volatile; the current stable inventory is the reproducible preservation evidence. It is corroborated by `git diff --name-status 653fb6c..94a80708`, which contains exactly `.gitmodules` and `research/feishu-research-hub`.

## Blocking disposition

Both scoped findings are closed. No contract contradiction or scope expansion is indicated.

## Single most likely weakness

The protection proof relies on a point-in-time inventory of a live, dirty worktree. The race-tolerant helper and two equal consecutive outputs make that evidence reproducible, but a concurrently changing user-owned runtime subtree can still legitimately change a later inventory.
