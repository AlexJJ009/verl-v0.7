# Training Script Index

This file is the collaboration index for runnable training scripts in this
branch. Keep it factual and short: script purpose, creation time, last use, and
current status. Update it whenever a training script is created or used.

The shared policy is
`docs/joint_training/constraints/experiment_tracking/training_script_index_policy.md`.

Do not put full launch commands, monitor commands, or operational playbooks in
this index. Those belong in the relevant on-demand guide or workflow, so agents
and collaborators can load run mechanics only when they need them.

Times are local server time (`Asia/Shanghai`, CST).

## Dual-Submodel Rollout WDL-SFT

| Script | Purpose | Created | Last used | Status / notes |
|---|---|---:|---:|---|
| `recipe/on_policy_wdl_sft/dual_submodel_rollout/` | Planned dual-submodel rollout experiment script directory referenced by `docs/joint_training/plans/active/dual_submodel_rollout_wdl_sft.md`. | before 2026-05-25 | needs audit | Pending concrete script audit. In this worktree the `recipe` submodule is not checked out (`git submodule status` shows `-e473020... recipe`), so script rows must be filled after submodule initialization or recipe branch checkout. |
