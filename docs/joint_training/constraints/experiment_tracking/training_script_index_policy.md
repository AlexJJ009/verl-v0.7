# Training Script Index Policy

This policy defines the invariant for tracking runnable training scripts across
branches. The policy is shared; each branch's concrete index remains
branch-local.

## Source of Truth

- **Policy**: this file.
- **Branch-local index**: `docs/joint_training/guides/training_script_index.md`.

Do not use one shared index file across branches. Different branches may have
different active algorithms, launchers, experiment names, and script lifecycles.

## When to Update the Branch Index

Update the branch-local training script index whenever any of the following
happens:

1. A runnable training script is created.
2. A shared launcher used by training scripts is created or materially changed.
3. A training script is used for a real run.
4. A monitor or queue script becomes part of the real run workflow.
5. A script becomes obsolete, superseded, broken, or host-specific.

## Required Fields

Each entry should stay short and factual:

- Script path.
- Purpose.
- Creation time or first known introduction date.
- Last verified real use.
- Current status and important notes.

Times should use the local server timezone unless the branch documents another
timezone explicitly.

## Boundary

The index is a directory of runnable artifacts, not an operations manual.

Keep full launch commands, tmux session setup, monitor instructions, Meituan
platform steps, checkpoint transfer procedures, and troubleshooting playbooks in
the relevant guide or workflow document. Link those documents from the index
only when useful.

## Cross-Branch Rule

When this policy is migrated to another branch, create or update that branch's
own `docs/joint_training/guides/training_script_index.md`. Do not copy script
rows from another branch unless those scripts also exist and are valid in the
target branch.
