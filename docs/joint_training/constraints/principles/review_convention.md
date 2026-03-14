# Code Review Convention

## Rule

Code review must happen in a **new, separate Claude Code session** — never in the same session that wrote the code.

## Why

When the same agent reviews its own work in the same session, it has full context of its own reasoning and intent. This creates two failure modes:

1. **Reward hacking**: The agent confirms its own assumptions instead of challenging them.
2. **Lazy review**: The agent skips checking things it "remembers" doing correctly, rather than verifying from scratch.

A fresh session forces the reviewer to understand the code from the diff alone, which is the same position a human reviewer or a future agent would be in.

## How to Apply

1. Finish the development task and commit (or stage) the changes.
2. Open a new Claude Code session.
3. In the new session, use the built-in review capability to review the diff.
4. Save review results to `docs/joint_training/codereview/active/`.
5. When issues from the review are resolved, move the review to `codereview/completed/` and append a resolution section.
