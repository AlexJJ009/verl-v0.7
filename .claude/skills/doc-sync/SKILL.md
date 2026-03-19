---
name: doc-sync
description: >
  Synchronize documentation (docs/joint_training/ and CLAUDE.md) after development work.
  Trigger when: user runs /doc-sync, or after completing a development task
  that changed code in verl/models/joint_model/, verl/trainer/, verl/workers/,
  or tests/joint_training/.
  Compares recent code changes against docs to find and fix stale documentation.
---

# doc-sync

工作路径: `/data-1/verl07/verl`

Synchronize `docs/joint_training/specs/`, `docs/joint_training/constraints/`, `docs/joint_training/plans/`, and `CLAUDE.md` with the actual codebase after development changes.

## Execution mode

- **Standalone conversation** (user opened a session just for this): execute directly in the working directory.
- **Mid-development** (main agent is working on a task): spawn a background subagent to avoid context pollution.

To spawn as background subagent, use the Agent tool with:
```
subagent_type: "general-purpose"
model: "sonnet"
run_in_background: true
```

Pass the full prompt below as the agent's task, including the gathered git context.

All file paths below are relative to `/data-1/verl07/verl`.

## Procedure

### 1. Gather change context

Working directory: `/data-1/verl07/verl`

```bash
conda activate verl07

# If there are recent commits, use the latest commit diff
git log --oneline -5
git diff HEAD~1 --name-only
git diff HEAD~1 --stat

# If no commits (uncommitted work), use working tree changes
git diff --name-only
git diff --stat
git status --short
```

Use whichever produces results. If the user provides a custom range (e.g., `/doc-sync HEAD~3`), use that range instead.

Only proceed if changes touch files in: `verl/models/joint_model/`, `verl/trainer/`, `verl/workers/`, `verl/utils/`, `verl/checkpoint_engine/`, `tests/joint_training/`, or `recipe/joint_training/`. If changes are docs-only, skip spec updates and just report "no code changes detected."

### 2. Read current documentation

Read these files (all paths under `/data-1/verl07/verl`):
- `docs/joint_training/README.md` (directory structure guide)
- All files in `docs/joint_training/specs/`
- All files in `docs/joint_training/constraints/`
- `docs/joint_training/plans/active/` (check if any tasks should be marked complete or if plans should move to completed)
- `CLAUDE.md`

### 3. Read changed code

For each changed source file identified in step 1, read the file to understand the actual behavior. Focus on:
- Public API changes (function signatures, class methods, config fields)
- New/removed/renamed files in joint_model/ or joint-training paths
- Changed behavior in the training loop, rollout, or checkpoint paths
- New or removed metrics under `jointTraining/`
- Changes to test coverage

### 4. Compare and update

For each spec in `docs/joint_training/specs/`, compare the spec's claims against the actual code:

- **Algorithm description**: Does the described fusion logic match `modeling_joint_qwen3.py`?
- **Config fields**: Do documented config options match `configuration_joint_qwen3.py`?
- **Code path references**: Do file paths in specs still exist and contain the described logic?
- **Metric names**: Do documented metric names match what the code actually emits?
- **Stage status**: Does the described stage progress match reality?
- **Temporal words**: No "当前"/"目前" without a date. Replace with direct statements or specific dates.

For `docs/joint_training/constraints/`:
- **Modification boundaries**: Do the "safe to modify" and "do not modify" lists match reality?
- **Development principles**: Are principles still accurate given recent changes?

Edit rules:
- **Only edit files in `docs/joint_training/` and `CLAUDE.md`**. Never edit source code.
- If a spec is accurate, do not touch it.
- Preserve the document structure (headings, sections) when editing.

### 5. Handle plans

- If an `active/` plan's objective has been fully achieved (verify by checking the code and tests), move it to `completed/` with today's date prefix.
- If new issues are discovered during the sync, add them to the appropriate active plan or create a new plan in `active/`.
- Update progress sections in active plans to reflect what has been accomplished.

### 6. Handle code reviews

- If a completed code review's issues have all been resolved (verify by checking the code), append a resolution section at the end of the review document:
  ```markdown
  ## Resolution

  **Date**: YYYY-MM-DD
  **Status**: All issues resolved
  **Resolution**: <brief description of how issues were fixed>
  ```

### 7. Generate report

After all edits are complete, output a report in this format:

```markdown
# Doc-Sync Report

Date: YYYY-MM-DD
Change scope: <git range used>

## Modified Documentation

| File | Change type | Description |
|------|------------|-------------|
| docs/joint_training/specs/xxx.md | Updated | Updated metric names |
| docs/joint_training/plans/active/xxx.md | Moved to completed | Objective achieved |

## Unchanged Documentation

The following docs are consistent with the code:
- docs/joint_training/specs/xxx.md
- ...

## Issues Found But Not Addressed

<If any issues are outside docs/ scope, list them here>

---

Review changes with: `git diff docs/ CLAUDE.md`
```

Present this report to the user and ask them to review with `git diff`.
