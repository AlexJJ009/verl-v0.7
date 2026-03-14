# Development Principles

These principles govern all development work on the joint-training branch. They apply equally to human engineers and coding agents.

## Core Principles

1. **Preserve upstream behavior** unless joint training truly requires a divergence. Do not refactor unrelated code.
2. **Prefer reusable framework fixes** over recipe-only guards when the issue is generic. Recipe fixes are for host- or run-shape-specific issues; framework fixes are for semantic or reusable issues.
3. **Treat per-GPU memory, filesystem topology, and optional dependencies as design constraints**, not incidental environment details.
4. **Add tests with each real fix**, especially when the failure was first seen in end-to-end GPU runs. Stabilization work is not finished until guards are covered by regression tests.
5. **Make debugging observable**. If a metric is needed to understand training, it must be surfaced explicitly in logs or persisted artifacts. Important metrics must be printed at meaningful milestones during the run.

## Agent-First Operating Principles

Adapted from OpenAI Harness Engineering practices:

1. **Repository knowledge is the system of record**. Context that lives in chat threads, people's heads, or external docs is invisible to coding agents. Push decisions and constraints into the repo.
2. **CLAUDE.md is the table of contents, not the encyclopedia**. Keep it short (~100 lines) with pointers to deeper sources of truth in `docs/joint_training/`.
3. **Enforce invariants, not implementations**. Define strict boundaries (what files not to modify, what tests must pass), but allow autonomy in how solutions are expressed within those boundaries.
4. **Progressive disclosure**. Agents start with CLAUDE.md and are taught where to look next, rather than being overwhelmed up front.
5. **Plans are first-class artifacts**. Active plans, completed plans, and known gaps are all versioned and co-located in `docs/joint_training/plans/`.
6. **Entropy requires garbage collection**. Periodically scan for stale documentation, drifted patterns, and accumulated technical debt. Keep docs current with the real code.

## Coding Style

1. Only add comments where the logic is not self-evident.
2. No over-engineering. Only make changes that are directly requested or clearly necessary.
3. Avoid backwards-compatibility hacks. If something is unused, delete it.
4. Do not create abstractions for one-time operations.
