# Joint Training Documentation

This directory is the knowledge base for the joint-training effort in verl. It is organized for both human engineers and coding agents.

## Directory Structure

```
docs/joint_training/
├── README.md                  ← You are here
├── codereview/                ← Code review results
│   ├── active/                   Reviews in progress
│   └── completed/                Resolved reviews (with resolution notes)
├── constraints/               ← Development rules and boundaries
│   ├── principles/               Core development principles
│   ├── commit_rules/             Commit conventions and git workflow
│   └── boundaries/               What to modify, what not to touch
├── courses/                   ← Educational docs for human learners
│   ├── stabilization_experience_notes.md
│   └── verl_infra_deep_dive.md
├── guides/                    ← Practical how-to guides
│   ├── testing.md
│   └── server_migration.md
├── plans/                     ← Development plans
│   ├── active/                   Current plans with progress tracking
│   └── completed/                Archived plans
├── references/                ← External materials
│   ├── external/                 Third-party articles and resources
│   └── media/                    Images, diagrams, etc.
└── specs/                     ← Technical specifications (agent-facing)
    ├── joint_training_algorithm_v1.md
    └── branch_status_2026-03-14.md
```

## How to Use This Directory

**For coding agents**: Start with `CLAUDE.md` at the repo root, which points here. Read the relevant `specs/` and `constraints/` files for your task. Check `plans/active/` for current priorities.

**For human learners**: Start with `courses/verl_infra_deep_dive.md` for a comprehensive walkthrough of the verl architecture and joint-training implementation. Read `courses/stabilization_experience_notes.md` for engineering lessons from Stage 1.

**For reviews**: Active code reviews go in `codereview/active/`. When issues are resolved, move to `codereview/completed/` and append a resolution section.

**For plans**: New plans go in `plans/active/`. When completed, move to `plans/completed/`. Keep progress sections updated during execution.

## Conventions

- **Specs** are agent-facing: precise, structured, and kept current with the code.
- **Courses** are human-facing: educational, detailed, and assume beginner-level knowledge.
- **Constraints** are enforced: they define what agents and humans must follow.
- **Plans** are living documents: update progress as work proceeds, move to completed when done.
- **Code reviews** track resolution: when an issue is fixed, append the resolution to the review document.
