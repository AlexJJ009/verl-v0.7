# Joint Training Documentation (ARCHIVAL)

> **Note**: This documentation was created during the `feature/joint-training` branch development (Stages 1 & 2). The current experiment is **On-Policy WDL-SFT** on branch `feature/on-policy-wdl-sft`. The infrastructure docs here (joint model, logit fusion, FSDP workers) remain relevant as background, but the training algorithm and recipe have changed. See `CLAUDE.md` at the repo root for the current experiment context.

This directory is the knowledge base for joint-training infrastructure in verl. It is organized for both human engineers and coding agents.

## Current Focus

**Math-first On-Policy Weak-Driven Learning causal validation** — the active
P60 matrix tests continuous WDL mixture C against ordinary Stage1 continuation
A, direct strong-only D, and the matched-scale no-weak D0 control. Stage2 uses
Model2-only rollout and a joint teacher-forced WDL objective; `strong_scaled`
implements D0 as $0.8z_2$. See
`plans/active/qwen3_1p7b_math_stage123.md` for the registered hypotheses,
telemetry, and admission contract.

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
│   ├── boundaries/               What to modify, what not to touch
│   └── experiment_tracking/      Experiment and training-script index policies
├── courses/                   ← Educational docs for human learners
│   ├── joint_minirl_reinforce_logit_mixing_theory.md
│   ├── minirl_vs_grpo_comparison.md
│   ├── stabilization_experience_notes.md
│   └── verl_infra_deep_dive.md
├── guides/                    ← Practical how-to guides
│   ├── testing.md
│   ├── sample_efficiency_tuning.md
│   ├── sample_efficiency_iteration_log.md
│   └── server_migration.md
├── plans/                     ← Development plans
│   ├── active/                   Current: on_policy_wdl_sft.md
│   └── completed/                Archived: stage1, stage2, minirl_migration, etc.
├── references/                ← External materials
│   ├── external/                 Third-party articles, papers, sft_analysis.tex
│   └── media/                    Images, diagrams, etc.
├── reports/                   ← Training run analysis reports
└── specs/                     ← Technical specifications (agent-facing)
    └── joint_training_algorithm_v1.md
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
- **Experiment tracking** lives in `recipe/joint_training/`: `EXPERIMENT_INDEX.md` (scripts, logs, checkpoints, weights) and `INFERENCE_RESULTS.md` (offline eval results). Maintenance rules are in `constraints/experiment_tracking/`.
- **Training script indexes** are branch-local. The shared policy is in `constraints/experiment_tracking/training_script_index_policy.md`; each branch maintains its own `guides/training_script_index.md`.
