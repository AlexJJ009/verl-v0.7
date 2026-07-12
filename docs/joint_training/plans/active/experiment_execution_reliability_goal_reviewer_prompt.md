# Independent Reviewer Prompt: Experiment Execution Reliability Goal

You are the independent plan reviewer. You did not author this plan and must not
accept intent in place of executable evidence.

## Inputs

- Plan: `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- Goal branch context: `codex/experiment-execution-reliability`
- Parent experiment branch: `feature/on-policy-wdl-sft`
- Current runtime family: Qwen3-1.7B Stage123
- Global playbook: `/root/agent-core/workflows/TEST_ENGINEERING_PLAYBOOK.md`
- Goal-plan contract: `/data-1/agent-tools/goal_plan/codex/skills/goal-plan/SKILL.md`

## Review Tasks

1. Build an AC table for AC-01 through AC-27 with one verdict each:
   `PASS`, `NEEDS SCAFFOLDING`, or `EXTERNAL`.
2. Identify every P0 contradiction between the plan and current code/docs with
   file and line evidence.
3. Identify ACs that are not sandbox-verifiable, are circular, rely on real
   external services, permit self-certification, or use subjective performance
   language without a measurable comparison.
4. Check that milestones are hard ordered and that every consumed artifact is
   produced by an earlier milestone.
5. Decide whether this is one serial Goal or must be split into multiple Goals.
6. Audit the preflight performance criteria for semantic downscoping: full LCB,
   `MAX_RESPONSE_LENGTH=8192`, reward semantics, and Stage1/2/3 resource-profile
   identity must remain intact.
7. Audit the manifest boundary: historical unresolved scripts must not block the
   Goal, but every new experiment after the cutoff must be enforceably native.
8. Audit the principles migration against the three-tier rule: structural
   elimination, machine checker, then trigger/action/failure prose.
9. Produce a preflight checklist covering test runner, fixtures/mocks, Docker
   fallback, branch/commit isolation, dirty-worktree handling, and reviewer-owned
   final commands.
10. Audit the real-L40S calibration acceptance contract (do not run calibration during
    plan review), removal of every automatic override path, guarded
    WxPusher user-decision notification, content-addressed dirty baselines, immutable
    dual-repository cutoff, completion states, concrete no-preflight-bypass coverage,
    controlled Stage123 dirty-file adoption, scaled-clock validation cleanup/GPU
    release, the three-event notification state machine, and submodule-first
    transaction.

## Required Output

Write the review to:

```text
docs/joint_training/codereview/active/experiment_execution_reliability_goal_plan_review.md
```

End with exactly one plan verdict:

```text
PLAN VERDICT: READY
```

or:

```text
PLAN VERDICT: NOT READY
```

`READY` is allowed only when every required AC is sandbox-verifiable or has an
explicitly accepted operational-only evidence boundary, no P0 contradiction
remains, and the execution order is implementable without weakening an AC.

Do not implement code. Do not contact real W&B, WxPusher, Hugging Face, or any
other external service. Do not accept the plan merely because its author supplied
verification commands; determine whether those commands can genuinely prove each
claim.
