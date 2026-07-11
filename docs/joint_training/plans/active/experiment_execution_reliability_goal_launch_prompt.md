# Codex Goal Launch Prompt: Experiment Execution Reliability

Execute the accepted plan at:

```text
docs/joint_training/plans/active/experiment_execution_reliability_goal.md
```

Do not start unless its latest independent review says `PLAN VERDICT: READY`.

## Objective

Deliver a machine-checkable VERL experiment workflow that provides representative
Stage1/Stage2/Stage3 preflight performance estimation, makes a versioned experiment
manifest the single source of truth for new queue/monitor/preflight/release flows,
and migrates project engineering principles into structural checks, machine gates,
or explicit trigger/action/failure rules.

## Contract

- The plan file and AC-01 through AC-27 are the single source of truth.
- Execute milestones 0, 1, 2, 3, 4, and 5 in that exact order.
- Do not begin a milestone until all required ACs in the previous milestone pass.
- If reality requires changing scope or an AC, update the plan and obtain a fresh
  independent `READY` review before continuing.
- Preserve full HumanEval+/MBPP+/LiveCodeBench, `MAX_RESPONSE_LENGTH=8192`, and
  the common Stage1/2/3 L40S resource profile.
- Use local fixtures/fakes for W&B, WxPusher, Hugging Face, and network behavior.
- Do not use real external-service success as acceptance evidence.
- There is no `--force`, signed approval file, or automatic override. On a soft
  threshold failure, send a guarded WxPusher decision notification when unattended,
  stop, and wait for the user to return. User approval becomes a reviewed manifest/
  policy commit followed by a newly passing preflight.
- Remove `SKIP_STAGE123_PREFLIGHT`, `SKIP_STAGE123_MACHINE_GATE`, and equivalent
  bypasses. Formal queue and direct phase launchers must consume the same fresh,
  content-addressed passing preflight receipt before any runtime side effect.
- Enforce a 30-minute hard wall-clock budget for each Stage1/Stage2/Stage3 real
  pre-training measurement and formal `val_before_train`. A 30-minute overrun without
  complete validation metrics is a blocked failure and must release the GPUs.
- WxPusher is event-only: verified run start, run failure, or user decision required.
  Do not send periodic healthy-progress notifications; tmux/container creation is not
  sufficient evidence for a run-start notification.
- The 30-minute validation deadline must execute verified idempotent runtime cleanup:
  stop queue advancement, terminate phase/Ray/Docker/tmux descendants, and prove the
  run no longer owns GPU resources. Cleanup failure remains blocked.
- Use `/data-1/tmp/verl_agent_scratch/experiment_workflow/` for scratch output.
- Do not revert, stage, or commit unrelated dirty-worktree changes.

## Branch and Commits

Create `codex/experiment-execution-reliability` unless the user explicitly directs
otherwise. Commit each independently verifiable milestone separately. A commit is
not complete without its AC verification output.

## Stop and Ask

Stop and ask when any plan stop trigger fires, when a contradiction appears, when
an AC requires a real external service, when the same blocker repeats three times,
or when passing would require weakening/skipping/trivializing a test.

## Definition of Done

All AC-01 through AC-27 pass, including approved local L40S calibration and both
repository isolation checks; fast/full gates preserve child exit codes,
Stage123 dry-run is manifest-native, no real external service was contacted for
acceptance, each milestone has a focused commit, and a fresh independent reviewer
runs the commands and reports every AC `PASS` with none `WEAKENED`.

The implementing agent must not mark the Goal complete based on its own checks.
