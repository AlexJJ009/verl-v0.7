# Codex Goal Resume Prompt: Experiment Execution Reliability

Resume and complete the long-running VERL Experiment Execution Reliability Goal.

## Single Source of Truth

```text
/data-1/code/verl/docs/joint_training/plans/active/experiment_execution_reliability_goal.md
```

Read the entire plan before acting. Treat its `Resume Snapshot - 2026-07-12` and
AC-01 through AC-27 as the execution contract. Do not restart from Milestone 0 or
repeat accepted work unless live verification proves its evidence is stale or invalid.

## Current Repository State

```text
superproject:                  /data-1/code/verl
branch:                        codex/experiment-execution-reliability
accepted implementation base: af1a407fda562f1cef8fd9d4471f73f3de91814e

recipe:                       /data-1/code/verl/recipe
branch:                       codex/experiment-execution-reliability
accepted implementation HEAD: cec05371fe17d42bb80722b5608c3cecbe4785b6
```

Verify these identities first. The superproject is expected to have one or more later
plan-only resume commits; classify commits after the accepted implementation base and
continue when they only update this Goal's plan/prompts. Do not create another branch
and do not revert the two pre-existing untracked user assets recorded in the plan. If
there are later implementation or unrelated commits, audit them and update the plan
before continuing.

## Accepted Work To Inherit

- Independent plan review: `READY`.
- CPU/sandbox review: `CPU ACCEPTED`.
- Dynamic sampled-validation contract, immutable history selection, candidate-only
  assembler, checker-owned canonical deployability receipt, dual receipt admission,
  strict Git isolation, and recipe-first transaction are implemented and committed.
- Fast/full gates and focused receipt/isolation tests passed.
- Formal validation remains sampled pass@1 over full HumanEval+, MBPP+, and
  LiveCodeBench with `MAX_RESPONSE_LENGTH=8192` and one shared L40S resource profile.
- There is no automatic override, `--force`, skip variable, approval file, or phone
  authorization path.

## Current Blocker - Resolve Before GPU Work

The first operational run completed 18 bootstrap repetitions under:

```text
/data-2/experiment_registry/calibration_runs/af1a407f
```

The queue then failed closed before creating trusted history because Stage2
`timing_s/testing` did not match the instrumented `validation_ready` to
`metrics_complete` interval within one second. Stage2 repetition 0 records about
`88.794s` versus `80.58s`.

Do not restart calibration immediately. First:

1. read the preserved status, metrics, timeline, resource, scorer, and host-log
   artifacts for all three phases;
2. trace both timing measurements to their code-level start and end points;
3. determine the normative interval required by AC-05/AC-19: validation rollout
   readiness through complete validation metrics;
4. amend the plan if the exact evidence/consistency rule is not already unique;
5. send that amendment to a fresh independent GPT-5.5 medium Reviewer and require
   `READY` before implementation or GPU work;
6. add CPU regression fixtures and run fast/full gates;
7. commit recipe first when recipe changes exist, then commit the superproject pointer
   and implementation with path-scoped staging.

Never pick a timing source or tolerance because it makes the observed runs pass. Never
widen an interval or consistency threshold after measurement. Preserve all 18 runs.

## Resume Operational Calibration

Only after the timing contract is reviewed and CPU gates pass:

1. generate a fresh machine report, bounded preflight report, budget decision, and
   preflight receipt bound to the current commits and manifest hashes; the old receipt
   under `/data-1/tmp/verl_agent_scratch/experiment_workflow/preflight/af1a407f` is
   expired historical evidence;
2. verify GPUs, Docker, model/data/scorer provenance, disk space, and no conflicting
   run-owned runtime;
3. run all long GPU work in tmux;
4. according to the reviewed timing contract, either content-address and reuse every
   complete eligible bootstrap run or rerun only the required repetitions;
5. freeze the immutable history snapshot before acceptance measurements;
6. generate and verify `prediction_contract.json`;
7. run exactly three new acceptance repetitions per Stage1/Stage2/Stage3, excluding
   them from history;
8. assemble only a `candidate` report; checker alone may issue `deployable` and its
   canonical receipt;
9. preserve the 30-minute readiness-to-complete-metrics hard timeout and verified GPU
   cleanup; any timeout, incomplete metric, scorer safety failure, semantic/hash drift,
   interval failure, or stale receipt is blocked with no bypass.

## External Services And Notifications

- Do not use real W&B, Hugging Face, WxPusher, or any network service as acceptance
  evidence.
- Failed/incomplete/bootstrap calibration must not enter experiment SQLite or W&B.
- WxPusher is notification-only and event-driven: verified run start, run failure, or
  user decision required. It never authorizes continuation.
- On a soft-threshold failure, stop the Goal and ask the user; do not auto-continue.

## Definition of Done

Do not mark the Goal complete until:

- every AC-01 through AC-27 is covered by current evidence;
- real local L40S calibration returns checker-owned `deployable`;
- both receipts are fresh and exactly hash-matched;
- fast/full gates pass from committed code;
- dual-repository isolation and recipe-first transaction pass;
- a fresh independent Reviewer runs the commands and reports every AC `PASS`, with no
  `FAIL` or `WEAKENED`;
- the completion-state checker passes;
- only then call the Goal completion tool.

The implementer must not self-accept.
