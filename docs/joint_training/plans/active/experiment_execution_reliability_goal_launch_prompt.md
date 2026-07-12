# Codex Goal Resume Prompt: Experiment Execution Reliability

Resume and complete the long-running VERL Experiment Execution Reliability Goal.

## Single Source of Truth

```text
/data-1/code/verl/docs/joint_training/plans/active/experiment_execution_reliability_goal.md
```

Read the entire plan before acting. Treat its `Resume Snapshot - 2026-07-12 (Outcome Schema V2)` and
AC-01 through AC-28 as the execution contract. Do not restart from Milestone 0 or
repeat accepted work unless live verification proves its evidence is stale or invalid.

## Current Repository State

```text
superproject:                  /data-1/code/verl
branch:                        codex/experiment-execution-reliability
original CPU-accepted base:    af1a407fda562f1cef8fd9d4471f73f3de91814e
current committed HEAD:        b796ab3c (verify exact full hash live)

recipe:                       /data-1/code/verl/recipe
branch:                       codex/experiment-execution-reliability
original accepted base:       cec05371fe17d42bb80722b5608c3cecbe4785b6
current committed HEAD:       67b930199745518ce0433a0e7ca4c3289ac1fb0c
```

Verify these identities first. The committed continuation after the original baseline
contains the reviewed timing contract/fix, outcome-schema-v2 plan, native trainer
telemetry, safe HF hashing, and workload descriptors listed in the plan. Do not reset
to `af1a407f` and do not create another branch. Preserve the two pre-existing untracked
user assets and continue the intentional uncommitted v2 implementation in place. Audit
any additional drift before editing it.

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

## Current Implementation Point - Resolve Before GPU Work

The first operational run completed 18 bootstrap repetitions under:

```text
/data-2/experiment_registry/calibration_runs/af1a407f
```

They are diagnostic-only because their generation JSONL lacks native token count,
EOS, and finish telemetry. Never use this root to authorize acceptance, and never
retrofit or infer missing outcome-schema-v2 telemetry from decoded text.

Timing semantics are already resolved and reviewed: the canonical elapsed time is
`metrics_complete.monotonic_seconds - validation_ready.monotonic_seconds`, and
`timing_s/testing` must contain that interval. Do not reopen this decision without new
contradictory evidence. Resume the existing outcome-schema-v2 worktree instead:

1. inspect `git status` and preserve every listed user/uncommitted path;
2. finish upgrading assembler/checker fixtures from schema v1 to v2;
3. add the missing checker boundary tests named in the plan;
4. reconcile queue-native acceptance directories with assembler input semantics without
   fabricating a predictor repetition;
5. run the focused outcome/predictor/runner/assembler/checker suite, then fast/full,
   isolation, and dual-repository transaction gates;
6. commit independently verifiable units with recipe-first ordering when applicable;
7. obtain fresh independent GPT-5.5 medium CPU acceptance before preflight or GPU work.

For any persistent local CI, queue monitor, or reliability checker, use PM2 for
keepalive and restart management. Do not create systemd units or use `systemctl`.
Commit and test the PM2 ecosystem/command contract, status/log/restart checks, and
`pm2 save`. Use `pm2 resurrect` only through an already available non-systemd host
bootstrap; if reboot persistence would require `pm2 startup` to install systemd, stop
and report that boundary instead.

Never weaken sampled validation, outcome telemetry, thresholds, interval semantics, or
tests to make old evidence pass. Preserve all 18 old runs as diagnostic evidence.

## Resume Operational Calibration

Only after outcome-schema-v2 and the PM2 keepalive contract are committed, the
focused/fast/full/isolation/transaction gates pass from committed code, and a fresh
independent CPU acceptance is obtained:

1. generate a fresh machine report, bounded preflight report, budget decision, and
   preflight receipt bound to the current commits and manifest hashes; the old receipt
   under `/data-1/tmp/verl_agent_scratch/experiment_workflow/preflight/af1a407f` is
   expired historical evidence;
2. verify GPUs, Docker, model/data/scorer provenance, disk space, and no conflicting
   run-owned runtime;
3. run all long GPU work in tmux;
4. create a completely new calibration root and run six new bootstrap repetitions per
   phase; the old `af1a407f` root is never eligible for v2 trusted history;
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

- every AC-01 through AC-28 is covered by current evidence;
- real local L40S calibration returns checker-owned `deployable`;
- both receipts are fresh and exactly hash-matched;
- fast/full gates pass from committed code;
- dual-repository isolation and recipe-first transaction pass;
- AC-28's sandbox PM2 keepalive tests and no-systemd checker pass from committed code,
  including fail-closed evidence when no non-systemd reboot bootstrap exists;
- a fresh independent Reviewer runs the commands and reports every AC `PASS`, with no
  `FAIL` or `WEAKENED`;
- the completion-state checker passes;
- only then call the Goal completion tool.

The implementer must not self-accept.
