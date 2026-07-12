# Experiment Execution Reliability Final Acceptance

- Review date: 2026-07-12
- Reviewer role: fresh independent final reviewer
- Superproject: `codex/experiment-execution-reliability` at `9bfcd2638042544205305f2d50cc776362d8112a`
- Recipe submodule: `codex/experiment-execution-reliability` at `08639d5352d3e9377fc408ad1c6384d4c4cf8da4`
- Contract: `docs/joint_training/plans/active/experiment_execution_reliability_goal.md`
- Final verdict: **REJECT**

## Blocking Findings

### P1: AC-19 accepts a bounded 16/16/32 shard as the real deployable calibration

The real report is internally hash-consistent, but it does not prove the required
full HumanEval+, MBPP+, and LiveCodeBench validation contract. The calibration
launcher explicitly sets one 64-row shard and `VAL_MAX_SAMPLES=64` at
`recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh:11`.
The shard creator selects only 16 HumanEval+, 16 MBPP+, and 32 LiveCodeBench rows at
`recipe/on_policy_wdl_sft/code_task/create_code_calibration_validation.py:8-15`.
The assembler then hard-codes `rows: 64` while writing `decision: deployable` at
`scripts/assemble_code_task_operational_calibration.py:151-158`.

The checker verifies artifact hashes, repetition counts, three metric names, deadline,
and prediction error, but it never verifies source dataset hashes and complete row
counts against the manifest's three full validation files. Consequently the existing
64-row report passes even though the final-review request requires full-suite evidence.

Required correction: run the real 1-warmup + 3-measured calibration for every phase
against the complete three validation datasets, record their source hashes and row
counts, and make the checker reject any sampled/downscoped validation report.

### P1: AC-21, AC-22, and AC-25 prescribed isolation commands fail on the submitted state

The exact AC-21 command and the isolation command shared by AC-22 exit `1`. They report
the reviewed Goal contract files and all changed Stage123 recipe paths as protected
content whose size and SHA-256 changed. The AC-25 command also exits `1`: passing only
the recipe adoption manifest leaves the three reviewed Goal contract paths unadopted.

`scripts/check_goal_git_isolation.py:35-37` skips a changed baseline path only when an
adoption manifest supplied by the caller places it in `allowed`. The submitted fast
gate supplies both recipe and superproject adoption manifests, but the AC-21/22/25
commands in the controlling contract do not. Therefore the required reviewer commands
are not self-contained and do not pass after the controlled adoption transaction.

Required correction: make the contract commands and checker interface agree on the
final adopted state. The exact commands listed under AC-21, AC-22, and AC-25 must pass
without weakening content-addressed detection, full-scope adoption, parentage, or
pointer-order checks.

## Review Environment

Host `python3` does not contain `pytest`. For every contract command written as
`python3 -m pytest`, I used the repository's own authoritative test path:

```bash
/data-1/verl07/run_train.sh /opt/venv/bin/python -m pytest ...
```

This is the same path used by `scripts/check_experiment_workflow_fast.sh`. No test was
skipped. All acceptance network-sensitive commands ran with the repository's local
socket-deny/fake-service gates; no real W&B, WxPusher, or Hugging Face endpoint was
used.

Reviewer logs are under:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/final_reviewer_9bfcd263/
```

## AC Verdicts

| AC | Verdict | Executed evidence |
|---|---|---|
| AC-01 | PASS | Baseline generator exited 0; fixture contract `2 passed`. |
| AC-02 | PASS | Failure classifier `3 passed`; all eight primary classes and unknown input covered. |
| AC-03 | PASS | Benchmark tests `2 passed`; three-phase report and budget checker exited 0. |
| AC-04 | PASS | Preflight budget gate `4 passed`; failed/missing/force cases remain blocking. |
| AC-05 | PASS | Comparison contract `3 passed`; semantic downscope and neutral optimization cases covered. |
| AC-06 | PASS | GPU smoke shell syntax passed; guard tests `3 passed`; no unapproved runtime started. |
| AC-07 | PASS | Manifest tests `6 passed`; validate/render exited 0 with manifest SHA `7276b817...afb`. |
| AC-08 | PASS | Queue/monitor contract `6 passed`; both shell syntax checks passed. |
| AC-09 | PASS | Manifest release gate `4 passed`; ineligible and mismatched provenance blocked. |
| AC-10 | PASS | Legacy inventory command exited 0; inventory tests `2 passed`. |
| AC-11 | PASS | Fast gate passed (`77 passed`); new-experiment gate `3 passed`. |
| AC-12 | PASS | Real catalog checker exited 0; catalog tests `2 passed`. |
| AC-13 | PASS | Rule enforcement tiers `2 passed`. |
| AC-14 | PASS | Migration-source test `1 passed`. |
| AC-15 | PASS | Startup context budget `3 passed`. |
| AC-16 | PASS | Fresh invocation completed in `5.17s`; `77 passed`; no remote call observed. |
| AC-17 | PASS | Fresh scratch full gate exited 0; reward `9 passed`, release `4 passed`; exit propagation `3 passed`. |
| AC-18 | PASS | Manifest-native dry run exited 0; end-to-end test `1 passed`; artifacts stayed in reviewer scratch. |
| AC-19 | **FAIL** | Checker exits 0 and hashes/repetitions/profile/8K/deadline/errors are consistent, but evidence is a 64-row 16/16/32 shard, not full validation. |
| AC-20 | PASS | User-decision notification test `1 passed`; fake, deduplicated, notification-only flow. |
| AC-21 | **FAIL** | Exact isolation checker command exited 1 before range `diff --check`; changed adopted paths are treated as protected. |
| AC-22 | **FAIL** | Mutation tests `3 passed`, but required live baseline checker exited 1 on current submitted state. |
| AC-23 | PASS | Goal completion-state test `1 passed`; only deployable reviewer acceptance can complete. |
| AC-24 | PASS | No-bypass tests `4 passed`; repeated fast gate `77 passed`; formal launch requires receipt. |
| AC-25 | **FAIL** | Adoption fixture tests `2 passed`, but prescribed live adoption/isolation command exited 1. |
| AC-26 | PASS | Deadline cleanup `5 passed`; includes TERM/KILL, residual ownership, cleanup failure, idempotence, and historical 4560-second step-0 fixture. |
| AC-27 | PASS | Notification state machine `4 passed`; exactly three events, deduplication, non-events, and fake-delivery failure covered. |

## AC-19 Artifact Audit

The following parts of the existing report were independently recomputed and passed:

- Report SHA-256: `21d2a96d32e5dcff940ed7d02bc0725a0f905796e466ddaed0d73673bc0b1344`.
- Validation shard SHA-256 matches: `c3eaf3374661fba71d1132f0de7a8dbdbd3d90295d4fabeb77b5e9dd7c221608`.
- All 36 repetition artifacts exist and match their recorded hashes: status,
  resources, and metrics for four repetitions across three phases.
- Every phase records flags `[warmup, measured, measured, measured]`.
- One profile is shared: `9fa4f1d08e1c3037e90c4bd6e26ab7c80ab6864462b898a92653c7fb7400c6a2`.
- Every launch log records `MAX_RESPONSE_LENGTH=8192`.
- Maximum validation elapsed time is below 1800 seconds in every phase.
- Recomputed elapsed-time, peak-RSS, and GPU-wait prediction errors are all at most 20%.
- Model provenance config hashes match, and all repetitions returned 0 without timeout.

These facts prove that the sampled calibration is authentic and internally coherent.
They do not upgrade the sampled shard into full-validation evidence, so the deployment
decision remains unacceptable under this final-review contract.

## Special Contract Checks

- Historical regression: `historical_76m_step0.json` records 4560 seconds, step 0,
  incomplete validation, blocked release, zero SQLite rows, zero W&B sync markers,
  and no residual ownership; AC-26 executes this fixture.
- Cleanup: timeout tests require run-owned process/Ray/Docker/tmux cleanup and refuse
  `resources_released=true` when GPU ownership or cleanup failures remain.
- Override/bypass: static and behavioral AC-24 tests reject skip/force paths before
  Docker, Ray, tmux, or trainer startup. `ray stop --force` is cleanup semantics,
  not a launch override.
- Notifications: only `run_started`, `run_failed`, and `user_decision_required` are
  accepted; phone delivery/reply does not authorize execution.
- Release isolation: failed/incomplete provenance cannot invoke the release hook; the
  fresh full gate used local fakes and socket denial.
- Repository parentage: recipe commits precede corresponding superproject pointer
  updates in the observed history. This does not cure the failing required isolation
  commands.
- Dirty workspace: current dirty paths are limited to the recorded user baseline plus
  three calibration-updated fixture files and the Stage123 plan artifacts recognized
  by the fast gate. The content-addressed live checker nevertheless fails on adopted
  baseline paths, so this condition cannot be accepted by inspection alone.

## Final Decision

**REJECT.** AC-19, AC-21, AC-22, and AC-25 are not reviewer-owned PASS. The Goal must
remain active. Re-review requires a full-suite real calibration report and a submitted
state where every exact repository-isolation command in the contract exits 0.
