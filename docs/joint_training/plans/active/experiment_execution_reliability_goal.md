# Experiment Execution Reliability and GPU Utilization Goal

- Status: `PLAN REVIEW REQUIRED - STAGE1 WORKLOAD IDENTITY MISMATCH; GPU CALIBRATION NOT AUTHORIZED (2026-07-12)`
- Created: 2026-07-11
- Goal branch: `codex/experiment-execution-reliability`
- Parent experiment branch: `feature/on-policy-wdl-sft`
- Scope: future Stage1/Stage2/Stage3 experiment families, beginning with Qwen3-1.7B Stage123
- Execution mode: serial milestones; no implementation before independent plan acceptance
- External services: forbidden as acceptance evidence; W&B and WxPusher must use local fakes/mocks

## Objective

Build a machine-checkable experiment execution workflow that:

1. predicts scorer latency, timeout rate, score distribution, host-memory risk,
   and expected GPU starvation before a formal training launch;
2. uses one experiment manifest to generate or configure new queue, monitor,
   preflight, release, and documentation records;
3. rewrites project engineering principles into trigger/action/failure rules and
   replaces machine-checkable prose with executable gates.

The goal does not promise to normalize every historical queue. Historical assets
receive a best-effort inventory. The manifest contract is mandatory for Stage123
and every newly created experiment family after this goal is accepted.

## Single Source of Truth

This file and its numbered acceptance criteria are the execution contract. During
implementation, any scope or acceptance change must update this file before code.

The implementation must introduce one versioned experiment-manifest schema under:

```text
recipe/on_policy_wdl_sft/experiment_manifest/
```

New queue, monitor, preflight, and release behavior must consume that schema rather
than maintaining independent run-prefix/final-step/train-file arrays.

## Resume Snapshot - 2026-07-12 (Outcome Schema V2)

This section is the authoritative resume point for the next `/goal` run. It updates
transient execution state without weakening or replacing AC-01 through AC-30.

### Accepted State

1. The amended plan preserving sampled pass@1 validation and the history-fitted
   dynamic interval contract received a fresh independent `READY` review.
2. CPU/sandbox implementation received independent `CPU ACCEPTED` review. The final
   accepted CPU state includes AC-19F, AC-24, and AC-25; the focused receipt tests,
   fast/full gates, strict Git isolation, and dual-repository transaction passed.
3. The original independently CPU-accepted implementation baselines were:

   ```text
   superproject branch: codex/experiment-execution-reliability
   accepted implementation commit: af1a407fda562f1cef8fd9d4471f73f3de91814e
   recipe branch:       codex/experiment-execution-reliability
   accepted recipe commit:         cec05371fe17d42bb80722b5608c3cecbe4785b6
   ```

   The timing correction and outcome-schema-v2 plan amendments were subsequently
   reviewed independently as `READY`. The committed continuation chain now includes:

   ```text
   06aeb171 docs: define canonical calibration timing interval
   0b8699c4 fix: use readiness interval for calibration timing
   46c5c312 chore: refresh goal contract adoption receipt
   8a4b3124 docs: specify calibration outcome schema v2
   21be9dad feat: record native validation response telemetry
   484f344a docs: define safe HF snapshot hashing
   67b9301 recipe commit: add calibration workload descriptors
   b796ab3c feat: validate calibration workload descriptors
   ```

   At Goal startup, verify this chain and classify every later commit. Do not reset to
   `af1a407f`, discard the current worktree, or repeat accepted timing/telemetry/workload
   work merely because the earlier baseline is named above.

4. Preserve these pre-existing user assets recorded by the content-addressed baseline:

   ```text
   .claude/skills/experiment-registry
   docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
   ```

   The current worktree also contains an intentional, unfinished outcome-schema-v2
   implementation. Audit and continue it in place; do not revert, overwrite, or treat
   it as unrelated dirt. At this snapshot its expected paths are:

   ```text
   scripts/calibration_outcomes.py
   scripts/assemble_code_task_operational_calibration.py
   scripts/check_calibration_prediction_contract.py
   scripts/check_code_task_operational_calibration.py
   scripts/run_code_task_operational_calibration_queue.sh
   tests/experiment_workflow/test_calibration_outcomes.py
   tests/experiment_workflow/test_dynamic_calibration_interval.py
   tests/experiment_workflow/test_operational_calibration_runner.py
   docs/joint_training/plans/active/experiment_execution_reliability_goal.md
   docs/joint_training/plans/active/experiment_execution_reliability_goal_launch_prompt.md
   ```

5. A passing preflight snapshot was previously generated under:

   ```text
   /data-1/tmp/verl_agent_scratch/experiment_workflow/preflight/af1a407f
   ```

   It and the later preflight root ending in `46c5c312` are stale historical evidence.
   A fresh machine report, budget result, and preflight receipt bound to the final
   committed v2 implementation, recipe commit, manifest hash, workload descriptors,
   and outcome schema are required before any new GPU work.

### Preserved Operational Evidence

The first resumed calibration attempt used the approved sampled/full-validation
contract and completed all eighteen bootstrap repetitions without runtime timeout,
fatal termination, CUDA OOM, or leaked DB/W&B release side effects:

| Phase | Successful bootstrap repetitions | Worker elapsed seconds | Median |
| --- | ---: | --- | ---: |
| Stage1 | 6 | `1222, 1223, 1252, 1203, 1221, 1216` | `1221.5` |
| Stage2 | 6 | `285, 336, 326, 276, 283, 264` | `284.0` |
| Stage3 | 6 | `282, 245, 278, 297, 235, 237` | `261.5` |

Evidence root:

```text
/data-2/experiment_registry/calibration_runs/af1a407f
```

The queue later wrote a prediction contract with an empty workload feature map, but it
did not reach acceptance. All GPUs and run-owned containers/tmux sessions were
released. The eighteen runs lack native response token count, EOS presence, and finish
reason telemetry. They are immutable diagnostic evidence only and are categorically
ineligible for `outcome_schema_version = 2` trusted history or authorization.

### Resolved Timing Blocker

History assembly correctly failed closed because the pre-resume implementation imposed
an unjustified one-second equality tolerance between two clocks with different start
semantics. Code tracing establishes that `timing_s/testing` starts immediately before
calling `_validate()`, whereas `validation_ready` is emitted only after `_validate()`
has switched joint-model validation weights, entered the dataloader, and materialized
the first validation batch. Both end after `_validate()` returns, immediately before
metrics logging. The trainer-wide timer therefore includes a non-negative pre-readiness
preparation interval that the required readiness-to-complete deadline intentionally
excludes. For Stage2 bootstrap repetition 0, the preserved evidence includes:

```text
timing_s/testing:                  88.79418030567467
validation_ready -> metrics_complete: approximately 80.58 seconds
```

The normative contract is now fixed below in AC-05 and AC-19. The Goal must not choose
whichever value makes prediction easier, introduce a post-hoc equality tolerance, or
silently discard the eighteen runs. It must:

1. trace the exact start/end semantics of both measurements in code and artifacts;
2. use the timeline-derived field as the normative wall-clock interval from validation
   rollout readiness to complete validation metrics;
3. enforce the reviewed containment relationship and clock precision below before
   rebuilding history;
4. obtain a fresh independent `READY` review for this clarification before changing
   assembler/checker/runner behavior or restarting GPU work;
5. add regression fixtures for Stage1, Stage2, and Stage3 timing evidence, then rerun
   CPU gates and issue a fresh preflight.

That gate passed: the canonical interval is
`metrics_complete.monotonic_seconds - validation_ready.monotonic_seconds`, while
`timing_s/testing` is required diagnostic containment evidence and must be greater than
or equal to the canonical interval. Equality or a post-hoc tolerance is forbidden.
Focused timing tests, the fast gate, the full gate, and Git isolation passed for the
committed timing implementation.

Deterministic audit then confirmed that all eighteen bootstrap
repetitions are complete under the timing contract, but the later outcome-schema audit
found that they lack native response token/EOS/finish telemetry. They remain immutable
diagnostic evidence and must not enter trusted history v2. After the outcome-schema-v2
implementation and its fresh preflight pass, the Goal must run a new six-repetition
bootstrap cohort per phase, freeze immutable history, generate the prediction contract,
run three new acceptance repetitions per phase, obtain checker-owned `deployable`,
and complete fresh independent AC-01 through AC-30 acceptance.

### Current Blocking Condition

#### Runtime-port isolation incident - 2026-07-13

The current committed Stage1/Stage2 calibration queue at superproject `deae0c2b`
completed Stage1 bootstrap repetition 0, then failed repetition 1 before
`validation_ready`. Preserve both repetitions under
`/data-2/experiment_registry/calibration_runs/deae0c2b_stage12` as diagnostic-only
evidence. Ray-session logs prove that this is not a stale process from repetition 0:
the repetition-1 Ray core worker PID 12329 bound its gRPC server to port `43063`
before the FSDP process group attempted to create its TCPStore on the same port.
Rank 0 then reported `EADDRINUSE`; the remaining ranks connected to the Ray gRPC
listener and reported an invalid TCPStore ping value.

The defect is a run-internal port-allocation TOCTOU in which
`get_master_addr_port()` probes a free port with a temporary socket and releases it
before TCPStore binds. A teardown delay alone does not fix it. Before another GPU
probe or canonical cohort:

1. obtain independent plan `READY` for AC-26A below;
2. make the PPO colocated-worker path pass a calibration-supplied
   `trainer.ray_master_port_range` into every applicable `RayWorkerGroup` as
   `master_port_range`;
3. configure the calibration-local Ray cluster's worker ports through a supported
   Ray startup interface, not unsupported `ray.init()` kwargs and not runtime-env
   variables; Ray worker ports and TCPStore ports must occupy disjoint controlled
   ranges;
4. keep the queue serial and require both pre-repetition and post-repetition quiet
   gates for the run-owned tmux session, container, and controlled port ranges;
5. never use global `ray stop --force` as run-owned cleanup, and fail closed when
   ownership-scoped cleanup cannot prove release;
6. treat port isolation as runtime safety policy, not as a model/data/GPU resource
   profile input: the canonical L40S resource-profile hash must remain unchanged;
7. add CPU behavior tests for PPO range propagation, supported Ray startup command
   generation, disjoint ranges, busy-port rejection, queue blocking, ownership-only
   cleanup, and unchanged resource-profile hash;
8. obtain independent `READY FOR COMMIT`, commit, and obtain committed-state
   `CPU ACCEPTED`; then issue fresh preflight evidence and run two consecutive
   Stage1 diagnostic probes before creating a new canonical cohort.

Outcome-schema-v2, validation eligibility, stable source UID routing, and PM2-only CI
keepalive are committed and received fresh independent `CPU ACCEPTED` review at:

```text
superproject: c62b60932f2e93d4c0cea6e6fb4181e6db56b3be
recipe:       7a7a4983b8d6adc568936ca656118dcae3de8f08
```

A fresh preflight bound to those commits passed under
`/data-1/tmp/verl_agent_scratch/experiment_workflow/preflight/c62b6093`. The first
Stage1 bootstrap probe under `/data-2/experiment_registry/calibration_runs/c62b6093_v4`
then completed in 1165 seconds, passed the 1379-row ordered stable-source-UID check, and
cleaned up all run-owned GPU/runtime state. It is nevertheless diagnostic-only:

1. native telemetry measured `truncation_rate = 0.29731689630166785`, which exceeds
   the immutable `0.01` hard gate; the per-dataset length-finished counts were
   HumanEval+ `8/164`, MBPP+ `22/378`, and LiveCodeBench `380/837`;
2. the probe and formal Stage1 path used the base pretrained snapshot at
   `/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
   while the actual FRAC25 Stage1 experiment family initializes from the retained
   format-SFT model at
   `/data-1/model_weights/format_cold_start_fraction/qwen3-1p7b-kodcode-format-sft-frac25`;
3. the latter is provenance-bound by `format_cold_start_source.json` to
   `SFT-FORMAT-COLDSTART-Qwen3-1P7B-CODE-KODCODE-FRAC25-V1_1783390514/global_step_30`.

The mismatch is a workload-identity defect, not grounds to tune the truncation threshold,
shorten context, filter datasets, or reinterpret telemetry. Preserve `c62b6093_v4`
unchanged and never admit it to trusted history after the Stage1 model identity changes.
Before implementation or another GPU launch:

1. obtain independent plan `READY` for the exact phase model-role contract below;
2. add an explicit manifest path for the FRAC25 format-SFT Stage1 initialization and
   bind its safe artifact hash plus provenance-file hash;
3. make formal Stage1 and Stage1 calibration consume that same manifest identity;
4. classify Stage1 workload provenance as `sft_checkpoint`, not `base_pretrained`;
5. preserve Stage2 model1 as the base pretrained model and Stage2 model2 as the retained
   Stage1 handoff, matching the existing fixed-model2 joint topology; preserve Stage3
   as the extracted Stage2 model2;
6. make descriptor, queue, preflight, and eligibility checks fail closed on any path,
   role, provenance, or hash mismatch;
7. add regression fixtures, run focused/fast/full/PM2/isolation/transaction gates,
   commit recipe first, and obtain a new independent `CPU ACCEPTED` verdict;
8. generate a completely fresh preflight bound to the new commits before one new
   Stage1 probe in a new calibration root. Only a probe with UID, deadline, telemetry,
   truncation, timeout, and cleanup gates all passing may unlock the remaining queue.

## Non-Negotiable Boundaries

1. Stage1, Stage2, and Stage3 continue to use the same canonical L40S resource
   profile, including `MAX_RESPONSE_LENGTH=8192`.
2. Full HumanEval+, MBPP+, and LiveCodeBench remain the formal validation set.
   A bounded preflight sample is infrastructure evidence only, never experiment
   quality evidence.
3. The implementation must not increase reward-worker concurrency without a
   measured peak-RSS budget and a passing memory-safety gate.
4. Failed or incomplete runs must not enter SQLite or W&B.
5. Long-running GPU, Docker, Ray, or transfer commands run in tmux.
6. Existing user changes and historical scripts must not be reverted merely to
   make the new structure cleaner.
7. Acceptance must not contact real W&B, WxPusher, Hugging Face, or other remote
   services.
8. No hard or soft gate has an automatic override path. Structural, provenance,
   semantic, profile-identity, model/data hash, checkpoint-placement, and host-memory
   failures remain blocked. A soft performance-threshold failure stops the Goal,
   notifies the user through WxPusher when unattended, and requires an explicit
   user decision followed by a reviewed manifest/policy commit and a fresh preflight.
9. A single phase's real pre-training measurement or formal `val_before_train`
   validation has a hard 30-minute wall-clock budget from validation rollout readiness
   to complete validation metrics. Exceeding the budget is a blocking failure, not a
   slow-but-acceptable state.
10. WxPusher sends only event-driven notifications: one verified run-start event,
    one failure event, or one human-decision-required event. It sends no periodic
    healthy-progress messages. A run-start notification requires the first training
    step or complete formal validation metrics; tmux/container existence alone is not
    evidence that an experiment has started successfully.
11. Any local CI, queue-monitor, or reliability-check process that must survive shell
    disconnects or restart after failure is managed by PM2, not systemd. The Goal must
    not create a systemd unit or call `systemctl` for CI keepalive. It must provide a
    committed PM2 ecosystem definition (or equivalent committed PM2 command contract),
    machine-check `pm2 start`, `pm2 status`, log paths, restart behavior, and `pm2 save`.
    Host-reboot restoration uses an already provisioned non-systemd PM2 bootstrap plus
    `pm2 resurrect`; if none exists, record the limitation and stop rather than using
    `pm2 startup` to install a systemd-backed unit.

## Repository Transaction Baseline

The Goal spans two Git repositories and must treat them as one ordered transaction:

| Repository | Baseline branch | Baseline commit | Goal branch |
| --- | --- | --- | --- |
| Superproject `/data-1/code/verl` | `feature/on-policy-wdl-sft` | `6a86aa27591178f9acf774fe107b1219e099af99` | `codex/experiment-execution-reliability` |
| Submodule `/data-1/code/verl/recipe` | `codex/l40s-readiness` | `763aab506cd38bc7ff9fccfd9a079840620c37c5` | `codex/experiment-execution-reliability` |

At the original Goal baseline, before the Goal branches were created, path/status
snapshots for both repositories were written to:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline/superproject.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline/recipe.json
```

These snapshots are protected evidence of pre-existing user changes. Every affected
milestone commits the submodule first, then commits superproject files plus the exact
submodule pointer. Staging is path-scoped. If an intended file already contains
unrelated edits that cannot be separated safely, stop and ask the user.

The following pre-existing untracked recipe paths are the only initial adoption set:

```text
on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh
on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh
on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh
on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh
on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh
on_policy_wdl_sft/code_task/stage123_gpu_idle_watchdog.py
on_policy_wdl_sft/code_task/stage123_preflight.py
```

They are not treated as immutable unrelated work. AC-25 governs their controlled
adoption. Every other pre-existing dirty path in either repository remains protected.

## Milestone 0: Baseline and Fixtures

Produce a read-only inventory and deterministic local fixtures before changing
runtime behavior.

Artifacts:

```text
docs/joint_training/reports/experiment_workflow_baseline.md
tests/experiment_workflow/fixtures/
tests/experiment_workflow/test_fixture_contract.py
```

### AC-01 - Baseline Records the Current Bottleneck

- Given the current Stage123 logs and local runtime artifacts,
- When the baseline collector runs,
- Then it emits machine-readable JSON plus a Markdown summary containing scorer
  timeout count/rate, score count/distribution, elapsed scorer time, peak observed
  RewardLoopWorker RSS, GPU idle intervals when evidence exists, profile hash,
  Docker image ID, and explicit `unknown` values where evidence is unavailable.

Verification:

```bash
python3 scripts/experiment_workflow_baseline.py \
  --fixture tests/experiment_workflow/fixtures/stage123_scorer_timeout \
  --json-out /data-1/tmp/verl_agent_scratch/goal/baseline.json \
  --md-out /data-1/tmp/verl_agent_scratch/goal/baseline.md
python3 -m pytest -q tests/experiment_workflow/test_fixture_contract.py
```

Expected evidence: both commands exit `0`; JSON schema validation passes; fixture
contains no secrets or remote-service credentials.

### AC-02 - Failure Taxonomy Is Deterministic

- Given fixtures for path/config failure, model provenance failure, CUDA OOM,
  Ray host-memory kill, scorer timeout, dependency failure, early queue exit, and
  release failure,
- When the classifier runs,
- Then each fixture receives exactly one primary terminal reason and zero or more
  secondary signals in JSON.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_failure_classifier.py
```

Expected evidence: all eight primary classes are asserted by name; unknown input
returns `unknown` rather than a fabricated diagnosis.

## Milestone 1: Preflight Performance Estimator

This milestone produces a bounded, representative scorer benchmark shared by
Stage1, Stage2, and Stage3. It must complete before manifest migration begins.

Artifacts:

```text
recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_benchmark.py
recipe/on_policy_wdl_sft/code_task/preflight_benchmark_schema.json
scripts/check_code_task_preflight_budget.py
tests/experiment_workflow/test_preflight_benchmark.py
```

### AC-03 - Representative Three-Phase Benchmark

- Given one Stage1 model source, one Stage2 joint/fixed-Model2 source, one Stage3
  model source, the canonical 8K resource profile, and bounded deterministic rows
  from HumanEval+, MBPP+, and LiveCodeBench,
- When the preflight benchmark runs in fixture/mock mode,
- Then it produces comparable per-phase metrics for rollout latency/tokens per
  second, scorer latency, timeout rate, score distribution, valid-score rate,
  peak RSS, and estimated GPU-wait fraction without changing context length or
  reward semantics.

The benchmark contract pins deterministic row IDs and counts per dataset, three
measured repetitions after one excluded warmup, timeout rate denominator as all
submitted scoring items, fixed score bins, RSS sampling at no more than one second,
and GPU-wait fraction as wall time with every GPU at or below 2% utilization while
the phase is active divided by measured phase wall time. Stage2 uses the manifest's
fixed Model2 source plus joint fused-loss topology, not a generic joint-model mock.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_preflight_benchmark.py
python3 recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_benchmark.py \
  --fixture tests/experiment_workflow/fixtures/preflight_three_phase \
  --output /data-1/tmp/verl_agent_scratch/goal/preflight.json
python3 scripts/check_code_task_preflight_budget.py \
  --report /data-1/tmp/verl_agent_scratch/goal/preflight.json \
  --policy tests/experiment_workflow/fixtures/preflight_policy.json
```

Expected evidence: all commands exit `0`; report identifies all three phases and
all three validation datasets; `max_response_length` equals `8192` in every phase.

### AC-04 - Launch Budget Blocks Predictable Failure

- Given a preflight report whose timeout rate, invalid-score rate, peak RSS, or
  estimated GPU-wait fraction exceeds its manifest policy threshold,
- When the launch-budget checker runs,
- Then it exits nonzero, names every exceeded threshold, and prevents formal queue
  launch. Hard-gate failures remain blocked. Soft-threshold failures enter the
  user-decision workflow in AC-20; neither `--force` nor an approval file may bypass
  the failed report.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_preflight_budget_gate.py
```

Expected evidence: passing, failing, missing-metric, hard/soft force-flag rejection,
and delegated AC-20 decision-notification cases are covered; no bypass exists.

### AC-05 - Performance Changes Require Before/After Evidence

- Given a proposed scorer/rollout/validation optimization,
- When its benchmark is compared with the frozen baseline fixture,
- Then the report records before/after timeout rate, valid scores per minute,
  p50/p95 scorer latency, peak RSS, and estimated GPU-wait fraction, and rejects
  a change that improves throughput by weakening dataset, context, timeout-result,
  or scoring semantics.

The primary optimization metric is valid scores per wall-clock minute. A performance
optimization is accepted only when its repeated median improves by at least 10%,
timeout rate does not regress by more than one absolute percentage point, peak RSS
does not increase by more than 10%, and semantic hashes are identical. Results below
that threshold are `neutral`: acceptable only for a separately justified correctness
or maintainability change, never evidence that performance was optimized.
In addition, every Stage1/Stage2/Stage3 formal `val_before_train` path must produce
complete full-validation metrics within 30 minutes after validation rollout readiness.
The historical 76-minute Stage2 pre-validation stall is a required regression fixture.
An implementation that improves relative throughput but still exceeds 30 minutes is
not accepted.

Formal code validation intentionally preserves sampled pass@1 decoding because the
experiment-quality contract takes precedence over making runtime prediction
artificially deterministic. The estimator therefore models a distribution rather than
claiming a single deterministic upper bound. Its inputs include phase, model parameter
count and provenance, canonical resource-profile hash, response-length distribution,
validation dataset composition and difficulty strata, scorer timeout/latency
distribution, and prior trusted runs with identical semantic hashes. Missing or
out-of-domain history widens the interval or returns `inconclusive`; it never silently
substitutes greedy decoding or a narrower validation set.

The dynamic estimator contract is fixed as `stage123_history_conformal_v1`:

0. The canonical `validation_elapsed_seconds` is the monotonic wall-clock interval
   `metrics_complete.monotonic_seconds - validation_ready.monotonic_seconds`. The
   `validation_ready` event is emitted after first-batch materialization and immediately
   before validation rollout preparation/dispatch; `metrics_complete` is emitted after
   `_validate()` has returned complete validation metrics and before external metrics
   logging. Event timestamps are captured from one process with `time.monotonic()` and
   retained as the original finite numeric values. Derived durations are computed from
   those unrounded timestamps and serialized with outward six-decimal rounding wherever
   they form an acceptance interval or hard-bound comparison. `timing_s/testing` is a
   required diagnostic whole-`_validate()` timer, not the predictor/deadline target: it
   starts before validation weight switching and dataloader/first-batch preparation, so
   it may exceed the canonical interval by any non-negative preparation duration. The
   evidence is consistent iff events occur exactly once in the order
   `validation_ready`, `generation_complete`, `metrics_complete`, all timestamps are
   finite and nondecreasing, the canonical interval is non-negative, and the unrounded
   `timing_s/testing` is greater than or equal to the unrounded canonical interval.
   Numerical equality and a fixed absolute-difference tolerance are forbidden because
   the intervals intentionally have different start points. Missing/duplicate events,
   clock reversal, non-finite values, or a trainer timer shorter than the canonical
   interval makes the repetition ineligible and `blocked`; it is never repaired by
   selecting another timing source.

1. A run is eligible history only when its release gate passed, its calibration
   artifacts remain content-addressed and readable, and its validation dataset hashes,
   sampled-decoding semantic hash, resource-profile hash, phase topology, scorer hash,
   timeout policy, and `MAX_RESPONSE_LENGTH` exactly match the requested phase.
2. Eligible history comes from a content-addressed immutable JSON snapshot, never a
   live database query. Before contract generation, the snapshot freezes a cutoff as
   UTC RFC3339 with whole seconds and `Z`; runs completed after that cutoff are ignored.
   Its canonical bytes and SHA-256 are recorded in the contract. Eligible runs are
   sorted by completion timestamp ascending and then run ID ascending, and the last
   twelve are selected. Operators cannot include, exclude, reorder, or weight individual
   runs. The current calibration's predictor and measured repetitions are never history
   inputs.
3. At least six eligible prior runs per phase are required. The estimator uses at most
   the latest twelve. Fewer than six returns `out_of_domain` and `inconclusive`; it
   cannot borrow from another semantic/profile/phase cohort.
   If a new exact-match cohort has fewer than six trusted runs, Milestone 5 first runs
   a bootstrap calibration batch of six independent repetitions per phase under the
   already reviewed semantic/profile contract and 30-minute timeout. Bootstrap runs
   are infrastructure calibration evidence only: they cannot produce `deployable`,
   cannot enter experiment-result SQLite/W&B, and any failed/timeout/incomplete run is
   ineligible and must be replaced by a fresh repetition. After six eligible runs are
   release-gated, their immutable snapshot and cutoff are frozen. Only then is
   `prediction_contract.json` generated, followed by three new acceptance repetitions
   per phase that are excluded from that snapshot. This ordered bootstrap is the only
   history-creation path; it cannot reuse acceptance measurements or weaken hashes.
4. Features are fixed to phase, log2 parameter count, model provenance class, per-dataset
   row counts and difficulty-stratum counts, response-length p50/p95/truncation rate,
   scorer-latency p50/p95, scorer-timeout rate, and peak RSS. This list is split to
   prevent future-information leakage:

   - launch-time exact-match workload descriptors are phase, base-2 `log2` model
     parameter count rounded to six decimals, model provenance class, per-dataset row
     counts, per-dataset difficulty-stratum counts, and the deterministic validation
     eligibility set produced by the pinned tokenizer/chat-template/prompt-length
     filter. They must be fully known from pinned model/config/dataset/code artifacts
     before the contract is generated;
   - response-length p50/p95/truncation rate, scorer-latency p50/p95,
     scorer-timeout rate, and peak RSS are run outcomes. They are stored in every
     historical repetition and receive frozen cohort predictions/intervals alongside
     elapsed time; they are never copied from the current acceptance run into its
     pre-execution feature vector. Acceptance records them after execution and checks
     them against the already frozen contract.

   The manifest schema adds `calibration_workloads.<phase>` for all three phases. Each
   object requires:

   ```text
   phase
   parameter_counter_version
   rollout_model_parameter_counts[]
   rollout_model_parameter_count_sum
   log2_rollout_model_parameter_count_sum
   model_provenance_class
   model_sources[] = {role, path, artifact_sha256, hash_algorithm}
   datasets[] = {name, path, sha256, row_count, uid_source,
                 difficulty_resolution, difficulty_mapping_sha256,
                 difficulty_stratum_counts}
   tokenizer = {path, config_sha256, tokenizer_sha256}
   validation_eligibility = {max_prompt_length, filter_enabled,
                             filter_implementation_sha256,
                             ordered_eligible_uid_sha256,
                             per_dataset_eligible_counts,
                             submitted_prompt_count}
   outcome_schema_version = 2
   ```

   Phase names and ordered source roles are exact-match fields. Stage1 has one
   `base_pretrained` rollout source; Stage2 has ordered `model1` and fixed `model2`
   sources with class `fixed_model2_joint_rollout`; Stage3 has one
   `stage2_model2_handoff` source. `hash_algorithm` is
   `sorted_relative_path_content_sha256_v1`: for a file it is ordinary SHA-256; for a
   directory, sort all regular files by POSIX relative path, then hash for each file
   the 8-byte big-endian path length, UTF-8 path bytes, and the file-content SHA-256
   bytes. A Hugging Face snapshot may contain relative symlinks only when each resolved
   target is a regular file under the same `models--*/blobs/` cache root; hash it under
   the symlink's snapshot-relative path using the resolved target content. Absolute,
   dangling, directory, special-file, symlink-chain, or cache-root-escaping links are
   rejected. Symlinks outside this HF snapshot rule and all special files are rejected.
   The manifest schema and renderer
   reject missing/extra descriptor keys, incorrect order, non-integer counts, and a
   declared artifact hash that does not match the pinned source.

   `model_parameter_count` is an explicit integer in the manifest, verified before
   contract generation by a schema-versioned architecture counter over the pinned
   Hugging Face `config.json`; the counter must account for tied embeddings and every
   architecture-specific projection exactly and must be covered by a known Qwen3-1.7B
   fixture. It describes the rollout model actually used by that phase; joint Stage2
   records both ordered submodel counts and their sum.
   The first version is `hf_qwen3_config_parameter_count_v1`; it accepts only
   `model_type=qwen3`, rejects unknown architecture/config keys that affect tensor
   shapes, uses `tie_word_embeddings` to count a shared LM head exactly once, and emits
   the ordered component formula plus total in verification evidence. The manifest's
   base-2 logarithm must equal `round(log2(sum), 6)` using round-half-even.
   `model_provenance_class` is one of `base_pretrained`, `sft_checkpoint`,
   `stage1_model2_handoff`, `stage2_model2_handoff`, or
   `fixed_model2_joint_rollout`, with the exact source artifact SHA-256 recorded
   separately. Dataset row counts come from Parquet metadata. Difficulty strata must
   come from an immutable, content-addressed dataset-side mapping keyed by dataset UID;
   score, reward, elapsed time, or output length from calibration may not define or
   relabel difficulty. If a dataset has no reviewed mapping, the only permitted value
   is a declared `unstratified` bucket containing every row, and the contract records
   `difficulty_resolution: unavailable`; it may establish exact workload identity but
   may not support a claim about difficulty-conditioned prediction.

   Dataset UID is the UTF-8 string in Parquet `extra_info.uid`. Every row must have a
   unique non-empty UID. A difficulty mapping is canonical JSON with schema version,
   dataset name/hash, and a UID-to-stratum object whose keys equal the dataset UID set;
   its SHA-256 is over canonical JSON bytes. For the permitted `unstratified` fallback,
   no external mapping file is invented: `difficulty_mapping_sha256` is the SHA-256 of
   canonical `{schema_version,dataset_name,dataset_sha256,strata:{unstratified:[sorted
   UIDs]}}`, and the sole count equals Parquet `row_count`.

   Full-validation provenance and submitted-outcome denominators are distinct. The
   three pinned source Parquet files and all 1422 source rows remain mandatory. Before
   contract generation, the descriptor reuses the canonical `RLHFDataset` message
   construction, pinned tokenizer/chat-template settings, `filter_overlong_prompts`,
   and `max_prompt_length=1024` to compute the ordered eligible UID sequence. It records
   the filter implementation SHA-256, ordered eligible UID SHA-256, per-dataset eligible
   counts, and total `submitted_prompt_count`. The observed 2026-07-12 Stage1 diagnostic
   fixture is 1422 source rows and 1379 deterministically eligible/submitted prompts;
   both values must be recomputed from artifacts, never hardcoded to make the run pass.
   A source row may be ineligible only through that pinned filter. Any tokenizer,
   chat-template, prompt construction, filter implementation/config, UID ordering, or
   eligible-count drift is an exact-match failure.
   `ordered_eligible_uid_sha256` is SHA-256 over canonical JSON bytes with one trailing
   newline for `{schema_version:1,datasets:[{name,source_index,ordered_uids}]}`; object
   keys are lexicographically sorted, separators are `,` and `:`, UTF-8 is unescaped,
   and dataset/source order plus UID order are preserved exactly.

   Eligibility has one versioned implementation entry point:
   `recipe/on_policy_wdl_sft/code_task/calibration_validation_eligibility.py`. It must
   instantiate the project-native `RLHFDataset` with the exact rendered manifest data
   config and pinned tokenizer, then read the filtered dataframe UID sequence; it must
   not duplicate prompt formatting or token-length logic. `filter_implementation_sha256`
   is the canonical SHA-256 over `{eligibility_tool_sha256, rl_dataset_sha256,
   normalized_manifest_data_config}`. The normalized config binds
   prompt key, chat-template kwargs, tool schemas, truncation mode, filter enabled flag,
   filter worker semantics, and max prompt length. Contract generation recomputes this
   descriptor from artifacts and rejects a manifest-provided value that differs. The
   superproject and recipe commits are bound separately by preflight/calibration
   receipts; they are not embedded in the recipe manifest descriptor because doing so
   would create a recipe-first submodule-pointer commit cycle.

   Response lengths are token counts of the complete generated response under the
   pinned rollout tokenizer, excluding padding and including EOS when present. p50 and
   p95 use nearest-rank order statistics: sort ascending and select one-based rank
   `ceil(q * N)` for `q=0.50` or `q=0.95`. Truncation means the response consumed exactly
   `MAX_RESPONSE_LENGTH` non-padding response tokens without EOS. Scorer latency and
   timeout use the existing per-sample reward telemetry and `submitted_prompt_count`
   denominator.
   Peak RSS keeps the readiness-to-completion cgroup sampling contract. Feature and
   outcome names, units, missing-value rejection, tokenizer/config hashes, ordering,
   and derivation algorithm are schema-versioned and hashed.

   New run evidence must write one row per submitted validation item with stable dataset
   `source_uid`, `response_token_count`, `response_eos_present`, `response_finish_reason`, response text,
   `code_reward_latency_seconds`, timeout/status, and score. Token count/EOS come from
   the rollout response tensor/mask before decoding, not tokenizer re-encoding. Missing
   `response_finish_reason` is exactly `stop` when EOS is present, `length` when the
   non-padding count equals `MAX_RESPONSE_LENGTH` without EOS, and `unknown` otherwise;
   `unknown` is incomplete telemetry and makes outcome-schema-v2 history ineligible.
   `RLHFDataset.__getitem__` must copy Parquet `extra_info.uid` into a distinct non-tensor
   `source_uid`. That field must survive DataProto batching, validation repeat, divisor
   padding/unpadding, generation, reward scoring, and JSONL dumping without replacement
   or reordering. The existing `uid` remains a transient random rollout/request-grouping
   identifier for advantage grouping and validation metric aggregation; it must never be
   treated as dataset provenance or substituted for `source_uid`.
   `build_validation_generation_samples` and the validation JSONL dump use `source_uid`
   for their externally visible `uid` column so existing consumers retain their schema
   while receiving the stable asset identity. Retaining source identity only in
   transient tracking samples is insufficient.
   Missing
   rows, duplicate UIDs, a row count other than `submitted_prompt_count`, an observed
   UID sequence that differs from `ordered_eligible_uid_sha256`,
   missing token/EOS/finish telemetry, or disagreement between runtime and artifact
   counts makes the repetition ineligible. Historical text-only JSONL may be re-encoded
   only for diagnostic p50/p95 analysis; it cannot prove EOS/truncation and therefore
   cannot enter a complete `outcome_schema_version=2` trusted cohort.
   The preserved `baaa596b_v2` Stage1 repetition 0 proves the deterministic
   1422-to-1379 count transition and native token/EOS/finish telemetry, but its JSONL
   omitted UID; it is diagnostic-only and must not enter trusted history. After this
   contract is implemented and committed, all bootstrap evidence starts from another
   fresh calibration root.
   The later `7c1ed4e1_v3` root is also diagnostic-only. Stage1 bootstrap repetition 0
   proved the 1422-to-1379 denominator, native response telemetry, deadline completion,
   and high GPU utilization, but its JSONL exposed random request UUIDs rather than
   Parquet source UIDs, so independent ordered source-UID verification failed.
   Repetition 1 was deliberately terminated immediately after discovery. Neither run
   nor any partial artifact in that root may enter trusted history or acceptance. The
   source-UID fix requires new committed code, fresh CPU acceptance, fresh preflight,
   and another completely new calibration root.

   For each continuous outcome
   `validation_elapsed_seconds`, `peak_rss_gib`, `response_length_p50_tokens`,
   `response_length_p95_tokens`, `scorer_latency_p50_seconds`, and
   `scorer_latency_p95_seconds`, point prediction, leave-one-out residual, finite-sample
   rank, interval construction, outward six-decimal rounding, measured three-run median,
   at-most-20% point error, per-repetition containment, median containment, overlap, and
   the 50%-of-midpoint informativeness limit are exactly the elapsed-time algorithm in
   items 5 and AC-19. Peak RSS retains its stricter 25%-of-midpoint width limit.
   `truncation_rate` and `scorer_timeout_rate` are bounded rates: their frozen predicted
   interval is `[max(0,min(raw)-1/N), min(1,max(raw)+1/N)]`, where `N` is the exact full
   submitted-item count, with outward six-decimal rounding. Each of three acceptance
   rates and their aggregate submitted-item rate must lie inside that interval;
   truncation rate above `0.01` or scorer timeout rate above `0.10` is `blocked`.
   A continuous diagnostic with a non-informative interval is `inconclusive`; a rate
   interval wider than `0.25` is `inconclusive`. These checks are additional to, and do
   not replace, elapsed deadline, scorer validity, RSS, and GPU-idle gates. Scorer
   latency p50/p95 and response p50/p95 both use nearest-rank statistics over all
   submitted items in a repetition.

   The current `stage123.yaml` lacks a complete descriptor schema and the pre-resume
   contract emitted empty phase features. That contract is preserved as failed evidence
   and must not authorize acceptance. After this amendment is independently `READY`,
   deterministic enrichment may derive diagnostic response-length and scorer summaries
   from all eighteen raw bootstrap generation/timeline/resource artifacts, provided
   every input path and content hash is recorded and all eighteen pass the same
   derivation. Because those artifacts lack original response token count/EOS/finish
   telemetry, they remain preserved diagnostic evidence but are ineligible for the new
   complete outcome schema. After instrumentation and CPU gates pass, the Goal must run
   a new six-repetition bootstrap cohort per phase under the unchanged sampled/full/8K
   semantics. The Goal must create a new history snapshot, prediction contract,
   preflight receipt, and calibration root after the implementing commit; it must never
   mutate or reuse the frozen `af1a407f` contract as an authorization artifact.
5. Elapsed-time and RSS point predictions use the cohort median. For each historical
   value `y_i`, the leave-one-out residual is exactly
   `r_i = abs(y_i - median(y_j for j != i))`. Residuals are sorted numerically ascending
   while preserving duplicates. The finite-sample rank is
   `k = min(n, ceil((n + 1) * 0.90))`; `q` is the one-based `k`th sorted residual. The
   prediction interval is exactly
   `[max(0, cohort_median - q), cohort_median + q]`. Equal values retain their duplicate
   ranks; no tie interpolation occurs. Exact internal decimal values are rounded only
   when serialized: lower bounds toward negative infinity and upper bounds toward
   positive infinity to six decimal places. GPU-idle uses the cohort raw-value interval
   `[max(0, min(values)-0.02), min(1, max(values)+0.02)]` with the same outward rounding.
6. Before any predictor or measured run starts, the history index, selected cohort,
   features, algorithm version, coverage `0.90`, interval parameters, hashes, and
   generated predictions are frozen in `prediction_contract.json`. Its SHA-256 is
   propagated into every run artifact. Changing it requires a new reviewed commit,
   new preflight, and a completely new calibration root.
7. A contract is non-informative and returns `inconclusive` if elapsed-time interval
   width exceeds 50% of its midpoint, RSS width exceeds 25% of its midpoint, GPU-idle
   width exceeds `0.75`, or leave-one-run-out empirical coverage is below 0.80. The
   `0.75` GPU-idle limit deliberately admits the current approximately `0.08` to `0.70`
   bimodal fixture while rejecting near-full-range evidence such as `[0.02,0.87]`.
   Coverage, margins, width limits, cohort size, and algorithm may not be lowered or
   widened after measured evidence exists. An elapsed upper bound at or above 1800
   seconds is a hard runtime-risk result and maps to `blocked`, not `inconclusive`.
8. Canonical JSON throughout this contract is UTF-8, keys sorted recursively, compact
   separators `(',', ':')`, no NaN/Infinity, and exactly one trailing newline. SHA-256
   is computed over those canonical bytes. Any manifest, policy, history-snapshot,
   semantic-contract, or prediction-contract byte change creates a new hash and
   invalidates downstream evidence.

Verification:

```bash
python3 -m pytest -q \
  tests/experiment_workflow/test_preflight_comparison.py \
  tests/experiment_workflow/test_dynamic_calibration_interval.py \
  tests/experiment_workflow/test_validation_readiness.py
```

Expected evidence: semantic-downscope fixtures fail even when their throughput is
higher; neutral results cannot satisfy a performance-optimization claim; the frozen
dynamic estimator contract is byte-reproducible across edge cases; and the historical
76-minute `val_before_train` fixture is classified as exceeding the 1800-second hard
deadline even if relative throughput improved.

### AC-06 - Real GPU Smoke Is Explicitly Approval-Gated

- Given the local L40S host and a plan-approved benchmark manifest,
- When an operator requests the representative GPU smoke,
- Then the command refuses to start without `ALLOW_CODE_PREFLIGHT_GPU_SMOKE=1`,
  runs in tmux when approved, writes only under declared scratch/report roots,
  and never labels bounded scores as formal experiment results.

Verification:

```bash
bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_preflight_gpu_smoke.sh
python3 -m pytest -q tests/experiment_workflow/test_gpu_smoke_guard.py
```

Expected evidence: guard test proves the unapproved path exits nonzero without
starting Docker/Ray. Actual GPU execution is operational evidence, not required
for sandbox acceptance.

## Milestone 2: Experiment Manifest and Generated Workflow

Artifacts:

```text
recipe/on_policy_wdl_sft/experiment_manifest/schema.json
recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
scripts/experiment_manifest.py
tests/experiment_workflow/test_experiment_manifest.py
docs/joint_training/reports/legacy_queue_monitor_inventory.md
```

### AC-07 - Manifest Owns Run Identity and Lifecycle

- Given a valid experiment manifest,
- When the manifest tool renders its normalized form,
- Then run prefix, phase, source model/checkpoint provenance, train/validation
  files, final step, resource-profile hash, preflight policy, tmux name, artifact
  root, release policy, and monitor policy come from the manifest exactly once.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_experiment_manifest.py
python3 scripts/experiment_manifest.py validate \
  recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
python3 scripts/experiment_manifest.py render \
  recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
  --format json > /data-1/tmp/verl_agent_scratch/goal/stage123.normalized.json
```

Expected evidence: duplicate run identities, missing provenance, invalid phase
order, mismatched profile hashes, and checkpoint roots outside the declared mount
fail validation.

### AC-08 - Queue and Monitor Consume the Same Manifest

- Given the Stage123 manifest,
- When queue and monitor dry-runs execute,
- Then both derive the same ordered run list, final steps, train files, tmux names,
  and release metadata without independent hard-coded arrays.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py
bash -n recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
bash -n recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh
```

Expected evidence: a fixture mutation changes queue and monitor output together;
grep/static test rejects reintroduction of Stage123 run-prefix arrays.

### AC-09 - Release Gate Uses Manifest Evidence

- Given a completed fixture run and its manifest,
- When the release hook is invoked against local fake SQLite/W&B directories,
- Then it verifies final checkpoint, metrics, manifest run identity, profile hash,
  dataset hashes, normalized-manifest hash, and content-addressed preflight report
  copied into immutable run provenance before recording release success. Every
  launched run must reference a passing preflight generated after the latest reviewed
  manifest/policy commit; there is no forced-run release path.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_manifest_release_gate.py
```

Expected evidence: incomplete, hash-mismatched, failed-preflight, and wrong-run
fixtures cannot be imported or marked W&B-synced.

### AC-10 - Historical Migration Is Bounded

- Given the existing queue/monitor/script index,
- When the inventory command runs,
- Then it classifies each discoverable family as `manifest-native`,
  `legacy-traceable`, or `legacy-unresolved`, records evidence paths, and does not
  block the goal on unresolved historical archaeology.

Verification:

```bash
python3 scripts/experiment_manifest.py inventory \
  --root recipe/on_policy_wdl_sft \
  --output /data-1/tmp/verl_agent_scratch/goal/legacy_inventory.json
python3 -m pytest -q tests/experiment_workflow/test_legacy_inventory.py
```

Expected evidence: Stage123 is `manifest-native`; unresolved legacy entries are
explicit rather than silently omitted.

### AC-11 - New Experiment Gate Rejects Legacy Structure

- Given version-controlled legacy inventories for both repositories containing
  schema version, the baseline commits above, and exact legacy runnable paths,
- When a runnable queue/monitor is added, changed, or renamed relative to those
  baselines,
- Then the gate fails unless the path is unchanged and allowlisted legacy, or is
  manifest-native. Renames are treated as delete plus add, and submodule paths are
  evaluated against the submodule baseline rather than filesystem dates or mtimes.

Verification:

```bash
bash scripts/check_experiment_workflow_fast.sh
python3 -m pytest -q tests/experiment_workflow/test_new_experiment_gate.py
```

Expected evidence: compliant and intentionally noncompliant fixture trees prove
the cutoff rule without rewriting historical scripts; tampered baseline/allowlist
hashes fail closed.

## Milestone 3: Project Principles and Engineering Experience Audit

Artifacts:

```text
docs/joint_training/constraints/principles/engineering_rule_catalog.md
scripts/check_engineering_rule_catalog.py
docs/joint_training/reports/engineering_rule_migration.md
tests/experiment_workflow/test_engineering_rule_catalog.py
```

### AC-12 - Rules Use Trigger, Action, Failure Structure

- Given project-level principles and operational rules selected for migration,
- When the catalog checker runs,
- Then every prose rule contains an explicit `When`, `Do`, and `Otherwise`
  failure field, a stable rule ID, scope, evidence source, and enforcement tier.

Verification:

```bash
python3 scripts/check_engineering_rule_catalog.py \
  docs/joint_training/constraints/principles/engineering_rule_catalog.md
python3 -m pytest -q tests/experiment_workflow/test_engineering_rule_catalog.py
```

Expected evidence: malformed, triggerless, actionless, and consequence-free rule
fixtures fail with line-specific diagnostics.

### AC-13 - Machine-Checkable Rules Point to Gates

- Given a rule whose condition can be checked deterministically,
- When the audit classifies it,
- Then the prose catalog contains only a one-line pointer to its checker, command,
  test ID, failure code, and fast/full gate reachability; duplicated normative prose
  is removed from the canonical source or clearly marked noncanonical.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_rule_enforcement_tiers.py
```

Expected evidence: structural, machine-check, and judgment-only examples are all
covered; a machine-checkable prose-only rule or unreachable checker fails the audit.

### AC-14 - Existing Principles Are Migrated Without Invented History

- Given `CLAUDE.md`, `AGENTS.md`, `docs/joint_training/constraints/`, active
  runbooks, and project memory indexes,
- When the migration report is generated,
- Then each selected rule is marked `retained`, `reworded`, `replaced-by-gate`,
  `project-local`, `global-candidate`, or `dropped-unverifiable`, with source-path
  evidence and no fabricated incident claims.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_rule_migration_report.py
```

Expected evidence: every migrated rule resolves to an existing source path; rules
without recoverable trigger/failure evidence may be dropped explicitly.

### AC-15 - Startup Context Remains Small

- Given the revised project entrypoint and rule catalog,
- When the documentation-size gate runs,
- Then startup-required files remain within their declared size budget and bulk
  inventories/results live in indexed Markdown or JSON rather than `AGENTS.md` or
  `CLAUDE.md`.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_context_size_budget.py
```

Expected evidence: oversized startup-document fixture fails; the real project
passes its documented budget. An exception requires owner, reason, plan/issue path,
reviewer approval, expiry, maximum permitted lines and bytes, and fails automatically
after expiry. No exception may exceed 400 lines or 128 KiB.

## Milestone 4: Integrated Fast and Full Gates

### AC-16 - Fast Gate Is Cheap and Deterministic

- Given a normal developer checkout with no Docker/GPU requirement,
- When `bash scripts/check_experiment_workflow_fast.sh` runs,
- Then it validates Python/shell syntax, manifest schema, generated contracts,
  fixture tests, release classification, and rule catalog in at most 120 seconds
  on the current L40S host.

Verification:

```bash
/usr/bin/time -f '%e' bash scripts/check_experiment_workflow_fast.sh
```

Expected evidence: exit `0`; elapsed time is recorded and no remote network call
occurs. Network isolation is proven with a network-disabled namespace/container or
an injected socket-deny layer whose attempted connection fixture fails the gate.

### AC-17 - Full Gate Preserves Exit Codes and External Isolation

- Given local Docker access and deterministic fixtures,
- When `bash scripts/check_experiment_workflow_full.sh` runs,
- Then it includes the fast gate plus container imports, reward mock execution,
  queue/monitor/release integration, preserves failing child exit codes, and uses
  fake W&B/WxPusher endpoints only.

Verification:

```bash
bash scripts/check_experiment_workflow_full.sh
python3 -m pytest -q tests/experiment_workflow/test_gate_exit_propagation.py
```

Expected evidence: both pass; an injected failing child command makes the full
gate nonzero even when output is piped or summarized.

### AC-18 - Stage123 End-to-End Dry Run Is Manifest-Native

- Given the accepted Stage123 manifest and local model/data artifacts,
- When the formal queue is run with `DRY_RUN=1`,
- Then preflight, phase order, handoff provenance, queue output, monitor output,
  checkpoint placement, and mocked release behavior all agree with the normalized
  manifest and write no durable experiment result. Dry-run must not start real
  Docker, Ray, or tmux; before/after snapshots compare both Git repositories plus
  SQLite row count/hash, release JSONL size/hash, W&B sync markers, checkpoint roots,
  monitor status roots, and other declared durable paths.

Verification:

```bash
STAGE123_MANIFEST=recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
DRY_RUN=1 bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
python3 -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py
git status --short --ignored
```

Expected evidence: commands pass; dry-run artifacts stay under declared scratch;
no SQLite/W&B success record is created.

## Milestone 5: Operational Calibration and Repository Acceptance

Milestone 5 runs only after sandbox AC-01 through AC-18 pass. It uses the local L40S
host but no remote service. Its metrics are infrastructure evidence, not experiment
quality evidence.

### AC-19 - Operational Calibration Produces a Deployment Decision

- Given sandbox acceptance is green, the canonical profile/image/model/data hashes
  are pinned, and the user explicitly approves bounded local L40S calibration,
- When Stage1, exact Stage2 fixed-Model2 joint topology, and Stage3 calibration run
  in tmux using the same content-addressed benchmark contract as AC-03 while preserving
  sampled pass@1 validation decoding,
- Then the report records full provenance, before/after observed metrics, estimator
  prediction error, and one decision: `deployable`, `blocked`, or `inconclusive`.
  `Deployable` requires the AC-05 improvement/safety budget. For validation elapsed time
  and peak RSS, both gates must pass: the point prediction relative error against the
  median of three valid measured repetitions is at most 20%, and all three measured
  values plus their median lie inside the frozen prediction interval. Sampled-validation
  `all_gpu_idle_fraction_during_validation` uses interval overlap rather than a
  single-point 20% requirement. Its measured interval from exactly three valid
  repetitions is `[max(0,min(raw)-0.02), min(1,max(raw)+0.02)]`, outward-rounded to six
  decimals; fewer than three valid repetitions is `blocked`. Overlap is exactly
  `max(pred_low, measured_low) <= min(pred_high, measured_high)`. GPU idle has no
  physical hard threshold. The conservative predicted validation-time upper bound must
  remain below 30 minutes. The report records raw per-run values, feature/provenance
  hashes, history selection, interval method,
  confidence/coverage target, and out-of-domain status. Status mapping is exhaustive:
  fewer than six trusted history runs, OOD features, a non-informative interval, or
  insufficient leave-one-out coverage returns `inconclusive`; semantic/profile/
  provenance mismatch, measured timeout, incomplete metrics, scorer safety failure,
  elapsed/RSS point-prediction relative error above 20%, any elapsed/RSS repetition or
  its median outside the frozen interval,
  elapsed upper bound at or above 1800 seconds, measured/predicted interval
  non-overlap, or stale/mismatched receipt returns `blocked`. It cannot be overridden.
  Each phase also has an
  independent 30-minute hard runtime timeout and must produce complete full-validation
  metrics; any timeout, incomplete metric set, or scorer stall returns `blocked`.

For AC-19 prediction, the 1800-second hard deadline, interval coverage, point error,
and valid-scores-per-minute denominator, `validation_elapsed_seconds` always means the
canonical timeline-derived readiness-to-`metrics_complete` interval defined in AC-05.
`timing_s/testing` remains required diagnostic evidence and must contain that interval,
but it is not substituted into prediction or deadline calculations.

#### AC-19A - Prediction Contract Is Frozen Before Execution

- Given a reviewed manifest/policy commit and a content-addressed immutable trusted-
  history snapshot whose cutoff, canonical SHA-256, and selection order are frozen,
- When any predictor or measured repetition is requested,
- Then `prediction_contract.json` is generated first with the exact history query,
  selected and excluded run IDs, feature vector, algorithm version, fixed parameters,
  hashes, OOD decision, and predictions; current calibration runs are excluded and the
  contract hash cannot change within the calibration root. Under outcome schema v2,
  predictor evidence is the frozen six-run-per-phase trusted-history cohort bound by
  `history_index` and `prediction_contract`; it is not copied into or relabeled as an
  acceptance repetition. Every v2 history row must contain
  `evidence_role="bootstrap_history"`, its canonical queue-native `artifact_root` under
  `report/bootstrap/<phase>/rep_0..5`, and content hashes for status, resources, metrics,
  generation, and timeline. Missing/unknown roles, roots outside that exact bootstrap
  subtree, root/phase/rep mismatches, duplicate roots, acceptance roots, or hashes that
  do not match files are rejected before cohort selection. A caller-provided role string
  alone never establishes provenance.

  The v2 assembler accepts one `--root` equal to the queue's `REPORT_ROOT` and resolves
  only `acceptance/<phase>/rep_0`, `rep_1`, and `rep_2`. Its candidate report records
  `predictor_repetitions: []`, contract predictor count `0`, and exactly those three
  acceptance repetitions. It rejects missing/extra acceptance repetitions and any mixed
  legacy directories (`<phase>/rep0_predictor*`, `<phase>/rep1..3`) within the supplied
  root. The checker verifies predictor provenance only from the exact bound immutable
  `history_index` and `prediction_contract`; it rejects nonzero report predictor counts,
  synthetic predictors, bootstrap-as-acceptance, acceptance-as-history, and any report
  repetition whose recorded root is not the canonical queue-native acceptance root.

Verification:

```bash
python3 -m pytest -q \
  tests/experiment_workflow/test_operational_calibration_assembler.py \
  tests/experiment_workflow/test_operational_calibration_checker.py \
  tests/experiment_workflow/test_operational_calibration_runner.py
```

Expected evidence: queue-native roots assemble without directory renaming; the report
contains zero predictor repetitions and three acceptance repetitions per phase; exact
history and prediction-contract bindings remain mandatory. Named fixtures cover zero
predictor count, exact `rep_0..2` discovery, missing/extra/mixed-legacy roots,
bootstrap-as-acceptance, acceptance-as-history with a missing role, acceptance with a
forged bootstrap role, artifact-root mismatch, artifact hash mismatch, and duplicate
artifact roots; every case except the canonical queue-native layout fails closed.

#### AC-19B - History Selection Is Deterministic

- Given trusted calibration history,
- When `stage123_history_conformal_v1` selects a cohort,
- Then it applies the exact semantic/profile/phase eligibility and chronological
  latest-twelve rule above; fewer than six eligible runs returns `inconclusive`, and
  operator-selected history or cross-cohort borrowing fails closed.

#### AC-19C - Interval Algorithm Is Byte-Reproducible

- Given the same frozen cohort and feature inputs,
- When the estimator runs repeatedly,
- Then it emits byte-identical median predictions, split-conformal intervals,
  empirical GPU-idle interval, coverage results, and contract hash using the fixed
  finite-sample and rounding rules above.

#### AC-19D - Intervals Cannot Be Widened After Evidence

- Given a frozen prediction contract,
- When coverage is insufficient, an interval exceeds its fixed informativeness limit,
  or elapsed-time upper bound is at or above 1800 seconds,
- Then insufficient history, OOD, non-informative width, or insufficient coverage is
  `inconclusive`; elapsed upper bound at or above 1800 seconds is `blocked`; coverage
  downgrade, cohort switching,
  outlier removal, margin changes, and post-hoc widening are rejected.

#### AC-19E - GPU Idle Is Distributional Diagnostic Evidence

- Given sampled validation can create bimodal rollout/scoring overlap,
- When deployability is checked,
- Then complete metrics, elapsed deadline, scorer health, timeout rate, and RSS remain
  hard gates; the three-repetition measured GPU-idle interval must overlap the frozen
  informative predicted interval but is not assigned an invented physical safety
  threshold. Raw values remain in the report and cannot be replaced by a selected run.

#### AC-19F - Checker Owns the Only Deployment Decision

- Given the assembler has produced a content-addressed calibration candidate,
- When the checker validates it,
- Then the assembler never self-declares `deployable`; the checker alone writes a
  receipt binding report, manifest, policy, history-index, and prediction-contract
  SHA-256 values to `deployable`, `blocked`, or `inconclusive`. The receipt uses the
  canonical JSON rule above, contains `issued_at` as UTC RFC3339 whole seconds with `Z`,
  and binds the exact report, manifest, policy, immutable history snapshot, semantic
  contract, and prediction-contract hashes. The manifest field
  `calibration_receipt_max_age_seconds` is fixed at `86400`; freshness is checked at
  formal-queue admission with at most `300` seconds future clock skew. A receipt is not
  single-use: within its TTL it may be reused only for repeated admission of the same
  formal queue identity and the exact same bound hashes. Reuse for another queue
  identity, after any bound-byte change, or outside the age/skew limits is replay and
  must be rejected. No mutable consumption ledger is introduced. Formal execution
  accepts only this fresh exact-matching checker receipt. The deployability receipt
  also binds the exact AC-24 preflight-receipt SHA-256; it supplements rather than
  replaces the preflight receipt.

Verification:

```bash
python3 -m pytest -q \
  tests/experiment_workflow/test_dynamic_calibration_interval.py \
  tests/experiment_workflow/test_operational_calibration_assembler.py \
  tests/experiment_workflow/test_operational_calibration_checker.py
python3 scripts/check_calibration_prediction_contract.py \
  --contract /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/prediction_contract.json \
  --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
  --history-index /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/trusted_history.json
python3 scripts/check_code_task_operational_calibration.py \
  --report /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/report.json \
  --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml \
  --contract /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/prediction_contract.json \
  --history-index /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/trusted_history.json \
  --receipt /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/deployability_receipt.json
```

Expected evidence: checker exits `0` only for `deployable`; `blocked` and
`inconclusive` preserve reports and keep formal deployment disabled. If GPU execution
is unavailable, sandbox implementation may be accepted but workflow deployment stays
`PENDING OPERATIONAL CALIBRATION`, not complete.
Fixtures must reject same-input nondeterminism, current-run leakage, operator-selected
history, checker invocation without `--history-index`, use of fewer than six bootstrap
runs, bootstrap-as-deployable, acceptance-run
leakage into the history snapshot, semantic/profile mismatch, coverage downgrade, post-hoc
widening, excessive interval width, upper bound equal to or above 1800, non-overlapping
intervals, elapsed/RSS point error above 20%, any elapsed/RSS repetition outside its
frozen interval, acceptance of the approximately `[0.08,0.70]` bimodal GPU-idle fixture,
rejection of `[0.02,0.87]`, assembler self-declaration, TTL/skew boundaries, permitted
same-queue exact-hash receipt reuse, and rejected cross-queue or changed-hash replay.

#### AC-19G - Stable Source UID Is Separate From Rollout Grouping UID

- Given a validation Parquet row with a unique non-empty UTF-8 `extra_info.uid`,
- When `RLHFDataset.__getitem__`, collation, DataProto repeat, divisor padding/unpadding,
  rollout generation, reward extraction, tracking-sample construction, and JSONL dump
  execute,
- Then `source_uid` equals the original Parquet value with exact order and multiplicity;
  the externally visible schema-compatible JSONL/tracking `uid` equals `source_uid`;
  transient random `uid` remains a separate rollout/request-grouping key passed to
  `process_validation_metrics`; and missing, empty, replaced, reordered, or duplicated
  source identity fails closed.

Given expected ordered eligible source UIDs and observed JSONL rows, when the sets match
but order differs, or a verifier derives or hardcodes expected order from the observed
artifact itself, then verification fails. A transient request UUID may be retained only
under a separately named diagnostic field such as `request_id`; it is never provenance.
The diagnostic-only roots `af1a407f`, `baaa596b_v2`, and `7c1ed4e1_v3`, including any
partial repetition, are explicitly ineligible as trusted-history or acceptance input.

Verification:

```bash
python3 -m pytest -q \
  tests/utils/dataset/test_rl_dataset_on_cpu.py \
  tests/joint_training/regression/test_validation_generation_logging.py \
  tests/experiment_workflow/test_calibration_outcomes.py \
  tests/experiment_workflow/test_operational_calibration_runner.py
```

Expected evidence: fixtures prove missing/empty Parquet UID rejection, exact source UID
survival across repeat and pad/unpad, transient UID retention for metrics, stable source
UID emission for tracking/JSONL, ordered-hash rejection despite set equality, rejection
of observed-artifact-derived expectations, and diagnostic-root exclusion. No test may
replace ordered comparison with a set-only check or pin the UUIDs produced by one run.

### AC-20 - Soft Threshold Failure Stops and Requests User Decision

- Given every hard gate passes but a report exceeds one or more soft performance
  thresholds during an unattended Goal,
- When the launch gate evaluates that report,
- Then it exits nonzero, records `USER_DECISION_REQUIRED`, and sends one deduplicated
  WxPusher notification containing the experiment/run identity, relevant background,
  failed thresholds and observed metrics, estimated GPU/time cost, recommended action,
  and local report/manifest paths. WxPusher is notification-only: phone delivery,
  clicks, or replies never authorize execution. The Goal resumes only after the user
  returns to the interactive task, explicitly decides, the agent updates the
  manifest/policy in Git with the decision rationale, a fresh independent review
  accepts that commit, and a newly generated preflight passes the revised policy.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_user_decision_notification.py
```

Expected evidence: hard and soft failures both block launch; notification payload is
complete, secret-free, deduplicated, and uses a local fake in acceptance; no response
channel or force flag can authorize launch; only a passing post-commit preflight can
clear the block.

The shared notification policy is event-only: `run_started`, `run_failed`, and
`user_decision_required`. `run_started` requires first-step or complete-validation
evidence; `run_failed` and `user_decision_required` include background, evidence,
cost, recommendation, and local paths. Healthy polling never sends WxPusher.

### AC-21 - Dual-Repository Change Isolation

- Given the recorded baseline commits and dirty-path snapshots for the superproject
  and recipe submodule,
- When each milestone is committed,
- Then only declared milestone paths differ from the snapshots, each affected
  submodule commit precedes its superproject pointer commit, unrelated dirty paths
  are unchanged, and final reviewer evidence covers both commit ranges.

Verification:

```bash
python3 scripts/check_goal_git_isolation.py \
  --superproject /data-1/code/verl \
  --submodule /data-1/code/verl/recipe \
  --baseline-root /data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline
git -C /data-1/code/verl diff --check 6a86aa27591178f9acf774fe107b1219e099af99..HEAD
git -C /data-1/code/verl/recipe diff --check 763aab506cd38bc7ff9fccfd9a079840620c37c5..HEAD
```

Expected evidence: path-scoped commits and pointer order pass; modification of a
protected pre-existing dirty path or an unrecorded commit fails.

### AC-22 - Dirty Baselines Are Content-Addressed

- Given both repositories contain tracked and untracked pre-existing user work,
- When the original Goal baseline is recorded before Goal-branch creation and that
  immutable evidence is revalidated on resume,
- Then every path entry includes repository-relative path, file type, mode, size,
  SHA-256 for file/symlink content, Git status, index blob ID when present, HEAD blob
  ID when present, and an aggregate sorted-manifest hash. Later checks detect content
  changes, deletion, type/mode changes, and replacement of protected untracked files.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_git_content_baseline.py
python3 scripts/check_goal_git_isolation.py \
  --superproject /data-1/code/verl \
  --submodule /data-1/code/verl/recipe \
  --baseline-root /data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline
```

Expected evidence: modification, deletion, replacement, chmod, symlink-target change,
and manifest truncation fixtures fail; unchanged protected work passes.

### AC-23 - Goal Completion Requires Real Deployable Calibration

- Given sandbox AC-01 through AC-18 pass,
- When real L40S calibration has not run or returns `blocked`/`inconclusive`,
- Then the highest allowed status is `READY FOR OPERATIONAL CALIBRATION`; the Goal
  remains active and cannot be marked complete. `GOAL COMPLETE` requires AC-19 to
  return `deployable` from real local measurements and an independent reviewer to
  mark AC-19 through AC-25 `PASS`. A phase exceeding the 30-minute validation budget
  can never yield `deployable`. Where exact-match history is initially insufficient,
  the ordered six-run bootstrap per phase in AC-05 must complete before the three
  acceptance repetitions; bootstrap evidence alone never satisfies AC-19 or this AC.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_goal_completion_states.py
```

Expected evidence: missing, blocked, and inconclusive calibration cannot produce a
complete status; only reviewer-accepted deployable evidence can.

### AC-24 - Formal Execution Cannot Bypass Preflight

- Given any Stage123 formal queue or direct phase launch,
- When `SKIP_STAGE123_PREFLIGHT`, `SKIP_STAGE123_MACHINE_GATE`, a force/skip alias,
  a missing/failed/stale report, a report for another manifest/run/profile, or direct
  phase invocation attempts to bypass the launch gate,
- Then execution exits nonzero before Docker, Ray, tmux, or trainer startup. Queue and
  every phase launcher require both receipts: the same content-addressed preflight
  receipt, whose report/manifest/profile hashes match and whose age is within the
  manifest policy, and the fresh `deployable` AC-19F checker receipt that binds that
  exact preflight-receipt SHA-256 plus the calibration evidence hashes. Neither receipt
  contains or replaces the other; failure of either independently blocks admission.
  `ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1` remains only a positive formal-launch guard
  and never substitutes for the receipt.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_no_preflight_bypass.py
bash scripts/check_experiment_workflow_fast.sh
```

Expected evidence: static checks reject skip/force variables in formal launch paths;
behavioral fixtures cover missing, failed, stale, mismatched, queue-skipped, and
direct-phase cases for each receipt independently, including a deployability receipt
bound to another preflight receipt; all fail before side effects.

### AC-25 - Stage123 Dirty Files Use a Controlled Adoption Transaction

- Given the eight exact Stage123 recipe paths listed in the repository baseline and
  their content-addressed AC-22 records,
- When Milestone 2 adopts and migrates them to the manifest-native workflow,
- Then an adoption manifest records each original hash/type/mode/size, intended
  ownership, allowed milestone, resulting hash, and baseline-to-adopted diff. Only
  those listed files may change under adoption; all other dirty paths remain
  byte-identical. A fresh reviewer accepts the recipe diff and commit parentage before
  the superproject records the new submodule pointer.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_stage123_adoption.py
python3 scripts/check_goal_git_isolation.py \
  --superproject /data-1/code/verl \
  --submodule /data-1/code/verl/recipe \
  --baseline-root /data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline \
  --adoption-manifest docs/joint_training/manifests/stage123_dirty_adoption.json
```

Expected evidence: listed baseline-to-adopted changes pass with a reviewer record;
unlisted changes, missing original hashes, altered adoption scope, wrong recipe parent,
or a superproject pointer committed before recipe review fail.

### AC-26 - Validation Deadline Cleans Up Runtime and Releases GPUs

- Given a Stage1/Stage2/Stage3 pre-training measurement or formal
  `val_before_train` runtime whose complete metrics are absent at the 30-minute
  deadline,
- When the hard-wall controller fires,
- Then it marks the run blocked, stops the owning queue from advancing, requests
  graceful phase termination, escalates to force termination after a bounded grace
  period, removes run-owned Ray/Docker/tmux/process descendants, and verifies no GPU
  process/allocation remains attributable to that run. Cleanup is idempotent; an
  incomplete cleanup remains a blocking failure and emits evidence rather than
  claiming resources were released.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_validation_deadline_cleanup.py
```

Expected evidence: scaled-clock fixtures cover completion before deadline, graceful
shutdown, forced shutdown, orphan descendants, cleanup-command failure, repeated
cleanup, and the historical 76-minute step-0 trace. Only complete cleanup reports
`resources_released=true`; the legacy run fixture also proves release gate blocked,
zero matching SQLite rows/W&B sync markers, and no residual runtime ownership.

#### AC-26A - Calibration Runtime Ports Are Ownership-Isolated

- Given the local Docker harness uses host networking and a calibration repetition
  creates both Ray worker gRPC listeners and an FSDP TCPStore,
- When a Stage1, Stage2, or Stage3 bootstrap or acceptance repetition starts and
  terminates,
- Then Ray worker listeners use a calibration-local controlled range and every PPO
  colocated `RayWorkerGroup` receives a disjoint controlled TCPStore range through
  `master_port_range`. The ranges are supplied through supported runtime interfaces,
  remain outside the canonical L40S resource-profile hash, and are recorded in local
  runtime evidence. Before launch and after termination, the queue proves that the
  run-owned tmux session and container are absent and both controlled ranges are
  quiet. A busy port, residual run-owned container/session, unsupported Ray argument,
  overlapping range, or incomplete ownership-scoped cleanup blocks the queue before
  the next repetition. Cleanup never invokes global `ray stop --force`, never kills
  an unrelated Ray process/container/session, and never claims release from process
  name matching alone.

Verification:

```bash
python3 -m pytest -q \
  tests/experiment_workflow/test_operational_calibration_runner.py \
  tests/experiment_workflow/test_operational_calibration_runtime_isolation.py \
  tests/experiment_workflow/test_validation_deadline_cleanup.py
```

Expected evidence: fixtures prove `trainer.ray_master_port_range` reaches the PPO
colocated `RayWorkerGroup` constructor; the generated calibration command uses a
Ray-version-supported worker-port mechanism; Ray and TCPStore ranges are disjoint;
busy controlled ports and residual run-owned resources prevent the next repetition;
quiet resources allow it; cleanup contains no global Ray stop; an unrelated listener
and container survive; and the canonical Stage123 L40S resource-profile hash is byte
identical before and after runtime isolation is enabled. After committed-state CPU
acceptance, two consecutive Stage1 diagnostic probes in one serial queue must both
reach complete validation metrics without `EADDRINUSE`, invalid TCPStore ping,
timeout, fatal termination, or residual run-owned runtime before AC-19 calibration
may resume.

### AC-27 - Notification Policy Is an Event State Machine

- Given local fake WxPusher delivery and run-state fixtures,
- When the shared notification policy observes runtime transitions,
- Then it sends at most one message per deduplication key for exactly three events:
  `run_started` after first training step or complete formal validation metrics,
  `run_failed` after terminal failure/cleanup evidence, and
  `user_decision_required` after a blocking decision gate. Tmux/container creation,
  model loading, incomplete validation, healthy polling, unchanged metrics, and phone
  delivery/reply state emit no authorization and no periodic notification.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_notification_policy.py
```

Expected evidence: all three positive events, every listed non-event, deduplication,
secret redaction, local-path inclusion, and fake-delivery failure are covered. A
delivery failure is recorded locally but never changes launch/release state.

### AC-28 - Persistent CI Uses PM2 and Never Systemd

- Given a committed PM2 ecosystem definition or equivalent committed PM2 command
  contract for every persistent local CI, queue-monitor, or reliability-check process,
- When the PM2 keepalive checker and its sandbox fixtures execute,
- Then they prove that the declared process can be started, inspected, restarted after
  a synthetic failure, mapped to explicit stdout/stderr log paths, and persisted with
  `pm2 save`; no keepalive path creates a systemd unit, calls `systemctl`, or invokes
  `pm2 startup`; and a host without an already provisioned non-systemd PM2 reboot
  bootstrap records `reboot_restore_available=false` and fails closed instead of
  installing systemd.

Verification:

```bash
python3 -m pytest -q tests/experiment_workflow/test_pm2_ci_keepalive.py
python3 scripts/check_pm2_ci_keepalive.py \
  --contract config/pm2/experiment-reliability-ci.json \
  --repo-root . \
  --require-no-systemd
```

Expected evidence: the tests use an isolated fake PM2 binary/state directory and cover
`start`, `status`, explicit log paths, synthetic crash/restart, `save`, `resurrect`,
missing reboot bootstrap, and command failure. The checker exits zero only when the
committed contract names every persistent process, contains no `systemctl`, systemd unit,
or `pm2 startup` path, and records either a verified existing non-systemd reboot
bootstrap or the fail-closed limitation. No real PM2 daemon or host service manager is
acceptance evidence.

### AC-29 - Phase Model Identity Matches the Formal Stage123 Experiment

- Given manifest fields that pin the exact FRAC25 format-SFT directory, its
  `format_cold_start_source.json`, the provenance schema version, the safe model-tree
  hash, and the provenance-file hash; the base pretrained Qwen3-1.7B snapshot; the
  retained Stage1 handoff; and, once produced, the current 20-step Stage2 model2 plus
  its run ID, final step, extraction path, model-tree hash, and provenance-file hash,
- When the manifest normalizer, workload-descriptor generator, formal queue dry-run,
  calibration phase sandbox launch, calibration queue sandbox launch, preflight receipt
  issue/verify path, deployability admission, and validation eligibility recomputation
  resolve model roles from a clean sandbox,
- Then Stage1 resolves only the manifest-pinned FRAC25 format-SFT initialization with
  provenance class `sft_checkpoint`; Stage2 resolves model1 to the base pretrained
  snapshot and model2 to the exact Stage1 handoff; Stage3 resolves only the model2
  extracted from the named current Stage2 run at `final_step = 20`; every consumer
  records identical canonical paths, safe artifact hashes, provenance schema versions,
  and provenance hashes; and any missing, not-yet-produced, substituted, role-swapped,
  stale, or hash-mismatched model fails before Docker, tmux, Ray, or trainer startup.
  Stage1 is admitted as the explicit source dependency of each Stage2 run rather than
  as a synthetic standalone entry in `runs`; the source-dependency admission is bound
  into the preflight receipt and is mandatory for direct Stage1 calibration. Until the
  current Stage2 20-step output exists, Stage3 descriptor generation, preflight, and
  calibration fail closed; the old `step_40_s2steps60` model is never an exact-match
  substitute.

Verification:

```bash
python3 -m pytest -q \
  tests/experiment_workflow/test_experiment_manifest.py \
  tests/experiment_workflow/test_calibration_workload_descriptor.py \
  tests/experiment_workflow/test_operational_calibration_runner.py \
  tests/experiment_workflow/test_no_preflight_bypass.py \
  tests/experiment_workflow/test_stage123_preflight_model_identity.py \
  tests/experiment_workflow/test_stage123_end_to_end.py
python3 recipe/on_policy_wdl_sft/code_task/calibration_validation_eligibility.py --help
python3 recipe/on_policy_wdl_sft/code_task/calibration_workload_descriptor.py --help
bash scripts/run_code_task_operational_calibration_queue.sh --sandbox-dry-run
bash recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh --sandbox-dry-run stage1
bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh --dry-run
python3 scripts/stage123_preflight_receipt.py verify --help
```

Expected evidence: positive fixtures bind the exact Stage1/2/3 role graph and hashes;
negative fixtures cover the previous base-pretrained Stage1 substitution, missing
format-SFT provenance/schema/hash, changed model bytes, Stage2 role swap,
calibration/formal queue drift, stale preflight bindings, a missing current Stage2
20-step output, and substitution of the legacy 60-step model. Sandbox commands must be
implemented as side-effect-free committed interfaces: no Docker, tmux, Ray, trainer,
GPU process, PM2 daemon, or external service contact may start.

### AC-30 - Stage2 Producer Uses a Checker-Owned Limited Receipt

- Given the only authorized producer identity:

  ```text
  run_id: frac25-stage2
  run_prefix: CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V1
  final_step: 20
  train_file_sha256: 160be1866e6c1dc439dcfbd594b54324f000f1f48db1f6a0fc88cf227c628dab
  expected_output_path: /data-2/model_weights/code_task/qwen3_1p7b_stage123/frac25_p40_s220_s340/stage2_final_model2
  expected_provenance_path: /data-2/model_weights/code_task/qwen3_1p7b_stage123/frac25_p40_s220_s340/frac25-stage3.provenance.json
  ```

  plus six eligible Stage1 and Stage2 bootstrap repetitions, three fresh Stage1 and
  Stage2 acceptance repetitions, a fresh preflight receipt authorizing only materialized
  Stage1/Stage2 calibration identities, and a candidate report containing no Stage3
  measurements because the current Stage2 20-step producer output does not yet exist,
- When the operational calibration checker evaluates the phase-scoped report with
  `--authorization-scope stage12_producer`, and the dedicated
  `stage123_stage12_producer_receipt.py` verifier
  evaluates its canonical receipt for the named producer run,
- Then checker may issue only decision `stage12_calibrated` with receipt type
  `code_task_operational_calibration_stage12_producer`, an exact allowlist containing
  `authorized_run_ids = ["frac25-stage2"]`, `authorized_final_steps =
  {"frac25-stage2": 20}`, and `phase_scope = ["stage1", "stage2"]`. It binds the exact
  run prefix, train-file hash, expected output/provenance paths, manifest, separate
  Stage1 and Stage2 selected-cohort hashes, phase-scoped history index, prediction
  contract, policy, preflight receipt, workloads, and candidate report. The limited
  receipt cannot authorize Stage1/Stage3 formal training,
  another Stage2 run, DB/W&B release, the full deployability verifier, or Goal
  completion. After Stage2 model2 and provenance are materialized, Stage3 descriptors
  and preflight evidence are regenerated, Stage3 bootstrap and acceptance execute, and
  only the existing complete three-phase checker path may issue final `deployable`.

Verification:

```bash
/data-1/verl07/run_train.sh /opt/venv/bin/python -m pytest -q \
  tests/experiment_workflow/test_stage12_producer_receipt.py \
  tests/experiment_workflow/test_operational_calibration_checker.py \
  tests/experiment_workflow/test_no_preflight_bypass.py \
  tests/experiment_workflow/test_manifest_release_gate.py
python3 scripts/check_code_task_operational_calibration.py --help
python3 recipe/on_policy_wdl_sft/code_task/stage123_stage12_producer_receipt.py --help
python3 recipe/on_policy_wdl_sft/code_task/stage123_deployability_receipt.py --help
```

Expected evidence: a new dedicated fixture module covers exact limited schema and
admission. Fixtures prove valid limited issuance, missing Stage1/Stage2 phase,
included Stage3 measurement, wrong producer run/final step, stale/mismatched hashes,
scope escalation, full-verifier misuse, Stage3 admission, release-hook use, and Goal
completion use all fail closed. The repo-native Docker test harness may provide Python
dependencies, but fixtures start no nested Docker container, GPU workload, tmux, Ray,
trainer, PM2, or external-service request.

#### AC-30A - Limited Receipt Schema Is Exact

- Given the exact producer identity above and a Stage1/Stage2-only candidate,
- When checker writes the limited receipt,
- Then its type, decision, phase scope, single-run allowlist, final-step map, run prefix,
  train hash, output/provenance paths, phase cohort hashes, and all input hashes match
  exactly; any extra/missing phase or producer field is blocked.

#### AC-30B - Limited Receipt Is Rejected Outside Producer Admission

- Given a valid `stage12_calibrated` receipt,
- When Stage1 admission, Stage3 admission, another Stage2 run, full deployability
  verifier, release hook, or completion-state checker evaluates it,
- Then each exits nonzero before side effects with `limited_receipt_scope_mismatch`.

#### AC-30C - Stage3 Regeneration Is Post-Producer Only

- Given `frac25-stage2` has not materialized model2 and provenance with matching hashes,
- When Stage3 descriptor, preflight, or calibration is requested,
- Then it fails closed; after materialization, descriptor and preflight are regenerated
  from the new hashes before any Stage3 bootstrap starts.

## Required Execution Order

Milestones 0 through 4 are implemented, committed, and independently CPU-accepted at
the commits recorded in the Resume Snapshot. On resume, execute this remaining order:

1. Revalidate the recorded commits, dirty-path baseline, preserved calibration evidence,
   and absence of conflicting run-owned runtime.
2. Treat the elapsed-time semantics amendment, shared timing parser, Stage1/2/3 timing
   fixtures, focused/fast/full CPU gates, and commit transaction as completed evidence.
3. Preserve the first frozen contract as failed diagnostic evidence: it exposed empty
   launch-time features and non-informative Stage2/Stage3 intervals. Do not mutate or
   authorize from that contract or its fresh-preflight predecessor.
4. Treat outcome schema v2, native telemetry, stable source UID, PM2-only keepalive,
   their CPU gates, and their accepted commits as completed evidence. Do not repeat them.
5. Obtain independent `READY` for AC-29, then implement the exact model identity,
   provenance schema/hash, source-dependency admission, Stage3 dynamic-output gate, and
   sandbox interfaces. Commit recipe first, run focused/fast/full/PM2/isolation/
   transaction gates, and obtain fresh independent `CPU ACCEPTED`.
6. Obtain independent `READY` for AC-26A, implement calibration-local disjoint Ray
   worker/TCPStore port domains plus ownership-scoped inter-repetition quiet gates,
   run the focused runtime-isolation and cleanup tests, and obtain independent
   committed-state `CPU ACCEPTED`. Do not alter the canonical L40S resource-profile
   hash and do not use global `ray stop --force`.
7. Generate fresh machine, budget, and preflight evidence bound to that committed state.
   Historical receipts must not authorize launch.
8. In a completely new diagnostic root, run two consecutive Stage1 bootstrap probes
   in one serial tmux queue. Both must pass AC-26A. The second probe specifically proves
   that repetition transition does not reproduce the `43063` collision.
9. In a completely new canonical calibration root, run exactly one Stage1 bootstrap probe in tmux.
   Only if UID, deadline, telemetry, timeout, truncation (`<= 0.01`), score, memory, and
   cleanup hard gates pass may the remaining Stage1 repetitions and Stage2 calibration
   proceed. A failed probe remains diagnostic and does not unlock the queue.
10. Complete six eligible Stage1/Stage2 bootstrap repetitions, freeze their history,
   generate their prediction contract, and run three fresh Stage1/Stage2 acceptance
   repetitions. Checker may issue only AC-30's `stage12_calibrated` limited receipt for
   the named 20-step Stage2 producer.
11. Run the authorized Stage2 producer. Materialize/hash/provenance-bind its model2,
   regenerate Stage3 descriptor and preflight evidence, then complete six Stage3
   bootstrap and three fresh Stage3 acceptance repetitions.
12. Assemble the complete three-phase candidate and require the original checker-owned
    `deployable` receipt. A limited receipt never satisfies this step.
13. A fresh independent Reviewer executes every AC-01 through AC-30 plus AC-26A command, the PM2
   keepalive checks, and the completion-state checker from committed code.

No milestone may start until all required ACs from the previous milestone pass.

## Commit and Workspace Rules

1. Continue on the existing `codex/experiment-execution-reliability` branches at the
   commits recorded in the Resume Snapshot. Do not create replacement branches. If
   either HEAD has advanced, audit the intervening commits and amend the snapshot
   before implementation.
2. Commit each milestone separately after its local verification passes; commit
   recipe first when affected, then the superproject and submodule pointer.
3. Stage only declared paths and do not stage or commit unrelated dirty changes.
4. Put test scratch under `/data-1/tmp/verl_agent_scratch/experiment_workflow/`.
5. Preserve failed-run logs and release evidence; do not clean checkpoints or
   W&B staging as part of this goal.

## Stop-and-Ask Triggers

Stop implementation and ask the user when:

1. an AC would require weakening full validation or the 8K response contract;
2. measured memory suggests more reward workers but no safe RSS budget can be
   demonstrated;
3. the manifest cannot represent an existing Stage123 behavior without changing
   experiment semantics;
4. acceptance would require a real external account/service;
5. the same blocker repeats three times;
6. a required upstream artifact is missing or its provenance cannot be proven;
7. implementation would need deleting or rewriting unrelated user changes.
8. any soft preflight threshold fails. When the Goal is unattended, send one guarded
   WxPusher `needs decision` message with background, evidence, cost, recommendation,
   and local paths, then stop. Do not treat WxPusher as approval.
9. any real pre-training measurement or formal `val_before_train` phase reaches 30
   minutes without complete validation metrics. Stop the affected runtime, preserve
   evidence, send one `run_failed` notification when unattended, and do not continue
   to consume all GPUs.
10. keeping CI alive would require creating or enabling a systemd service. Stop and
    report the missing non-systemd reboot bootstrap; do not substitute systemd for PM2.

## Approved Operational Decisions

The user approved these boundaries on 2026-07-11:

1. Stop the pre-Goal Stage123 run
   `CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V1_1783777744`.
   It stopped at observed step 0 after approximately 76 minutes in
   `val_before_train`; it is local diagnostic evidence only and must be rerun through
   the completed workflow.
2. L40S resources may be used for operational calibration.
3. Each Stage1/Stage2/Stage3 pre-training measurement has a maximum wall-clock budget
   of 30 minutes.
4. WxPusher sends once when a run is verified started, once when it fails, or once
   when human judgment is required; it does not send routine healthy-status updates.
5. On 2026-07-12 the user explicitly chose to preserve sampled pass@1 formal validation.
   Runtime prediction must use a deliberately loose, history-fitted dynamic model based
   on model size, phase, dataset difficulty/composition, response-length behavior,
   scorer behavior, and the shared resource profile. Experiment quality must not be
   weakened to make the runtime bound deterministic; the 30-minute wall-clock controller
   remains the final hard safety boundary.

## Independent Acceptance Contract

The implementer may report local verification but may not mark this goal accepted.
A fresh reviewer must:

1. run all AC verification commands in both repositories;
2. report AC-01 through AC-30 as `PASS`, `FAIL`, or `WEAKENED`;
3. inspect commits and tests for skipped, deleted, loosened, or trivial checks;
4. confirm acceptance used no real external service;
5. write the final review under `docs/joint_training/codereview/active/` and move
   it to `completed/` only after every required AC passes.

The goal is complete only when every AC is reviewer-owned `PASS` and none is
`WEAKENED`.
