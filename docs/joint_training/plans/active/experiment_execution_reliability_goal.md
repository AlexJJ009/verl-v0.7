# Experiment Execution Reliability and GPU Utilization Goal

- Status: `READY TO RESUME - CPU ACCEPTED; OPERATIONAL CALIBRATION BLOCKED ON ELAPSED-TIME EVIDENCE SEMANTICS (2026-07-12)`
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

## Resume Snapshot - 2026-07-12

This section is the authoritative resume point for the next `/goal` run. It updates
transient execution state without weakening or replacing AC-01 through AC-27.

### Accepted State

1. The amended plan preserving sampled pass@1 validation and the history-fitted
   dynamic interval contract received a fresh independent `READY` review.
2. CPU/sandbox implementation received independent `CPU ACCEPTED` review. The final
   accepted CPU state includes AC-19F, AC-24, and AC-25; the focused receipt tests,
   fast/full gates, strict Git isolation, and dual-repository transaction passed.
3. The independently CPU-accepted implementation baselines are:

   ```text
   superproject branch: codex/experiment-execution-reliability
   accepted implementation commit: af1a407fda562f1cef8fd9d4471f73f3de91814e
   recipe branch:       codex/experiment-execution-reliability
   accepted recipe commit:         cec05371fe17d42bb80722b5608c3cecbe4785b6
   ```

   The superproject may contain later plan-only resume commits. At Goal startup,
   classify every commit after the accepted implementation baseline; do not treat an
   expected plan-only commit as implementation drift.

4. The recipe-first transaction is complete and the recipe worktree is clean. The
   only remaining superproject untracked paths are pre-existing user assets recorded
   by the content-addressed baseline:

   ```text
   .claude/skills/experiment-registry
   docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
   ```

5. A passing preflight snapshot was generated under:

   ```text
   /data-1/tmp/verl_agent_scratch/experiment_workflow/preflight/af1a407f
   ```

   Its receipt has a one-hour TTL and is historical evidence only when the Goal is
   resumed. A fresh machine report, budget result, and preflight receipt bound to the
   then-current commits and hashes are required before any new GPU work.

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

The queue stopped before writing `trusted_history.json` or
`prediction_contract.json`. All GPUs and run-owned containers/tmux sessions were
released. These bootstrap artifacts are candidates for reuse only after the blocker
below is resolved and their completeness/content hashes are revalidated; they are not
trusted history merely because their phase status returned zero.

### Current Blocking Condition

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

After that gate passed, deterministic audit confirmed that all eighteen bootstrap
repetitions are complete under the timing contract, but the later outcome-schema audit
found that they lack native response token/EOS/finish telemetry. They remain immutable
diagnostic evidence and must not enter trusted history v2. After the outcome-schema-v2
implementation and its fresh preflight pass, the Goal must run a new six-repetition
bootstrap cohort per phase, freeze immutable history, generate the prediction contract,
run three new acceptance repetitions per phase, obtain checker-owned `deployable`,
and complete fresh independent AC-01 through AC-27 acceptance.

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
     counts, and per-dataset difficulty-stratum counts. They must be fully known from
     pinned model/config/dataset artifacts before the contract is generated;
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
   outcome_schema_version = 2
   ```

   Phase names and ordered source roles are exact-match fields. Stage1 has one
   `base_pretrained` rollout source; Stage2 has ordered `model1` and fixed `model2`
   sources with class `fixed_model2_joint_rollout`; Stage3 has one
   `stage2_model2_handoff` source. `hash_algorithm` is
   `sorted_relative_path_content_sha256_v1`: for a file it is ordinary SHA-256; for a
   directory, sort all regular files by POSIX relative path, then hash for each file
   the 8-byte big-endian path length, UTF-8 path bytes, and the file-content SHA-256
   bytes. Symlinks and special files are rejected. The manifest schema and renderer
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

   Response lengths are token counts of the complete generated response under the
   pinned rollout tokenizer, excluding padding and including EOS when present. p50 and
   p95 use nearest-rank order statistics: sort ascending and select one-based rank
   `ceil(q * N)` for `q=0.50` or `q=0.95`. Truncation means the response consumed exactly
   `MAX_RESPONSE_LENGTH` non-padding response tokens without EOS. Scorer latency and
   timeout use the existing per-sample reward telemetry and submitted-item denominator.
   Peak RSS keeps the readiness-to-completion cgroup sampling contract. Feature and
   outcome names, units, missing-value rejection, tokenizer/config hashes, ordering,
   and derivation algorithm are schema-versioned and hashed.

   New run evidence must write one row per submitted validation item with stable dataset
   UID, response token count, EOS-present boolean, finish reason, response text,
   `code_reward_latency_seconds`, timeout/status, and score. Token count/EOS come from
   the rollout response tensor/mask before decoding, not tokenizer re-encoding. Missing
   rows, duplicate UIDs, a row count other than the pinned full-validation count,
   missing token/EOS/finish telemetry, or disagreement between runtime and artifact
   counts makes the repetition ineligible. Historical text-only JSONL may be re-encoded
   only for diagnostic p50/p95 analysis; it cannot prove EOS/truncation and therefore
   cannot enter a complete `outcome_schema_version=2` trusted cohort.

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
  contract hash cannot change within the calibration root.

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
  --receipt /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/deployability_receipt.json
```

Expected evidence: checker exits `0` only for `deployable`; `blocked` and
`inconclusive` preserve reports and keep formal deployment disabled. If GPU execution
is unavailable, sandbox implementation may be accepted but workflow deployment stays
`PENDING OPERATIONAL CALIBRATION`, not complete.
Fixtures must reject same-input nondeterminism, current-run leakage, operator-selected
history, use of fewer than six bootstrap runs, bootstrap-as-deployable, acceptance-run
leakage into the history snapshot, semantic/profile mismatch, coverage downgrade, post-hoc
widening, excessive interval width, upper bound equal to or above 1800, non-overlapping
intervals, elapsed/RSS point error above 20%, any elapsed/RSS repetition outside its
frozen interval, acceptance of the approximately `[0.08,0.70]` bimodal GPU-idle fixture,
rejection of `[0.02,0.87]`, assembler self-declaration, TTL/skew boundaries, permitted
same-queue exact-hash receipt reuse, and rejected cross-queue or changed-hash replay.

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
4. Implement the independently reviewed `calibration_workloads` descriptor schema,
   outcome schema v2, native token/EOS/finish telemetry, predictor/checker coverage, and
   regression fixtures. Run focused and fast/full CPU gates and commit the result.
5. Generate fresh machine, budget, and preflight evidence bound to that committed state
   and a completely new calibration root. Historical receipts must not authorize launch.
6. Run a new six-repetition bootstrap cohort per phase. The eighteen preserved v1
   repetitions are diagnostic-only and cannot enter trusted outcome-schema-v2 history.
7. Freeze trusted history, generate the prediction contract, run exactly three new
   acceptance repetitions per phase, and require checker-owned `deployable` evidence.
8. A fresh independent Reviewer executes every AC-01 through AC-27 command and the
   completion-state checker from committed code.

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
2. report AC-01 through AC-27 as `PASS`, `FAIL`, or `WEAKENED`;
3. inspect commits and tests for skipped, deleted, loosened, or trivial checks;
4. confirm acceptance used no real external service;
5. write the final review under `docs/joint_training/codereview/active/` and move
   it to `completed/` only after every required AC passes.

The goal is complete only when every AC is reviewer-owned `PASS` and none is
`WEAKENED`.
