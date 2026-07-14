# Milestone 2 Independent Review

## Review Identity

- Reviewer: Codex independent reviewer, GPT-5.5 requested by prompt context; executed in current Codex reviewer session.
- Review type: Milestone Review.
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`.
- Frozen Plan version reviewed: `8`.
- Reviewed base/candidate: base commit `29089a6c1c63d017384b1ff09eba9821d10a2a7a`; candidate is the dirty worktree at the same HEAD, before Milestone 3 commit.
- Scope followed: independent inspection only; no production/test implementation edits; no GPU probe, training launch, or external service call.

## Overall Verdict

**FAIL**

Milestone 2 is not ready to advance. The required pytest suites pass, but reviewer-owned behavior probes found two fail-closed violations inside the Milestone 2 acceptance surface:

1. `admission render-launch` can render a launch command from a synthetic accepted bundle without invoking the common current-checkout validator, without canonical input paths, without protected-baseline comparison, and without strict `acceptance_report.json` schema v1.
2. `scripts/render_calibration_result.py render` can output a passed `calibration_result.json` from a minimal fake producer report with empty phase evidence and only minimal execution-core state.

These are blocking `IN_SCOPE_DEFECT`s against the frozen Plan v8 Architecture Contract and AC-01/AC-06/AC-08.

## Per-AC Verdict Table

| AC | Verdict | Evidence |
|---|---:|---|
| AC-01 - Re-Aligned Production Identity Is Freshly Qualified | FAIL | Required tests passed, but a reviewer fake producer probe made `render_calibration_result.py render` return `decision=passed` with `phase_evidence=[]` and no zero-step/formal-checkpoint evidence. |
| AC-02 - Fresh Preflight Covers Deployability | PASS | Required preflight/model/scorer/wrapper tests passed; static inspection shows host facts are produced by `scripts/stage123_host_facts.sh` and container preflight requires `--host-facts`. |
| AC-03 - Run Set Is Exactly The Primary Chain | PASS | Required manifest/queue/monitor/end-to-end tests passed; preflight shard constants are FRAC25-only. |
| AC-04 - Queue Lifecycle Is Python-Owned | PASS | Required execution-core/deadline/end-to-end tests passed; no contradictory reviewer evidence found in Milestone 2 changes. |
| AC-05 - Queue And Monitor Share One Event Authority | PASS | Required manifest queue/monitor and end-to-end tests passed; no duplicated Milestone 2 run-set authority found in inspected changes. |
| AC-06 - Admission Bundle Fails Closed | FAIL | `admission render-launch` accepted a synthetic bundle and printed a launch command despite no current-checkout validation, no `inputs`, no protected baseline, and legacy embedded acceptance. |
| AC-07 - Readiness Does Not Train Or Publish | PASS | Required tests passed; reviewer probes used only temp files under `/data-1/tmp/verl_agent_scratch`; no training child/GPU/external calls were run. |
| AC-08 - Launch Command Is Exact And Reproducible | FAIL | Same render-launch probe printed an exact launch command from untrusted synthetic bindings, so launch rendering is not gated by current checkout/commit/input hashes/strict acceptance. |
| Implementation readiness | FAIL | Blocking AC-01/AC-06/AC-08 defects remain. |

## Commands And Evidence

### Required Verification Commands

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py; REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_new_experiment_gate.py tests/experiment_workflow/test_calibration_milestone3.py
```

Relevant output:

```text
32 passed in 186.42s (0:03:06)
14 passed in 32.46s
33 passed in 47.46s
```

### Runtime Validator

Command:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness || true
```

Relevant output:

```json
{"current_milestone":"Milestone 2","goal_status":"ACTIVE","plan_status":"READY","plan_version":8,"pending_user_decisions":[]}
```

The validator reports the lifecycle is positioned at Milestone 2, but it does not prove the candidate behavior satisfies Milestone 2.

### Candidate Worktree

Command:

```bash
git status --short && git rev-parse HEAD && git diff --stat && git diff --name-only
```

Relevant output:

```text
29089a6c1c63d017384b1ff09eba9821d10a2a7a
M scripts/execution_results.py
M scripts/render_calibration_probe_command.py
M scripts/run_calibration_probe_zero_step.py
M tests/experiment_workflow/test_execution_results.py
M tests/experiment_workflow/test_stage123_preflight_model_identity.py
?? scripts/render_calibration_result.py
?? scripts/stage123_host_facts.sh
?? tests/experiment_workflow/test_stage123_admission_bundle.py
?? tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py
```

Other goal docs and support artifacts are also dirty/untracked; production/test behavior review focused on the Milestone 2 files above.

### Probe 1 - `render-launch` Bypasses Current-Checkout Validation

Reviewer-owned probe created a temporary synthetic bundle under `/data-1/tmp/verl_agent_scratch/stage123-review-*` with:

- `run_ids = ["frac25-stage2", "frac25-stage3"]`
- fake `implementation_tree_sha256`, fake result hashes, fake readiness commit
- no `inputs`
- no `protected_baseline_sha256`
- legacy embedded `acceptance` object, not schema v1 `acceptance_report.json`

Command shape:

```bash
python scripts/execution_results.py admission render-launch --bundle /data-1/tmp/verl_agent_scratch/stage123-review-ME4pPf/admission_bundle.json --repo-host /data-1/code/verl
```

Observed output:

```text
tmux new-session -d -s stage123_primary_chain env REPO_HOST=/data-1/code/verl ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1 STAGE123_ADMISSION_BUNDLE=/data-1/tmp/verl_agent_scratch/stage123-review-ME4pPf/admission_bundle.json STAGE123_IMPLEMENTATION_TREE_SHA256=3333333333333333333333333333333333333333333333333333333333333333 STAGE123_BUNDLE_SHA256=d65bd660da54df083e00f90618f01f07f9512039ae30f6ec8722ba19f4cba3ad bash /data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
exit_code=0
```

Relevant implementation evidence:

- `/data-1/code/verl/scripts/execution_results.py:527` sets `require_accepted=True` for `render-launch`.
- `/data-1/code/verl/scripts/execution_results.py:529` only calls `validate_current_checkout(...)` when `args.repo_root` is present.
- The `render-launch` subparser defines `--bundle` and `--repo-host`, but no `--repo-root`, so line 529 is false for launch rendering.
- `/data-1/code/verl/scripts/execution_results.py:312` `validate_admission_bundle(..., require_accepted=True)` still accepts the older embedded `acceptance` shape and does not invoke strict `validate_acceptance_report(...)` unless `validate_current_checkout(...)` is reached.

This violates the Plan v8 contract that one common current-checkout admission validator is used by candidate, `--bundle`, accepted-bundle validation, and `render-launch`, and that `acceptance_report.json` schema v1 is strict.

### Probe 2 - Calibration Result Renderer Accepts Insufficient Producer Evidence

Reviewer-owned probe created a temporary fake producer scratch under `/data-1/tmp/verl_agent_scratch/stage123-render-review-*` with:

- schema v2 `latest-probe.json`
- matching report SHA256
- `run_id` and `authorization_decision_id` matching the pointer
- execution-core state containing only `{"status":"succeeded","run_id":"run-fake","cleanup":{"note":"minimal fake cleanup"}}`
- producer report with `status=passed`, matching `manifest_sha256`, and `phases=[]`

Command shape:

```bash
python scripts/render_calibration_result.py render --run-id run-fake --state-root /data-1/tmp/verl_agent_scratch/stage123-render-review-1YddsZ/state --latest-probe /data-1/tmp/verl_agent_scratch/stage123-render-review-1YddsZ/scratch/latest-probe.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --resource-profile recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh --implementation-tree /data-1/tmp/verl_agent_scratch/stage123-render-review-1YddsZ/implementation-tree.jsonl --evidence-commit 29089a6c1c63d017384b1ff09eba9821d10a2a7a --runtime-ledger docs/joint_training/goals/stage123-execution-readiness/reviews/.tmp-ledger.jsonl --decision-id RD-FAKE --output /data-1/tmp/verl_agent_scratch/stage123-render-review-1YddsZ/calibration_result.json
```

Observed output summary:

```json
{"decision":"passed","phase_evidence":[],"cleanup":{"execution_state":{"note":"minimal fake cleanup"},"resources_released":true},"prediction_comparison":{"qualified":true,"source":"producer verification"}}
```

Relevant implementation evidence:

- `/data-1/code/verl/scripts/render_calibration_result.py` checks pointer schema/hash/status and state `status=succeeded`, but it does not require Stage2/Stage3 phase evidence, repetitions, zero optimizer steps, empty formal checkpoint lists, or a producer-report path/hash recorded in terminal execution-core state.
- The renderer synthesizes `prediction_comparison.qualified=True` and `cleanup.resources_released=True` instead of deriving them from strict producer and execution-core evidence.

This violates AC-01's requirement that rendering must bind exact phase/repetition evidence, prediction comparison, zero optimizer steps, empty formal-checkpoint lists, terminal cleanup, and reject absent/mismatched input as `blocked`, never `passed`.

## Blocking In-Scope Defects

### F-M2-01 - `render-launch` authorizes without the common current-checkout validator

- Classification: `IN_SCOPE_DEFECT`.
- Affected ACs: AC-06, AC-08; also weakens strict acceptance binding in the Architecture Contract.
- Evidence: Probe 1 rendered a launch command with fake hashes and legacy embedded acceptance, exit code `0`.
- Root cause: `render-launch` has no `--repo-root`, so `/data-1/code/verl/scripts/execution_results.py:529` skips `validate_current_checkout(...)`; strict schema v1 acceptance validation is only reached from that skipped function.
- Required fix direction: `render-launch`, `admission validate --bundle`, candidate construction, and accepted validation must all call the same current-checkout/protected-baseline/strict-acceptance validator and fail before printing a launch command when any binding is absent or fake.

### F-M2-02 - `render_calibration_result.py` renders `passed` from insufficient producer evidence

- Classification: `IN_SCOPE_DEFECT`.
- Affected ACs: AC-01 and downstream AC-06 admission freshness/binding.
- Evidence: Probe 2 rendered `decision=passed` with empty `phase_evidence` from a minimal fake report and minimal succeeded execution state.
- Root cause: the renderer validates pointer identity/hash/status but does not enforce the full post-probe evidence contract before constructing a passed `calibration_result.json`.
- Required fix direction: rendering must fail closed unless the producer report and terminal execution-core state jointly prove the exact run ID, decision ID, report path/hash, Stage2/Stage3 phase evidence, repetition counts, zero optimizer steps, no formal checkpoints, prediction comparison, and terminal cleanup.

## Deferred Suggestions

None. The blocking findings are within frozen ACs; no non-blocking suggestions are needed for this review.

## Contract Contradictions

None found in Plan v8 during this Milestone 2 review. The failures are implementation/test coverage defects, not Plan contradictions.

## Single Most Likely Weakness In This Review

I did not run the live host preflight command because it can touch Docker/tmux/GPU inventory and the prompt explicitly excluded GPU probes and training; AC-02 is therefore judged from required tests plus static inspection, not from live host artifact generation.
