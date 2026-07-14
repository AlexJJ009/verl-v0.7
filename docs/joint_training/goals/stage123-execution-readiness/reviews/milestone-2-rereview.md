# Milestone 2 Focused Re-Review

## Review Identity

- Reviewer: same independent Milestone 2 reviewer in this Codex session.
- Review type: Milestone Re-Review.
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`.
- Frozen Plan version reviewed: `8`.
- Candidate reviewed: dirty worktree before Milestone 3 commit; HEAD remains `29089a6c1c63d017384b1ff09eba9821d10a2a7a` from the prior review context.
- Scope: focused re-verification of `F-M2-01` and `F-M2-02` only, covering AC-01, AC-06, and AC-08. No unrelated ACs were re-reviewed.
- Safety: no GPU probe, no training launch, no external service calls. Temporary probe artifacts were written only under `/data-1/tmp/verl_agent_scratch/...`; the only repository file written by this review is this report.

## Overall Verdict

**PASS for focused re-verification.**

Both prior blocking findings are closed by reviewer-owned behavior probes:

- `F-M2-01` is closed: `admission render-launch` now rejects the prior synthetic accepted bundle before printing a launch command because it lacks canonical `inputs`.
- `F-M2-02` is closed: `scripts/render_calibration_result.py render` now rejects the prior fake producer report with empty phase evidence before writing a passed `calibration_result.json`.

The required focused pytest/runtime command also passed.

## Per-AC Verdict Table

| AC / Finding | Verdict | Evidence |
|---|---:|---|
| `F-M2-01` / AC-06 | PASS | Synthetic accepted bundle without canonical inputs returns `authorized=false`, code `admission_inputs`, exit code `1`; no launch command is printed. |
| `F-M2-01` / AC-08 | PASS | `render-launch` now reaches the common validation gate for launch rendering and blocks missing canonical inputs before rendering. |
| `F-M2-02` / AC-01 | PASS | Fake producer report with `phases=[]` is rejected with `producer report must contain exactly stage2 and stage3`; no calibration result output is written. |

## Commands And Evidence

### Required Focused Verification

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_calibration_milestone3.py; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness
```

Relevant output:

```text
39 passed in 65.49s (0:01:05)
```

Runtime validator relevant output:

```json
{"current_milestone":"Milestone 2","goal_status":"ACTIVE","plan_status":"READY","plan_version":8,"pending_user_decisions":[]}
```

The runtime ledger still lists `F-M2-01` and `F-M2-02` as open because this review does not mutate runtime/finding ledgers; behavior re-verification closes them from the reviewer perspective.

### F-M2-01 Re-Probe: Synthetic Bundle Must Not Render Launch

Probe setup: recreated the prior synthetic accepted bundle under `/data-1/tmp/verl_agent_scratch/stage123-rereview-fm201-rOgjli/admission_bundle.json` with FRAC25 run IDs, fake hashes, fake readiness commit, no canonical `inputs`, no protected baseline, and legacy embedded `acceptance`.

Command:

```bash
python scripts/execution_results.py admission render-launch --bundle /data-1/tmp/verl_agent_scratch/stage123-rereview-fm201-rOgjli/admission_bundle.json --repo-host /data-1/code/verl
```

Observed output:

```json
{"authorized": false, "code": "admission_inputs", "context": {}, "message": "admission bundle lacks canonical input paths"}
```

Observed exit:

```text
exit_code=1
```

Relevant implementation evidence:

- `/data-1/code/verl/scripts/execution_results.py:527` sets `require_accepted=True` for `render-launch`.
- `/data-1/code/verl/scripts/execution_results.py:529` now enters the follow-up validation block for all authorized bundles.
- `/data-1/code/verl/scripts/execution_results.py:530` uses `args.repo_host` as the validation root for `render-launch`.
- `/data-1/code/verl/scripts/execution_results.py:531` rejects bundles without canonical `inputs`.
- `/data-1/code/verl/scripts/execution_results.py:538` routes valid-input bundles to `validate_current_checkout(...)`.
- `/data-1/code/verl/scripts/execution_results.py:542` returns before line 545 can print a launch command when the decision is not authorized.

This directly verifies the prior bypass is closed for the original failing class.

### F-M2-02 Re-Probe: Fake Producer Report Must Not Render Passed Calibration Result

Probe setup: recreated the prior fake producer scratch under `/data-1/tmp/verl_agent_scratch/stage123-rereview-fm202-2E4gtn` with schema v2 pointer, matching report SHA256, matching run/decision IDs, minimal succeeded execution-core state, and a producer report containing `status=passed`, matching manifest hash, and `phases=[]`.

Command:

```bash
python scripts/render_calibration_result.py render --run-id run-fake --state-root /data-1/tmp/verl_agent_scratch/stage123-rereview-fm202-2E4gtn/state --latest-probe /data-1/tmp/verl_agent_scratch/stage123-rereview-fm202-2E4gtn/scratch/latest-probe.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --resource-profile recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh --implementation-tree /data-1/tmp/verl_agent_scratch/stage123-rereview-fm202-2E4gtn/implementation-tree.jsonl --evidence-commit 29089a6c1c63d017384b1ff09eba9821d10a2a7a --runtime-ledger /data-1/tmp/verl_agent_scratch/stage123-rereview-fm202-2E4gtn/ledger.jsonl --decision-id RD-FAKE --output /data-1/tmp/verl_agent_scratch/stage123-rereview-fm202-2E4gtn/calibration_result.json
```

Observed output:

```json
{"error": "producer report must contain exactly stage2 and stage3", "ok": false}
```

Observed file/exit evidence:

```text
no output
exit_code=1
```

Relevant implementation evidence:

- `/data-1/code/verl/scripts/render_calibration_result.py:56` reads producer `phases`.
- `/data-1/code/verl/scripts/render_calibration_result.py:57` requires exactly `stage2` and `stage3` in order.
- `/data-1/code/verl/scripts/render_calibration_result.py:59` requires three passed repetitions per phase.
- `/data-1/code/verl/scripts/render_calibration_result.py:61` requires zero optimizer steps and empty formal checkpoints.
- `/data-1/code/verl/scripts/render_calibration_result.py:63` requires qualified prediction and cleanup evidence.
- `/data-1/code/verl/scripts/render_calibration_result.py:112` now copies phase evidence from the verified producer report rather than synthesizing a passed empty result.
- `/data-1/code/verl/scripts/render_calibration_result.py:113` and `/data-1/code/verl/scripts/render_calibration_result.py:114` bind prediction and cleanup from report/state evidence.

This directly verifies the prior fake-report acceptance is closed for the original failing class.

## Blocking In-Scope Defects

None for this focused re-review. `F-M2-01` and `F-M2-02` are behaviorally closed.

## Deferred Suggestions

None.

## Contract Contradictions

None found in this focused re-review.

## Single Most Likely Weakness In This Review

The re-review intentionally tested the two original failing behavior classes and the required focused suites only; it did not construct a fully valid accepted admission bundle or a fully valid three-repetition producer report to prove the happy paths beyond the existing test coverage.
