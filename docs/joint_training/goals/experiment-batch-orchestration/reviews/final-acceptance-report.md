# Final Acceptance Report

## Review Identity

- Reviewer: `Codex independent final acceptance reviewer`
- Review type: `final-acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Plan version: `2`
- Plan SHA256: `15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `ce827f2a9d631197a5dac97484a7f5f6543f4535`
- Recipe commit/gitlink: `9b83f9f488ac4b34e87a36a40d969d98d7d025f7`

## Overall Verdict

`ACCEPTED`. AC-01 through AC-11 are individually `PASS` from reviewer-owned evidence. I found no blocking in-scope defects, deferred suggestions, contract contradictions, duplicate transition authority, retry/resume/tuning/AI authority, protected-asset mutation, unresolved finding, or unapproved training.

## Per-AC Verdicts

| AC | Verdict | Evidence |
|---|---|---|
| AC-01 | PASS | Focused replacement passed; authority inventory and deletion budget bind `scripts/experiment_execution_core.py` as sole transition authority; source audit found Stage123 shell delegates to `batch-run`. |
| AC-02 | PASS | `batch-validate` returned `ok: true`; fixture binds batch hash, admission hash, command hash, implementation tree, protected hashes, and recipe gitlink `9b83f9f...`. |
| AC-03 | PASS | Focused routing/core tests passed, including success ordering, cleanup-before-next, and no duplicate/invented item coverage. |
| AC-04 | PASS | Focused routing/core tests passed for local failure fallback; batch `RunSpec` uses `max_attempts=1` and empty resumable failures; Stage123 adapter rejects retry/resume overrides. |
| AC-05 | PASS | Focused routing/core tests passed for shared failure, repeated normalized failure stop, cleanup failure stop, corrupt state/event fail-closed, and no next launch. |
| AC-06 | PASS | Focused control/core tests passed for `pause_after_current`, `stop_now`, `continue_remaining`, stale/replayed controls, CAS revision, and control hash recording. |
| AC-07 | PASS | Focused monitor tests passed; `stage123_manifest_monitor.py` reads persisted state/events and has no tmux/checkpoint/W&B/registry inference authority. |
| AC-08 | PASS | Focused Stage123 migration/admission tests passed; recipe adapter is a thin parser/delegate and dry-run compatibility renderer is evidence projection only. |
| AC-09 | PASS | Focused policy/failure-classifier tests passed; source audit found no Codex/Agent/timer/tuning authority in batch paths and batch mode rejects recovery. |
| AC-10 | PASS | PM2/policy tests passed inside focused replacement; protected asset fingerprint compare returned `ok: true`; `git diff --check` passed. |
| AC-11 | PASS | This report and `acceptance.md`/`acceptance_report.json` bind Plan v2 hash, candidate commit, recipe commit, full CPU summary, focused replacement, deletion budget, and source audits. |

## Commands And Evidence

```text
$ git rev-parse HEAD
ce827f2a9d631197a5dac97484a7f5f6543f4535

$ git -C recipe rev-parse HEAD
9b83f9f488ac4b34e87a36a40d969d98d7d025f7

$ git ls-tree HEAD recipe
160000 commit 9b83f9f488ac4b34e87a36a40d969d98d7d025f7 recipe

$ sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md
15bc9c5120d671eac58dfacc549b6daf2ff67ef4c7a204489e4e8f2eff755cbb  docs/joint_training/goals/experiment-batch-orchestration/plan.md
```

```text
$ goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration
PASS
```

```text
$ goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration
PASS: plan_status READY; current_milestone null; pending_user_decisions []; all listed findings CLOSED.
```

```text
$ REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "00d731cb96aee9a9e2cb8171d3dba6a40233c01b7295d086c8af727f2114b066", "items": ["stage123-primary"], "ok": true}
```

```text
$ REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

```text
$ git diff --check
# no output; exit code 0
```

```text
$ REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_end_to_end.py::test_dry_run_is_scratch_only_and_manifest_consistent tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_monitor.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_experiment_batch_control.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_notification_policy.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_failure_classifier.py tests/experiment_workflow/test_pm2_ci_keepalive.py
63 passed in 33.48s
```

## Full CPU Gate Summary

I inspected `docs/joint_training/goals/experiment-batch-orchestration/full-cpu-gate-summary.json` and did not rerun the full CPU suite, per the final acceptance prompt. The summary records exactly one full CPU preservation run: exit code `1`, `207 passed`, one failure in `test_stage123_end_to_end.py::test_dry_run_is_scratch_only_and_manifest_consistent`, classified `IN_SCOPE`, repaired by `scripts/stage123_dry_run_compat.py`, followed by a focused replacement result of `63 passed`, with `full_gate_rerun: false`.

## Source And Mutation Audits

- Candidate diff from `2020531b470ec932d7b00afd13080e1318fc8429` to `ce827f2a9d631197a5dac97484a7f5f6543f4535` adds the Goal ledger/evidence, extends `scripts/experiment_execution_core.py`, adds `scripts/stage123_dry_run_compat.py`, updates `scripts/stage123_manifest_monitor.py`, updates the `recipe` gitlink, and adds focused batch/control/policy/monitor/routing tests.
- `scripts/experiment_execution_core.py` batch mode rejects `--resume` and `--recovery-policy`, creates batch item `RunSpec` with `max_attempts=1` and `resumable_failure_codes=()`, validates controls by sequence/revision/authorization/hash, and stops on malformed event/state evidence.
- `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh` at recipe commit `9b83f9f...` only parses `--batch-manifest`/`--state-root`, rejects retry/resume overrides, delegates batch execution to `scripts/experiment_execution_core.py batch-run`, and invokes dry-run compatibility only for `DRY_RUN=1` without a batch manifest.
- `scripts/stage123_manifest_monitor.py` consumes persisted `*.json` state and `events.jsonl`, emits notification-policy state, maintains an idempotent cursor, and does not transition lifecycle state.
- `scripts/stage123_dry_run_compat.py` writes normalized dry-run/provenance/status evidence only; it does not launch children, inspect checkpoints, route phases, or classify failures.
- Focused mutation tests cover manifest mutation, command/implementation binding, protected binding mutation, stale/replayed controls, malformed batch/atomic events, no agent/timer/tuning authority, and Stage123 retry/resume rejection.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness

This review did not rerun the full CPU preservation suite because the final acceptance prompt explicitly forbids rerunning it. The acceptance therefore relies on the inspected single-run summary plus reviewer-owned focused replacement, validator, protected-asset, batch-validation, diff, source, and mutation audits.
