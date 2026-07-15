# Milestone 2 Plan v9 Independent Review

## Review Identity

- Reviewer: independent Milestone 2 reviewer, Codex session, requested model `GPT-5.5`, reasoning effort `medium`
- Repo: `/data-1/code/verl`
- Branch: `codex/experiment-execution-reliability`
- Base commit: `7099830b`
- Candidate commit: `0540935b479c3042fe2567f14805c84b6064fc68`
- Frozen Plan: `docs/joint_training/goals/stage123-execution-readiness/plan.md`, Plan version `9`
- Review prompt: `docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-review-prompt.md`
- Report path: `docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-review.md`

## Overall Verdict

**FAIL**

The frozen verification commands all passed, and the matched-control test/fixture repair plus manifest-driven phase adapter close the two named Milestone 2 findings `F-M2-MC-01` and `F-M2-MC-02`. However, candidate commit `0540935b` still renders the authoritative calibration result with `authorization_identity.plan_version = 8` in `scripts/render_calibration_result.py`, while Plan v9 AC-01 requires authorization identity to bind the exact Plan v9 hash and current decision identity. This is a blocking in-scope defect for AC-01 before Milestone 3/4 can rely on the candidate implementation.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | FAIL | Three-phase calibration tests pass, but `scripts/render_calibration_result.py:141` hard-codes `plan_version: 8`; Plan v9 requires exact Plan v9 authorization identity. |
| AC-02 | PASS | Preflight/model/scorer/wrapper verification command passed: `33 passed in 195.01s`. |
| AC-03 | PASS | Manifest, batch, monitor, and end-to-end tests passed; static manifest inspection shows exact run order `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`. |
| AC-04 | PASS | Batch/core and operational runner tests passed; candidate routes admitted Stage123 commands through `scripts/stage123_phase_adapter.py` and keeps lifecycle in `experiment_execution_core.py`. |
| AC-05 | PASS | Monitor/queue contract tests passed in the third frozen command: total `45 passed in 33.13s`. |
| AC-06 | PASS | Admission/mutation tests passed in first and third frozen commands; protected-asset compare returned `ok: true`. |
| AC-07 | PASS | End-to-end/new-experiment gates passed; phase adapter dry-run tests do not start training, external services, or GPU work. |
| AC-08 | PASS | Launch-renderer and batch command tests passed; static diff shows canonical three-command adapter matrix and explicit `EXPERIMENT_BATCH_MANIFEST` binding. |

## Named Finding Verdicts

| Finding | Verdict | Rationale |
| --- | --- | --- |
| F-M2-MC-01 | CLOSED / PASS | Candidate updates tests and fixtures from the stale two-run assumptions to the Plan v9 three-run contract; focused suites now pass. |
| F-M2-MC-02 | CLOSED / PASS | Candidate introduces one deterministic manifest-driven phase adapter for all three canonical batch commands, sets per-run wrapper environment, forbids automatic retry/resume via existing checkpoint/provenance checks, and performs Stage2 merge/extract/provenance handoff before Stage3. |

## Commands And Evidence

### Frozen Command 1

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_phase_adapter.py tests/experiment_workflow/test_stage123_core_migration.py
```

Result:

```text
23 passed in 45.96s
```

### Frozen Command 2

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py
```

Result:

```text
33 passed in 195.01s (0:03:15)
```

### Frozen Command 3

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_monitor.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_new_experiment_gate.py
```

Result:

```text
45 passed in 33.13s
```

### Frozen Command 4

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json
```

Result:

```json
{"kind":"git_tree","path":"config/experiment_execution","tree_sha1":"4377c79f59537caf09f47bbc9ef6464e2b078d1f"}
{"gitlink_commit":"6fcccb353a87045a17f9d52b3821f0e20f7f9a9d","kind":"gitlink","mode":"160000","path":"recipe"}
{"kind":"git_tree","path":"scripts","tree_sha1":"04a9f38fc6799deb176d181fb0c2d10aa22ae99f"}
{"kind":"git_tree","path":"verl","tree_sha1":"40deac7dc6da65ef470c5e42c75fb2fd35b9335a"}
{"implementation_tree_sha256": "949ea66938be2f72b9ad518ce6c314a7fa58f4b41ebe96cf8a3b1414f65073da"}
```

### Frozen Command 5

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Result:

```json
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

## Static Inspection Evidence

- `git diff 7099830b..0540935b --stat` shows the candidate changes production code, tests, goal ledgers, and recipe gitlink; no protected asset diff is included.
- Recipe gitlink changes from `9b83f9f488ac4b34e87a36a40d969d98d7d025f7` to `6fcccb353a87045a17f9d52b3821f0e20f7f9a9d` with submodule commit `stage123: add matched Stage1 control admission`.
- `recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml` at candidate submodule commit enumerates runs in order: `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`.
- `scripts/experiment_execution_core.py` validates Stage123 admission into exactly three canonical commands using `/workspace/verl/scripts/stage123_phase_adapter.py` with the frozen manifest and run IDs.
- `scripts/stage123_phase_adapter.py` dry-run output is covered by tests for per-run environment, offline W&B, artifact roots outside repo, Stage2 handoff fields, and Stage3 `STAGE2_PROVENANCE_FILE`.
- Blocking evidence: `scripts/render_calibration_result.py:141` renders `"authorization_identity": {"decision_id": args.decision_id, "run_id": args.run_id, "plan_version": 8}`. Plan v9 AC-01 requires the authorization identity to bind the matching `USER_DECISION_RECORDED.decision_id` plus the exact Plan v9 hash and candidate implementation-tree SHA256.

## Blocking In-Scope Defects

### F-M2-MC-03 — Calibration result renderer still binds Plan version 8

- Classification: `IN_SCOPE_DEFECT`
- Blocks: AC-01
- Evidence: `scripts/render_calibration_result.py:141`
- Expected by Plan v9: authoritative calibration result must bind the matched-control Plan v9 authorization identity, not historical Plan v8.
- Actual in candidate: renderer emits `plan_version: 8` and does not include the Plan v9 hash in `authorization_identity`.
- Impact: a fresh Milestone 4 three-phase calibration result rendered by this candidate would carry stale authorization identity and fail the Plan v9 acceptance contract even if the probe itself passed.

## Deferred Suggestions

None.

## Contract Contradictions

None. The defect above is an implementation mismatch against AC-01, not a Plan contradiction.

## Test-Weakening / Trivialization Audit

- No added `pytest.mark.skip`, `pytest.mark.xfail`, `assert True`, broad early `return`, or deletion of relevant test files was found in `git diff 7099830b..0540935b -- tests`.
- Test count increased by three: `test_experiment_batch_core.py` `10 -> 11`, new `test_stage123_phase_adapter.py` `0 -> 2`.
- Existing test assertions were mostly retargeted from two-run Plan v8 assumptions to the three-run Plan v9 contract.
- One new unit test monkeypatches `execution_results.validate_admission_bundle` and `validate_current_checkout` to isolate `experiment_execution_core.validate_admission_bundle`; this is acceptable only as a command-mapping unit test because other frozen tests exercise admission/current-checkout behavior directly.
- The current suite did not catch the stale `plan_version: 8` renderer binding, so coverage needs one focused assertion for Plan v9 authorization identity before Milestone 4.

## Single Most Likely Weakness In This Review

I did not run a live GPU calibration, live preflight, or final admission bundle rendering because the Milestone 2 prompt froze only CPU/unit verification plus implementation-tree and protected-asset checks. The PASS verdicts outside AC-01 are therefore bounded to Milestone 2 behavior and static inspection, not final readiness acceptance.
