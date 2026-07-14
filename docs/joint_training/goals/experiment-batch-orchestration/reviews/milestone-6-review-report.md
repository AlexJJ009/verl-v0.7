# Milestone 6 Independent Review Report

## Review Identity

- Reviewer: independent GPT-5.5 medium milestone reviewer
- Review type: Milestone 6 review
- Goal: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: 2
- Implementation candidate reviewed: `98fed9681265a79dbbf5b023ab08e9a2550aa914`
- Evidence commit reviewed: `2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd`
- Recipe gitlink reviewed: `9b83f9f488ac4b34e87a36a40d969d98d7d025f7`
- Applicable ACs: AC-01 through AC-10

## Overall Verdict

**PASS**

`F-M6-01` is closed. I did not rerun the full `tests/experiment_workflow` suite. I inspected the recorded single full CPU gate, reran the focused replacement command from `full-cpu-gate-summary.json`, reran the requested validations, and performed a scratch dry-run probe. The Stage123 dry-run compatibility renderer restores the accepted scratch/provenance/status evidence contract without child launch, lifecycle state, checkpoint inference, registry/W&B writes, GPU use, training, or external services.

## Finding Closure Status

| Finding | Status | Evidence |
| --- | --- | --- |
| `F-M6-01` | CLOSED | Focused replacement passed `63 passed`; direct dry-run probe returned `rc=0`, produced normalized manifest/provenance/status evidence in scratch only, left registry/release files unchanged, and did not create lifecycle/checkpoint outputs. |

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Prior accepted core remains unchanged by implementation candidate except Stage123 dry-run evidence renderer is inventoried as non-authoritative; focused replacement covers core/migration suites. |
| AC-02 | PASS | `batch-validate` returned `ok: true`; fixture bundle, admission file, manifest hash, implementation tree, and recipe gitlink recomputed successfully. |
| AC-03 | PASS | Focused replacement includes routing tests and passed. |
| AC-04 | PASS | Focused replacement includes local-failure routing tests and passed. |
| AC-05 | PASS | Focused replacement includes shared-failure/event-corruption tests and passed. |
| AC-06 | PASS | Focused replacement includes operator-control tests and passed. |
| AC-07 | PASS | Focused replacement includes monitor contract and batch monitor tests; monitor remains read-only. |
| AC-08 | PASS | Stage123 dry-run compatibility test passed; batch adapter still delegates to core when manifest is supplied and rejects retry/resume. |
| AC-09 | PASS | Focused replacement includes policy/failure-classifier tests; renderer only projects evidence and does not tune/retry/launch agents. |
| AC-10 | PASS | Protected asset compare returned `ok: true`; scratch dry-run left registry/release files unchanged. |

## Commands And Evidence

### Candidate, Evidence, And Recipe State

```bash
git rev-parse HEAD
git rev-parse 98fed9681265a79dbbf5b023ab08e9a2550aa914
git rev-parse 2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd
git merge-base 98fed9681265a79dbbf5b023ab08e9a2550aa914 2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd
git ls-tree 98fed9681265a79dbbf5b023ab08e9a2550aa914 recipe
git ls-tree 2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd recipe
git -C recipe rev-parse HEAD
```

Result:

```text
HEAD=2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd
impl=98fed9681265a79dbbf5b023ab08e9a2550aa914
evidence=2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd
merge-base impl evidence=98fed9681265a79dbbf5b023ab08e9a2550aa914
160000 commit 9b83f9f488ac4b34e87a36a40d969d98d7d025f7 recipe
160000 commit 9b83f9f488ac4b34e87a36a40d969d98d7d025f7 recipe
9b83f9f488ac4b34e87a36a40d969d98d7d025f7
```

### Goal Validation

```bash
goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
```

Result: `validate-plan` returned `PASS`; `validate-runtime` reported Plan status `READY`, current milestone `Milestone 6`, and `F-M6-01` open pending this review.

### Full CPU Gate Summary Inspection

File inspected: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration/full-cpu-gate-summary.json`

Key fields:

```text
run_count=1
initial_exit_code=1
initial_passed=207
initial_failed=1
initial failure=test_stage123_end_to_end.py::test_dry_run_is_scratch_only_and_manifest_consistent
focused_exit_code=0
focused_passed=63
full_gate_rerun=False
```

Recorded full gate log inspected: `/data-1/tmp/experiment-batch-m6-full-cpu.log`

Log tail confirms the single full run ended as:

```text
FAILED tests/experiment_workflow/test_stage123_end_to_end.py::test_dry_run_is_scratch_only_and_manifest_consistent
1 failed, 207 passed, 5 warnings in 470.47s (0:07:50)
```

This matches the summary and supports that there was one full gate followed by a focused replacement, not a full rerun.

### Focused Replacement Command

Command rerun exactly from `full-cpu-gate-summary.json`:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_stage123_end_to_end.py::test_dry_run_is_scratch_only_and_manifest_consistent \
  tests/experiment_workflow/test_experiment_batch_core.py \
  tests/experiment_workflow/test_experiment_batch_monitor.py \
  tests/experiment_workflow/test_experiment_batch_routing.py \
  tests/experiment_workflow/test_experiment_batch_control.py \
  tests/experiment_workflow/test_experiment_batch_policy.py \
  tests/experiment_workflow/test_stage123_core_migration.py \
  tests/experiment_workflow/test_manifest_queue_monitor_contract.py \
  tests/experiment_workflow/test_stage123_admission_bundle.py \
  tests/experiment_workflow/test_notification_policy.py \
  tests/experiment_workflow/test_operational_calibration_runner.py \
  tests/experiment_workflow/test_failure_classifier.py \
  tests/experiment_workflow/test_pm2_ci_keepalive.py
```

Result:

```text
63 passed in 32.32s
```

### Requested Validations

Batch validation:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "00d731cb96aee9a9e2cb8171d3dba6a40233c01b7295d086c8af727f2114b066", "items": ["stage123-primary"], "ok": true}
```

Protected asset comparison:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Result:

```text
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

Diff checks:

```bash
git diff --check b50c3f83cd7664f0dbb6d611324c35d21974db15 98fed9681265a79dbbf5b023ab08e9a2550aa914
git diff --check 98fed9681265a79dbbf5b023ab08e9a2550aa914 2f7aab5e7bd3b58e5a6598090bb4c48e86f523bd
```

Result:

```text
diff_check_impl_rc=0
diff_check_evidence_rc=0
```

Fixture recomputation:

```text
impl_paths=['scripts/experiment_execution_core.py', 'scripts/stage123_manifest_monitor.py']
impl_tree_match=True
bundle_hash_match=True
admission_file_hash_match=True
manifest_hash_match=True
recipe_gitlink=9b83f9f488ac4b34e87a36a40d969d98d7d025f7
evidence_commit=b50c3f83cd7664f0dbb6d611324c35d21974db15
```

### Reviewer-Owned Dry-Run Probe

Command summary:

```bash
DRY_RUN=1 STAGE123_MANIFEST_PYTHON=/opt/venv/bin/python STAGE123_SCRATCH_ROOT=/data-1/tmp/verl_agent_scratch/m6_review_dry_run REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
```

Result:

```text
rc=0
registry_unchanged=true
release_unchanged=true
stdout=[STAGE123 QUEUE] DRY_RUN PASS; Stage3 blocked: pending current manifest_hash=e665049cc67a40c32f0b104058bfe4e20c2529dc22328a485622bed78d3c8f0c
scratch files:
/data-1/tmp/verl_agent_scratch/m6_review_dry_run/frac25-stage2.provenance.json
/data-1/tmp/verl_agent_scratch/m6_review_dry_run/stage123.normalized.json
/data-1/tmp/verl_agent_scratch/m6_review_dry_run/status.tsv
status.tsv contains one stage3 pending_producer row
provenance count=1
forbidden outputs=<none>
```

Interpretation: the renderer writes normalized manifest, Stage2 provenance, and pending Stage3 status evidence under scratch only. It does not create `stage2_final_model2`, execution-state outputs, registry/release mutations, training, GPU, or external-service artifacts.

### Authority Boundary Audit

Implementation diff from `b50c3f83...` to `98fed968...` adds:

- `scripts/stage123_dry_run_compat.py`
- recipe gitlink update to `9b83f9f...`
- deletion-budget and authority-inventory entries for the compatibility renderer
- fixture hash updates and `F-M6-01` finding records

Recipe diff from `d717672...` to `9b83f9f...` adds only a DRY_RUN/no-manifest branch in `run_code_task_qwen3_1p7b_stage123_queue_impl.sh` that executes `scripts/stage123_dry_run_compat.py`; the normal batch path still delegates to `scripts/experiment_execution_core.py batch-run`, and retry/resume overrides remain rejected.

`stage123_dry_run_compat.py` contains `subprocess.check_output` only to render the manifest through `scripts/experiment_manifest.py`; source scan found no `tmux`, `docker`, `nvidia-smi`, `wandb`, `registry`, `checkpoint`, `run_train`, `ray`, `codex`, `crontab`, `systemd`, `auto_tune`, `parameter_mutation`, `batch-run`, `--resume`, or `recovery-policy` lifecycle/tuning authority tokens in that renderer.

The deletion budget now classifies `scripts/stage123_dry_run_compat.py` as `compatibility_renderer` with `authority: evidence_projection_only`; the authority inventory records removed authority categories `child launch`, `phase transition`, `checkpoint interpretation`, and `failure routing`.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None found.

## Single Most Likely Weakness In This Review

I did not rerun the full `tests/experiment_workflow` suite by instruction. I relied on the recorded full-gate log plus the focused replacement command and targeted dry-run/source probes, so this review validates the F-M6-01 repair and final evidence envelope without independently reproducing the entire 208-test full gate.
