# Milestone 2 V18 Port-Fix Batch Mechanical Rereview

## Review Identity

- Reviewer: independent GPT-5.5 medium scoped mechanical rereviewer.
- Review type: Mechanical Rereview.
- Scope: sole prior blocker, corrected `experiment_batch_manifest.json` binding.
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`.
- Frozen Plan version: 18.
- Base commit: `425f844734607b6e02bcd83a1de702d6e3239a30`.
- Candidate commit: `425f844734607b6e02bcd83a1de702d6e3239a30`.
- Applicable ACs: AC-01, AC-02, AC-07, AC-08, AC-12.

## Overall Verdict

PASS.

The prior blocker is resolved: `experiment_batch_manifest.json` now binds the accepted admission bundle file hash `b0f232241479adcd3ad8fde8e99eb2e3c06f2cc0ffa96f11b4645a885898e6b7`, implementation tree `7b0c9449ac66a0842007fc30169e833c19b37c11cb5f56761a4d878cc2d80bff`, Plan v18 hash `471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852`, unchanged command hash `5d22e664fc4666133c67b946c74d3cf91d13938ef6eb26daeccb9604423197a2`, and the exact three-run set. Batch validation passes against the supplied new empty state root. No training was launched.

## Per-AC Verdict Table

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Reused prior PASS for port fix/admission. Required accepted admission validation returned `authorized=true`; batch manifest now binds accepted admission file SHA `b0f232...`. |
| AC-02 | PASS | Batch manifest and loader show exactly `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3` in order; `batch-validate` returned `ok=true`. |
| AC-07 | PASS | Reused prior PASS for no-retry policy; corrected batch binding preserves unchanged command hash `5d22e664...` and `stage123_queue_v1`. |
| AC-08 | PASS | Required `nvidia-smi` query returned no rows; supplied state root `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state` is absent, so no persisted entries exist. |
| AC-12 | PASS | Reused prior PASS for port fix/tests/applicability/admission; batch now binds implementation tree `7b0c9449...` and unchanged V18 Plan hash. |

## Commands And Evidence

### Required Commands

1. Runtime validator:
   ```bash
   goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
   ```
   Result: exit 0; `plan_status=READY`, `plan_version=18`, `goal_status=ACTIVE`, no pending user decisions.

2. Accepted admission validation:
   ```bash
   REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl
   ```
   Output:
   ```json
   {"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}
   ```

3. Batch validation:
   ```bash
   REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state --repo-root /data-1/code/verl
   ```
   Output:
   ```json
   {"batch_id": "stage123-primary-chain-frac25-p40-v1", "batch_manifest_sha256": "f129078f00e6e5924f1ec4cc27812543e374b4639fc6c037c00eaed13a79198c", "items": ["stage123-primary"], "ok": true}
   ```

4. GPU process query:
   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Output: no rows.

### Focused Batch-Binding Evidence

- `git rev-parse HEAD` and `git rev-parse 425f844734607b6e02bcd83a1de702d6e3239a30` both returned `425f844734607b6e02bcd83a1de702d6e3239a30`.
- Manifest item fields:
  - `admission_bundle_sha256`: `b0f232241479adcd3ad8fde8e99eb2e3c06f2cc0ffa96f11b4645a885898e6b7`.
  - `implementation_tree_sha256`: `7b0c9449ac66a0842007fc30169e833c19b37c11cb5f56761a4d878cc2d80bff`.
  - `plan_sha256`: `471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852`.
  - `command_sha256`: `5d22e664fc4666133c67b946c74d3cf91d13938ef6eb26daeccb9604423197a2`.
  - `expected_run_ids`: `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`.
- File hashes:
  - `docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json`: `b0f232241479adcd3ad8fde8e99eb2e3c06f2cc0ffa96f11b4645a885898e6b7`.
  - `docs/joint_training/goals/stage123-primary-chain-execution/plan.md`: `471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852`.
  - `docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl`: `7b0c9449ac66a0842007fc30169e833c19b37c11cb5f56761a4d878cc2d80bff`.
- Batch manifest self-hash recomputation matched:
  - Expected: `f129078f00e6e5924f1ec4cc27812543e374b4639fc6c037c00eaed13a79198c`.
  - Computed: `f129078f00e6e5924f1ec4cc27812543e374b4639fc6c037c00eaed13a79198c`.
- Loader-bound commands remain unchanged:
  - `frac25-stage1-control :: /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage1-control`
  - `frac25-stage2 :: /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage2`
  - `frac25-stage3 :: /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage3`
- Supplied state root check: `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state` is absent, so the state root is new/empty.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

This rereview intentionally reused prior PASS findings and checked only the corrected batch-binding blocker; it did not rerun the full port-fix test/applicability/admission suite beyond the required accepted admission and batch validation commands.
