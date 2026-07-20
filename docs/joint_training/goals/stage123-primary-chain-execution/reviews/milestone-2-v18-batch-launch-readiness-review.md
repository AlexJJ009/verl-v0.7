# Milestone 2 V18 Batch Launch Readiness Review

## Review Identity

- Reviewer: independent GPT-5.5 medium mechanical launch-readiness reviewer.
- Review type: Launch Readiness Review.
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution`.
- Frozen Plan version: 18.
- Reviewed commit: `6a069213f3467529530217fa14a473d0671859f6`.
- Base commit: `6a069213f3467529530217fa14a473d0671859f6`.
- Candidate commit: `6a069213f3467529530217fa14a473d0671859f6`.
- Applicable ACs: AC-01, AC-02, AC-07, AC-08, AC-12.
- Report path: `/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-batch-launch-readiness-review.md`.

## Overall Verdict

PASS.

The accepted fresh admission bundle is file-hash-bound into a self-hashed batch manifest; the manifest binds the current Plan v18 hash, implementation tree hash, `stage123_queue_v1` adapter, exact three-run order, and one admitted launch command. Required validators pass. The supplied state root is absent/new with no persisted state entries. I found no active Stage123 tmux/process/GPU training and did not launch training.

## Per-AC Verdict Table

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | `admission validate --require-accepted` returned `{"authorized": true, "code": "authorized", "message": "current checkout matches admission bundle"}`. Rendered launch is one tmux command with `STAGE123_ADMISSION_BUNDLE`, `STAGE123_IMPLEMENTATION_TREE_SHA256=f97f5478...`, `STAGE123_BUNDLE_SHA256=ddcedb...`, and `EXPERIMENT_BATCH_MANIFEST=.../experiment_batch_manifest.json`. |
| AC-02 | PASS | `batch-validate` returned `ok=true`, item `stage123-primary`; manifest `expected_run_ids` are exactly `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`; loader prints bound commands in that order through `stage123_phase_adapter.py`. |
| AC-07 | PASS | `batch-validate` passes; `BatchExecutor` constructs each phase with `max_attempts=1` and `resumable_failure_codes=()`. Queue wrapper rejects `--resume` and `--recovery-policy`. Stage2 extraction target pre-existence fails closed. |
| AC-08 | PASS | Supplied state root `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T115009Z/state` is absent before launch, so no persisted state entries exist. `nvidia-smi --query-compute-apps=...` returned no rows; Stage123 tmux/process scans returned no rows. |
| AC-12 | PASS | Fresh V16 roots and run names are used in `stage123.yaml`: artifact/scratch roots end in `_v16`, run prefixes end in `V16`, run order is control order 0, Stage2 order 10, Stage3 order 20. Stage2 finalization extracts `stage2_final_model2`; Stage3 requires Stage2 provenance and matching extracted model2 path before running. No V13/V14/certified/reuse literal appears in launch/bundle/manifest/adapter files inspected. |

## Commands And Evidence

### Required Commands

1. Runtime validator:
   ```bash
   goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
   ```
   Relevant output: `plan_status: READY`, `plan_version: 18`, `goal_status: ACTIVE`, `current_milestone: Milestone 2`, exit 0.

2. Accepted admission validation:
   ```bash
   REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl
   ```
   Output: `{"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}`.

3. Batch validation:
   ```bash
   REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T115009Z/state --repo-root /data-1/code/verl
   ```
   Output: `{"batch_id": "stage123-primary-chain-frac25-p40-v1", "batch_manifest_sha256": "10dfd81813675ed31af5f3adc1e6beb20b15639002da5966c833315a40b37bb5", "items": ["stage123-primary"], "ok": true}`.

4. Render launch:
   ```bash
   REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission render-launch --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-host /data-1/code/verl
   ```
   Output printed, not executed:
   ```bash
   tmux new-session -d -s stage123_primary_chain env REPO_HOST=/data-1/code/verl ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1 STAGE123_ADMISSION_BUNDLE=/workspace/verl/docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json STAGE123_IMPLEMENTATION_TREE_SHA256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b STAGE123_BUNDLE_SHA256=ddcedb365e17c95ee86913ae4aa9e8a17935ae215f84815c47cecdb813904ede EXPERIMENT_BATCH_MANIFEST=/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json bash /data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
   ```

5. GPU process query:
   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Output: no rows.

### Additional Mechanical Evidence

- Git identity: `git rev-parse HEAD` and `git rev-parse 6a069213f3467529530217fa14a473d0671859f6` both returned `6a069213f3467529530217fa14a473d0671859f6`; `git diff --stat 6a069213f3467529530217fa14a473d0671859f6..HEAD` was empty.
- Admission bundle file hash: `ca58ce1b6f20d8fa3da7263420921fb70edca0f56238b24b9626d0a75d93912d`, matching `experiment_batch_manifest.json` line 11.
- Plan v18 file hash: `471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852`, matching `experiment_batch_manifest.json` line 21.
- Batch manifest self-hash recomputation with `sha256_json(_without_hash(raw, "batch_manifest_sha256"))` produced `10dfd81813675ed31af5f3adc1e6beb20b15639002da5966c833315a40b37bb5`, matching the file.
- Batch manifest item binds `adapter_type=stage123_queue_v1`, `implementation_tree_sha256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b`, and `command_sha256=5d22e664fc4666133c67b946c74d3cf91d13938ef6eb26daeccb9604423197a2`.
- Validator loader printed exact phase commands:
  - `frac25-stage1-control :: /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage1-control`
  - `frac25-stage2 :: /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage2`
  - `frac25-stage3 :: /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /workspace/verl/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --run-id frac25-stage3`
- `recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml` lines 28-29 use fresh roots `/data-2/model_weights/code_task/qwen3_1p7b_stage123_v16` and `/data-1/tmp/verl_agent_scratch/qwen3_1p7b_stage123_v16`; lines 59-115 list the three V16 run records and Stage3 source `run_id: frac25-stage2`.
- `scripts/stage123_phase_adapter.py` lines 162-177 merge Stage2 and extract submodel index 1 to `stage2_final_model2`; lines 178-179 bind Stage3 provenance to `stage2_model2`.
- `recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh` lines 12-28 require Stage2 provenance and reject an extracted model2 path mismatch before Stage3 starts.
- State root check: `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T115009Z/state` does not exist; no persisted state entries exist.
- Stage123 live-process checks: `tmux list-sessions | rg -i 'stage123|primary_chain|qwen3_1p7b'` returned no rows; `ps -eo ... | rg -i 'stage123|run_code_task_qwen3_1p7b|main_ppo|primary_chain'` returned no rows after excluding the probe itself.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None from this mechanical launch-readiness review.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

This review is mechanical and pre-launch: it verifies bindings, validators, command rendering, empty state, and absence of live Stage123/GPU activity, but it does not execute training and therefore cannot validate future runtime behavior after the rendered tmux command is actually launched.
