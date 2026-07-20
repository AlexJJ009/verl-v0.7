# Milestone 3-4 Stage2 Extraction And Stage3 Launch Review

Reviewer: independent Codex reviewer
Scope: post-launch review of `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z`, Stage2 completion/extraction, and Stage3 live launch.
Candidate commit: `1f8ccf9f93902b30857cac063f4859be3a7b5e21`
Restriction followed: I did not modify training, did not stop/restart any process, and did not publish registry or W&B results.

## Verdict

PASS.

AC-04, AC-05, AC-06, and the no-publication/no-weakening part of AC-07 pass for this post-launch milestone boundary. Stage2 succeeded exactly once without resume, produced 20-step metrics/validation/provenance and extracted `stage2_final_joint` / `stage2_final_model2` trees. Stage3 is running from the extracted Stage2 model2, has an active trainer process and 8-GPU allocation, and its trainer command contains `actor_rollout_ref.rollout.val_kwargs.temperature=0.2`. The batch runner uses the active item admission path after item start and does not reapply wall-clock freshness at the Stage3 boundary.

## AC Results

| AC | Result | Evidence |
| --- | --- | --- |
| AC-04 Stage2 Execution Completes | PASS | `frac25-stage2.json` has `status=succeeded`, `attempt=1`, `resume_from_checkpoint=false`, transitions `pending -> running -> succeeded`, and batch phase 0 succeeded. |
| AC-05 Stage2 Extraction Provenance | PASS | Stage2 provenance is release-eligible, final step is 20, metrics hash matches, validation `20.jsonl` exists, and extracted joint/model2 tree hashes were recomputed. |
| AC-06 Stage3 Starts From Extracted Model2 | PASS | Treatment manifest binds Stage3 source to `{"type":"stage2_model2","run_id":"frac25-stage2"}`; phase adapter derives `STAGE2_MODEL2_PATH` from Stage2 artifact dir; live trainer command uses the extracted model2 path. |
| AC-07 No Retry/Resume/Weakening/Publication | PASS | Stage2 and Stage3 state both show `attempt=1` and `resume_from_checkpoint=false`; Stage3 is running, not terminal/published; no registry or W&B cloud publication was performed by this review. |

## Commands And Evidence

### Stage2 State

Command:

```bash
python -m json.tool /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/state/frac25-stage2.json
```

Evidence:

```json
{
  "attempt": 1,
  "child_id": "3605502",
  "failure": null,
  "max_attempts": 1,
  "resume_from_checkpoint": false,
  "run_id": "frac25-stage2",
  "status": "succeeded",
  "transitions": [
    {"from": "pending", "to": "running"},
    {"from": "running", "to": "succeeded"}
  ]
}
```

Batch state evidence:

```json
{
  "status": "running",
  "current_run_id": "frac25-stage3",
  "phases": [
    {
      "run_id": "frac25-stage2",
      "phase_index": 0,
      "status": "succeeded",
      "attempt": 1,
      "failure": null
    }
  ]
}
```

The event ledger has exactly one Stage2 running event and one Stage2 succeeded event before Stage3 starts.

### Stage2 Metrics, Validation, And Provenance

Stage2 provenance path:

```text
/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2.provenance.json
```

Provenance summary:

```json
{
  "checkpoint": "/data-1/checkpoints/CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V16-TREATMENT-v18-recovery-20260716T1740Z_1784223880",
  "final_step": 20,
  "manifest_sha256": "9ba794a9d2dd504dd25c3d64a617e14d5d7d55d46515dc771c7751165341b0cd",
  "metrics_sha256": "fabba885b079d2ad06bf855ef5925fb602c70442ac2da112c538d87ea2cd4134",
  "phase": "stage2",
  "release_eligible": true,
  "run_id": "frac25-stage2",
  "train_file_sha256": "160be1866e6c1dc439dcfbd594b54324f000f1f48db1f6a0fc88cf227c628dab"
}
```

Reviewer recomputation:

```json
{"label": "provenance", "sha256": "52dd51e217eacac53f0dca914e168e879e9f8d335470638525025ceb3e7a5756"}
{"label": "metrics", "actual": "fabba885b079d2ad06bf855ef5925fb602c70442ac2da112c538d87ea2cd4134", "expected": "fabba885b079d2ad06bf855ef5925fb602c70442ac2da112c538d87ea2cd4134", "ok": true}
{"metrics_lines": 21, "validation_lines": 1379, "validation_sha256": "0614344c6874b147ed780a4e9ebcbbbdbaf66bb3c460a42ba233f91e4577b51b"}
```

Validation path:

```text
/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/runtime/frac25-stage2/logs/validation/20.jsonl
```

### Stage2 Extraction Trees

Reviewer-owned sorted relative-path content tree hashes:

```json
{"label": "joint_model", "path": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/stage2_final_joint", "tree_sha256": "64313ed1c212dfbd2c711be34c0894bd8c29cd67d833230de39e557f76418724", "file_count": 14}
{"label": "extracted_model2", "path": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/stage2_final_model2", "tree_sha256": "9ef39170674724ca46e21eec301580fb00d72213b053470f1708bca26d4ec7fc", "file_count": 6}
```

The Stage2 provenance source section points to the same two extracted paths:

```json
{
  "type": "stage2_complete",
  "joint_model": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/stage2_final_joint",
  "extracted_model2": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/stage2_final_model2"
}
```

### Stage3 Manifest And Source Binding

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/treatment-manifest.yaml --format json
```

Evidence:

```json
{
  "manifest_sha256": "9ba794a9d2dd504dd25c3d64a617e14d5d7d55d46515dc771c7751165341b0cd",
  "stage2_artifact_dir": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2",
  "stage2_provenance_file": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2.provenance.json",
  "stage3_artifact_dir": "/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage3",
  "stage3_source": {"run_id": "frac25-stage2", "type": "stage2_model2"}
}
```

Derived Stage3 model2 source:

```text
/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/stage2_final_model2
```

The live Stage3 trainer command confirms the same source path:

```text
actor_rollout_ref.model.path=/data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery_1740/frac25-stage2/stage2_final_model2
```

### Stage3 Running State, Process, And GPU Allocation

Stage3 state:

```json
{
  "attempt": 1,
  "child_id": "3856653",
  "failure": null,
  "max_attempts": 1,
  "resume_from_checkpoint": false,
  "run_id": "frac25-stage3",
  "status": "running",
  "transitions": [
    {"from": "pending", "to": "running"}
  ]
}
```

Process evidence:

```text
PID 3856653: docker run ... verl-harness:latest python /workspace/verl/scripts/stage123_phase_adapter.py --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/treatment-manifest.yaml --run-id frac25-stage3
PID 3857535: python3 -m verl.trainer.main_ppo ... trainer.experiment_name=CODE-S3-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16-TREATMENT-v18-recovery-20260716T1740Z_1784227624 ...
tmux: stage123_v18_recovery_1740
```

GPU allocation from `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader` showed Stage3 Ray/vLLM workers on all 8 GPUs. Per-GPU summary at review time:

```text
GPU0 95% util, 21047 MiB used
GPU1 98% util, 20375 MiB used
GPU2 95% util, 20333 MiB used
GPU3 96% util, 20377 MiB used
GPU4 99% util, 20303 MiB used
GPU5 98% util, 20423 MiB used
GPU6 98% util, 20787 MiB used
GPU7 99% util, 20577 MiB used
```

### Stage3 Trainer Validation Parameters

The live trainer command and Stage3 log both show:

```text
actor_rollout_ref.rollout.val_kwargs.temperature=0.2
actor_rollout_ref.rollout.val_kwargs.top_p=0.95
actor_rollout_ref.rollout.val_kwargs.top_k=-1
actor_rollout_ref.rollout.val_kwargs.do_sample=True
actor_rollout_ref.rollout.val_kwargs.n=1
```

The Stage3 log contains the rendered config:

```text
'val_kwargs': {'_target_': 'verl.workers.config.SamplingConfig',
               'do_sample': True,
               'temperature': 0.2,
               'top_p': 0.95}
```

### Wall-Clock Freshness Boundary

Evidence that wall-clock freshness was not reapplied at Stage3 boundary:

- `events.jsonl` has one `item_started` event and one active admission record for the entire treatment item, followed by `phase_started` for Stage2 and Stage3.
- The active admission record hash is valid: `3effc5e52cb7be6497b18ca0e9ab763f62e7c76437a0b3f08c820502d256b713`.
- `scripts/experiment_execution_core.py` validates active item admissions with `static_after_item_start=True`, which calls admission validation with `static_after_item_start=True`; this disables freshness re-enforcement after item start while preserving static Plan/tree/admission/command bindings.

Relevant runtime events:

```text
item_started: 1
phase_started: 2
phase_terminal: 1
running atomic events: 2
succeeded atomic events: 1
```

Active admission record check:

```json
{"record_sha256": "3effc5e52cb7be6497b18ca0e9ab763f62e7c76437a0b3f08c820502d256b713", "canonical_without_field": "3effc5e52cb7be6497b18ca0e9ab763f62e7c76437a0b3f08c820502d256b713", "ok": true, "status": "active"}
```

### Runtime Validation

Command:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Evidence:

```json
{
  "current_milestone": "Milestone 2",
  "goal_status": "ACTIVE",
  "pending_user_decisions": [],
  "plan_status": "READY",
  "plan_version": 18
}
```

`validate-runtime` passed. The runtime ledger still reports the latest prior review as Milestone 2 because this file is a post-launch independent review artifact only; I did not append ledger events.

## Non-Actions

- Did not publish to registry.
- Did not sync W&B cloud.
- Did not modify training, checkpoint, state, or runtime ledger files.
- Did not restart, stop, or otherwise control the running Stage3 process.
