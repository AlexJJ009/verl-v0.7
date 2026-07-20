# Milestone 2 V18 Control-Reuse Treatment Readiness Review

Reviewer: independent Codex reviewer
Scope: Goal Plan v18, committed state `1f8ccf9f93902b30857cac063f4859be3a7b5e21`, uncommitted certificate `docs/joint_training/goals/stage123-primary-chain-execution/certified-control-reuse-v18.json`, and prepared treatment artifacts under `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/`.
Restriction followed: I did not implement or modify production/runtime ledger files. No training was launched.

## Verdict

PASS.

The V18 certified-control reuse evidence is reviewer-recomputable; the old root remains terminal `completed_with_failures` with Control succeeded and Stage2 failed once after child start; the refreshed accepted admission is bound to commit `1f8ccf9f93902b30857cac063f4859be3a7b5e21`; the new treatment identity is prepared but not authorized, executes only `frac25-stage2` and `frac25-stage3` when later authorized, starts Stage2 from the original admitted P40 source rather than Control final weights, and has isolated state/artifact/monitor/provenance roots. Current GPU workload is idle.

## Commands And Evidence

### Repository Identity

Command:

```bash
git branch --show-current
git rev-parse HEAD
```

Evidence:

```text
codex/stage123-validation-protocol-rerun
1f8ccf9f93902b30857cac063f4859be3a7b5e21
```

### Runtime And Accepted Admission

Command:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Evidence:

```text
plan_status: READY
plan_version: 18
current_milestone: Milestone 2
goal_status: ACTIVE
pending_user_decisions: []
latest_review.verdict: PASS
latest_review.candidate_commit: 1f8ccf9f93902b30857cac063f4859be3a7b5e21
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted --repo-root /data-1/code/verl
```

Evidence:

```json
{"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}
```

Command:

```bash
sha256sum docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Evidence:

```text
5e8fa729c946d8226cbc107025ca896cbbeb03da42e2c73961f75986fab4b0d4  docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json
b6ccaaaa99acaafcde44122c64a3aa1ae9a7fe6b0851b673c6490f9deae19aee  docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json
bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee  docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl --format json
```

Evidence:

```text
{"implementation_tree_sha256": "bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee"}
```

### Certified Control Reuse

Command:

```bash
sha256sum docs/joint_training/goals/stage123-primary-chain-execution/certified-control-reuse-v18.json
```

Evidence:

```text
0cc4dcbbc2df9f030586d52ac6032856224ef6104391c2f49e9531987e6155f3  docs/joint_training/goals/stage123-primary-chain-execution/certified-control-reuse-v18.json
```

Reviewer-owned recomputation used the same sorted relative-path content tree hash algorithm as `scripts/stage123_control_reuse.py`: each file under the checkpoint root is hashed, rows are canonical JSON with sorted keys and compact separators, then SHA256 is computed over that canonical payload.

Evidence from independent recomputation:

```text
control provenance: actual 1bfa9fdd4ae57434d2a739f8aafe9cc7f44a9d7684152df112b556dafed2790d, expected 1bfa9fdd4ae57434d2a739f8aafe9cc7f44a9d7684152df112b556dafed2790d, ok true
control checkpoint tree: actual e1bf0524d8a7b635711f4bcbfb14c4000f96e1f8485194bd188077707c1ef6e1, expected e1bf0524d8a7b635711f4bcbfb14c4000f96e1f8485194bd188077707c1ef6e1, ok true
control metrics: actual dbedcafd86ff357e0a444a10042e171e421829fe4f181be6a664fb159eebc527, expected dbedcafd86ff357e0a444a10042e171e421829fe4f181be6a664fb159eebc527, ok true
control final validation: actual 5d07fb2ddaa5d20ec09268cab92b8e4441376943cd0f5f7292d2e54771b75381, expected 5d07fb2ddaa5d20ec09268cab92b8e4441376943cd0f5f7292d2e54771b75381, ok true
control train file: actual a73593277c2997b579e64fa786a93a504a086465b20e7965bfd7724eabbf65f4, expected a73593277c2997b579e64fa786a93a504a086465b20e7965bfd7724eabbf65f4, ok true
old batch state: actual 6421a6870ebea395a19c99d1ddc113f6da4ede068be3b25b5e91ba328aec075d, expected 6421a6870ebea395a19c99d1ddc113f6da4ede068be3b25b5e91ba328aec075d, ok true
old events: actual 5dd4201e6ab07a9dfdc61179cb3034fa8d1c54c15d19aa52f25c32db7f8ada0a, expected 5dd4201e6ab07a9dfdc61179cb3034fa8d1c54c15d19aa52f25c32db7f8ada0a, ok true
old stage2 state: actual 406a1f785090641010766b24b94102e7c9da3c200a0614e8196c50c9dd66e27d, expected 406a1f785090641010766b24b94102e7c9da3c200a0614e8196c50c9dd66e27d, ok true
recovery admission file: actual 5e8fa729c946d8226cbc107025ca896cbbeb03da42e2c73961f75986fab4b0d4, expected 5e8fa729c946d8226cbc107025ca896cbbeb03da42e2c73961f75986fab4b0d4, ok true
recovery implementation tree file: actual bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee, expected bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee, ok true
recovery acceptance report: actual b6ccaaaa99acaafcde44122c64a3aa1ae9a7fe6b0851b673c6490f9deae19aee, expected b6ccaaaa99acaafcde44122c64a3aa1ae9a7fe6b0851b673c6490f9deae19aee, ok true
overall_ok: true
```

### Old Root Terminal Failure Boundary

Inspected root:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix
```

Evidence:

```json
{"batch_status": "completed_with_failures", "phases": [["frac25-stage1-control", "succeeded"], ["frac25-stage2", "failed"]]}
{"stage2_status": "failed", "stage2_child_id": "3570606", "stage2_attempt": 1, "transitions": [["pending", "running"], ["running", "failed"]], "failure": {"code": "child_exit", "context": {"returncode": 1}, "message": "child process exited unsuccessfully"}}
```

Evidence that the old root contains only state files, not Stage2 checkpoint/metrics/provenance/extraction:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state/events.jsonl
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state/frac25-stage1-control.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state/frac25-stage2.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state/stage123-primary-chain-frac25-p40-v1.json
```

Certificate-forbidden Stage2 outputs are absent:

```text
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/stage2_final_joint
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/stage2_final_model2
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/frac25-stage2.provenance.json
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/frac25-stage2
```

Additional checkpoint search for V18 Stage2 checkpoint candidates under `/data-1/checkpoints` and `/data-2/checkpoints` returned no matches. The certificate field `old_failure.stage2_checkpoint_matches` is `[]`.

### Old Admission And Implementation Binding

The certificate binds the old failed root to:

```text
source_admission_bundle_sha256: b0f232241479adcd3ad8fde8e99eb2e3c06f2cc0ffa96f11b4645a885898e6b7
source_implementation_tree_sha256: 7b0c9449ac66a0842007fc30169e833c19b37c11cb5f56761a4d878cc2d80bff
source_plan_sha256: 471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852
```

The refreshed accepted recovery admission is bound to:

```text
file_sha256: 5e8fa729c946d8226cbc107025ca896cbbeb03da42e2c73961f75986fab4b0d4
bundle_sha256: 2d0052fed26d58a5a071ce6c4683c852d0fda1c0195f81cf71109712bda40000
implementation_tree_file_sha256: bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee
evidence_commit: 1f8ccf9f93902b30857cac063f4859be3a7b5e21
recipe_gitlink: 324a6aef2433f0163bf58e14be9d537fa7410388
```

### Treatment Prepared State

Treatment root:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z
```

Prepared files:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/provenance/treatment-reuse.provenance.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-batch-manifest.json
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-manifest.yaml
```

State and monitor directories exist but contain no old copied state:

```text
d /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/monitor
d /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/state
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json --allow-prepared
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json --allow-prepared --run-id frac25-stage2
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json --allow-prepared --run-id frac25-stage3
```

Evidence:

```json
{"execution_id": "v18-recovery-20260716T1724Z", "ok": true, "status": "prepared_not_authorized"}
{"execution_id": "v18-recovery-20260716T1724Z", "ok": true, "status": "prepared_not_authorized"}
{"execution_id": "v18-recovery-20260716T1724Z", "ok": true, "status": "prepared_not_authorized"}
```

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-batch-manifest.json --repo-root /data-1/code/verl
```

Evidence:

```json
{"failure": {"code": "invalid_batch_request", "context": {}, "message": "treatment admission is prepared but not authorized"}, "ok": false}
```

This is expected for the requested review because the treatment is explicitly `prepared_not_authorized` and training must not start from a prepared-only batch.

Treatment admission and batch internal canonical hashes are valid:

```json
{"canonical_without_field": "0ed4765628bb0dec5d2b591687bb1bce21c88a1b502263b752fba1fe7fd9a3d1", "field": "admission_sha256", "field_hash": "0ed4765628bb0dec5d2b591687bb1bce21c88a1b502263b752fba1fe7fd9a3d1", "file_sha256": "de139dc511b2d2e971ba709e164fb5bbc696b419518a54b46dd8788db197c158", "ok": true, "path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json"}
{"canonical_without_field": "9332804af10a166f4ca6e4bd8a3e956dfae977cc90f3543f024724426c51ebc8", "field": "batch_manifest_sha256", "field_hash": "9332804af10a166f4ca6e4bd8a3e956dfae977cc90f3543f024724426c51ebc8", "file_sha256": "8c558391580f042881ba89cc611b7f452f4ec045ed14d7b5e55d2a1c3585c9e6", "ok": true, "path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-batch-manifest.json"}
```

The treatment admission and batch item authorize only the treatment phases:

```json
{"expected_run_ids": ["frac25-stage2", "frac25-stage3"], "status": "prepared_not_authorized", "state_root": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/state", "monitor_path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/monitor/treatment-monitor.log", "provenance_path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/provenance/treatment-reuse.provenance.json"}
```

The batch item has:

```text
adapter_type: stage123_treatment_reuse_v1
expected_run_ids: frac25-stage2, frac25-stage3
prepared_not_authorized: true
```

### Treatment Manifest Source And Isolation

Original manifest render:

```text
manifest_sha256: 1d2c5d55da7d758784b7383a574449cecc9bd67281a2b17752150dcb542d14fc
run_ids: frac25-stage1-control, frac25-stage2, frac25-stage3
frac25-stage2 source.model2_path: /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/beta01/stage1_model2
frac25-stage2 source.checkpoint_root: /data-2/checkpoints/ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783425947
frac25-stage3 source: {"run_id": "frac25-stage2", "type": "stage2_model2"}
```

Treatment manifest render:

```text
manifest_sha256: e56d71458d246e43788a80f9f184720b147c00d33e8ae176c4d3ccf1d6d4de0d
frac25-stage2 run_prefix: CODE-S2-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-LAMBDA08-V16-TREATMENT-v18-recovery-20260716T1724Z
frac25-stage3 run_prefix: CODE-S3-QWEN3-1P7B-STAGE123-FRAC25_P40_S220_S340-BETA01-V16-TREATMENT-v18-recovery-20260716T1724Z
frac25-stage2 artifact_dir: /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage2
frac25-stage2 provenance_file: /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage2.provenance.json
frac25-stage2 source.model2_path: /data-2/model_weights/code_task/qwen3_1p7b_stage123_v16/frac25_p40_s220_s340_v16/beta01/stage1_model2
frac25-stage2 source.checkpoint_root: /data-2/checkpoints/ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783425947
frac25-stage3 source: {"run_id": "frac25-stage2", "type": "stage2_model2"}
frac25-stage3 artifact_dir: /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage3
frac25-stage3 provenance_file: /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage3.provenance.json
```

Conclusion: Stage2 remains bound to the original admitted P40 source, not the Control final weights. Stage3 is bound to the new Stage2 output by `source.run_id = frac25-stage2`. New treatment artifact/provenance paths use the `qwen3_1p7b_stage123_v18_recovery` root, separate from the old V18 failed state root and the prior V16 artifact root.

A grep over the new treatment root for old reuse roots found no references to:

```text
primary-v18-20260716T123423Z-portfix
treatment-reuse-20260716T0317Z
stage3-handoff
```

### GPU And Workload State

Command:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
tmux ls 2>/dev/null | rg -i 'stage123|train|qwen|verl|code_task' || true
```

Evidence:

```text
0, 0 %, 0 MiB, 46068 MiB
1, 0 %, 0 MiB, 46068 MiB
2, 0 %, 0 MiB, 46068 MiB
3, 0 %, 0 MiB, 46068 MiB
4, 0 %, 0 MiB, 46068 MiB
5, 0 %, 0 MiB, 46068 MiB
6, 0 %, 0 MiB, 46068 MiB
7, 0 %, 0 MiB, 46068 MiB
```

`--query-compute-apps` emitted no compute rows, and the tmux filter emitted no Stage123/train/qwen/verl/code_task sessions. No GPU workload is currently running.

## Review Notes

- The prepared treatment batch is intentionally not launchable yet: `batch-validate` fails closed with `treatment admission is prepared but not authorized`. This supports the requested boundary that no training was launched.
- The treatment manifest file still carries the full three-run matrix, but the treatment admission and batch item authorize only `frac25-stage2` and `frac25-stage3`; this satisfies the treatment-only execution boundary.
- I did not append runtime events, edit `runtime.jsonl`, edit `findings.jsonl`, or modify production/runtime ledger state.

## Authorized Launch Re-verification

Mechanical re-verification result: PASS.

Scope: same V18 treatment root, after authorization was written to `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json` and `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/authorized-treatment-batch-manifest.json`. I did not start training, did not run `batch-run`, did not invoke a phase wrapper, and did not modify runtime ledgers.

### Authorized Artifacts

Command:

```bash
sha256sum /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/authorized-treatment-batch-manifest.json
```

Evidence:

```text
c57f3d8934b4f82a11787ea088aa7cb2a77e5c0eb8951e3ba1b9e3dec05e931c  /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json
79a0018dc25044975705ec644ce8b21ad41663e7dae48eee2fc3425bac0cf901  /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/authorized-treatment-batch-manifest.json
```

Admission summary:

```json
{
  "admission_sha256": "1cbee429effc738c5cc0b2e51543ca2d8b6e50801f8b46108dd8678057759e08",
  "authorization": {
    "authorized_at": "2026-07-16T17:35:13Z",
    "decision_id": "user-authorized-stage123-v18-treatment-2026-07-16",
    "host_facts_path": "docs/joint_training/goals/stage123-execution-readiness/host_facts.json",
    "host_facts_sha256": "86f778707462c19ca19b2f6ad7724626c9c049025fa5016ca197d0cc6a2ed37e",
    "resource_profile_sha256": "d687caf1146c9b32a2f51dcf71e876cf5f987fe62b4db69f1043caaff52ddbe8"
  },
  "execution_id": "v18-recovery-20260716T1724Z",
  "expected_run_ids": ["frac25-stage2", "frac25-stage3"],
  "status": "authorized"
}
```

Authorized batch item bindings:

```json
{
  "admission_bundle_path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json",
  "admission_bundle_sha256": "c57f3d8934b4f82a11787ea088aa7cb2a77e5c0eb8951e3ba1b9e3dec05e931c",
  "command_sha256": "a2690fb454361cccc1d8f0423dede429b4eed19bc3b1176d71a16aa546bb3d0b",
  "expected_run_ids": ["frac25-stage2", "frac25-stage3"],
  "implementation_tree_sha256": "bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee",
  "item_id": "stage123-treatment-reuse-v18-recovery-20260716T1724Z",
  "plan_sha256": "471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852"
}
```

Internal canonical hashes are valid:

```json
{"canonical_without_field": "1cbee429effc738c5cc0b2e51543ca2d8b6e50801f8b46108dd8678057759e08", "field": "admission_sha256", "field_hash": "1cbee429effc738c5cc0b2e51543ca2d8b6e50801f8b46108dd8678057759e08", "file_sha256": "c57f3d8934b4f82a11787ea088aa7cb2a77e5c0eb8951e3ba1b9e3dec05e931c", "ok": true, "path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json"}
{"canonical_without_field": "99d2f4be951a7a5771f1ae6184207c195bcc449c528182bcdd00d85a2c2d854a", "field": "batch_manifest_sha256", "field_hash": "99d2f4be951a7a5771f1ae6184207c195bcc449c528182bcdd00d85a2c2d854a", "file_sha256": "79a0018dc25044975705ec644ce8b21ad41663e7dae48eee2fc3425bac0cf901", "ok": true, "path": "/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/authorized-treatment-batch-manifest.json"}
```

### Validators

Commands:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json --run-id frac25-stage2
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/treatment-admission.json --run-id frac25-stage3
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/authorized-treatment-batch-manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/state --repo-root /data-1/code/verl
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Evidence:

```json
{"execution_id": "v18-recovery-20260716T1724Z", "ok": true, "status": "authorized"}
{"execution_id": "v18-recovery-20260716T1724Z", "ok": true, "status": "authorized"}
{"batch_id": "stage123-treatment-reuse-v18-recovery-20260716T1724Z", "batch_manifest_sha256": "99d2f4be951a7a5771f1ae6184207c195bcc449c528182bcdd00d85a2c2d854a", "items": ["stage123-treatment-reuse-v18-recovery-20260716T1724Z"], "ok": true}
```

Runtime summary:

```json
{
  "current_milestone": "Milestone 2",
  "goal_status": "ACTIVE",
  "latest_review": {
    "candidate_commit": "1f8ccf9f93902b30857cac063f4859be3a7b5e21",
    "plan_version": 18,
    "review_id": "milestone-2-v18-control-reuse-treatment-readiness-review",
    "verdict": "PASS"
  },
  "pending_user_decisions": [],
  "plan_status": "READY",
  "plan_version": 18
}
```

### Launch Boundary Checks

State and monitor roots are still empty directories:

```text
d /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/state
d /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z/monitor
```

GPU and tmux checks:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
tmux ls 2>/dev/null | rg -i 'stage123|train|qwen|verl|code_task' || true
```

Evidence:

```text
--query-compute-apps emitted no rows.
0, 0 %, 0 MiB, 46068 MiB
1, 0 %, 0 MiB, 46068 MiB
2, 0 %, 0 MiB, 46068 MiB
3, 0 %, 0 MiB, 46068 MiB
4, 0 %, 0 MiB, 46068 MiB
5, 0 %, 0 MiB, 46068 MiB
6, 0 %, 0 MiB, 46068 MiB
7, 0 %, 0 MiB, 46068 MiB
tmux filter emitted no Stage123/train/qwen/verl/code_task sessions.
```

Pre-training treatment artifact outputs remain absent:

```text
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage2
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage3
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage2.provenance.json
ABSENT /data-2/model_weights/code_task/qwen3_1p7b_stage123_v18_recovery/frac25_p40_s220_s340_v18_recovery/frac25-stage3.provenance.json
```

Conclusion: the authorized launch artifacts are mechanically valid and bound to the expected Plan/tree/admission/command/state identities. The system is ready for launch authorization consumption, but this review did not launch training.

## F-EX-LAUNCH-10 Mechanical Rereview

Mechanical rereview result: PASS.

Scope: I reviewed only the launch-boundary correction after the failed 1724Z authorized attempt. I did not launch training, did not run `batch-run`, and did not modify runtime ledgers.

Old terminal root:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1724Z
```

Evidence:

```text
batch status: completed_with_failures
item status: inconclusive_operational_failure
failed phase: frac25-stage2
returncode: 126
skipped phases: frac25-stage3
queue log: /data-1/verl07/run_train.sh: line 20: exec: docker: cannot execute: Is a directory
```

The old root has only state/provenance/manifest/log files under the treatment root; no Stage2/Stage3 training artifact, checkpoint, metrics, validation, or provenance output was present. The V18 recovery artifact outputs checked for both the 1724Z and 1740Z roots remained absent before training.

New authorized root:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z
```

Identity evidence:

```text
treatment-admission file sha256: 8cf140929a9e2370c1945da74693b0366827599f93330a500eb49a6b3652cf72
authorized batch file sha256: 8a9855c2c406336d2d17123ddfd8b034e180888707c48e228bf363d996a52458
treatment manifest sha256: 725d6c469a79d773ec05a53996a5556ea1f5a330248f312dfa77cddcc75c6cbf
admission status: authorized
execution_id: v18-recovery-20260716T1740Z
expected_run_ids: frac25-stage2, frac25-stage3
plan_sha256: 471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852
implementation_tree_sha256: bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee
evidence_commit: 1f8ccf9f93902b30857cac063f4859be3a7b5e21
recipe_gitlink: 324a6aef2433f0163bf58e14be9d537fa7410388
authorized item admission_bundle_sha256: 8cf140929a9e2370c1945da74693b0366827599f93330a500eb49a6b3652cf72
authorized item command_sha256: 39761a857a20de280f2652fa94f4f36b4ecb5cdbf8d3f9a4ae63526457d3971e
```

Internal hash checks:

```text
admission_sha256 canonical check: 809398edea982420786052ce93ffa037c77410450b3e8b597dcce1d0880c3a22 == field, ok true
batch_manifest_sha256 canonical check: 6e923a16901fc72d6bdb2ea27b1ca04828e56fb0cb3182a6507140cffe863507 == field, ok true
```

Validator evidence:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/treatment-admission.json --run-id frac25-stage2
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/stage123_control_reuse.py validate-treatment --admission /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/treatment-admission.json --run-id frac25-stage3
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/authorized-treatment-batch-manifest.json --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-v18-20260716T1740Z/state --repo-root /data-1/code/verl
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Outputs:

```json
{"execution_id": "v18-recovery-20260716T1740Z", "ok": true, "status": "authorized"}
{"execution_id": "v18-recovery-20260716T1740Z", "ok": true, "status": "authorized"}
{"batch_id": "stage123-treatment-reuse-v18-recovery-20260716T1740Z", "batch_manifest_sha256": "6e923a16901fc72d6bdb2ea27b1ca04828e56fb0cb3182a6507140cffe863507", "items": ["stage123-treatment-reuse-v18-recovery-20260716T1740Z"], "ok": true}
```

Runtime validator remained `READY`, Plan v18, with no pending user decisions.

State and launch boundary:

```text
1740Z state root: empty directory
1740Z monitor root: empty directory
GPU compute apps: no rows
GPU summary: all 8 GPUs 0% util, 0 MiB used
tmux Stage123/train/qwen/verl/code_task filter: no sessions
```

Correct launch boundary:

```text
Host boundary: python3 scripts/experiment_execution_core.py batch-run --manifest <authorized-treatment-batch-manifest.json> --state-root <1740Z/state> --repo-root /data-1/code/verl
Wrapper evidence: recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh ends with exec python3 "${REPO_ROOT}/scripts/experiment_execution_core.py" batch-run ...
Phase child boundary: each treatment phase command is exactly /data-1/verl07/run_train.sh python /workspace/verl/scripts/stage123_phase_adapter.py --manifest <1740Z treatment-manifest.yaml> --run-id <frac25-stage2|frac25-stage3>
Command hash recomputation: 39761a857a20de280f2652fa94f4f36b4ecb5cdbf8d3f9a4ae63526457d3971e
```

Conclusion: F-EX-LAUNCH-10 mechanical correction is verified. The failed 1724Z root is terminal returncode 126 without GPU/training artifacts. The fresh 1740Z authorized identities validate, have empty state/artifact roots, and preserve the corrected host launch boundary so `run_train.sh` is entered once by each phase child rather than wrapping the batch runner itself.
