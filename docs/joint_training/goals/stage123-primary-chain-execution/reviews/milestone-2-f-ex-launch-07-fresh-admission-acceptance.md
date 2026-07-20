# Independent Fresh Admission Acceptance — F-EX-LAUNCH-07

## Review Identity

- Reviewer: independent GPT-5.5 medium Fresh Admission Acceptance reviewer.
- Review type: separate Fresh Admission Acceptance for Stage123 Plan v18.
- Candidate commit: `1f8ccf9f93902b30857cac063f4859be3a7b5e21`.
- Stage123 execution Plan version: 18.
- Readiness Plan version: 9.
- Candidate admission bundle path: `docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json`.
- Rewritten acceptance report path: `docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json`.

## Overall Verdict

**PASS.**

I accepted the fresh admission bundle for candidate `1f8ccf9f93902b30857cac063f4859be3a7b5e21`, rewrote `acceptance_report.json`, rebuilt the canonical accepted `admission_bundle.json`, and validated the accepted bundle with `--require-accepted`.

- Accepted candidate bundle SHA256: `2d0052fed26d58a5a071ce6c4683c852d0fda1c0195f81cf71109712bda40000`.
- Accepted admission file SHA256: `5e8fa729c946d8226cbc107025ca896cbbeb03da42e2c73961f75986fab4b0d4`.
- Acceptance report file SHA256: `b6ccaaaa99acaafcde44122c64a3aa1ae9a7fe6b0851b673c6490f9deae19aee`.
- Acceptance report self-hash: `ed0a1d560bfd7c502dfcb284287e2fd076c554a4f4ac6e56b101b5c2f58d284e`.

## Acceptance Checks

| Check | Verdict | Evidence |
| --- | --- | --- |
| Candidate commit | PASS | `git rev-parse HEAD` returned `1f8ccf9f93902b30857cac063f4859be3a7b5e21`. |
| Recipe gitlink | PASS | `git -C recipe rev-parse HEAD` returned `324a6aef2433f0163bf58e14be9d537fa7410388`; admission bundle binds the same value. |
| Implementation tree | PASS | `implementation-tree.jsonl` file SHA is `bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee`; compare command recomputed the same `implementation_tree_sha256`. |
| Calibration applicability | PASS | `calibration_result.json` embeds `stage123_calibration_applicability` with `decision=applicable` and `capacity_differences=[]`; standalone `calibration_applicability.json` matches the same decision and empty differences. |
| Host facts | PASS | `host_facts.json` has `ok=true`, completed at `2026-07-16T17:04:00Z`, file SHA `86f778707462c19ca19b2f6ad7724626c9c049025fa5016ca197d0cc6a2ed37e`. |
| Preflight | PASS | `preflight_result.json` has `decision=passed`, `ok=true`, exact three run IDs, implementation tree `bbb960...`, and host facts SHA `86f778...`. |
| Protected baseline | PASS | Protected baseline compare returned `{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}`. |
| Three-run manifest | PASS | Rendered manifest contains exactly `frac25-stage1-control`, `frac25-stage2`, and `frac25-stage3`; admission bundle binds the same run set. |
| Candidate admission | PASS | Candidate `admission validate --bundle` returned `authorized=true` before acceptance; `--require-accepted` correctly failed before rewrite with `admission_not_accepted`. |
| Accepted admission | PASS | After rewrite/rebuild, `admission validate --bundle --require-accepted` returned `authorized=true`. |

## Commands And Evidence

### Current Candidate And Input Hashes

```bash
git rev-parse HEAD
git -C recipe rev-parse HEAD
sha256sum docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json docs/joint_training/goals/stage123-execution-readiness/host_facts.json docs/joint_training/goals/stage123-execution-readiness/preflight_result.json
```

Relevant output:

```text
1f8ccf9f93902b30857cac063f4859be3a7b5e21
324a6aef2433f0163bf58e14be9d537fa7410388
bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee  implementation-tree.jsonl
86f778707462c19ca19b2f6ad7724626c9c049025fa5016ca197d0cc6a2ed37e  host_facts.json
fac6ae98010fa6fa3abd6a73dc30b9c8ee67006042ff82be351c8ae799e4072e  preflight_result.json
```

### Admission Validation Before Acceptance

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-root /data-1/code/verl
```

Output:

```json
{"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}
```

Expected pre-acceptance fail-closed check:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-root /data-1/code/verl --require-accepted
```

Output:

```text
{"authorized": false, "code": "admission_not_accepted", "context": {}, "message": "admission bundle lacks independent acceptance"}
exit=1
```

### Implementation Tree

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Relevant output:

```text
{"gitlink_commit":"324a6aef2433f0163bf58e14be9d537fa7410388","kind":"gitlink","mode":"160000","path":"recipe"}
{"implementation_tree_sha256": "bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee"}
exit=0
```

### Protected Baseline

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Output:

```text
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
exit=0
```

### Calibration And Focused Tests

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json
```

Output:

```text
{"ok": true, "sha256": "b557476277e61bbfa0e6deda2f8ee70edb41995439956ba22bfb06b5a888c37b"}
exit=0
```

Focused pytest command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_calibration_applicability.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py
```

Output:

```text
35 passed in 79.37s (0:01:19)
exit=0
```

### Accepted Report And Bundle Rebuild

I rewrote `docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json` with schema v1, reviewer model `GPT-5.5`, reasoning effort `medium`, Plan v9 hash, candidate/readiness commit `1f8ccf9f93902b30857cac063f4859be3a7b5e21`, exact run IDs, input hashes, protected baseline SHA, and AC-01 through AC-08 `PASS`.

Rebuild command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml --resource-profile recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh --calibration-result docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --preflight-result docs/joint_training/goals/stage123-execution-readiness/preflight_result.json --readiness-evidence-commit 1f8ccf9f93902b30857cac063f4859be3a7b5e21 --output docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --acceptance-report docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl
```

Output:

```text
{"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}
exit=0
```

### Accepted Bundle Validation

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-root /data-1/code/verl --require-accepted
```

Output:

```text
{"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}
exit=0
```

Rendered launch command only, not executed:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission render-launch --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-host /data-1/code/verl
```

Relevant output:

```text
tmux new-session -d -s stage123_primary_chain env REPO_HOST=/data-1/code/verl ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1 STAGE123_ADMISSION_BUNDLE=docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json STAGE123_IMPLEMENTATION_TREE_SHA256=bbb960c9867b9132fe943497dd8bf70ad2ab5557c34c338564eaad35835255ee STAGE123_BUNDLE_SHA256=2d0052fed26d58a5a071ce6c4683c852d0fda1c0195f81cf71109712bda40000 EXPERIMENT_BATCH_MANIFEST=/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json bash /data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
exit=0
```

## Written Artifacts

| Artifact | SHA256 |
| --- | --- |
| `docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json` | `b6ccaaaa99acaafcde44122c64a3aa1ae9a7fe6b0851b673c6490f9deae19aee` |
| `docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json` | `5e8fa729c946d8226cbc107025ca896cbbeb03da42e2c73961f75986fab4b0d4` |

## Blocking Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Review Weakness

This acceptance did not start tmux training or mutate runtime/finding ledgers. It validates the admission surface, not live execution progress after launch.
