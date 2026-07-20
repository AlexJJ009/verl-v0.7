# Milestone 2 V18 Fresh Admission Acceptance

## Review Identity

- Reviewer: independent GPT-5.5 medium fresh admission acceptance reviewer
- Review type: Milestone Review and Admission Acceptance
- Goal: `stage123-primary-chain-execution`
- Frozen Plan version: 18
- Candidate commit: `6a069213f3467529530217fa14a473d0671859f6`
- Applicable ACs: execution `AC-01`, `AC-02`, `AC-07`, `AC-08`, `AC-12`; readiness `AC-01` through `AC-08`
- Review ID: `milestone-2-v18-fresh-admission-acceptance-01`

## Overall Verdict

`PASS`

The fresh V18/V16 admission is accepted. The accepted bundle binds current HEAD, implementation tree, recipe gitlink `aa972ba489f75b9faebf42ae91307a542749faa3`, protected baseline, derived calibration applicability, fresh preflight, exact three-run manifest, and fresh V16 output identities. I did not launch training.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| Readiness AC-01 | PASS | Current implementation tree and evidence commit match the admission bindings. |
| Readiness AC-02 | PASS | Protected baseline comparison passed with SHA `c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207`. |
| Readiness AC-03 | PASS | Derived calibration result validates as passed and binds V18 applicability. |
| Readiness AC-04 | PASS | Fresh preflight is passed and bound to current manifest/profile/tree. |
| Readiness AC-05 | PASS | Admission bundle validates with exact run IDs `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`. |
| Readiness AC-06 | PASS | Render-launch succeeds only after accepted report is embedded; command was rendered, not executed. |
| Readiness AC-07 | PASS | `nvidia-smi --query-compute-apps` returned no rows; no GPU process observed. |
| Readiness AC-08 | PASS | Accepted report and bundle bind all current inputs and reviewer identity. |
| Execution AC-01 | PASS | `--require-accepted` admission validation returns authorized for current checkout. |
| Execution AC-02 | PASS | Bundle run set is exactly the frozen three-run matrix. |
| Execution AC-07 | PASS | Admission remains fail-closed before acceptance; unsigned render-launch returned `admission_not_accepted`. |
| Execution AC-08 | PASS | No training/GPU process observed; no launch command executed. |
| Execution AC-12 | PASS | Manifest uses fresh V16 roots/names and recipe gitlink `aa972ba`; applicability is resource-neutral and empty-diff. |

## Commands And Evidence

- `goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution` -> `PASS`, exit `0`.
- `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution` -> exit `0`; Plan v18 `READY`.
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl` -> exit `0`, `implementation_tree_sha256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b`.
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl` -> `ok=true`, SHA `c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207`, exit `0`.
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl` -> `authorized=true`, current checkout matches admission bundle, exit `0`.
- `jq '{decision,capacity_differences,source_capacity_sha256,candidate_capacity_sha256,implementation_tree_sha256,evidence_commit,plan_sha256}' docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json` -> `decision=applicable`, `capacity_differences=[]`, source/candidate capacity SHA `7cab911b63caba6c001e29e1b0a7cb7d7bacf04b0e23a642e7638ed0cd91e2f5`.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` -> no rows, exit `0`.
- Unsigned render-launch probe before acceptance -> `admission_not_accepted`, exit `1`.
- Rebuilt accepted bundle with `scripts/execution_results.py admission validate` using `--acceptance-report` -> `authorized=true`, exit `0`.
- Accepted-bundle validation: `scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --require-accepted ...` -> `authorized=true`, exit `0`.
- Render-launch after acceptance printed this command and did not execute it:

```bash
tmux new-session -d -s stage123_primary_chain env REPO_HOST=/data-1/code/verl ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1 STAGE123_ADMISSION_BUNDLE=/workspace/verl/docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json STAGE123_IMPLEMENTATION_TREE_SHA256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b STAGE123_BUNDLE_SHA256=ddcedb365e17c95ee86913ae4aa9e8a17935ae215f84815c47cecdb813904ede EXPERIMENT_BATCH_MANIFEST=/data-1/code/verl/docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json bash /data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
```

## Accepted Artifacts

- `docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json`
  - internal `acceptance_report_sha256`: `5da278f52004441333c0f21327a6b954a0ae67ea546d69343385bdb6fd0c39c6`
  - file SHA bound in bundle: `ad129d19f23d7688cca12c9bb79bf66ff3917e772191e79efcaf895905417128`
- `docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json`
  - `bundle_sha256`: `ddcedb365e17c95ee86913ae4aa9e8a17935ae215f84815c47cecdb813904ede`
  - `acceptance.decision`: `accepted`

## Files Changed By This Review

- `docs/joint_training/goals/stage123-execution-readiness/acceptance_report.json`
- `docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json`
- `docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-v18-fresh-admission-acceptance.md`

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

The review depends on the current uncommitted fresh `host_facts.json`, `preflight_result.json`, and derived calibration artifacts already present at HEAD; I verified their hashes and bundle bindings rather than regenerating them, because the prompt limited reviewer-owned writes to the acceptance/report artifacts.
