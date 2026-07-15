# Stage123 Execution Readiness Acceptance

- Status: `PASS`
- Decision: `ACCEPTED`
- Plan version: `8`
- Plan SHA256: `fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Readiness evidence commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Reviewer: `independent final-acceptance Reviewer`
- Reviewer model: `GPT-5.5`
- Reasoning effort: `medium`

## Per-AC Verdicts

| AC | Verdict |
| --- | --- |
| AC-01 | PASS |
| AC-02 | PASS |
| AC-03 | PASS |
| AC-04 | PASS |
| AC-05 | PASS |
| AC-06 | PASS |
| AC-07 | PASS |
| AC-08 | PASS |
| AC-09 | PASS |

## Required Verification

- `goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness` -> `PASS`
- `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness` -> `PASS`, `goal_status=ACTIVE`, `current_milestone=Milestone 6`, `pending_user_decisions=[]`
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow tests/joint_training/regression/test_validation_generation_logging.py` -> `194 passed, 8 warnings in 468.28s`
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json` -> `ok=true`, `sha256=80d09f5aad98d3d834162663cd6532a8da5f27a9b903421b364d14d4f18e340d`
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl` -> `authorized=true`, `current checkout matches admission bundle`
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl` -> `ok=true`, `sha256=c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207`
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json` -> `implementation_tree_sha256=91b8e419933b13d521faaf6eedf6f04cccdfdeccfad4d808759896f61c0fbf7e`
- `git status --short` -> expected pre-existing protected/untracked assets plus Goal runtime/findings modifications; no protected-baseline mismatch
- `tmux list-sessions` -> no `stage123_primary_chain` or Stage123 training session
- `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` -> empty output

## Accepted Artifacts

- Candidate bundle SHA256: `97e8745829795232078533df19e069769c15972adcd43dacdab65cc44911233e`
- Accepted bundle file SHA256: `5308d3ff43b3e6cbf5c1470770a70377ade45bba4389a0350815f1e52b0cabf7`
- Acceptance report canonical SHA256: `cdc0766a541150d42f89e4c63a0f8b6e7c3931195d17802f8576d29747343148`
- Acceptance report file SHA256: `81d5d12756d3134a70c5dd154ae17e0ad0d6876847845a49ce7ba4856436febc`

## Rendered Launch Command

Rendered for reproducibility only; not executed:

```bash
tmux new-session -d -s stage123_primary_chain env REPO_HOST=/data-1/code/verl ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1 STAGE123_ADMISSION_BUNDLE=docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json STAGE123_IMPLEMENTATION_TREE_SHA256=91b8e419933b13d521faaf6eedf6f04cccdfdeccfad4d808759896f61c0fbf7e STAGE123_BUNDLE_SHA256=97e8745829795232078533df19e069769c15972adcd43dacdab65cc44911233e bash /data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh
```
