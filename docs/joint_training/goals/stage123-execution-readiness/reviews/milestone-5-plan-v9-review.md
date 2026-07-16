# Milestone 5 Plan v9 Independent Review

## Review Identity

- Reviewer: Codex independent Milestone 5 reviewer
- Requested model/effort: GPT-5.5 medium
- Review type: Milestone Review
- Goal: `stage123-execution-readiness`
- Frozen Plan version: `9`
- Plan SHA256: `29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c`
- Candidate commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Base commit from prompt: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Applicable ACs: `AC-02`, `AC-03`, `AC-04`, `AC-05`, `AC-06`, `AC-07`, `AC-08`

## Overall Verdict

PASS.

Milestone 5 Plan v9 passes on reviewer-owned evidence. The current candidate admission bundle is correctly pre-acceptance: `admission validate` authorizes the current checkout, while `admission render-launch` fails closed with `admission_not_accepted` until independent acceptance is added in Milestone 6. I found no blocking in-scope defects, no contract contradictions, no protected-asset drift, and no evidence of formal training, Ray execution, registry mutation, W&B sync, or external publication.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-02 | PASS | Fresh `host_facts.json` and `preflight_result.json` are bound by SHA256; all 19 structured preflight checks are `ok=true`, including scorer dependencies, mount, GPU, Docker image, release paths, and no conflicting run. |
| AC-03 | PASS | Bundle and preflight both enumerate exactly `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`; focused manifest/queue/monitor tests passed. |
| AC-04 | PASS | Shared batch/core tests passed; collected tests include Python-owned lifecycle, restart, cleanup, corrupt-state, and failure-routing checks. |
| AC-05 | PASS | Monitor/queue tests passed; collected tests include no hard-coded run arrays, manifest-owned queue reads, no shell lifecycle authority, and event-policy monitor behavior. |
| AC-06 | PASS | `admission validate --bundle` returned authorized; admission/test suites passed 16 tests including mutation/fail-closed bundle behavior. |
| AC-07 | PASS | No active Stage123 tmux session, no GPU compute process, protected baseline compare passed, and readiness/new-experiment gate tests passed. |
| AC-08 | PASS | Current candidate is reproducible enough to validate against HEAD and protected baseline; pre-acceptance `render-launch` fails closed with `admission_not_accepted`, which is the correct Milestone 5 state before Milestone 6 acceptance. |

## Commands And Evidence

### Frozen Required Commands

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl --repo-root /data-1/code/verl
```

Output:

```text
{"authorized": true, "code": "authorized", "context": {}, "message": "current checkout matches admission bundle"}
```

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_end_to_end.py
```

Output:

```text
16 passed in 16.27s
```

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_monitor.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_new_experiment_gate.py
```

Output:

```text
33 passed in 16.18s
```

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Output:

```text
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

### Additional Reviewer Checks

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission render-launch --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-host /data-1/code/verl
```

Output:

```text
{"authorized": false, "code": "admission_not_accepted", "context": {}, "message": "admission bundle lacks independent acceptance"}
```

This is the expected Milestone 5 fail-closed behavior: the candidate bundle validates, but launch rendering is blocked until Milestone 6 independent acceptance.

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness
sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md
```

Output:

```text
PASS
plan_status=READY current_milestone=Milestone 5 pending_user_decisions=[]
29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c  docs/joint_training/goals/stage123-execution-readiness/plan.md
```

```bash
git status --short --branch
git diff --name-status 9c736bc0 9c736bc0
git show --name-status --oneline --no-renames 9c736bc0
```

Output summary:

```text
## codex/experiment-execution-reliability
 M docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json
 M docs/joint_training/goals/stage123-execution-readiness/host_facts.json
 M docs/joint_training/goals/stage123-execution-readiness/preflight_result.json
?? .claude/skills/experiment-registry
?? docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-5-plan-v9-review-prompt.md
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/
```

`git diff --name-status 9c736bc0 9c736bc0` is empty because the frozen prompt names the same commit as base and candidate. Commit `9c736bc0` itself modifies only Goal review/runtime files for Milestone 4 acceptance:

```text
M docs/joint_training/goals/stage123-execution-readiness/findings.jsonl
A docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-4-plan-v9-review-prompt.md
A docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-4-plan-v9-review.md
M docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl
```

The untracked protected paths match the Plan's protected baseline and are not failures. The only modified Goal evidence files are the current host facts, preflight result, and candidate bundle expected by the prompt.

## Evidence Bindings

- `host_facts.json` SHA256: `4356f6fa26e3f3fb26a8b4778e873bfba137c5cdf77e92fb4667d0a845726df5`
- `preflight_result.json` SHA256: `92da8dcda08eacd064eca83534c7f4f5bf3b3a02cef2471883234375adec383e`
- `calibration_result.json` SHA256: `24647c4c5031ab199e40a2338b18dec02788c355600bed96192bb22faf43f880`
- `admission_bundle.json` file SHA256: `21294ba9823f957de90bd65d35a6d925f37554d5c8ca8364353c97ec67ee1c98`
- Candidate canonical `bundle_sha256`: `5578540d602ae8ba01e4c79ee7b2c6ac1cdaae87b5b41f29620cce18b8f21b44`
- Protected baseline SHA256: `c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207`
- HEAD: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Bundle `readiness_evidence_commit`: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`
- Bundle `implementation_tree_sha256`: `0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc`
- Bundle `manifest_sha256`: `323bcc6084c1b01653bdf3cb5b299cab51c76864c8ca24878a9e5b002cb76278`
- Bundle `resource_profile_sha256`: `d9b6a36dd9fcc4307f7b502e5511989e60c1a257f57c9ac70574acaf12eee2b5`

Hash equality checks passed:

```text
preflight host_facts_sha256 matches True
bundle preflight sha matches True
bundle calibration sha matches True
bundle protected sha matches True
```

## Fresh Host Facts And Preflight

`host_facts.json`:

- `artifact_type`: `stage123_host_facts`
- `schema_version`: `1`
- `repo_host`: `/data-1/code/verl`
- `generated_at`: `2026-07-15T06:25:47Z`
- `completed_at`: `2026-07-15T06:25:47Z`
- `producer.command`: `scripts/stage123_host_facts.sh`
- `producer.host_owned`: `true`
- Docker reference: `verl-harness:latest`
- Docker immutable ID: `sha256:c9d525a1f4b33267bd00be60fe00693338253537cac78151e4c55a6d3a7e5708`
- tmux sessions observed by host facts: `docker-export`, `harness`
- `stage123_conflicts`: `[]`
- checkpoint mount: `/data-2/checkpoints`
- storage evidence: `/data-1` 82% used with `372429164` 1K blocks available; `/data-2` 28% used with `1448490964` 1K blocks available

`preflight_result.json`:

- `schema_version`: `1`
- `result_type`: `preflight_result`
- `gate`: `stage123_preflight`
- `decision`: `passed`
- `ok`: `true`
- `started_at`: `2026-07-15T06:25:55Z`
- `completed_at`: `2026-07-15T06:26:06Z`
- `host_facts_sha256`: `4356f6fa26e3f3fb26a8b4778e873bfba137c5cdf77e92fb4667d0a845726df5`
- `calibration_evidence_commit`: `8a5402e5a31c7810c7c0c77b4d8dcd1aa6129f56`
- `implementation_tree_sha256`: `0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc`
- `run_ids`: `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`

All 19 preflight checks passed:

```text
host_facts=True
repo_topology=True
compat_symlink=True
checkpoint_mount=True
scorer_dependencies=True
resource_profile_command=True
run_identity=True
model_identity=True
dataset_shards=True
matched_control_union=True
matched_control_source=True
gpu_inventory=True
host_ram=True
docker_image=True
release_paths=True
no_conflicting_run=True
implementation_tree=True
resource_profile_binding=True
calibration_binding=True
```

Scorer dependency evidence includes imports for `evalplus.evaluate`, `evalplus.gen.util`, `evalplus.eval._special_oracle`, `lcb_runner.benchmarks.code_generation`, and `lcb_runner.evaluation.compute_code_generation_metrics`; `lcb_index` is `/data-2/evaluator_assets/livecodebench_cache/index/release_v5_input_output.sqlite`. GPU inventory records eight `NVIDIA L40S, 46068` entries.

## Mutation And Fail-Closed Audit

- The required admission validation returned `authorized=true` for the exact candidate bundle and current checkout.
- Focused suites passed `16 + 33 = 49` tests.
- `pytest --collect-only -q` collected 49 focused tests; representative mutation/fail-closed tests include:
  - `test_mutated_protected_binding_is_rejected_without_touching_asset`
  - `test_tampered_inventory_fails_closed`
  - `test_corrupt_state_and_event_ledgers_fail_closed`
  - admission bundle mutation tests in `test_stage123_admission_bundle.py`
  - lifecycle/failure routing tests in `test_experiment_batch_routing.py`
- Skip/xfail audit over the focused test files found no `pytest.mark.skip`, `pytest.mark.skipif`, `pytest.mark.xfail`, `unittest.skip`, bare `pass`, or trivial bare `return` matches.
- `admission render-launch` before acceptance returned `authorized=false` with `code=admission_not_accepted`, so launch rendering remains fail-closed until Milestone 6.

## No-Training / No-Publication Audit

Read-only runtime audit:

```bash
tmux ls
ps -eo pid,ppid,stat,comm,args | rg -i 'ray|stage123|wandb|experiment_execution_core|run_qwen3_1p7b_stage123' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
```

Observed:

- tmux sessions: `docker-export`, `harness`
- no Stage123 training tmux session
- no Ray, Stage123, W&B, `experiment_execution_core`, or Stage123 wrapper process after excluding the audit command itself
- no NVIDIA compute processes
- preflight release-path checks reference local release assets only; no registry mutation, W&B sync, Hugging Face upload, GitHub publication, or WxPusher call was observed

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

I did not regenerate `host_facts.json` or `preflight_result.json`, because the user explicitly limited reviewer writes to this review file and the frozen prompt asked me to review the current fresh evidence files. I instead verified freshness and bindings from the existing artifacts, checked their SHA256 links through admission validation, and ran the required fail-closed validators independently.
