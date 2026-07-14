# Milestone 3 Focused Rereview Report

## Review Identity

- Reviewer: independent GPT-5.5 medium same-reviewer focused rereview
- Review type: Milestone 3 rereview, focused on `F-M3-01` only
- Goal: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: 2
- Base commit: `066c34da0d41254db22e9db542b1b83c7a34f460`
- Candidate commit reviewed: `1c61e75f0348467c7d00eb48e7948d616a333796`
- Recipe gitlink reviewed: `d717672fb671edb86e504ba11e15b742686d7ef8`
- Applicable ACs: AC-01, AC-02, AC-08

## Overall Verdict

**PASS**

`F-M3-01` is closed. The committed fixture now binds recipe gitlink `d717672fb671edb86e504ba11e15b742686d7ef8`, all dependent fixture hashes recompute exactly, direct `batch-validate` passes, and the required focused test matrix passes. The candidate diff from the failed Milestone 3 review is limited to `findings.jsonl` plus the two fixture files; no core, monitor, Stage123 adapter, or recipe authority path changed.

## Finding Closure Status

| Finding | Status | Evidence |
| --- | --- | --- |
| `F-M3-01` | CLOSED | Fixture `recipe_gitlink`, bundle hash, admission file hash, and manifest hash all recompute and match; direct `batch-validate` returns `ok: true`; required pytest matrix passes. |

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Required matrix passes; no authority file changed between base and candidate (`authority_diff_rc=0`). |
| AC-02 | PASS | `batch-validate` returns `ok: true`; independent recomputation confirms recipe gitlink and all dependent hashes match. |
| AC-08 | PASS | Stage123 compatibility fixture now binds the reviewed recipe gitlink; required Stage123/admission suites pass. |

## Commands And Evidence

### Candidate State

```bash
git rev-parse HEAD
git cat-file -t 1c61e75f0348467c7d00eb48e7948d616a333796
git ls-tree 1c61e75f0348467c7d00eb48e7948d616a333796 recipe
git -C recipe rev-parse HEAD
```

Result:

```text
1c61e75f0348467c7d00eb48e7948d616a333796
commit
160000 commit d717672fb671edb86e504ba11e15b742686d7ef8 recipe
d717672fb671edb86e504ba11e15b742686d7ef8
```

### Goal Validation

```bash
goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
```

Result: `validate-plan` returned `PASS`; `validate-runtime` reported Plan status `READY`, current milestone `Milestone 3`, and `F-M3-01` as the open in-scope finding pending this rereview.

### Candidate Diff Scope

```bash
git diff --name-only 066c34da0d41254db22e9db542b1b83c7a34f460 1c61e75f0348467c7d00eb48e7948d616a333796 | sort
```

Result:

```text
docs/joint_training/goals/experiment-batch-orchestration/findings.jsonl
tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json
tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

```bash
git diff --quiet 066c34da0d41254db22e9db542b1b83c7a34f460 1c61e75f0348467c7d00eb48e7948d616a333796 -- \
  scripts/experiment_execution_core.py \
  scripts/stage123_manifest_monitor.py \
  recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh \
  recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh \
  recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh; printf 'authority_diff_rc=%s\n' "$?"
```

Result:

```text
authority_diff_rc=0
```

### Required Verification

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json && \
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_experiment_batch_core.py \
  tests/experiment_workflow/test_experiment_batch_monitor.py \
  tests/experiment_workflow/test_manifest_queue_monitor_contract.py \
  tests/experiment_workflow/test_stage123_core_migration.py \
  tests/experiment_workflow/test_stage123_admission_bundle.py \
  tests/experiment_workflow/test_notification_policy.py
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "748214d4d28aeac48dd4c6d18c9a62881c03affea75ae237c8946d5e5ceac1be", "items": ["stage123-primary"], "ok": true}
..................................                                       [100%]
34 passed in 15.97s
```

### Independent Hash Recompute

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -c '...recompute fixture recipe gitlink, bundle sha256, admission file sha256, manifest sha256...'
```

Result:

```text
recipe_head=d717672fb671edb86e504ba11e15b742686d7ef8
fixture_recipe_gitlink=d717672fb671edb86e504ba11e15b742686d7ef8
recipe_gitlink_match=True
bundle_sha256_expected=e9deda6fac421201963d1d4dc35d13e61925e9247d6b2ead3f0b5c6c42c9d313
bundle_sha256_actual=e9deda6fac421201963d1d4dc35d13e61925e9247d6b2ead3f0b5c6c42c9d313
bundle_sha256_match=True
admission_file_sha256_expected=c508ed16774b8ac5739f4ea4ded14d9599338ba1f55f657419cfb3163edb6294
admission_file_sha256_manifest=c508ed16774b8ac5739f4ea4ded14d9599338ba1f55f657419cfb3163edb6294
admission_file_sha256_match=True
manifest_sha256_expected=748214d4d28aeac48dd4c6d18c9a62881c03affea75ae237c8946d5e5ceac1be
manifest_sha256_actual=748214d4d28aeac48dd4c6d18c9a62881c03affea75ae237c8946d5e5ceac1be
manifest_sha256_match=True
```

### Patch Hygiene

```bash
git diff --check 066c34da0d41254db22e9db542b1b83c7a34f460 1c61e75f0348467c7d00eb48e7948d616a333796
```

Result: no output, exit code 0.

## Blocking In-Scope Defects

None. The only rereviewed blocker, `F-M3-01`, is closed.

## Deferred Suggestions

None for this focused rereview.

## Contract Contradictions

None found.

## Single Most Likely Weakness In This Review

This was intentionally scoped to `F-M3-01`; I did not reopen broader Milestone 3 design questions or rerun probes outside the required command and direct fixture/hash/authority-diff checks.
