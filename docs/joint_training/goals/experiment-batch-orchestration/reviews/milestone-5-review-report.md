# Milestone 5 Independent Implementation Review Report

## Review Identity

- Reviewer: independent GPT-5.5 medium implementation reviewer
- Review type: Milestone 5 implementation review
- Goal: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: 2
- Base commit: `1c1e56f3adde1922175f7343c5f593038ffb1dff`
- Candidate commit reviewed: `74f73f987a65e21b9d43f7d87e06b0e8c2dc03bc`
- Recipe gitlink reviewed: `d717672fb671edb86e504ba11e15b742686d7ef8`
- Applicable ACs: AC-01 through AC-10

## Overall Verdict

**PASS**

The committed candidate passes all exact AC-01 through AC-10 verification commands from Plan v2. Reviewer-owned mutation probes also confirmed event-corruption fail-closed behavior with preserved evidence, repeated normalized failure stopping after restart, and stale control rejection without launching work. Source and inventory inspection found the Python core remains the sole transition authority, Stage123 remains a thin adapter, the monitor remains read-only/idempotent, and batch mode rejects retry/resume paths.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Exact core/Stage123 migration command passed; deletion budget and inventory still name `scripts/experiment_execution_core.py` as sole transition authority. |
| AC-02 | PASS | Exact `batch-validate` command returned `ok: true`; fixture hashes and recipe gitlink recompute. |
| AC-03 | PASS | Exact success-routing command passed; routing tests verify ordered advancement. |
| AC-04 | PASS | Exact local-failure command passed; reviewer probe confirmed no retry in restart/fallback path. |
| AC-05 | PASS | Exact shared-failure command passed; reviewer probes confirmed malformed event fail-closed and repeated normalized failure shared-stop across restart. |
| AC-06 | PASS | Exact control command passed; stale-control probe stayed paused and launched nothing. |
| AC-07 | PASS | Exact monitor command passed; monitor consumes persisted events and uses batch-first discrimination without runtime inference authority. |
| AC-08 | PASS | Exact Stage123 compatibility command passed; adapter delegates to `batch-run` and rejects `--resume` / `--recovery-policy`. |
| AC-09 | PASS | Exact policy/failure-classifier command passed; source scan found no agent/timer/tuning authority in batch paths. |
| AC-10 | PASS | Exact PM2/policy plus protected-asset compare command passed; protected compare returned `ok: true`. |

## Commands And Evidence

### Candidate and Runtime State

```bash
git rev-parse HEAD
git cat-file -t 74f73f987a65e21b9d43f7d87e06b0e8c2dc03bc
git ls-tree 74f73f987a65e21b9d43f7d87e06b0e8c2dc03bc recipe
git -C recipe rev-parse HEAD
```

Result:

```text
74f73f987a65e21b9d43f7d87e06b0e8c2dc03bc
commit
160000 commit d717672fb671edb86e504ba11e15b742686d7ef8 recipe
d717672fb671edb86e504ba11e15b742686d7ef8
```

```bash
goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
```

Result: `validate-plan` returned `PASS`; `validate-runtime` reported Plan status `READY`, current milestone `Milestone 5`, and all prior findings closed.

Candidate diff from base:

```text
1 file changed, 1 insertion(+)
M docs/joint_training/goals/experiment-batch-orchestration/runtime.jsonl
```

This candidate is an implementation-review checkpoint over the previously accepted implementation tree plus the Milestone 5 runtime start event.

### Exact AC-01 Through AC-10 Commands

AC-01:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_experiment_batch_core.py
```

Result: `17 passed in 0.49s`.

AC-02:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Result:

```text
{"batch_id": "experiment-batch-v1-fixture", "batch_manifest_sha256": "7d03b45a276e4acb81d9546a28eaefdeafd0ee7f9fa0d2a7e1a88d595c5ac0c7", "items": ["stage123-primary"], "ok": true}
```

AC-03:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k success
```

Result: `1 passed, 3 deselected in 0.03s`.

AC-04:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k local_failure
```

Result: `1 passed, 3 deselected in 0.03s`.

AC-05:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_routing.py -k shared_failure
```

Result: `2 passed, 2 deselected in 0.04s`.

AC-06:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_control.py
```

Result: `2 passed in 0.04s`.

AC-07:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_experiment_batch_monitor.py
```

Result: `9 passed in 15.62s`.

AC-08:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_admission_bundle.py
```

Result: `11 passed in 0.24s`.

AC-09:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_failure_classifier.py
```

Result: `7 passed in 0.04s`.

AC-10:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_pm2_ci_keepalive.py tests/experiment_workflow/test_experiment_batch_policy.py && \
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Result:

```text
10 passed in 0.75s
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

Patch hygiene:

```bash
git diff --check 1c1e56f3adde1922175f7343c5f593038ffb1dff 74f73f987a65e21b9d43f7d87e06b0e8c2dc03bc
```

Result: no output, exit code 0.

### Reviewer-Owned Mutation Probes

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python /data-1/tmp/m5_review_probe.py
```

Result:

```json
{"event_count": 2, "failure": {"code": "event_corruption", "context": {}, "message": "atomic event schema mismatch at line 1"}, "first_event_preserved": true, "probe": "malformed_atomic_event", "started": [], "status": "shared_failure"}
{"failure": {"code": "shared_failure", "context": {}, "message": "batch stopped after child_exit"}, "first_status": "paused_after_current", "item_ids": ["one", "two"], "probe": "repeated_failure_restart", "second_status": "shared_failure", "started_after_restart": [["fixture", "two", "1"]]}
{"probe": "stale_continue", "rejection": {"code": "control_rejected", "message": "stale batch revision"}, "started": [], "status": "paused_after_current"}
```

Interpretation:

- Malformed atomic event: shared failure, no child start, original evidence preserved.
- Repeated normalized failure after pause/restart: second failed item stops the batch before third item starts.
- Stale continue control: rejected without mutation and no child start.

### Binding and Authority Audits

Fixture recomputation:

```text
impl_tree_match=True
bundle_hash_match=True
admission_file_hash_match=True
manifest_hash_match=True
recipe_gitlink=d717672fb671edb86e504ba11e15b742686d7ef8
```

Source evidence:

- `scripts/experiment_execution_core.py:692` through `scripts/experiment_execution_core.py:718` validates the shared event ledger, including batch-first discrimination and strict atomic event schema.
- `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh:14` through `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh:16` rejects `--resume` and `--recovery-policy`.
- `recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh:27` delegates to `scripts/experiment_execution_core.py batch-run`.
- `scripts/stage123_manifest_monitor.py:58` through `scripts/stage123_manifest_monitor.py:65` validates persisted events as a read-only consumer with batch-first discrimination.

Deletion budget and authority inventory:

- `deletion-budget.json` identifies `scripts/experiment_execution_core.py` as sole authority and the Stage123 queue script as `reduce_to_adapter`.
- `authority-inventory.json` records removed Stage123 authority paths: status TSV writes, phase loop, tmux launch, validation deadline, checkpoint interpretation, phase routing, and failure fallback.

Source grep notes:

- Stage123 adapter contains no `tmux`, `latest_checkpoint`, `wandb`, `registry`, `codex`, `crontab`, `systemd`, `auto_tune`, or `parameter_mutation` tokens; it contains only rejection handling for `--resume` / `--recovery-policy` and one `batch-run` delegation.
- `scripts/training_queue_monitor.sh` still contains legacy observational tokens (`tmux`, `latest_checkpoint`, `wandb`, `registry`), consistent with the deletion budget retaining it as read-only observer and not a transition authority.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None found.

## Single Most Likely Weakness In This Review

The review is CPU-only and does not launch a real Stage123 training batch, as required by the Goal. It therefore validates orchestration semantics, bindings, and no-authority boundaries through focused tests and scratch probes rather than through live training integration.
