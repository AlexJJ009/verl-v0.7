# Milestone 3 Independent Review Report

## Review Identity

- Reviewer: independent GPT-5.5 medium milestone reviewer
- Review type: Milestone 3 review
- Goal: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Frozen Plan version: 2
- Main candidate reviewed: `066c34da0d41254db22e9db542b1b83c7a34f460`
- Recipe gitlink reviewed: `d717672fb671edb86e504ba11e15b742686d7ef8`
- Base commit: `ce8d3fc35ddf131cc985f6fd47744771d6a536de`
- Applicable ACs: AC-01, AC-03, AC-04, AC-07, AC-08

## Overall Verdict

**NOT_READY**

The required Milestone 3 verification matrix fails on the committed batch fixture because the admission bundle still binds recipe gitlink `a40ed88cc6e0c3983cc4627c90865ace0f22425c`, while the reviewed candidate and submodule gitlink are `d717672fb671edb86e504ba11e15b742686d7ef8`. The core correctly rejects this as `recipe gitlink mismatch`, so the committed candidate cannot satisfy the required verification or the Stage123 manifest/admission compatibility evidence for this milestone.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | FAIL | Required matrix includes `test_experiment_batch_core.py`; it fails in committed fixture validation. Core authority appears centralized in inspected code, but required core evidence does not pass. |
| AC-03 | PASS | Focused scratch probe `test_successful_items_advance_once_in_manifest_order` and `test_core_owns_ordered_phases_inside_an_item` passed; no duplicate shell phase loop found in Stage123 adapter. |
| AC-04 | PASS | Focused scratch probe `test_local_failure_is_inconclusive_and_falls_forward_without_retry` passed; queue adapter rejects `--resume` and recovery override. |
| AC-07 | PASS | Monitor consumes persisted events with cursor idempotence in tests; source inspection found no tmux/checkpoint/W&B/registry authority in `scripts/stage123_manifest_monitor.py`. |
| AC-08 | FAIL | Required Stage123/admission matrix fails because committed fixture/admission binding is stale against the reviewed recipe gitlink; the adapter itself is thin, but committed compatibility evidence is invalid. |

## Commands And Evidence

### Candidate and Runtime State

```bash
git rev-parse HEAD
git cat-file -t 066c34da0d41254db22e9db542b1b83c7a34f460
git -C recipe rev-parse HEAD
git ls-tree 066c34da0d41254db22e9db542b1b83c7a34f460 recipe
```

Result:

```text
066c34da0d41254db22e9db542b1b83c7a34f460
commit
d717672fb671edb86e504ba11e15b742686d7ef8
160000 commit d717672fb671edb86e504ba11e15b742686d7ef8 recipe
```

Note: the worktree also had unrelated dirty/untracked files outside this review. I did not modify them.

```bash
goal-plan-runtime validate-plan /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration
```

Result: `validate-plan` returned `PASS`; `validate-runtime` reported Plan status `READY`, current milestone `Milestone 3`, and prior findings closed.

### Required Verification Command

```bash
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
.........F........................                                       [100%]
FAILED tests/experiment_workflow/test_experiment_batch_core.py::test_committed_batch_fixture_validates_without_starting_child
E assert 2 == 0
E where 2 = CompletedProcess(... stdout='{"failure": {"code": "invalid_batch_request", "context": {}, "message": "recipe gitlink mismatch"}, "ok": false}\n', stderr='').returncode
1 failed, 33 passed in 15.96s
```

Direct reproduction:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Result:

```json
{"failure": {"code": "invalid_batch_request", "context": {}, "message": "recipe gitlink mismatch"}, "ok": false}
```

Relevant committed evidence:

- `tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json:12` binds `recipe_gitlink` to `a40ed88cc6e0c3983cc4627c90865ace0f22425c`.
- `scripts/experiment_execution_core.py:313` through `scripts/experiment_execution_core.py:320` recomputes `git -C recipe rev-parse HEAD` and rejects a mismatch.
- Candidate `066c34da0d41254db22e9db542b1b83c7a34f460` records recipe gitlink `d717672fb671edb86e504ba11e15b742686d7ef8`.

### Focused Adversarial Probes

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_batch_core.py -k 'local_failure or core_owns_ordered_phases or batch_cli_rejects_resume'
```

Result:

```text
...                                                                      [100%]
3 passed, 7 deselected in 0.12s
```

```bash
bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh --resume; printf 'resume_rc=%s\n' "$?"
```

Result:

```text
ERROR: Stage123 batch execution forbids retry/resume and recovery-policy overrides
resume_rc=2
```

```bash
bash recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh; printf 'missing_rc=%s\n' "$?"
```

Result:

```text
ERROR: EXPERIMENT_BATCH_MANIFEST or --batch-manifest is required
missing_rc=2
```

```bash
python3 - <<'PY'
from pathlib import Path
checks = {
 'queue_impl': Path('recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh'),
 'monitor': Path('scripts/stage123_manifest_monitor.py'),
 'legacy_monitor': Path('scripts/training_queue_monitor.sh'),
}
for name,path in checks.items():
    text=path.read_text().lower()
    print(f'[{name}] {path}')
    for token in ['status.tsv','queue_status','launch_and_wait','validation_deadline','docker inspect','tmux','latest_checkpoint','wandb','registry','terminate(']:
        if token in text:
            print('  contains', token)
    print('  batch-run count', text.count('batch-run'), 'persisted_events count', text.count('persisted_events'))
PY
```

Result:

```text
[queue_impl] recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh
  batch-run count 1 persisted_events count 0
[monitor] scripts/stage123_manifest_monitor.py
  batch-run count 0 persisted_events count 2
[legacy_monitor] scripts/training_queue_monitor.sh
  contains tmux
  contains latest_checkpoint
  contains wandb
  contains registry
  batch-run count 0 persisted_events count 0
```

Interpretation: the Stage123 batch adapter is reduced to a core delegate and does not contain the probed legacy lifecycle authority. The legacy monitor still contains observational tmux/checkpoint/W&B/registry logic, but it is not referenced as a batch transition authority in the inspected Milestone 3 path.

```bash
git diff --check ce8d3fc35ddf131cc985f6fd47744771d6a536de 066c34da0d41254db22e9db542b1b83c7a34f460
```

Result: no whitespace errors.

## Blocking In-Scope Defects

### F-M3-01 — Committed batch admission fixture binds stale recipe gitlink

- Classification: `IN_SCOPE_DEFECT`
- Affected ACs: AC-01, AC-08
- Evidence: required verification fails because `tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json:12` binds `recipe_gitlink` to `a40ed88cc6e0c3983cc4627c90865ace0f22425c`, but the reviewed candidate gitlink is `d717672fb671edb86e504ba11e15b742686d7ef8`.
- Behavioral impact: `batch-validate` rejects the committed fixture with `invalid_batch_request: recipe gitlink mismatch`, so the milestone’s committed CPU verification and Stage123 admission compatibility evidence are not valid.
- Why in-scope: this is a committed Milestone 3 fixture/admission binding defect under the frozen manifest/admission compatibility contract; it does not require changing the Plan.

## Deferred Suggestions

- The monitor cursor currently records an event digest after `subprocess.run(..., check=False)` regardless of whether the notification policy command succeeds (`scripts/stage123_manifest_monitor.py:100` through `scripts/stage123_manifest_monitor.py:127`). I did not block on this because AC-07 focuses on read-only persisted-event consumption and idempotence, but a later reliability pass should consider not advancing the cursor on policy emission failure.

## Contract Contradictions

None found. The blocker is an in-scope committed artifact/fixture defect, not a frozen Plan contradiction.

## Single Likeliest Weakness In This Review

The required test matrix failed early, so my behavioral probes emphasized the failure boundary, Stage123 adapter authority removal, no-retry rejection, monitor read-only/idempotent evidence, and selected core routing tests rather than re-running a complete scratch-rebuilt equivalent of the stale committed fixture. This is sufficient for `NOT_READY`, but a same-reviewer rereview should re-run the full required matrix after the fixture/admission binding is corrected.
