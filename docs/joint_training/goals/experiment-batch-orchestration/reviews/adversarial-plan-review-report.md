# Adversarial Plan Review Report

- Review identity: independent adversarial Plan Reviewer
- Review type: `Adversarial Plan Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration`
- Reviewed Plan version: `1`
- Reviewed Plan SHA256: `baec753e892f85e186be4690c2d550252758e930fb3dca82590aec286a8ca785`
- Base commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Candidate commit: `2020531b470ec932d7b00afd13080e1318fc8429`
- Overall verdict: `NOT_READY`

## Summary

The Plan is directionally coherent, and the required CPU-only baseline passes, but the frozen contract still contains blocking contradictions around retry/resume policy and an under-specified migration boundary for existing Stage123 lifecycle authority. I do not accept the Plan as `READY`.

The strongest blocker is that the Plan excludes retry/resume and requires no retry/resume in batch routing, while also requiring preservation of the existing atomic API and Stage123 compatibility; the current accepted Stage123 path explicitly carries `--resume`, a recovery policy with `max_attempts: 2`, and resumable failure codes. The Plan never states whether batch mode disables this policy, replaces it, or preserves it outside batch. That makes AC-04, AC-08, and AC-09 mutually ambiguous.

## Per-AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | `WEAKENED` | Single-authority target is clear, but the existing Stage123 queue implementation still owns phase launching, checkpoint/metric interpretation, status TSV, deadline ownership, and phase-to-phase routing; the Plan does not freeze a measurable deletion/migration boundary for this authority. |
| AC-02 | `PASS` | Manifest immutability, binding, mutation rejection, and no-child validation are testable as written. |
| AC-03 | `PASS` | Ordered success routing and cleanup-before-next evidence are testable as written. |
| AC-04 | `FAIL` | The no-retry/no-resume requirement conflicts with the preserved atomic API/current Stage123 recovery policy unless the Plan specifies a batch-specific disabling or replacement rule. |
| AC-05 | `PASS` | Shared-stop behavior and two-equal-code anti-cascade routing are testable with CPU mocks. |
| AC-06 | `WEAKENED` | Pause/stop controls are required, but the Plan does not define the operator-control schema, freshness rule, authorization binding, or race boundary needed to test stale control rejection deterministically. |
| AC-07 | `WEAKENED` | One persisted-event monitor is a clear target, but the Plan does not state whether existing notification/release side effects remain only observational or must be moved behind the core event schema. |
| AC-08 | `FAIL` | Stage123 compatibility says no second state root or fallback policy, while current wrappers expose `--resume` and a separate recovery policy; the Plan does not resolve that conflict. |
| AC-09 | `FAIL` | The no-retry/no-resume/no-agent policy conflicts with the current resumable recovery path and is not scoped tightly enough to distinguish batch fallback from atomic recovery. |
| AC-10 | `PASS` | PM2/tmux/REPO_HOST/external-service/protected-asset checks are testable by focused tests and fingerprint comparison. |
| AC-11 | `WEAKENED` | Independent acceptance is structurally testable, but it depends on resolving AC-01/04/06/08/09 first. |

## Commands And Evidence

### Runtime/Plan Validation

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
```

Result:

```text
PASS
EXIT:0
```

Command:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-batch-orchestration; echo EXIT:$?
```

Result:

```json
{
  "current_milestone": null,
  "goal_status": "ACTIVE",
  "latest_review": null,
  "open_findings": {},
  "pending_user_decisions": [],
  "plan_status": "UNREVIEWED",
  "plan_version": 1
}
EXIT:0
```

Command:

```bash
sha256sum docs/joint_training/goals/experiment-batch-orchestration/plan.md; echo EXIT:$?
```

Result:

```text
baec753e892f85e186be4690c2d550252758e930fb3dca82590aec286a8ca785  docs/joint_training/goals/experiment-batch-orchestration/plan.md
EXIT:0
```

Command:

```bash
git diff --check -- docs/joint_training/goals/experiment-batch-orchestration/plan.md; echo EXIT:$?
```

Result:

```text
EXIT:0
```

### Frozen 23-Test Baseline

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_failure_classifier.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_pm2_ci_keepalive.py; echo EXIT:$?
```

Result:

```text
.......................                                                  [100%]
23 passed in 16.65s
EXIT:0
```

### Candidate Diff And Worktree Context

Command:

```bash
git diff --stat 2020531b470ec932d7b00afd13080e1318fc8429 2020531b470ec932d7b00afd13080e1318fc8429
```

Result: no diff, because base and candidate commits are identical.

Command:

```bash
git status --short
```

Relevant result:

```text
?? docs/joint_training/goals/experiment-batch-orchestration/
```

The reviewed Goal directory is untracked in the current worktree; I treated its own Plan/runtime files as the frozen review inputs because the reviewer prompt names this exact directory and Plan version.

### Existing Recovery/Retry Evidence

Evidence from `/data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh`:

- Line 16 invokes `scripts/experiment_execution_core.py` with `--recovery-policy "${REPO_ROOT}/config/experiment_execution/stage123_recovery_policy_v1.json"`.
- Line 17 appends `--resume` when the public entrypoint receives `--resume`.

Evidence from `/data-1/code/verl/config/experiment_execution/stage123_recovery_policy_v1.json`:

- Line 4 sets `"max_attempts": 2`.
- Lines 5-9 define resumable failure codes: `checkpoint_available_child_exit`, `container_runtime_interruption`, and `host_interruption`.

Evidence from `/data-1/code/verl/scripts/experiment_execution_core.py`:

- Lines 132-142 compute a resume decision from `spec.resumable_failure_codes` and `spec.max_attempts`.
- Lines 207-220 resume a terminal state by moving it back to `pending` and rerunning the spec when the recovery decision permits it.
- Lines 273-281 expose only `queue`/`phase` modes and an explicit `--resume` flag; no batch mode exists yet.

Conflicting Plan text:

- `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration/plan.md` lines 42-43 says the existing atomic API is preserved.
- Lines 51-52 require no-retry/no-parameter-change policy.
- Lines 61-62 exclude automatic retry and checkpoint resume.
- Lines 170-172 require next-item fallback without retry or resume.
- Lines 227-228 require no retry or resume in batch routing.
- Lines 307-308 classify adding retry/resume/tuning as `USER_DECISION`.

### Existing Nested Lifecycle Evidence

Evidence from `/data-1/code/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh`:

- Lines 56-57 write a queue status TSV independently of the execution-core event ledger.
- Lines 204-268 define `launch_and_wait`, which starts tmux sessions, inspects Docker/GPU ownership, watches validation deadlines, finds checkpoints, reads latest step, and checks metrics.
- Lines 270-278 call `stage123_require_formal_admission` and then launch Stage2 through the shell helper, outside the current Python core's item-level transition model.

Conflicting/weak Plan text:

- `/data-1/code/verl/docs/joint_training/goals/experiment-batch-orchestration/plan.md` lines 74-79 require the Python core to be the only transition authority and shell entrypoints to be adapters only.
- Lines 214-217 require existing Stage123 public queue and phase launchers to delegate to the unified core and not create a second state root or fallback policy.
- Lines 274-275 say to "remove/reduce duplicate lifecycle logic under an explicit deletion budget", but the Plan does not define the deletion budget or list the existing authority-bearing functions/files that must be removed, reduced, or retained.

## Blocking Findings

### F-PLAN-01 - CONTRACT_CONTRADICTION - Retry/resume policy conflicts with no-retry batch semantics

The Plan requires no retry/resume for batch-local failures and excludes automatic retry/checkpoint resume, but it also preserves the existing atomic execution API and Stage123 compatibility. The current Stage123 public entrypoint passes a recovery policy with `max_attempts: 2` and supports `--resume`; the current core implements terminal-state resume.

This is not a mere implementation gap. The Plan does not specify whether:

- batch mode must ignore `stage123_recovery_policy_v1.json`;
- Stage123 compatibility must remove or bypass `--resume`;
- atomic mode may keep retry/resume while batch items forbid it;
- host interruption and checkpoint-available exits are shared stops, local failures, or legacy-only recovery cases.

Affected ACs: AC-04, AC-08, AC-09. The Plan is not READY until the policy boundary is frozen.

### F-PLAN-02 - IN_SCOPE_DEFECT - Existing Stage123 shell lifecycle authority has no measurable deletion/migration boundary

The Plan correctly identifies single authority as the core requirement, but the current Stage123 implementation still has authority-bearing shell logic: status TSV writes, phase loop control, tmux launch, validation deadline ownership, checkpoint/metric interpretation, and phase-to-phase routing. The Plan only says "remove/reduce duplicate lifecycle logic under an explicit deletion budget" without freezing the deletion budget.

This leaves a path where implementation adds batch tests while retaining enough shell lifecycle behavior to remain a second authority. The acceptance evidence phrase "inventory showing no duplicate transition implementation" is not enough unless the Plan names the inventory target and the existing constructs that must be absent, thin, or read-only.

Affected ACs: AC-01, AC-07, AC-08.

### F-PLAN-03 - IN_SCOPE_DEFECT - Operator-control freshness and race semantics are under-specified

The Plan requires `pause_after_current`, `stop_now`, and `continue_remaining`, plus rejection of stale or unauthorized controls, but it does not define the control-file schema, freshness token/epoch, operator identity binding, compare-and-swap rule, or exact boundary for a control that arrives during cleanup or between item terminal event and next-item launch.

Because the Plan uses the phrase "documented boundary" without documenting the boundary, AC-06 can be satisfied by tests that encode an implementation's private interpretation rather than the frozen Plan contract.

Affected AC: AC-06.

### F-PLAN-04 - IN_SCOPE_DEFECT - Mutable command/path binding is underspecified for shell launch commands

The batch manifest requires "exact launch command JSON" and input hashes, and validation rejects current-checkout mismatch, but the Plan does not name the implementation-tree hash field, whether command path contents are hashed, or whether a shell command may point to a mutable worktree path. Existing Stage123 public launch constructs command JSON as `["bash", <queue_impl_path>]`, so binding only the argv string is insufficient unless the Plan freezes how script content and repo identity are hashed.

Affected ACs: AC-02, AC-08, AC-11.

## Deferred Suggestions

- `DEFERRED_SUGGESTION`: Consider renaming the legacy `queue` CLI mode once batch mode exists, because "queue" currently means atomic wrapping of the Stage123 queue script, while the new batch mode will own item routing. This is non-blocking if the Plan freezes semantics elsewhere.
- `DEFERRED_SUGGESTION`: Consider requiring a CPU fixture that simulates a malformed event line and confirms the batch stops as shared failure before processing further items. This is likely useful but already implied by AC-05.

## Contract Contradictions

- `F-PLAN-01` is a contract contradiction because the Plan's no-retry/no-resume acceptance criteria conflict with the current recovery policy that the Plan also says to preserve through existing atomic API and Stage123 compatibility.

## Single Likeliest Weakness In This Review

The single likeliest weakness is that I may be over-weighting current Stage123 recovery behavior during Plan review. A narrower intended interpretation could be "preserve atomic retry/resume outside batch, but batch mode forbids retry/resume." However, that interpretation is not written in the frozen Plan, and the Plan explicitly names Stage123 compatibility as part of this Goal, so the ambiguity is still blocking for an adversarial review.
