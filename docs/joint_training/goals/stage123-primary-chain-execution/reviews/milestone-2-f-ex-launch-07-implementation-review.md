# Independent Mechanical Rereview — F-EX-LAUNCH-07

## Review Identity

- Reviewer: same independent GPT-5.5 medium implementation Reviewer.
- Review lane: mechanical rereview only for the prior blocking fixture binding defect.
- Frozen Plan version: 18.
- Base commit for original implementation review: `425f844734607b6e02bcd83a1de702d6e3239a30`.
- Prior failed candidate: `45802ffa7492e09c95d7d7fa7fbdc3cc35b79383`.
- Mechanical correction candidate: `1f8ccf9f93902b30857cac063f4859be3a7b5e21`.
- Reviewed prompt: `docs/joint_training/goals/stage123-primary-chain-execution/reviews/milestone-2-f-ex-launch-07-implementation-review-prompt.md`.

## Overall Verdict

**PASS.**

The prior blocking defect is resolved. Candidate `1f8ccf9f93902b30857cac063f4859be3a7b5e21` changes only the committed batch fixtures relative to `45802ffa`: `tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json` now binds `recipe_gitlink=324a6aef2433f0163bf58e14be9d537fa7410388`, updates the evidence commit to `45802ffa7492e09c95d7d7fa7fbdc3cc35b79383`, and refreshes the admission and batch manifest hashes. The exact required verification command now exits `0`.

This rereview did not edit implementation code, runtime ledgers, or findings.

## Per-AC Verdict Table

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | The stale fixture binding that made `batch-validate` fail is repaired; the required focused pytest suite now passes all 53 tests, including `test_committed_batch_fixture_validates_without_starting_child`. |
| AC-07 | PASS | Same as prior review: no retry/resume, parameter, data, evaluator, or protected-asset change was introduced by the mechanical correction; required suite and runtime validation pass. |
| AC-08 | PASS | Same as prior review: batch/monitor contract tests pass; `goal-plan-runtime validate-runtime` exits `0`. |
| AC-12 | PASS | The fixture now binds the candidate recipe gitlink `324a6aef2433f0163bf58e14be9d537fa7410388` and refreshed hashes; required suite passes. |

## Mechanical Diff Evidence

Command run:

```bash
git diff --stat 45802ffa7492e09c95d7d7fa7fbdc3cc35b79383..1f8ccf9f93902b30857cac063f4859be3a7b5e21
git diff --name-status 45802ffa7492e09c95d7d7fa7fbdc3cc35b79383..1f8ccf9f93902b30857cac063f4859be3a7b5e21
git diff --unified=80 45802ffa7492e09c95d7d7fa7fbdc3cc35b79383..1f8ccf9f93902b30857cac063f4859be3a7b5e21 -- tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Relevant output:

```text
.../experiment_workflow/fixtures/experiment_batch_admission_v1.json | 6 +++---
tests/experiment_workflow/fixtures/experiment_batch_v1.json         | 4 ++--
2 files changed, 5 insertions(+), 5 deletions(-)

M tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json
M tests/experiment_workflow/fixtures/experiment_batch_v1.json
```

Relevant binding changes:

```text
tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json:
  evidence_commit: 425f844734607b6e02bcd83a1de702d6e3239a30 -> 45802ffa7492e09c95d7d7fa7fbdc3cc35b79383
  recipe_gitlink: aa972ba489f75b9faebf42ae91307a542749faa3 -> 324a6aef2433f0163bf58e14be9d537fa7410388
  bundle_sha256: b050a65cf1f79179c09dd35da780a44bd59342d6795c91e2a91edd42e4ba884c -> e181b409794417deef256bd03dcfd52d780eaa59cf762c2918fc7ee0d72b9daa

tests/experiment_workflow/fixtures/experiment_batch_v1.json:
  batch_manifest_sha256: fec3c5064c1ffbc193d726a6dcca4f8b84a36e785948018cc42d8a36396fdcb1 -> c19f38bb51e5d4ca82d3bdc778d0d00600e290c0e754f8c24689aa77b7b878e1
  admission_bundle_sha256: 6d880d61054d9e0fdaf0bf9a853fc4fe9289bd5c885eb94458ad93c6f7e3be45 -> 7f9b4f2b5aa73d06f4c99a07b5cc5b2aa19c2e4a550d9746205c6e1f2bf27e71
```

## Required Verification

Command run:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_experiment_batch_core.py tests/experiment_workflow/test_experiment_batch_routing.py tests/experiment_workflow/test_experiment_batch_policy.py tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_control_reuse.py tests/experiment_workflow/test_stage123_calibration_applicability.py; bash -n recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh; goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Relevant output:

```text
.....................................................                    [100%]
53 passed in 19.22s
...
COMMAND_STATUS pytest=0 bash_n=0 validate_runtime=0
```

Runtime validator output summary:

```text
plan_status: READY
plan_version: 18
goal_status: ACTIVE
current_milestone: Milestone 2
pending_user_decisions: []
```

## Blocking In-Scope Defects

None. The previous blocker `F-EX-LAUNCH-07-REVIEW-01` is mechanically resolved by candidate `1f8ccf9f93902b30857cac063f4859be3a7b5e21`.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

This was intentionally scoped to the prior fixture-binding blocker and the exact required verification command. I did not rerun unrelated GPU/training launch evidence and did not modify runtime/finding ledgers.
