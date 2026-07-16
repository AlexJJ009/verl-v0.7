# Independent Plan Amendment and Milestone 2 Review

- Reviewer: independent GPT-5.5 medium reviewer
- Candidate commit: `04863301cc480b5ce95b099a882245b9f1e27822`
- Plan version: `13`
- Verdict: `READY`

## Verdict

The treatment-only requalification clarification passes. It permits only the exact
ordered `stage2,stage3` zero-step validation for the repaired treatment wrappers;
it does not invoke, recreate, or modify the completed Stage1 control. The allowed
phase-set checks reject arbitrary subsets and ordering changes. The existing zero-step
contract still rejects nonzero training steps and requires zero optimizer steps and no
formal checkpoints.

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

## Reviewer-Owned Verification

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_calibration_milestone3.py \
  tests/experiment_workflow/test_manifest_queue_monitor_contract.py
goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

- Focused tests: `28 passed in 64.54s`.
- Plan validation: `PASS`.
- Runtime validation exited `0` and correctly reported `UNREVIEWED` before this review
  event was recorded.
- Additional behavioral checks accepted `stage2,stage3`; rejected `stage1`, every
  noncontiguous or reordered subset, extra phases, and `--training-steps 1`.

## Findings

- Blocking in-scope defects: none.
- Deferred suggestions: record this review and close `F-EX-M2-19` before the next
  runtime validation.
- Contract contradictions: none.
- Review limitation: the full zero-step probe was not run during review because it
  requires the separately authorized GPU requalification; static guards and focused
  behavioral tests were independently verified.
