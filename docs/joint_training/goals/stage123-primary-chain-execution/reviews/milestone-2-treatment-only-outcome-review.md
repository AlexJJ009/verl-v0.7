# Independent Milestone 2 Treatment-Only Outcome Review

- Reviewer: independent GPT-5.5 medium reviewer
- Candidate commit: `3b8a62a245ffad53f3b7d9e8ea5227eb097c79f6`
- Plan version: `13`
- Verdict: `PASS`

## Scope and Verdict

This review covers `F-EX-M2-20` and `F-EX-M2-21`. The treatment-only calibration
outcome path now accepts exactly ordered `stage2,stage3` evidence while retaining
fail-closed behavior for every other partial or reordered phase set. The corrected
test binds its assertion to the manifest phase it mutates.

The preserved six-repetition zero-step evidence may be CPU-reclassified and
re-rendered without another GPU probe: all three Stage2 and all three Stage3
repetitions passed, recorded zero optimizer/training steps, no formal checkpoints,
and released owned resources. The original failed pointer/report must remain
preserved because it records the previous classification defect.

| AC | Verdict |
| --- | --- |
| AC-01 | PASS |
| AC-04 | PASS |
| AC-06 | PASS |
| AC-07 | PASS |
| AC-08 | PASS |

## Reviewer-Owned Verification

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_calibration_milestone3.py \
  tests/experiment_workflow/test_operational_calibration_checker.py
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

- Focused tests: `30 passed in 239.26s`.
- Runtime validation: exit `0`, Plan v13 `READY`, Milestone 2 active.
- CPU negative probes accepted only the exact full and treatment-only phase sets and
  rejected partial, extra, and reordered sets.

## Findings

- Blocking in-scope defects: none.
- Contract contradictions: none.
- Review limitation: no GPU probe was rerun; the conclusion is based on preserved
  scratch evidence plus reviewer-owned CPU validation.
