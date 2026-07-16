# F-EX-M2-24 Convergence Review

- Review type: `Convergence Review`
- Reviewer: independent GPT-5.5 medium reviewer
- Plan version: `13`
- Base commit: `3b8a62a245ffad53f3b7d9e8ea5227eb097c79f6`
- Candidate commit: `c436e5a56047b135bb03530706acc773bf1821a5`
- Recipe gitlink: `241e141efd84183ebf6cde9960baa05ecbf4d6fd`
- Verdict: `PASS`

## Scope Decision

Removing host-facts wall-clock age checks is an `IN_SCOPE` architectural repair.
Plan v13 states that qualification is valid by training-plane identity rather than a
wall-clock TTL, and that elapsed time after `item_started` is not a phase failure.
The repair does not change an acceptance criterion, scientific parameter, manifest,
or retry/resume policy.

## Independent Evidence

The reviewer inspected the committed candidate and recipe gitlink, then ran:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_stage123_control_reuse.py \
  tests/experiment_workflow/test_stage123_preflight_model_identity.py \
  tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py
```

Result: `11 passed in 65.76s`.

The review verified that treatment authorization and direct Stage3 invocation retain
certificate, manifest, run-id, host-facts hash, host-facts `ok`, GPU inventory, and
resource-profile checks. No phase TTL remains in the treatment path, and no
retry/resume or scientific-parameter change was introduced.

## AC Verdicts

| AC | Verdict |
| --- | --- |
| AC-01 | PASS |
| AC-04 | PASS |
| AC-06 | PASS |
| AC-07 | PASS |
| AC-08 | PASS |

No blocking in-scope defect, deferred suggestion, or contract contradiction remains.
