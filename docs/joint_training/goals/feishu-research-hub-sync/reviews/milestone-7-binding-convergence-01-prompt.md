# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Convergence Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `593b4bac5ab46bde0b732e2b705c7aa0a79d2846`
- Candidate commit: `66d6f5aaf804d17c0f37f312f28277dea88a2c98`
- Applicable ACs: `AC-03, AC-04, AC-10, AC-12`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && .venv/bin/python -m pytest -q tests/gate/test_live_binding_privacy.py tests/feature_stories/test_fs03_local_publish.py tests/feature_stories/test_fs04_remote_edit_pull.py tests/git/test_attribution.py tests/runtime/test_sync_once.py && .venv/bin/python -m pytest -q && .venv/bin/python -m hubctl check --root . && git diff --check 593b4bac5ab46bde0b732e2b705c7aa0a79d2846 66d6f5aaf804d17c0f37f312f28277dea88a2c98; goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync

## Additional Task-Specific Focus

Convergence review before a third ordinary repair round for F-M7-R01.

The two related review rounds are final-acceptance-01 and
milestone-7-binding-fix-review-01. Determine whether the recurrence is caused
by duplicated sources of truth, experiment-specific policy in a shared layer,
reviewer scope expansion, or a second independently useful outcome. Confirm
whether `.hub/live-bindings.json` is the single runtime-private authority for
live root/object bindings and whether the remaining correction is an IN_SCOPE
architectural fix under AC-10 and AC-12 without a Plan amendment.

Review the current candidate only; do not implement, push, call Feishu, or edit
the append-only ledgers. Require an explicit convergence disposition for each
of these cases in live mode:

- missing binding store;
- empty `entries` mapping;
- malformed binding document or entry shape;
- missing or empty object token;
- fixture-shaped token in the live store;
- binding for an unknown entry ID;
- a configured managed entry with no matching private binding.

The expected safety property is fail closed with stable nonzero binding errors,
no adapter call, no Git commit/push, no cursor advance, and a truthful ERROR
status from `scripts/sync_once.sh`. Do not require object/folder tokens or
private URLs in tracked evidence. Preserve AC-04: revision/editor identifiers
remain allowed and required in structured Git trailers/private audit, while
their concrete values remain redacted from tracked review and acceptance
reports.


## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.
9. For a Plan review, reject `READY` when any AC declares an absolute numeric performance or resource budget that has no recorded feasibility probe in the Plan's `Feasibility Probes` section, or whose budget contradicts the probe's measured floor.
10. When your only blocking findings are purely mechanical (formatting, patch context offsets, artifact or directory placement) with no behavioral or contract impact, say so explicitly and offer a light same-reviewer re-verification scoped to those findings instead of demanding a fresh full round.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
