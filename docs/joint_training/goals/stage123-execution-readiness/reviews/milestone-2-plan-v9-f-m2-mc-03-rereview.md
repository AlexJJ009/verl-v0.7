# Milestone 2 Plan v9 F-M2-MC-03 Focused Re-review

## Review Identity

- Reviewer: independent focused Milestone 2 re-reviewer, Codex session, requested model `GPT-5.5`, reasoning effort `medium`
- Repo: `/data-1/code/verl`
- Branch: `codex/experiment-execution-reliability`
- Focused base commit: `0540935b`
- Focused candidate commit: `1d28852921b1c8caf963b70ccb0f71cab6a75c01`
- Frozen Plan: `docs/joint_training/goals/stage123-execution-readiness/plan.md`, Plan version `9`
- Prompt: `docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-f-m2-mc-03-rereview-prompt.md`
- Report path: `docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-f-m2-mc-03-rereview.md`

## Overall Verdict

**PASS**

`F-M2-MC-03` is closed. Candidate `1d288529` removes the Plan v8 hard-code from `scripts/render_calibration_result.py`, derives `plan_version` and `plan_sha256` from the matching `USER_DECISION_RECORDED` event, validates the recorded hash against the current Goal `plan.md`, and fails closed for missing, malformed, or stale authorization identity. Milestone 2 is now **PASS** for this focused re-review scope.

## Per-AC Verdict Table

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Focused tests pass, the reviewer-owned runtime probe returns Plan version `9` and Plan hash `29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c`, and static inspection confirms renderer output binds those values. |

## Finding Verdicts

| Finding | Verdict | Rationale |
| --- | --- | --- |
| F-M2-MC-03 | CLOSED / PASS | `matching_authorization()` now selects the matching decision, validates `plan_version`, validates a 64-hex-like hash length, rejects current-plan hash mismatch, and render output includes `plan_version`/`plan_sha256` from that decision. |
| F-M2-MC-01 | PRESERVED / PASS | Focused diff touches only `scripts/render_calibration_result.py` and `tests/experiment_workflow/test_execution_results.py`; prior three-run fixture/test repair is unchanged. |
| F-M2-MC-02 | PRESERVED / PASS | Focused diff does not touch `scripts/stage123_phase_adapter.py`, batch command mapping, manifest, or recipe gitlink; prior phase-adapter PASS conclusion is unchanged. |

## Commands And Evidence

### Focused Command 1

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_calibration_outcomes.py
```

Result:

```text
..................                                                       [100%]
18 passed in 0.09s
```

### Focused Command 2

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -c 'from pathlib import Path; import importlib.util; p=Path("scripts/render_calibration_result.py"); s=importlib.util.spec_from_file_location("renderer",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); d=m.matching_authorization(Path("docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl"),"RD-MATCHED-CONTROL-REQUAL-01"); assert d["plan_version"] == 9; print(d["plan_sha256"])'
```

Result:

```text
29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c
```

### Focused Command 3

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json
```

Result:

```json
{"kind":"git_tree","path":"config/experiment_execution","tree_sha1":"4377c79f59537caf09f47bbc9ef6464e2b078d1f"}
{"gitlink_commit":"6fcccb353a87045a17f9d52b3821f0e20f7f9a9d","kind":"gitlink","mode":"160000","path":"recipe"}
{"kind":"git_tree","path":"scripts","tree_sha1":"a14fa2208cacd87294c2ffe2088752b510164b27"}
{"kind":"git_tree","path":"verl","tree_sha1":"40deac7dc6da65ef470c5e42c75fb2fd35b9335a"}
{"implementation_tree_sha256": "903f637ac931a3969f1e3beff09864728887ab66bbd17a4180867ddb0faab0fd"}
```

## Static Inspection Evidence

- Focused diff `0540935b..1d288529` touches exactly two files: `scripts/render_calibration_result.py` and `tests/experiment_workflow/test_execution_results.py`.
- `scripts/render_calibration_result.py:41` adds `matching_authorization(ledger_path, decision_id)`.
- `scripts/render_calibration_result.py:49` fails closed when no matching `USER_DECISION_RECORDED` event exists.
- `scripts/render_calibration_result.py:54` fails closed when `plan_version` is absent, non-integer, or invalid.
- `scripts/render_calibration_result.py:56` fails closed when `plan_sha256` is absent or malformed length.
- `scripts/render_calibration_result.py:58` resolves the Goal-local `plan.md`; `scripts/render_calibration_result.py:59` rejects hash mismatch against the current Plan.
- `scripts/render_calibration_result.py:123` calls `matching_authorization()` during render, and `scripts/render_calibration_result.py:156` emits `authorization_identity.plan_version` and `authorization_identity.plan_sha256` from the matching decision.
- `tests/experiment_workflow/test_execution_results.py:206` adds a regression test that accepts Plan v9 and rejects a mutated current Plan hash.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Test-Weakening / Trivialization Audit

- No test was skipped, xfailed, deleted, or loosened in the focused diff.
- The focused diff adds one new regression test and does not modify the earlier matched-control phase-adapter or fixture tests.
- The new test includes a positive Plan v9 assertion and a negative stale-plan-hash assertion, matching the F-M2-MC-03 defect.

## Single Most Likely Weakness In This Review

This was a focused CPU re-review of `F-M2-MC-03` only. I did not rerun the full Milestone 2 suite from the prior review or execute GPU calibration/preflight because the focused prompt froze three specific commands and limited the scope to AC-01 authorization binding.
