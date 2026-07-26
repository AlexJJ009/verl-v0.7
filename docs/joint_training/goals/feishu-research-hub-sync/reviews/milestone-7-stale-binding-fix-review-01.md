# Milestone 7 stale-binding fix review 01

## Review identity

- Independent reviewer: Codex (`/root/stale_binding_review01`)
- Review type: Milestone Review
- Frozen Plan: v2
- Base: `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`
- Candidate: `e354f950338bed8750e77653395efec4dc473907`
- Scope: only `F-M7-R02` guarded `binding-refresh` repair

## Overall verdict: FAIL

The candidate supplies the private-binding gate, adapter-version check, redacted
result, and a no-write fetch path.  However, it permits a refresh when the
recorded common edition is absent: it falls back to current local content.
That is not the required `remote == recorded common edition` precondition and
can refresh stale CAS metadata after an unrecorded local edit.  It therefore
does not close the stale-binding readiness defect without weakening the
common-edition/CAS guard.

## Applicable AC verdicts

| AC | Verdict | Reviewer evidence |
| --- | --- | --- |
| AC-03 | FAIL | `refresh_entry_binding` uses the local file when `.hub/common/<entry>.md` is absent, so it can pass remote content that equals a changed local edition rather than the recorded common edition. |
| AC-10 | PASS | The command is a one-fetch, protected local runtime-metadata operation; it invokes the deterministic writer/root gate and performs no adapter publish/write operation. |
| AC-12 | FAIL | Final acceptance requires privacy-redacted token/revision/editor bindings and correct revision-checked behavior. The privacy behavior is covered, but the absent-common path weakens the required CAS condition. |

## Commands and evidence

Executed in `/data-1/code/verl/research/feishu-research-hub`:

```text
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q tests/gate/test_live_binding_privacy.py
7 passed in 0.33s

PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q
146 passed in 14.88s

PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root .
PASS

git diff --check 7078fdd..e354f95
# exited 0

git show -s --format=fuller e354f95
Author:     Codex Agent <codex-agent@example.invalid>
Commit:     Codex Agent <codex-agent@example.invalid>
...
    Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>
```

The shell emitted unrelated login-profile warnings about a read-only `fnm`
multishell path; all required commands completed successfully.

Candidate inspection:

- `src/hubctl/operations.py:147-154` calls `gated_adapter`, validates adapter
  version, and requires a runtime-private binding.
- `src/hubctl/operations.py:156-176` only fetches before storing refreshed
  metadata; it does not call `publish`.
- `src/hubctl/operations.py:177-183` returns/audits only a private-binding
  marker, content-state, boolean, and adapter version; it omits token,
  revision, and editor values.
- `src/hubctl/operations.py:157-160` selects `content.md` when the common
  snapshot is missing. This is the blocking defect.
- `tests/gate/test_live_binding_privacy.py:137-165` calls the positive case on
  `valid_hub`. That fixture copies only `config`, `entries`, and `generated`,
  so it has no `.hub/common/FS03-DOC.md`. The supposed positive control thus
  proves the disallowed local-file fallback rather than the required recorded-
  common path.
- The negative test at `tests/gate/test_live_binding_privacy.py:168-185`
  correctly proves `E_BINDING_DIVERGED` and unchanged private state for a
  semantically changed remote, but it does not cover absent-common state or a
  changed local file with remote equal to that local content.

## Blocking in-scope defect

- `IN_SCOPE_DEFECT F-M7-R02`: make a missing recorded common snapshot fail
  closed (with a stable error), and add a negative control showing that this
  leaves runtime-private binding bytes untouched. Add a separate positive
  control that creates the recorded common snapshot and verifies a stale
  revision/editor refresh only when canonical remote content equals it. This
  is an AC-03 revision/CAS behavior fix, not a new acceptance requirement.

## Deferred suggestions

None.

## Contract contradictions

None. The frozen Plan and the repair prompt are consistent; the candidate's
fallback is the defect.

## Single most likely weakness in this review

I used the deterministic fake adapter and did not call Feishu, as required.
Consequently this review cannot independently observe a live lark-cli fetch;
the blocking result rests on the locally inspectable branch and its fixture
coverage, which are sufficient for the frozen no-live-review scope.
