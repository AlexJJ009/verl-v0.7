# Milestone 7 stale-binding fix review 02

## Review identity

- Independent same reviewer: Codex (`/root/stale_binding_review01`)
- Review type: focused mechanical re-verification of `F-M7-R02`
- Frozen Plan: v2
- Base: `e354f950338bed8750e77653395efec4dc473907`
- Candidate: `c024c980e3a4773de7bd1361853a70eea8074b3b`
- Scope: only the missing-common-edition blocking finding from review 01

## Overall verdict: PASS

The candidate removes the unsafe local-content fallback. A missing common
snapshot now fails `E_BINDING_COMMON_MISSING` after the read-only fetch and
before any binding/audit mutation. The positive control creates the recorded
common edition; the divergent-remote control remains fail-closed. This closes
the review-01 in-scope defect without changing the frozen contract.

## Applicable AC verdicts

| AC | Verdict | Reviewer evidence |
| --- | --- | --- |
| AC-03 | PASS | Refresh requires a runtime-private binding and canonical remote equality with `.hub/common/<entry>.md`; missing common and semantic divergence leave the private binding bytes unchanged. |
| AC-10 | PASS | The repaired command still enters through the normal writer/root gate, verifies the adapter version, fetches only, and writes no Feishu object. |
| AC-12 | PASS | The response/audit remain redacted to private-binding marker, content state, boolean, and adapter version; token/revision/editor stay outside tracked output. |

## Commands and evidence

Executed in `/data-1/code/verl/research/feishu-research-hub`:

```text
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q tests/gate/test_live_binding_privacy.py
8 passed in 0.73s

PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q
147 passed in 14.97s

PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root .
PASS

git diff --check e354f95..HEAD
# exited 0

git show -s --format=fuller HEAD
Author:     Codex Agent <codex-agent@example.invalid>
Commit:     Codex Agent <codex-agent@example.invalid>
...
    Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>
```

The shell emitted unrelated login-profile warnings about a read-only `fnm`
multishell path; all required commands completed successfully.

Code and test inspection:

- `src/hubctl/operations.py:157-160` now requires
  `.hub/common/<entry>.md` and raises `E_BINDING_COMMON_MISSING` if absent;
  no fallback to `content.md` remains.
- `src/hubctl/operations.py:161-176` compares canonical remote content to the
  recorded common edition before storing revision/editor metadata.
- `tests/gate/test_live_binding_privacy.py:137-168` explicitly creates the
  common snapshot for the successful refresh case.
- `tests/gate/test_live_binding_privacy.py:171-191` retains the divergent
  remote `E_BINDING_DIVERGED` / byte-for-byte unchanged state control.
- `tests/gate/test_live_binding_privacy.py:194-210` supplies the missing-common
  `E_BINDING_COMMON_MISSING` / byte-for-byte unchanged state control.

## Blocking in-scope defects

None.

## Deferred suggestions

None.

## Contract contradictions

None.

## Single most likely weakness in this review

The re-verification used the deterministic fake adapter and did not call
Feishu, as required. It therefore does not independently exercise a live
lark-cli fetch, but the repaired branch and the required no-live test suite
directly cover the original stale-binding defect.
