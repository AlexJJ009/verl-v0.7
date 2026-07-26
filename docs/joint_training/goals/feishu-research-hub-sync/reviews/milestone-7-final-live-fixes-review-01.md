# Milestone 7 Final Live Fixes Review 01

## Review identity and scope

Independent focused reviewer, reviewing frozen Plan v2 at Hub candidate
`79bf9a0e7f7f14ec0069f3ef9f06a08872c28fd1`. Scope is limited to
`F-M7-R03`, `F-M7-R04`, and AC-04/05/09/10/12. No Feishu request, Git push,
ledger update, acceptance update, or parent-repository update was performed.

## Verdict

`F-M7-R03: PASS`.

`F-M7-R04: FAIL`.

The candidate records a resolved-conflict ref and commit for `FS03-DOC`, but
neither object is present in the candidate checkout. The same ref/object is
present in the dedicated runtime clone, which is not a reproducible substitute
for the committed Hub candidate. Thus the frozen AC-05 requirement for a
reachable conflict ref cannot be proven from the proposed accepted Hub revision.

| Acceptance criterion | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-04 | PASS | Candidate contains the human-edit commit; Author matches `GongxunLi <lgxma01@buaa.edu.cn>`, Committer is the service, and structured Feishu revision/editor trailers are present (values redacted). |
| AC-05 | FAIL | Active FS03 content is clean, includes `FINAL-AC04`, and matches its common hash, but its recorded resolved conflict ref and commit are absent from the candidate object database. |
| AC-09 | PASS | Full suite and focused local-CI tests pass; dedicated runtime clone is clean and its cursor is bound to the candidate. |
| AC-10 | PASS | Focused runtime tests pass; latest scheduled event is terminal `PASS` with `changed=false`, and cursor SHA matches candidate. |
| AC-12 | FAIL | End-to-end evidence is not bound to fully reproducible committed Hub state because AC-05 fails. |

## Commands and evidence

Executed from the Hub candidate checkout:

```bash
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root .
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q -k 'auth and (verified or envelope or identity)'
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q tests/feature_stories/test_fs04_remote_edit_pull.py tests/feature_stories/test_fs05_concurrent_remote_wins.py tests/git/test_attribution.py tests/local_ci/test_runner_e2e.py tests/local_ci/test_verdict_query.py tests/runtime/test_sync_once.py tests/runtime/test_locking.py tests/runtime/test_status.py
```

Results: `154 passed`; root `hubctl check: PASS`; auth-envelope focus `6
passed`; AC-focused suite `40 passed`.

The candidate and `origin/main` both resolve to `79bf9a0`. The dedicated runtime
clone is clean and resolves to the same SHA. Its latest sync event is `PASS`
with `changed=false`; its cursor matches the candidate SHA. The tracked active
FS03 state is `clean`; its canonical content includes the requested marker and
its recorded common hash matches that content. The live-pull commit has the
required human Author, service Committer, and both required structured trailer
classes, verified without emitting private revision/editor values.

For the contract defect, I independently checked the tracked
`resolved_conflict_ref` and `resolved_conflict_commit` using `git show-ref
--verify` and `git cat-file -e`. Both are missing in the candidate checkout;
both are present only in the dedicated runtime clone. The new test checks that
the YAML fields have a syntactically valid shape, but does not verify the ref or
object exists in the candidate repository.

No deleted files were found between the M7 base and candidate; no skip/xfail
markers were found in the Hub tests; a tracked-content privacy scan found no
candidate Docx URL, OAuth, or credential match outside excluded test/lock data.

## Blocking in-scope defect

`F-M7-R04`: preserve the resolved conflict snapshot/ref in the committed Hub
repository (or otherwise make it reachable from the candidate) and make the
gate prove reachability, not merely YAML syntax. This is an in-scope AC-05
reproducibility defect.

## Deferred suggestions

None.

## Contract contradictions

None.

## Most likely review weakness

This is a local, read-only review of the committed candidate and protected
runtime state; it did not repeat a live Feishu call, by scope instruction.
