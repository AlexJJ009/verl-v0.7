# Milestone 7 Conflict Retention Convergence Review 01

## Review identity and scope

Independent convergence reviewer. Frozen Plan v2; candidate Hub commit
`88ac17c2181d22f1e33373c54763ea8e6351bf74`; scope limited to convergence of
`F-M7-R04` and AC-05/09/10/12. No Feishu call, push, parent change, acceptance
change, or ledger mutation was performed.

## Overall verdict

**PASS — IN_SCOPE_ARCHITECTURAL_FIX.**

The two-parent retention merge makes the former conflict snapshot a parent of
the normal `main` history. The retained named ref targets that same snapshot.
This fixes the prior non-reproducibility without adding another outcome or a
second content source of truth: normal `main` history preserves the snapshot;
the named ref is an explicit recovery locator.

| Criterion | Verdict | Evidence |
| --- | --- | --- |
| AC-05 | PASS | Active FS03 state is clean and remote-active; the recorded common hash matches active content; the pre-edit snapshot is reachable from fresh private `main`, distinct from active content, nonempty, and lacks the human marker; the named ref resolves to the recorded snapshot commit. |
| AC-09 | PASS | Full candidate suite and root gate pass; pristine local-CI state is recorded PASS at the candidate. |
| AC-10 | PASS | Runtime clone is clean; latest scheduled sync is terminal PASS with `changed=false`; cursor matches candidate. |
| AC-12 | PASS for this convergence scope | Candidate state, retained snapshot, deterministic gate, private-origin evidence, local-CI, and no-op runtime cursor agree. Final end-to-end acceptance remains the separately required final-acceptance step. |

## Reviewer-owned commands and evidence

```bash
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q
PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root .
git show -s --format='parents=%P%nsubject=%s' 88ac17c
git cat-file -e <recorded-conflict-commit>^{commit}
git show <recorded-conflict-commit>:entries/FS03-DOC/content.md
git ls-remote --exit-code <private-origin> <recorded-conflict-ref>
git clone <private-origin> <temporary-clone>
git -C <temporary-clone> fetch origin +<recorded-conflict-ref>:<recorded-conflict-ref>
PYTHONPATH=<temporary-clone>/src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q tests/gate/test_tracked_common_snapshot.py
```

Results:

- `88ac17c` has exactly two parents: the prior candidate and the retained
  conflict snapshot commit.
- Full suite: `154 passed`; `hubctl check --root .`: `PASS`.
- Candidate local ref target, private-origin advertised ref target, and explicit
  fetched-ref target all match the recorded conflict commit. A fresh private
  `main` clone contains that commit before fetching the named ref.
- The recovered pre-edit bytes are nonempty, distinct from active content, and
  do not contain `FINAL-AC04`; their deterministic content hash was recomputed
  locally without publishing or emitting the value.
- The reachability test passed on the candidate. In a disposable temporary clone
  I replaced only the recorded commit value with a nonexistent object; the same
  test exited nonzero. This proves the new object-reachability assertion is not
  merely syntactic.
- Runtime clone HEAD and cursor match candidate; worktree is clean; latest sync
  event is PASS and no-op.
- Privacy scan found zero tracked candidate Docx-URL, OAuth, or credential
  matches outside excluded test/lock data. No test deletion or skip weakening
  was observed in the retention change.

## Runtime validation note

`goal-plan-runtime validate-runtime` currently returns the expected pending
error that `F-M7-R04` requires this convergence review before another fix
round. This reviewer cannot append the corresponding convergence event. The
coordinator must record this report's convergence result, then rerun the
validator; the pre-recording error is not a behavioral failure of the candidate.

## Blocking in-scope defects

None.

## Deferred suggestions

None.

## Contract contradictions

None.

## Most likely review weakness

Private-origin verification was read-only and used a temporary clone; no live
Feishu operation was repeated, as required by this focused convergence scope.
