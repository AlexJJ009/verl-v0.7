# Milestone 2 Independent Review 01

## Review identity

- Reviewer: fresh independent reviewer, `gpt-5.6-terra` / medium fallback.
- Review type: Milestone Review, Milestone 2.
- Candidate repository: `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub`.
- Base / candidate: `cf9e69f2c87566b3b49ac8b8fbb5bf4c6ac2d2ee` /
  `d924fe728002d6b76b0001bed6562823d7e109bb`.
- Frozen Plan: version 1, SHA-256
  `5ffdc332b89f8de424bbff66a6bca9ffa396e8adc3506a7adf25bc9eb011789a`.
- Write boundary: this add-only report is the sole parent-worktree write. No
  implementation, ledger, Feishu, GitHub, credential, identity, or shared
  remote mutation was made. Local bare-remote pushes used only scratch repos.

## Overall verdict: FAIL

The core publish, stale-revision rejection, pull commit/push/no-op, conflict
preservation, structural fixture transitions, root gate, and a real local
bare-remote pre-push red/green probe work. Two frozen AC requirements remain
unmet: pull audit records omit remote editor IDs, and the Docx canonical/diff
contract does not implement or test tables, selected media, or Mermaid source.

| AC | Verdict | Reviewer-owned basis |
| --- | --- | --- |
| AC-02 | PASS | `71 passed`; `hubctl check --root .` returned `PASS`; the pre-existing every-detector green-before-red matrix ran in that suite. The gate is local-only and writer construction is gated. |
| AC-03 | PASS | In a fresh scratch clone, the first fake publish returned `changed:true`, the second `changed:false` with the same token/revision, and a stale remote revision exited 2 before `publish`. Read-back mismatch and no-duplicate behavior are independently covered by the executed suite. |
| AC-04 | FAIL | Attributable commit/push/no-op and unknown-editor neutrality work, but the audit record does not contain the required Feishu editor IDs. `reconcile_document` stores them only in mutable entry sync metadata, then writes an audit event lacking that field. |
| AC-05 | PASS | The executed FS-05 test proves remote-wins content, a reachable `refs/hub-conflicts/...` commit with recoverable local bytes, and publish blocking before adapter construction. |
| AC-06 | WEAKENED | Fixture tests prove rename/in-root metadata move/detach/tombstone and no fake publish. The tombstone is a copied entry directory plus a global audit line; no test proves a retained history/deletion-evidence bundle inside the tombstone as required by the Feature Story. |
| AC-07 | FAIL | The implementation canonicalizes only `document.content`; it has no representation for Docx tables, assets/media, or Mermaid source, and no golden/probe covers them. Formatting-only, malformed, unsupported-version, and basic B/L/R text diff portions pass. |
| AC-08 | PASS | A real local bare-remote green `git push` passed; a secret canary made ordinary push fail with the hook's exit 1; `--no-verify` bypassed only that Git hook, after which `hubctl publish` stopped at `E_GATE_CHECK` before adapter construction. |

## Commands and evidence

Required commands, run by this reviewer from `/data-1/code/verl`:

```text
REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m pytest -q tests
71 passed in 2.08s

REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m hubctl check --root .
PASS

git -C /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub status --short
<empty output>

git -C /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub remote get-url origin
/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub-origin.git

goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 2; plan_status=READY;
pending_user_decisions=[]
```

Additional fresh-clone adversarial probes used only
`/data-1/tmp/verl_agent_scratch/m2-review2-dGO0X4` and the candidate's local
bare origin:

```text
python -m hubctl publish FS03-DOC ... --fixture fs03_remote.json --adapter fake --json
{"changed": true, "object_token": "fixture-token-fs03", "revision": "fixture-9446a03379cb", ...}
second invocation
{"changed": false, "object_token": "fixture-token-fs03", "revision": "fixture-9446a03379cb", ...}

python -m hubctl publish FS03-DOC ... --fixture stale.json --adapter fake --json
E_ADAPTER_STALE_REVISION: remote revision changed before publication
exit=2

python -m hubctl pull FS04-DOC ... --commit --push --json
first: changed=true, pushed=true, revision=fixture-rev-fs04-2
second: changed=false and no commit
git show -s --format='Author=%an <%ae>; Committer=%cn <%ce>; %B'
Author=Fixture Human <fixture-human@example.invalid>
Committer=Fixture Sync Service <fixture-sync@example.invalid>
Hub-Entry-Id: FS04-DOC
Feishu-Revision: fixture-rev-fs04-2
Feishu-Editor-Ids: fixture-editor-1

git push origin HEAD:refs/heads/review-green
PASS
exit=0

# committed local secret canary, then ordinary push
git push origin HEAD:refs/heads/review-red
E_SECRET_DETECTED entries/FS02-DOC/content.md: credential-like value detected and redacted
exit=1

git push --no-verify origin HEAD:refs/heads/review-red
exit=0
python -m hubctl publish FS03-DOC ... --adapter fake --json
E_GATE_CHECK: E_SECRET_DETECTED ... credential-like value detected and redacted
exit=2
```

The parent worktree's protected dirty-path list was observed before this report
and remained unchanged by the review. Candidate `origin` is the local bare
scratch repository above; fixtures and identities use `example.invalid`; no
real token, user email, or external endpoint appeared in the candidate.

## Blocking in-scope defects

- **F-M2-R01 — pull audit lacks editor identity binding (AC-04).**
  [`operations.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/operations.py:138)
  receives a remote document, but `reconcile_document` records only hashes,
  revision, and adapter version in its result/audit event
  ([`sync.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/sync.py:98)).
  Editor IDs are written only to mutable `entry.yaml`
  ([`sync.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/sync.py:119)).
  AC-04 requires revision/editor audit fields, so the append-only audit cannot
  independently bind the pulled revision to its Feishu editor(s).

- **F-M2-R02 — Docx canonical diff omits required content surfaces (AC-07).**
  [`canonicalize.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/canonicalize.py:20)
  accepts solely a string at `document.content` and strips volatile IDs; it has
  no table, asset/media, or Mermaid extraction/reference model. The CLI merely
  diffs the resulting strings ([`__main__.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/__main__.py:109)).
  The only CLI-diff test asserts text and hashes
  ([`test_cli_diff.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/tests/diff/test_cli_diff.py:19)), not the frozen table/media/Mermaid requirements.

## Test-strength audit

The candidate adds tests rather than deleting or skipping existing ones; the
full suite ran with no skipped-test output. The reviewer also made the hook go
green and red through actual Git transactions, avoiding reliance on the static
hook-text test. The AC-06 test strength is incomplete: it proves copied content
but not explicit preserved history/deletion evidence inside its tombstone
([`test_fs06_remote_structure_changes.py`](/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/tests/feature_stories/test_fs06_remote_structure_changes.py:55)).

## Deferred suggestions

None. The two failures and AC-06 evidence gap are within the frozen Milestone
2 scope.

## Contract contradictions

None. Both defects are implementable and verifiable in the fixture-only local
repository; no decision-gated external/shared write is needed.

## Single most likely weakness in this review

The fake adapter is necessarily narrower than native Feishu Docx behavior.
That limitation does not weaken the verdict: the frozen Plan expressly requires
captured fixture coverage for the missing table/media/Mermaid surfaces, which
the candidate does not provide.
