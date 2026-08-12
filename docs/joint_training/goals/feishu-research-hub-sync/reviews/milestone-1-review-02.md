# Milestone 1 Independent Review 02

## Review identity

- Reviewer: fresh independent reviewer, `gpt-5.6-terra` / medium; requested
  project reviewer models were unavailable, so this is the documented fallback.
- Review type: full-lane Milestone 1 re-review.
- Base commit: `1cbecf7a15f8612d27469785e4a98ea79bacecbd`.
- Candidate: `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub`
  at `10e2746b21d687405a3620341b754dcd96517fe3`.
- Frozen Plan: version 1, SHA-256
  `5ffdc332b89f8de424bbff66a6bca9ffa396e8adc3506a7adf25bc9eb011789a`.
- Scope: AC-02, AC-05, AC-06, AC-07, and AC-08 in Milestone 1 fixture/harness
  scope. This review made no implementation, ledger, Feishu, GitHub, parent,
  or other shared-state change. This report is the only write.

## Overall verdict: FAIL

Round 02 closes the four prior implementation gaps: it provides executable
remote-wins preservation, tombstones, a structured `hubctl diff`, and
gate-before-adapter writer coverage. Two new blocking evidence gaps remain:
the claimed in-root move test is only a same-parent metadata update, and the
root gate still lacks known-bad committed canaries for many implemented
detectors. The latest round therefore has new blocking findings and cannot
return PASS.

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-02 | FAIL | The valid control passes and several red paths are real, but the frozen requirement is a known-bad fixture for every detector. The test suite asserts only 16 of the checker error codes; for example it has no committed canary for invalid result authority, malformed source, malformed links, missing content, duplicate ID, invalid representation/status/sensitivity/tags/title, or invalid YAML/JSON. |
| AC-05 | PASS | `reconcile_document` preserved the locally changed bytes in a reachable `refs/hub-conflicts/...` commit, installed the remote edition, marked the entry `conflict`, and `gated_adapter` blocked publication before adapter construction. The direct reviewer CLI probe confirmed all four conditions. |
| AC-06 | FAIL | Delete/out-of-root fixtures are executable and make no fake writer call, but the purported in-root-move case leaves `parent_token` equal to the managed root and only changes title. No model field or assertion records a new in-root location/logical target; the frozen rename/move behaviour is not proved. |
| AC-07 | PASS | Captured Docx fixtures exercise formatting-only normalization and a semantic table/Mermaid/asset change. `hubctl diff --json` returned the common/local/remote hashes, change flags, and readable unified diff; malformed and unsupported payloads fail closed in tests. |
| AC-08 | PASS | A malformed repository returned `E_GATE_CHECK` before adapter construction. In a fresh local bare remote, a valid real `git push` ran the hook and succeeded; a malformed manifest made the hook reject the push. This demonstrates both green-before-red hook behaviour and gate-before-fake-adapter behaviour without a shared remote. |

## Required commands and evidence

Executed from `/data-1/code/verl` unless the command supplies another root:

```text
REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m pytest -q tests
............................                                             [100%]
28 passed in 0.22s
exit=0

REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m hubctl check --root .
PASS
exit=0

git -C /data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub status --short
<empty output; exit=0>

goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
goal_status=ACTIVE; current_milestone=Milestone 1; plan_status=READY;
pending_user_decisions=[]; exit=0
```

Adversarial reviewer-owned probes were run only in fresh local scratch clones:

```text
PYTHONPATH=src python -m hubctl diff FS02-DOC --fixture tests/fixtures/docx_semantic_after.json --json
{"changed": true, "common_hash": "6cca...011d", "local_changed": false,
 "local_hash": "6cca...011d", "remote_changed": true,
 "remote_hash": "285a...cdfe", "unified": "--- L ... +| score | 0.80 ..."}
exit=0

# A malformed entry before `hubctl publish`:
E_GATE_CHECK: E_SCHEMA_REQUIRED .../entry.yaml: missing required entry fields
exit=2

# Fresh local bare remote and real Git hook invocation:
valid git push -> hook printed PASS; exit=0
malformed-manifest git push -> hook printed E_SCHEMA_REQUIRED; exit=1

# Locally change the entry, set its common snapshot to the original bytes,
# then pull docx_semantic_after.json through the CLI:
{"changed": true, "state": "conflict",
 "conflict_ref": "refs/hub-conflicts/FS02-DOC/evt-...", ...}
git show <conflict_ref>:entries/FS02-DOC/content.md
# Local Agent Edition
```

The candidate `origin` is a local bare repository under the scratch root.
Fixture configuration retains `shared_writes_authorized: false`, unresolved
production identity, and `example.invalid` fixture identities. No real token,
real identity configuration, external service call, or shared write was found
or performed. The round-02 diff adds 510 lines and removes only two blank lines
from implementation; it deletes no test or test assertion.

## Prior finding disposition

- **F-M1-R01 (AC-05): closed by evidence.** `src/hubctl/sync.py` and the FS05
  fixture now prove remote-wins preservation, reachable recovery, and blocked
  publication.
- **F-M1-R02 (AC-06): partly repaired but superseded by F-M1-R05 below.**
  Detach/tombstone behaviour is now covered; in-root move is not.
- **F-M1-R03 (AC-07): closed by evidence.** `hubctl diff` and captured fixture
  coverage exist and were run by the reviewer.
- **F-M1-R04 (AC-08): closed by evidence.** Writer construction is gated and
  a real local `git push` red/green hook probe passed.

## Blocking in-scope defects

- **F-M1-R05 — AC-06 in-root move is neither represented nor tested.**
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/tests/feature_stories/test_fs06_remote_structure_changes.py:29-35`
  labels a test “rename and in-root move,” but its fixture uses
  `parent_token` set to `fixture-research-hub`, exactly the `managed_root` passed at
  line 32. `reconcile_inventory` then only updates `title` and writes the same
  parent token (`src/hubctl/sync.py:146-152`). `RemoteDocument` has no child
  location/name field beyond `title` and parent (`src/hubctl/models.py:18-27`).
  It cannot establish that an object moved within the managed root updates the
  entry's logical target rather than being ignored.

- **F-M1-R06 — AC-02 does not provide a known-bad canary for every root-gate detector.**
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/check.py:72-212`
  implements distinct schema/source/link/asset/identity/catalog detectors, but
  the committed canaries at
  `tests/gate/test_gate_canaries.py:22-67` and the adjacent tests do not cover
  many of them. A reviewer mutation of `source: {}` independently returned
  `E_SCHEMA_SOURCE` / exit 2, proving the detector exists but also that its
  committed red-canary proof is missing. This contradicts AC-02's explicit
  “one known-bad fixture for every detector” acceptance requirement.

## Test-strength audit

No candidate test was deleted or loosened relative to `1cbecf7`; the diff adds
the FS05, FS06, CLI-diff, writer-gate, and malformed-payload tests. All 28 tests
were collected and passed. The issue is missing coverage, not a skipped or
weakened existing test. The hook was additionally shown green before red by a
real local `git push`, so exit propagation is behavioural evidence rather than
a static-text claim.

## Deferred suggestions

None. Both findings are inside frozen acceptance criteria.

## Contract contradictions

None. The missing evidence is fixture-only and can be supplied without any
external or shared mutation.

## Single most likely weakness in this review

The real hook test uses a local bare remote, not the future private GitHub
remote. That is intentionally within Milestone 1 scope and does not affect the
blocking findings: both are local fixture-contract omissions.
