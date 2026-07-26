# Milestone 1 Independent Review 01

## Review identity

- Reviewer: fresh independent reviewer, `gpt-5.6-terra` / medium.  The
  requested project reviewer models were unavailable; this is the recorded
  fallback.
- Review type: Milestone Review (Milestone 1 only).
- Candidate repository: `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub`
- Candidate commit: `1cbecf7a15f8612d27469785e4a98ea79bacecbd`
- Frozen Plan: version 1, SHA-256
  `5ffdc332b89f8de424bbff66a6bca9ffa396e8adc3506a7adf25bc9eb011789a`.
- Scope: AC-02, AC-05, AC-06, AC-07, and AC-08 in their Milestone 1
  contract/fixture/harness form. No implementation, ledger, Feishu, GitHub,
  parent-repository, or other shared-state mutation was performed by this
  review. The sole write is this reviewer testimony.

## Overall verdict: FAIL

The deterministic control is green, its tested red canaries turn red, and the
hook propagates the gate exit code.  However, three applicable Feature Story
behaviours have no executable Milestone 1 harness, and no writer entrypoint
exists to prove that a `--no-verify` Git-hook bypass cannot bypass the root
gate.  Green tests therefore do not establish AC-05 through AC-08.

| AC | Verdict | Reviewer-owned basis |
| --- | --- | --- |
| AC-02 | PASS | All 19 tests passed; `hubctl check --root .` returned `PASS`; direct unsafe-link canary returned `E_LINK_UNSAFE` and exit 2. `check_root` only reads local files and the purity test monkeypatches fake-adapter construction. |
| AC-05 | FAIL | No `pull` or `sync --once` CLI command or reconciliation implementation exists. `python -m hubctl pull FS02-DOC --adapter fake --json` exited 2; there is no test for remote-wins, snapshot/ref reachability, blocked publish, or byte recovery. |
| AC-06 | FAIL | The fake adapter exposes no reconciliation or delete operation and no tests exercise rename, in-root move, out-of-root detachment, or tombstone retention. It cannot prove no automatic recreation/loss. |
| AC-07 | FAIL | Canonicalization and a bare `unified_diff` helper have unit coverage, including unsupported version, but `hubctl diff` is absent (exit 2). No captured fixtures/goldens cover tables, media, Mermaid, malformed payload, or the common/local/remote structured diff contract. |
| AC-08 | FAIL | The tracked hook correctly returns the gate exit code, but the only CLI command is `check`; no publish/pull/sync writer exists to run the root gate before its first remote call. Thus the required `--no-verify` writer-boundary proof is absent. |

## Required commands and evidence

Executed from `/data-1/code/verl` unless the command supplies another root:

```text
REPO_HOST=/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub REPO_CONTAINER=/workspace/hub /data-1/verl07/run_train.sh env PYTHONPATH=/workspace/hub/src python -m pytest -q tests
...................                                                      [100%]
19 passed in 0.12s
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

Additional adversarial reviewer probes:

```text
# In a fresh local clone, replace the fixture link with http://unsafe.example.invalid
PYTHONPATH=src python -m hubctl check --root .
E_LINK_UNSAFE .../entries/FS02-DOC/entry.yaml: only HTTPS links are accepted
exit=2

# Invoke the tracked hook directly in a fresh local clone.
.githooks/pre-push
PASS
exit=0

# Replace the entry manifest with "kind: malformed", then invoke it again.
.githooks/pre-push
E_SCHEMA_REQUIRED .../entries/FS02-DOC/entry.yaml: missing required entry fields
exit=2

PYTHONPATH=src python -m hubctl diff FS02-DOC
PYTHONPATH=src python -m hubctl publish FS02-DOC --adapter fake --json
PYTHONPATH=src python -m hubctl pull FS02-DOC --adapter fake --json
# Each: argparse invalid choice, choose from {check}; exit=2.
```

No remote credential or real identity was found in the candidate. Its sole
`origin` is a local bare repository under the scratch root. Fixture identities
use `example.invalid`, and `config/hub.yaml` fixes
`shared_writes_authorized: false` and `production_human_identity: unresolved`.
The only source match for a real-looking email is the deliberately mutated
negative test, not committed fixture configuration. The deterministic gate has
no adapter import/call path; its candidate purity test and direct source review
support that conclusion.

## Blocking in-scope defects

- **F-M1-R01 — AC-05 concurrent-change harness missing.**
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/__main__.py:13-25`
  registers only `check`; no code or test implements `pull`/`sync --once`,
  remote-wins activation, conflict snapshots/refs, publication blocking, or
  recovery. This leaves the frozen AC without executable evidence.

- **F-M1-R02 — AC-06 structure-change harness missing.**
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/adapters/fake_lark.py:10-70`
  contains fixture fetch/history/inventory/publish only; it has neither a
  reconciliation transition nor tombstone/snapshot behaviour. There is also no
  `tests/feature_stories/test_fs06_remote_structure_changes.py` equivalent.

- **F-M1-R03 — AC-07 executable diff contract missing.**
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/__main__.py:13-25`
  exposes no `diff` command, while
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/diffing.py:6-16`
  is only a two-string helper. The required structured common/local/remote
  output and fixture coverage are absent.

- **F-M1-R04 — AC-08 writer-boundary enforcement cannot be verified.**
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/.githooks/pre-push:1-4`
  is correct as a hook, but there is no writer command in
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub/src/hubctl/__main__.py:13-25`
  to invoke `hubctl check` independently. The static hook test at
  `tests/hooks/test_pre_push.py:4-8` would not catch a writer that performs a
  remote call before (or without) the gate.

## Test-strength audit

The initial candidate is a root commit, so there is no earlier candidate test
suite to compare for deletion/weakening. No test was skipped in the reviewer
run. The tests are nevertheless insufficient for the four failed ACs: they
only cover the happy-path fake adapter, local check canaries, basic text
normalization/diffing, and static hook text. The reviewer deliberately ran a
green-before-red probe for both the root gate and the hook; those controls are
real, but they do not substitute for missing Feature Story harnesses.

## Deferred suggestions

None. The missing behaviours are frozen, applicable acceptance evidence, not
phase-two suggestions.

## Contract contradictions

None identified. The Plan permits fixture-only implementation and requires the
missing behaviour to be proved there; it does not require any live Feishu or
shared Git action for this review.

## Single most likely weakness in this review

The direct hook probe demonstrates shell exit propagation, but it invokes the
hook rather than a real `git push` transaction. That is a bounded limitation:
the more consequential blocker is independent and already demonstrated—the
writer entrypoints and their first-remote-call gate do not exist.
