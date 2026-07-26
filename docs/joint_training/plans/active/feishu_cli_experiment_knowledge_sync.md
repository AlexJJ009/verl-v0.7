# Feishu Research Hub Architecture

- Status: ACTIVE IMPLEMENTATION; MILESTONE 4 PLAN RE-ENTRY
- Design date: 2026-07-23
- Initial host project: `/data-1/code/verl`
- Planned private repository: `AlexJJ009/feishu-research-hub`
- Planned submodule path: `research/feishu-research-hub`
- Goal Plan: `docs/joint_training/goals/feishu-research-hub-sync/plan.md`

## 1. Positioning

Feishu Research Hub is a curated publication, progress-reporting, sharing, and
limited-collaboration surface. It is not a second Obsidian and it is not a
mirror of every local file.

The existing systems keep their current jobs:

- project Git repositories own experiment source documents and policies;
- the experiment registry and release gate own run/result authority;
- W&B owns uploaded telemetry and run history;
- Hugging Face owns published model weights;
- a Win11 knowledge base may keep working notes and private reading notes;
- Feishu Research Hub keeps the smaller, reviewed edition that is useful to
  read, present, share, query, and collaboratively refine.

The Hub's private Git repository is the local backup, synchronization control
plane, and machine-readable audit history for that published edition. Once an
entry has been published, an authorized human edit in Feishu is authoritative
for the Hub edition. It does not silently rewrite the originating experiment
document in another repository.

## 2. Why a Separate Private Repository and Submodule

The Hub has a different lifecycle from `verl`: it contains reusable sync code,
publication manifests, normalized Feishu snapshots, paper notes, and local CI
configuration. Keeping it in a private repository avoids mixing those concerns
with training code while a submodule gives the current project a pinned,
reviewable integration point.

```text
/data-1/code/verl/
└── research/feishu-research-hub/     # private Git submodule
    ├── entries/                      # one stable directory per published entry
    ├── generated/                    # validator-owned indexes
    ├── src/                          # hubctl, adapters, canonicalizers
    ├── tests/                        # feature stories and failure fixtures
    ├── .githooks/pre-push            # tracked fast gate
    └── pyproject.toml
```

Each entry uses a small, uniform shape instead of a deep PARA hierarchy:

```text
entries/<entry-id>/
├── entry.yaml                        # identity, type, source, ownership, state
├── content.md                        # normalized human-readable edition
└── assets/                           # only selected figures/attachments
```

Classification lives in `entry.yaml`: `kind`, `batch_id`, `tags`, `status`,
`source`, and `sensitivity`. A generated catalog provides views by experiment
batch, paper topic, status, and tag. Moving an entry between categories does not
require moving a large local directory tree.

The Feishu side is intentionally shallow:

```text
Research Hub/
├── 00 Home and Catalog
├── Experiments/<batch-id>/
├── Papers/<topic>/
├── Reports/
└── Archive/
```

Sheets, Base, Slides, and Minutes are supported object types when a publication
needs them; they are not mandatory components of the first release. The first
release must prove Docx, native Markdown, selected images, and links before
adding more representations.

## 3. Content Contract

### 3.1 Content worth publishing

- reviewed experiment designs and material amendments;
- authoritative result reports and clearly labelled negative results;
- comparisons that change the next research decision;
- compact result tables, selected figures, and Mermaid/whiteboard diagrams that
  support a conclusion;
- verified W&B run links and immutable Hugging Face revision links;
- reusable runbooks needed by collaborators;
- curated paper notes, literature comparisons, and legally shareable paper
  links or files;
- concise progress reports and meeting outputs that have continued value.

Raw logs, checkpoint shards, W&B offline directories, full datasets, temporary
screenshots, generated intermediates, credentials, and unreviewed private notes
do not enter the Hub by default.

### 3.2 Entry kinds

The initial schema supports `experiment_design`, `experiment_result`,
`comparison`, `paper_note`, `workflow`, `report`, and `attachment`. Every entry
has a stable ID independent of title and Feishu folder location. One live entry
maps to at most one live Feishu object.

An experiment result additionally records `diagnostic`, `authoritative_local`,
or `externally_verified`. W&B or Hugging Face links appear only after the exact
remote identity is verified. A failed or incomplete run may be published only
as a visibly labelled diagnostic or negative result; Feishu publication never
bypasses the existing training result release gate.

### 3.3 Representations

| Content | Default Feishu representation | Local representation |
| --- | --- | --- |
| Curated plan/result/paper note | Docx | normalized Markdown |
| Exact Markdown requiring native diff | Drive `.md` file | exact Markdown |
| Small summary table | Docx table | Markdown table plus schema check |
| Filterable structured dataset | Sheet/Base after a later capability probe | selected CSV/JSON plus schema |
| Figure | embedded image with caption | selected asset plus checksum |
| Mermaid diagram | rendered whiteboard/preview plus source block | fenced Mermaid source |
| W&B/HF/paper reference | verified link and short context | typed link record |

## 4. Synchronization Model

### 4.1 One three-way algorithm

For every managed entry, synchronization compares:

- `B`: last verified common snapshot;
- `L`: current normalized local content;
- `R`: current normalized Feishu content and revision.

The only automatic outcomes are:

| State | Outcome |
| --- | --- |
| `L=B`, `R!=B` | pull `R`, update the entry, commit, and push Git |
| `L!=B`, `R=B` | validate, revision-check, publish `L`, verify, commit state |
| `L=B`, `R=B` | no-op |
| `L!=B`, `R!=B` | keep `R` as the active Hub edition, preserve `L` in a conflict snapshot/branch, and require explicit reconciliation |

This implements “Feishu human edit wins” without deleting concurrent local
work. A user rename or move inside the managed Research Hub updates local
metadata on the next pull. A move outside the managed root marks the entry
`detached` and alerts; it is not recreated automatically. Polling absence is
not deletion proof: the documented Drive metadata code `970005` conflates type
mismatch and nonexistence, so polling fails closed and retains the entry
unchanged. A future deletion-event adapter may create a Git tombstone only from
the deletion-exclusive `drive.file.trashed_v1` event, retaining last content,
revision, editor evidence, and deletion time. Until that later capability is
implemented, trusted fixtures verify tombstone retention and live sync never
infers deletion or erases Git history.

### 4.2 Docx diff adapter

`lark-cli 1.0.76` currently provides:

- `markdown +diff` for Drive-native `.md` files;
- `docs +fetch` in XML or Markdown form, including a requested revision;
- `docs +history-list`, including `revision_id`, `edit_time`, and
  `editor_ids`;
- Drive file version history and editor identity for ordinary files.

It does not provide a first-class Docx-to-Docx diff. The Hub therefore owns a
thin, versioned adapter:

```text
lark-cli fetch/history
        -> canonical Docx Markdown + asset references
        -> deterministic canonicalization
        -> structural/unified diff
        -> Git snapshot and review output
```

The adapter wraps public CLI commands and records the tested CLI version. It
does not fork or reimplement lark-cli. Unknown output/schema changes fail
closed until a capability probe passes.

### 4.3 Pull and publish commands

The planned command surface is deliberately small:

```text
hubctl check                    # deterministic schema/content gate
hubctl diff <entry>             # local/common/remote comparison
hubctl pull [<entry>]           # remote-first reconciliation, no silent loss
hubctl publish <entry>          # revision-checked local publication
hubctl sync --once              # pull first, then safe publish candidates
hubctl status [--json]          # operator state, conflicts, CI and last sync
hubctl import-project ...       # create reviewed candidates from a project
```

All network writes re-run `hubctl check` themselves. A hook verdict is never
trusted as proof that a later publish is safe.

## 5. Git History and Attribution

The Hub repository has repo-local Git identity configuration; the current
`verl` and global Git identities are not changed.

The tracked identity map separates source authorship from automation:

- Feishu changes by the user: Git `Author` is `李功勋` with the selected human
  email; the sync service is `Committer`.
- Agent-authored local changes: Git `Author` is the named agent using its
  configured agent email; the human is added with a standard `Co-authored-by`
  trailer when the commit incorporates the user's authored direction/content.
- Mixed Feishu revisions: the latest mapped human editor is `Author`; all
  mapped contributors appear in structured trailers, while raw Feishu
  `editor_ids` remain in the audit event.
- Unknown editor IDs never impersonate the user; they use a neutral mapped
  identity and require identity-map review.

No existing native `cooperator` Git facility or project convention was found.
The implementation therefore uses standard `Author`, `Committer`, and
`Co-authored-by` semantics, exposed through one identity-map module. The human
email must be chosen before repository initialization from the user-provided
candidates `2665631223@qq.com` and `lgxma01@buaa.edu.com`; the Plan treats that
as a `USER_DECISION`, not a global Git config change.

## 6. Structural Enforcement

### 6.1 Root validator

One deterministic command, `hubctl check`, owns all machine-checkable rules:

- entry schema, enums, IDs, and uniqueness;
- batch and source references;
- table column/type constraints;
- selected asset existence and checksum;
- generated catalog drift;
- link syntax and immutable-link requirements;
- secrets, tokens, private infrastructure paths, and restricted content;
- sync-state fixtures and unsupported remote operations;
- commit-author identity policy.

Every detector has a known-bad fixture that must turn the gate red. A green
scanner that cannot catch its canary is an error, not a pass.

### 6.2 Pre-push, not pre-commit

The repository uses a tracked `.githooks/pre-push`, installed through
`core.hooksPath`. It runs the fast deterministic profile and propagates the
validator exit code unchanged. There is no mandatory pre-commit hook.

`--no-verify` remains technically possible, so the runtime publish/sync entry
point re-runs the root validator. A bypass cannot result in a Feishu write. Any
future explicit force path requires a reason in the append-only audit log.

### 6.3 Local CI

Local CI runs on this server, not in a cloud runner:

1. detect a new private-repository commit;
2. create a pristine temporary clone;
3. install the pinned toolchain;
4. run the full deterministic gate without live Feishu writes;
5. append a `PASS`, `RED`, or `ERROR` JSONL verdict;
6. advance the cursor only after a verdict is safely recorded.

Only `PASS` can authorize the scheduled sync worker. `RED` is a real content or
test failure; `ERROR` is an infrastructure failure. The sync worker also
re-runs the gate before writing, so local CI is not a single point of failure.

## 7. Runtime Pull and Git Push

The first production scheduler is one locked `sync --once` job under the
project's existing PM2-only persistent-process policy. PM2 is installed or
repaired during its deployment milestone if required; systemd, cron, a public
listener, and inbound webhook are excluded. No supervisor is installed merely
by approving this architecture.

Each cycle:

1. acquire a process lock;
2. verify lark-cli authentication and supported version;
3. list managed objects and histories since the stored cursor;
4. pull and canonicalize Feishu changes;
5. preserve conflict snapshots before applying remote-wins;
6. run the deterministic gate;
7. commit with mapped author/committer identities;
8. push the private repository;
9. append a structured event and expose it through `hubctl status`.

Network timeouts, bounded retries, lock ownership, and a cursor that advances
only after a successful Git push prevent a live but non-functional loop from
silently losing updates.

## 8. Operator Visibility and UI Finding

The server was inspected on 2026-07-23. It is headless (`DISPLAY` and
`WAYLAND_DISPLAY` unset, `XDG_SESSION_TYPE=tty`) and has no installed
code-server, VNC/noVNC, Xpra, Chromium/Firefox, JupyterLab, File Browser, or
Cockpit management surface. Current listeners belong primarily to training,
Ray, and vLLM processes. There is no existing UI to reuse safely.

The first operator surface is therefore `hubctl status --json/pretty` plus
Git history and append-only CI/sync events. A later read-only Web dashboard may
bind to localhost and be accessed through an SSH tunnel. It must consume the
same status API and may not become a second editing system. Building a general
knowledge-management frontend is outside this architecture.

## 9. Security and Deletion

- OAuth scope breadth does not broaden document sharing automatically.
- Tokens, private URLs, and credentials live outside Git in mode-`0600`
  runtime configuration.
- Project import creates candidates; publishing still validates sensitivity,
  evidence authority, and exact target.
- Permission changes, public sharing, owner transfer, and collaborator changes
  are separate high-risk actions requiring explicit approval.
- Archive is the default retirement action.
- An Agent may never interpret remote absence as permission to delete local
  history.
- A remote delete initiated by CLI requires exact target resolution, a verified
  Git snapshot, link-impact check, dry-run where supported, current-turn human
  approval, and lark-cli's high-risk confirmation.
- A deletion performed manually in Feishu is never inferred from ambiguous
  polling absence. Live tombstone ingestion is deferred until the
  deletion-exclusive `drive.file.trashed_v1` event adapter is separately
  planned and deployed; restoring or recreating remains a human decision.

## 10. Project and Win11 Integration

`verl` integration is downstream of existing experiment authority. A project
adapter may create or refresh candidate entries only after it reads the local
release/eval evidence. It cannot mark a result externally verified, fabricate a
W&B/HF link, or upload checkpoints.

A Win11 knowledge base integrates as another clone/import source, not as a
second synchronization authority. Manual notes and Agent-assisted paper
thinking can be curated into the same entry format and then reviewed/published.
The repository format and core Python CLI must remain cross-platform, but an
automatic Win11 watcher and Obsidian plugin are deferred until the server path
is accepted.

## 11. Delivery Order

1. Freeze schemas, Feature Stories, CLI capability probes, and known-bad gate
   fixtures.
2. Implement the private-repo core, deterministic validator, canonicalizer,
   diff, and fake-Feishu integration tests.
3. After explicit approval, create the private GitHub repository and add the
   pinned project submodule.
4. On disposable Feishu objects, prove local publish, remote edit pull,
   concurrent-change preservation, move/rename/detach, fail-closed suspected
   absence, and attribution; prove tombstone retention with trusted fixtures.
5. Add the `verl` candidate importer and publish one approved seed batch.
6. Deploy the locked periodic pull, automatic Git commit/push, and local CI;
   prove one scheduler-owned end-to-end cycle.
7. Obtain independent acceptance before enabling broader content types or
   additional machines.

The detailed frozen contract, Feature Stories, verification commands, review
rules, and `USER_DECISION` gates live in the Goal Plan linked above.

## 12. Deferred Work

- a read-only localhost Web dashboard;
- event/webhook-based sync after polling is proven;
- live `drive.file.trashed_v1` consumption and deletion-event-backed tombstones;
- automatic Win11/Obsidian ingestion;
- Sheets/Base/Slides/Minutes writers beyond individually probed use cases;
- public sharing automation or external-tenant permission management;
- semantic merge of concurrent local and Feishu prose;
- high-frequency W&B metric mirroring.
