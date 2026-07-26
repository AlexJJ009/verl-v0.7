# Milestone 1 Contract

- Status: implementation contract for fixture-only/local-only Milestone 1
- Runtime root: `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/local-hub`
- External/shared writes: forbidden

## Boundary

Milestone 1 freezes the schema, fake adapter protocol, deterministic gate
contract, canaries, and the tests that consume them. Milestone 2 implements the
full `hubctl` behavior. A minimal `hubctl check` runner is part of Milestone 1
because AC-02 and AC-08 require executable red/green gate evidence.

The implementation lives in a disposable local Git repository under the Goal
scratch root until D-01 through D-03 authorize a private GitHub repository,
human identity, and parent submodule. The local repository must never contain
real Feishu tokens, private URLs, OAuth material, or the user's undecided email.

## Local Repository Shape

```text
local-hub/
├── pyproject.toml
├── config/
│   ├── hub.yaml
│   └── identity-map.yaml
├── entries/<entry-id>/
│   ├── entry.yaml
│   ├── content.md
│   └── assets/
├── generated/catalog.json
├── src/hubctl/
│   ├── __init__.py
│   ├── __main__.py
│   ├── check.py
│   ├── models.py
│   ├── canonicalize.py
│   ├── diffing.py
│   ├── identity.py
│   ├── sync.py
│   └── adapters/fake_lark.py
├── tests/
│   ├── fixtures/
│   ├── schema/
│   ├── gate/
│   ├── security/
│   ├── canonicalization/
│   ├── diff/
│   ├── adapters/
│   ├── hooks/
│   └── feature_stories/
└── .githooks/pre-push
```

Runtime snapshots, conflicts, tombstones, cursors, and audit events live below
an ignored `.hub/` root. Tests use committed fixture copies under
`tests/fixtures/`; they do not treat runtime state as source.

## Entry Contract

Every entry has exactly one `entry.yaml` and `content.md`. Required fields:

- `entry_id`: `^[A-Z][A-Z0-9-]{2,63}$`, globally unique;
- `title`: non-empty string;
- `kind`: `experiment_design`, `experiment_result`, `comparison`,
  `paper_note`, `workflow`, `report`, or `attachment`;
- `status`: `draft`, `current`, `archived`, or `withdrawn`;
- `batch_id`: nullable stable identifier;
- `tags`: unique strings;
- `sensitivity`: `internal`, `restricted`, or `approved_external`;
- `representation`: `docx` or `markdown`;
- `source`: typed mapping containing the local/external source identity;
- `links`: typed link records with verification state;
- `assets`: records with repo-relative path, SHA256, and media type;
- `sync`: state, common snapshot hash, remote revision, and editor IDs.

`experiment_result` additionally requires `result_authority` with exactly
`diagnostic`, `authoritative_local`, or `externally_verified`.

Tracked fixture entries may use explicit `fixture_*` remote identities. A live
remote token is optional for an unpublished entry and is never required in Git.
One live remote identity may map to at most one live entry.

## Sync and Audit Contract

The common snapshot `B` records canonical content/hash, stable asset references,
revision, fixture/live binding key, and fetch time. `L` is derived from the
working entry. `R` is the normalized adapter response.

Audit events contain an ID, timestamp, entry ID, action, `B/L/R` hashes and
revisions, adapter version, outcome, and conflict/tombstone reference when
present.

Concurrent `L!=B` and `R!=B` preserves the complete local payload and metadata
in a Git commit reachable through `refs/hub-conflicts/<entry-id>/<event-id>`.
Remote content becomes active, `sync.state=conflict`, and publish remains
blocked. Tombstones retain the last snapshot plus deletion evidence and never
trigger recreation.

## Identity Contract

The tracked fixture identity map contains only fictional test identities and an
explicit unresolved production-human marker. D-02 is the only path that may
select one of the user's real emails.

Resolution produces a Git Author, service Committer, and structured trailers:

- `Hub-Entry-Id`;
- `Feishu-Revision`;
- `Feishu-Editor-Ids`;
- `Co-authored-by` only when Agent-authored content incorporates human-authored
  direction or content.

An unknown Feishu editor resolves to a neutral identity and an identity-review
flag. It never impersonates the selected human.

## Fake Adapter Contract

The fake adapter exposes `version`, `fetch`, `history`, `publish`, `inventory`,
and an inspectable call log. It must enforce expected revisions, idempotency
keys, read-back verification, scripted malformed responses, unsupported
versions, move/rename/out-of-root/deletion cases, and zero-call assertions.

`hubctl check` is pure local validation: it must not construct or call any
adapter.

## Gate Canaries

Each canary has a stable failure code and an executable test that first proves
the valid control passes and then proves the mutation fails:

- malformed schema, unknown enum, duplicate ID, or invalid entry shape;
- unsafe/unverified link or invalid result-authority transition;
- credential/token or private infrastructure path;
- missing asset or SHA256 mismatch;
- stale generated catalog;
- unsupported sync state or operation;
- malformed identity map or selected-human impersonation;
- malformed/unsupported adapter output/version;
- formatting-only canonicalization noise;
- concurrent remote-wins without recoverable conflict evidence;
- rename/move/delete causing an unexpected writer call;
- pre-push exit-code loss or writer bypass after `--no-verify`.

Scanner reports must never echo a matched secret. Tests assert failure codes and
redacted paths/fingerprints only.
