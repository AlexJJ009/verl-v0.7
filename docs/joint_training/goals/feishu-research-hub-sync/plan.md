# Feishu Research Hub Sync Submodule

- Goal ID: `feishu-research-hub-sync`
- Plan version: `3`
- Authorization policy version: `2`
- Plan status: `READY — RESUME AT REMAINING STEP R2`
- Architecture:
  `docs/joint_training/plans/active/feishu_cli_experiment_knowledge_sync.md`
- Planned private repository: `AlexJJ009/feishu-research-hub`
- Planned parent path: `/data-1/code/verl/research/feishu-research-hub`

## Outcome

Deliver one independently accepted private Git-backed Feishu Research Hub
submodule that safely publishes curated research entries, pulls authorized
Feishu human edits as the Hub edition without losing concurrent local work,
records attributable Git history, blocks malformed or unsafe content before
remote writes, and exposes verifiable synchronization and local-CI status.

## Starting Evidence

- The private Hub repository and parent submodule already exist. Hub `HEAD` and
  `origin/main` are both
  `88ac17c2181d22f1e33373c54763ea8e6351bf74`; prior Milestone 7 convergence
  evidence reports `154 passed`, root-gate PASS, reachable retained conflict
  history, and a matching no-op runtime cursor.
- The parent working tree contains unrelated user experiment changes. They stay
  protected and must not be reformatted, staged, committed, or overwritten by
  this Goal. R3 stages only the exact Hub gitlink and Goal-owned records.
- Git identity and attribution were already selected and exercised: Agent
  implementation commits, human edit attribution as
  `GongxunLi <lgxma01@buaa.edu.cn>`, and structured Feishu revision/editor
  trailers. Identity selection is complete and is not a remaining decision.
- `lark-cli` user identity and current auth-envelope compatibility were already
  verified during Milestone 7. Exact tokens and URLs remain runtime-private.
- The server remains headless; CLI/JSON status is the frozen operator surface and
  a Web UI remains deferred.
- Deletion semantics remain unchanged: ambiguous polling metadata cannot prove
  deletion, live tombstoning requires the deferred deletion-exclusive event
  capability, and this Goal performs no live deletion.
- The Goal ledger has no pending user decisions and all findings are closed.
  Plan v3 re-entry review returned `READY`; execution resumes at R2 rather than
  replaying completed milestones.

## Scope

### Included

- Create one private GitHub repository and add it as a pinned submodule at the
  frozen parent path after the corresponding `USER_DECISION`.
- Bootstrap a small cross-platform Python CLI named `hubctl`, tracked schemas,
  one-entry-per-directory storage, generated catalogs, tests, and project entry
  documentation in the private repository.
- Support curated experiment designs, experiment results, comparisons, paper
  notes, workflows, reports, selected attachments, verified W&B links, verified
  Hugging Face revisions, and paper links.
- Implement Docx and Drive-native Markdown canonicalization, three-way diff,
  pull, revision-checked publish, status, and append-only audit records through
  a versioned adapter over public lark-cli commands.
- Treat an authorized human edit in Feishu as authoritative for the Hub edition;
  preserve a concurrently changed local edition before applying remote-wins.
- Reconcile human rename/move within the managed root and mark out-of-root
  moves detached. Polling-based suspected absence fails closed and leaves the
  entry untouched. Trusted deletion-event fixtures still exercise recoverable
  Git tombstones, while live deletion evidence consumption is deferred.
- Implement repo-local human/Agent identity mapping and deterministic Git
  authorship/committer/co-author trailers.
- Implement one deterministic root validator, a tracked pre-push hook, and a
  pristine-clone local CI runner with `PASS`, `RED`, and `ERROR` verdicts.
- Implement a locked periodic `sync --once` deployment, automatic attributable
  Git commit/push, cursor safety, timeouts, and CLI/JSON operator status.
- Integrate a read-only `verl` candidate importer downstream of existing
  release/eval authority and validate one approved disposable/seed publication
  workflow.
- Provide Feature Story fixtures, known-bad canaries, exact commands, and
  independent milestone/final review evidence.

### Excluded

- Replacing Obsidian, the Win11 knowledge base, Git, the experiment registry,
  W&B, Hugging Face, or the Feishu UI.
- A general-purpose knowledge-management frontend, public Web service, browser
  editor, semantic search engine, or graph database.
- Automatic Win11/Obsidian file watching or bidirectional Win11 sync.
- High-frequency W&B telemetry mirroring, checkpoint/model-weight payload
  upload to Feishu, raw-log mirroring, or dataset archival.
- Automatic semantic merge of concurrent prose. Concurrent local/remote change
  preserves local evidence and makes remote the active Hub edition, then
  requires explicit reconciliation.
- Production-wide sharing changes, public-link enablement, owner transfer,
  secure-label changes, or collaborator management.
- Automatic remote deletion, bulk deletion, destructive local cleanup, or
  deletion of existing Feishu/GitHub objects.
- Live `drive.file.trashed_v1` subscription/consumption and live deletion
  tombstoning. The documented polling metadata result `970005` conflates type
  mismatch with nonexistence and is never accepted as deletion proof.
- Mandatory Sheets, Base, Slides, Minutes, Wiki, or Whiteboard write support in
  the initial accepted capability. Their adapter contracts may be added only by
  a later Plan after an individual probe.
- Cloud-hosted CI or a public inbound webhook.
- Modification of current repository/global Git identity or unrelated dirty
  files in `/data-1/code/verl`.

## Architecture Contract

### Repository and entry model

- The private Hub repository is the synchronization control plane, local
  backup, and Git audit history for the curated Hub edition.
- Each live entry has exactly one stable `entry_id`, one `entry.yaml`, one
  normalized `content.md`, optional selected assets, and at most one live
  Feishu object.
- Classification is metadata, not a deep local folder tree. Generated indexes
  are derived and fail the validator when stale.
- Private Feishu tokens, URLs when classified private, OAuth material, and
  cursors live in mode-`0600` runtime state outside Git.
- The initial managed remote representations are `docx` and `markdown` plus
  selected image/file attachments. Unknown operations fail closed.

### Synchronization state machine

- Every entry stores the last verified common snapshot `B`, current local
  normalized content `L`, and fetched remote normalized content/revision `R`.
- `L=B,R!=B`: remote-only change is pulled, validated, committed, and pushed.
- `L!=B,R=B`: local-only change may publish only after root validation, current
  remote-revision check, read-back verification, and Git recording.
- `L=B,R=B`: no-op and no duplicate commit or remote write.
- `L!=B,R!=B`: save local content and metadata in a recoverable conflict
  snapshot/branch, make `R` the active Hub edition, set an unresolved conflict,
  and block local publish until explicit reconciliation. No side is discarded.
- Remote rename/move inside the managed root updates metadata; out-of-root move
  becomes `detached`. Polling absence or ambiguous metadata becomes a
  fail-closed error that leaves content, history, and sync state unchanged and
  never recreates the object. Only a trusted deletion-event fact can create a
  tombstone; this Goal verifies that downstream behavior with fixtures and
  defers live deletion-event consumption. No automatic recreation, hard delete,
  or local-history deletion follows.
- `sync --once` pulls before it considers publication. The cursor advances only
  after content, audit, commit, and Git push succeed.

### CLI adapter and canonical diff

- lark-cli is an upgradable external dependency. The Hub records the tested
  version and calls documented commands; it does not fork lark-cli.
- Docx fetch output is canonicalized deterministically to normalized Markdown
  plus stable asset/reference records. Volatile IDs, ordering, or formatting
  that do not change reader content must not create a semantic change.
- `hubctl diff` reports common/local/remote identities, structured summary, and
  a readable unified diff. Unknown CLI schema/version changes produce `ERROR`.
- Native Markdown may use `markdown +diff`, but its result is normalized into
  the same internal diff model as Docx.
- Network and live Feishu operations are not part of the deterministic root
  gate; acceptance uses a fake CLI fixture and separately authorized disposable
  remote objects.

### Enforcement and local CI

- `hubctl check` is the only root deterministic gate and owns schema, source,
  asset, catalog, link, identity, secret, and sync-state validation.
- Each detector has a known-bad canary. Acceptance proves that the canary turns
  the gate red before relying on a green run.
- `.githooks/pre-push` is tracked and installed with repo-local
  `core.hooksPath`; it propagates the gate's exit code unchanged. There is no
  mandatory pre-commit hook.
- Every remote-writing entrypoint runs `hubctl check` itself. `--no-verify`
  cannot bypass a Feishu write gate.
- Local CI judges each new target commit in a pristine clone and records exactly
  `PASS`, `RED`, or `ERROR`. It never marks an unjudged commit seen and never
  performs Feishu writes.
- The sync worker accepts only a matching `PASS` verdict or performs the full
  gate itself. A status narrative without a recorded exit code is not evidence.

### Git attribution

- The selected user identity is stored only in repo-local Hub configuration.
- A Feishu human revision maps editor identity to Git `Author`; the sync service
  is `Committer`; the Feishu revision/editor IDs are recorded in structured
  trailers/audit.
- Agent-authored content uses a named Agent author and configured Agent email.
  When the commit incorporates user-authored direction/content, add the user as
  a standard `Co-authored-by` trailer.
- Unknown Feishu editors use a neutral non-user author mapping and open an
  identity-review finding. They never impersonate the selected user.

### Safety and protected state

- All external writes and shared-state mutations named as `USER_DECISION`
  below remain unauthorized until the matching decision is recorded.
- A CLI-created remote delete requires exact target/backup/link-impact evidence,
  current-turn explicit approval, dry-run where supported, and the lark-cli
  high-risk confirmation. Remote deletion is not needed to satisfy this Goal.
- Existing `verl` release gates remain authoritative. The importer emits
  candidates and cannot fabricate W&B/HF URLs, externally verified status, or
  result eligibility.
- Parent-repo changes are limited to `.gitmodules`, the submodule gitlink, and
  necessary plan/index entry files after approval. All pre-existing dirty files
  are preserved byte-for-byte.

## Acceptance Criteria

Each criterion is expressed as a Feature Story plus Given/When/Then evidence.

### AC-01 - Private Repository and Submodule Are Reproducible

Feature Story FS-01: As a project maintainer, I can initialize a new clone and
obtain the exact private Hub revision through the project submodule without
changing global Git identity.

- Given recorded decisions for repository creation, submodule addition, and the
  selected human email,
- When a fresh authenticated parent clone initializes the Hub submodule,
- Then the URL is the approved private GitHub repository, the path is exactly
  `research/feishu-research-hub`, the gitlink is pinned, Hub setup installs only
  repo-local hooks/identity configuration, and current global/parent identities
  remain unchanged.
- Verification commands:

```bash
git -C /data-1/code/verl config --get-regexp '^submodule\.feishu-research-hub\.(path|url)$'
git -C /data-1/code/verl ls-files --stage research/feishu-research-hub
git -C /data-1/code/verl/research/feishu-research-hub config --local --get core.hooksPath
git -C /data-1/code/verl/research/feishu-research-hub remote get-url origin
gh repo view AlexJJ009/feishu-research-hub --json nameWithOwner,visibility
```

- Expected evidence: `visibility=PRIVATE`, exact URL/path/gitlink, local
  `.githooks`, selected Hub-local identity, and unchanged before/after
  parent/global Git config hashes.

### AC-02 - Entry Schema and Content Gate Fail Closed

Feature Story FS-02: As a publisher, I receive fast, deterministic rejection for
a malformed entry, unsafe link, secret, missing asset, invalid result authority,
or stale generated catalog before any remote API call.

- Given valid fixtures and one known-bad fixture for every detector,
- When `hubctl check` runs and each fixture is mutated independently,
- Then valid content passes, every invalid state returns nonzero with a stable
  error code, known-bad canaries are detected, generated indexes are exactly
  reproducible, and the fake lark-cli call log stays empty.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/gate tests/schema tests/security && python -m hubctl check --root .
```

- Expected evidence: reviewer-owned passing tests, explicit red-canary output,
  exact regenerated-index diff of zero, and zero remote calls.

### AC-03 - Local Publication Is Idempotent and Revision Checked

Feature Story FS-03: As a researcher, I can publish one reviewed local entry to
Feishu and rerun the command without duplicates or lost collaborator edits.

- Given a valid local-only change, a fake remote object at common revision, and
  a disposable Feishu object approved for the live probe,
- When `hubctl publish <entry>` runs twice and a stale-revision mutation is also
  exercised,
- Then the first run updates/creates at most one object, read-back content and
  revision match the plan, the second run is a no-op, and the stale revision
  blocks before content mutation.
- Verification commands:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/feature_stories/test_fs03_local_publish.py
python -m hubctl publish FS03-DOC --adapter fake --json
python -m hubctl publish FS03-DOC --adapter fake --json
```

- Expected evidence: one write, verified normalized hash/revision, second-run
  `changed=false`, stale-revision structured conflict, and no duplicate token.
  The independent live reviewer repeats the same story against only the
  user-approved disposable object.

### AC-04 - Feishu Human Edit Pulls to Attributable Git History

Feature Story FS-04: As a user editing in Feishu, I can change a managed Docx and
have the server pull the exact edition into Git with my mapped authorship.

- Given `L=B`, a newer remote Docx revision, editor IDs, and selected human
  identity mapping,
- When `hubctl pull <entry>` runs,
- Then canonical content and assets become the local Hub edition, the new common
  snapshot is recorded, one Git commit is created/pushed with the human as
  `Author` and service as `Committer`, revision/editor audit fields are present,
  and a second pull is a no-op.
- Verification commands:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/feature_stories/test_fs04_remote_edit_pull.py tests/git/test_attribution.py
python -m hubctl pull FS04-DOC --adapter fake --commit --json
git show -s --format=fuller HEAD
python -m hubctl status --entry FS04-DOC --json
```

- Expected evidence: content diff equals the fixture edit, exact author and
  committer fields, Feishu revision/editor audit binding, push success in a
  local bare-remote fixture, and idempotent second pull.

### AC-05 - Concurrent Changes Preserve Local Work While Remote Wins

Feature Story FS-05: As a collaborator, my Feishu edit becomes the active Hub
edition even when an Agent changed the local file, while the Agent change
remains recoverable and cannot be silently published over me.

- Given `L!=B` and `R!=B` with independently identifiable changes,
- When `hubctl pull` or `hubctl sync --once` runs,
- Then remote content becomes active, the complete local edition and metadata
  are retained in a conflict snapshot/branch referenced by the audit event, the
  entry is `conflict`, publish is blocked, and reconciliation can recover both
  exact byte streams.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/feature_stories/test_fs05_concurrent_remote_wins.py
```

- Expected evidence: two distinct pre/post hashes, reachable conflict ref,
  remote-active content, blocked publish, and successful recovery assertions for
  both sides.

### AC-06 - Structure Changes and Suspected Absence Never Cause Silent Recreation or Loss

Feature Story FS-06: As a Feishu organizer, I can rename or move a managed
document in the Research Hub and see Git metadata follow it; if I move it out or
polling can no longer prove where it is, the system preserves evidence and asks
before any recreation.

- Given remote rename, in-root move, out-of-root move, ambiguous absence,
  permission/type failure, and trusted confirmed-deletion fixtures,
- When a pull cycle reconciles remote inventory,
- Then rename/in-root move update the logical target, out-of-root move becomes
  `detached`, all polling absence/error classes fail closed without changing
  entry state or content, and a trusted confirmed-deletion fixture creates a
  tombstone retaining last content/history. None of the cases creates, deletes,
  or recreates a remote object automatically. Live deletion evidence
  consumption is outside this Goal because documented polling metadata code
  `970005` is not deletion-exclusive.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/feature_stories/test_fs06_remote_structure_changes.py
```

- Expected evidence: stable entry ID, expected rename/move/detach transitions,
  permission/type/`970005`/empty-metadata canaries leaving entry bytes and sync
  state unchanged, readable retained fixture tombstone, and adapter call logs
  containing no create/delete/recreate write. Disposable live evidence covers
  rename/in-root move/out-of-root metadata lookup only; it does not delete an
  object or claim live tombstone proof.

### AC-07 - Docx Canonical Diff Is Stable and CLI-Version Safe

Feature Story FS-07: As a reviewer, I can inspect a readable diff for Docx
content, tables, selected media, and Mermaid source without noise from volatile
Feishu metadata.

- Given captured lark-cli fixtures for two semantic revisions, formatting-only
  variation, assets, unknown fields, malformed output, and an unsupported
  version,
- When canonicalization and `hubctl diff` run,
- Then semantic changes produce deterministic structured/unified diffs,
  formatting-only volatility is a no-op, assets have stable references,
  malformed/unsupported output is `ERROR`, and native Markdown uses the same
  internal diff contract.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/canonicalization tests/diff tests/adapters/test_lark_cli_contract.py
```

- Expected evidence: golden canonical snapshots, exact diff goldens,
  formatting-only `changed=false`, and explicit version/schema failure codes.

### AC-08 - Pre-push and Every Writer Enforce the Same Root Gate

Feature Story FS-08: As a maintainer, I can rely on pushes catching structural
errors, while an Agent using `--no-verify` still cannot write unsafe content to
Feishu.

- Given a valid branch, a malformed manifest, a broken table, and a secret
  canary,
- When pushes run with and without hook bypass and write entrypoints are called,
- Then valid pre-push succeeds, each invalid case blocks an ordinary push, hook
  exit status is preserved, `--no-verify` may bypass Git's hook but every writer
  independently rejects before its first remote call.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/hooks/test_pre_push.py tests/gate/test_writer_gate.py
```

- Expected evidence: local bare-remote push fixture, known-bad failures, exact
  exit codes, and zero fake remote writes after hook bypass.

### AC-09 - Local CI Records Honest PASS, RED, and ERROR Verdicts

Feature Story FS-09: As an operator, I can distinguish a real content/test
failure from runner infrastructure failure and know that no unjudged commit was
silently accepted.

- Given a local bare origin and three commits that respectively pass, fail a
  deterministic test, and trigger clone/install infrastructure failure,
- When the deployed-shape runner processes them,
- Then it uses a pristine clone, records one truthful `PASS`, `RED`, or `ERROR`
  JSONL verdict with command/exit/log identity, does not advance the cursor on
  unrecordable/unjudged work, and never calls Feishu.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/local_ci/test_runner_e2e.py tests/local_ci/test_verdict_query.py
```

- Expected evidence: all three verdicts, cursor assertions, pristine-clone
  identity, zero live-network calls, and a query CLI returning the exact verdict
  for each SHA.

### AC-10 - Periodic Sync Is Single-Writer, Recoverable, and Observable

Feature Story FS-10: As an operator, I can tell whether scheduled synchronization
is alive, which revision it last pushed, and why it stopped without inspecting a
Python process tree.

- Given remote-only changes, a concurrent invocation, fetch timeout, failed Git
  push, expired/refreshable auth, and a matching local-CI verdict,
- When the deployed `sync --once` job runs,
- Then only one writer owns the lock, timeouts/retries are bounded by config, the
  cursor advances only after successful Git push, failed cycles remain
  retryable, and `hubctl status --json/pretty` exposes last attempt, last success,
  current revision, CI verdict, conflicts, and stable error code.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/runtime/test_sync_once.py tests/runtime/test_locking.py tests/runtime/test_status.py
```

- Expected evidence: concurrent-lock rejection, retry without data loss after
  failed push, unchanged cursor on failure, advanced cursor on success, and
  status output matching append-only events.

### AC-11 - Project Import Is Curated and Downstream of Result Authority

Feature Story FS-11: As a `verl` researcher, I can generate a Hub candidate from
an approved design/result and verified links without publishing raw artifacts or
overstating a failed/incomplete run.

- Given released, diagnostic, W&B-pending, verified-W&B, verified-HF, secret,
  invalidated-design, and raw-artifact fixtures,
- When the `verl` importer scans and plans candidates,
- Then it selects only approved summaries/assets, preserves diagnostic and
  local/external authority labels, omits unverified links, rejects secrets and
  invalid current designs, never uploads weights/log directories, and performs
  no remote write until an explicit approved publish action.
- Verification command:

```bash
cd /data-1/code/verl/research/feishu-research-hub && python -m pytest -q tests/importers/test_verl_candidates.py tests/feature_stories/test_fs11_curated_seed.py
```

- Expected evidence: exact candidate goldens, zero remote calls in scan/plan,
  omitted W&B-pending URL, immutable verified HF URL, diagnostic banner, and
  explicit exclusion list for raw/protected paths.

### AC-12 - Independent End-to-End Acceptance Is Bound to Committed State

Feature Story FS-12: As the owner, I can make one human edit in an approved
disposable Feishu Docx and see an independently verified server pull, Git
commit/push, status update, and no-op rerun from the exact accepted commits.

- Given committed Hub and parent candidates, a private disposable Feishu folder
  and object explicitly approved for this test, valid lark-cli auth, selected
  identity mapping, and a passing deterministic gate,
- When an independent reviewer performs local publish, manual Feishu edit,
  periodic-job-shaped pull, Git push, status query, and no-op rerun,
- Then AC-01 through AC-11 are individually `PASS`, the final remote content,
  local snapshot, Git author/committer, audit revision/editor, CI verdict, and
  cursor agree, no unrelated remote object or parent dirty file changes, and
  acceptance is bound to Plan hash plus Hub/parent commit IDs.
- Verification commands:

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync
cd /data-1/code/verl/research/feishu-research-hub && python -m hubctl check --root . && python -m pytest -q
git -C /data-1/code/verl status --short
git -C /data-1/code/verl/research/feishu-research-hub status --short
```

- Expected evidence: reviewer-owned commands, AC matrix, disposable object
  token/revisions redacted from tracked files, exact commit bindings, clean Hub
  worktree, unchanged protected-path hashes, and `ACCEPTED` only if every AC is
  `PASS`.

## Feasibility Probes

- None: no acceptance criterion declares an absolute numeric performance,
  latency, throughput, memory, disk, polling-frequency, or resource budget.
- Timeouts, retry counts, polling intervals, and retention durations are
  implementation configuration, not frozen acceptance budgets. They must be
  selected after measurement in the deployment host and recorded in runtime
  config without amending these ACs unless they become part of the definition of
  done.

## Milestones

The architecture and acceptance criteria are frozen. This Plan revision changes
only execution state, remaining order, and authorization handling. Completed
milestones are evidence-bearing history and must not be re-executed.

1. **Contract and fake harness — COMPLETED**: schemas, fixtures, Feature Stories,
   canaries, and deterministic gate were implemented and reviewed.
2. **Core repository implementation — COMPLETED**: `hubctl`, sync semantics,
   conflict retention, attribution, catalogs, tests, and pre-push gate were
   implemented and reviewed.
3. **Private repository and parent submodule bootstrap — COMPLETED**: the private
   Hub exists, reviewed history was pushed, and the parent submodule was created.
4. **Disposable Feishu capability probe — COMPLETED**: the approved disposable
   object envelope exercised the required Docx/Markdown and failure behaviors;
   live deletion remains excluded.
5. **Project candidate integration — COMPLETED**: the read-only `verl` importer
   and authority/privacy gates passed independent review; no real seed batch was
   published.
6. **Local CI and scheduled runtime implementation — COMPLETED**: tracked
   launchers and deployed-shape behavior passed review. The original PM2 jobs are
   not currently running; service restart belongs to Remaining Step R2 only if
   final acceptance needs the deployed path.
7. **Milestone 7 repairs through conflict retention — COMPLETED**: binding
   privacy, full-store preflight, current lark auth envelopes, remote-wins state,
   and conflict snapshot reachability were repaired. Hub `origin/main` is
   `88ac17c2181d22f1e33373c54763ea8e6351bf74`; the conflict-retention convergence
   review returned PASS.
8. **Remaining Step R1 — reconcile Plan/runtime state**: append the Plan v3
   amendment, record the already completed convergence PASS for `F-M7-R04`, close
   that finding, and obtain one independent execution-state re-entry review. Do
   not repeat Milestones 1-7 or prior live writes.
9. **Remaining Step R2 — final acceptance only**: from clean Hub and parent
   candidates, run the reviewer-owned deterministic gate and only the minimum
   still-missing disposable live/deployed evidence. Reuse valid prior evidence;
   do not repeat a passed live operation merely to reproduce history. Complete
   `acceptance.md`, append acceptance/completion events, and validate runtime.
10. **Remaining Step R3 — finalize parent pointer and records**: after acceptance,
    commit the parent gitlink at the accepted Hub commit plus this Goal's tracked
    Plan/ledger/review/acceptance artifacts, then fast-forward push the current
    parent branch. No unrelated dirty parent files enter that commit.

The only execution order now is `R1 -> R2 -> R3`. Earlier milestones remain
completed unless fresh evidence proves a regression relevant to final acceptance.

## Authorization Policy

- This user request starts the remaining execution. Default is
  `DEFAULT_AUTHORIZED`: every Plan-defined exact-target action in R1-R3 is
  authorized unless explicitly `HOLD` or `DENIED`; silence means authorized.
- Whole-Goal authorization: `AUTHORIZED` for R1-R3 and ordinary in-scope repairs,
  reviews, commits, private fast-forward pushes, disposable-object read/update,
  local runtime restart, and verification.
- Milestone overrides: `None`.
- `RISK_NOTICE`: append `RISK_NOTICE_RECORDED` with risk, mitigation, and exact
  target, then continue; a reported risk is not an approval request.
- `PREAUTHORIZED_STOP_ACTION`: existing recorded decisions continue to cover the
  exact private Hub, parent submodule path, approved disposable FS03 objects,
  local runtime paths, normal private pushes, and final acceptance operation set.
  They do not authorize deletion, trash, public sharing, permission expansion,
  owner transfer, force/history rewrite, unrelated/non-disposable Feishu objects,
  credential exposure, or real seed publication.
- `USER_DECISION` is allowed only for an uncovered stop class: deletion or another
  hard-to-reverse action; exposure/permission expansion; owner transfer; history
  rewrite; non-disposable live-object access; credential/sensitive-data exposure;
  tool-enforced current-turn confirmation; new outcome/out-of-scope work; or an
  unresolved `CONTRADICTION`/`AC_CHANGE`. A changed target or broader boundary
  requires a new decision. Milestone boundaries, reviews, failures, retries,
  ordinary repairs, and private fast-forward pushes do not.

## Runtime Contract

- Implementation starts only after Plan status is `READY` and the user
  explicitly starts this Goal. Plan preparation/review does not authorize
  implementation.
- Completed decisions D-01 through D-07 and the later exact-target Milestone 7
  decisions are historical evidence, not gates to ask again. The Authorization
  Policy above governs the remaining R1-R3 execution.
- `AUTO_ADVANCE` covers plan/runtime validation, reviewer prompt construction,
  fixture work, local bare-remote operations, candidate generation, finding
  classification, authorized `IN_SCOPE` fixes, due reviews, and milestone
  transitions after a passing review/validator.
- Every new finding is appended to `findings.jsonl` and classified before
  action. Run `validate-runtime` after classification, review, and milestone
  transition and before acceptance/completion.
- `IN_SCOPE` fixes existing AC behavior autonomously. `DEFERRED` is recorded and
  not implemented. `CONTRADICTION` and `AC_CHANGE` stop for Plan amendment and a
  fresh independent Plan review.
- If two related implementation-review rounds leave the same finding open,
  perform a convergence review before a third ordinary fix round.
- Use scratch only under
  `/data-1/tmp/verl_agent_scratch/feishu-research-hub-sync/`; never use the
  parent repo root as scratch.
- Long-lived CI/sync installation must place executables/config outside a
  disposable clone and prove its deployed command/provenance. Persistent
  processes use PM2 only; no systemd, cron, container, public listener, or
  inbound webhook is added.
- Remote-writing commands use user identity only for the exact resources in the
  recorded decision. lark-cli confirmation exit code 10 is never converted to
  `--yes` without current-turn explicit approval.
- Polling metadata code `970005`, empty metadata, fetch/history failure, missing
  permission, and type mismatch are suspected absence only. They must fail
  closed and may not produce a tombstone. Live deletion tombstoning requires
  the deletion-exclusive `drive.file.trashed_v1` event and is deferred to a
  later Goal with its own subscription/runtime decisions.
- Stop on credential exposure, broader remote permission need, unapproved shared
  mutation, protected dirty-file overlap, invalid runtime transition, or the
  need for a second independently useful outcome.
- The implementer cannot self-review or self-accept.

## Progression Policy

- `AUTO_ADVANCE`: every action covered by the Authorization Policy proceeds
  immediately, including R1-R3 transitions, risk-noticed actions, validators,
  reviewer construction, evidence reuse, in-scope repair, disposable exact-target
  operations, local runtime restart, commits, and private fast-forward pushes.
- `USER_DECISION`: only an uncovered stop-class action pauses dependent work.
  Append a structured v2 request with `stop_category`, exact `target`,
  `operation`, `risk`, and `decision_needed`. Do not use it for a risk notice,
  reviewer rejection, validator failure, retry, or milestone boundary.

## Reviewer Contract

- Plan Review checks that this is one capability rather than a hidden Obsidian
  clone; that every live dependency has a fake acceptance path plus an explicit
  decision-bound disposable probe; and that Feature Stories prove failures, not
  only happy paths.
- Milestone reviewers inspect only applicable ACs, run the frozen commands
  themselves, verify known-bad canaries really fail, and audit zero unexpected
  remote calls or protected-path changes.
- The reviewer must verify `remote-wins` never means “discard local”; conflict
  evidence must remain reachable and byte-recoverable.
- The reviewer must inspect Git `Author`, `Committer`, trailers, editor/revision
  bindings, and unknown-editor behavior, not accept prose claims.
- The reviewer distinguishes deterministic CI evidence from disposable live
  Feishu capability evidence. Live services cannot substitute for fake failure
  coverage, and fakes cannot substitute for the final explicitly authorized
  end-to-end story.
- A reviewer may add `DEFERRED_SUGGESTION`; an opinion outside the frozen ACs is
  non-blocking. A required completion-definition change is
  `CONTRACT_CONTRADICTION` and the reviewer must not amend or implement it.
- Final acceptance reports each AC as `PASS`, `FAIL`, or `WEAKENED`, includes
  exact commands and relevant output, checks for skipped/loosened/deleted tests,
  binds Plan/Hub/parent commits, and returns `ACCEPTED` only when all ACs are
  `PASS`.

## Verification Commands

- Plan:
  `goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync`
- Runtime:
  `goal-plan-runtime validate-runtime docs/joint_training/goals/feishu-research-hub-sync`
- Plan hash:
  `sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md`
- Architecture cross-reference:
  `rg -n 'feishu-research-hub-sync|Feishu Research Hub' docs/joint_training/plans/active/feishu_cli_experiment_knowledge_sync.md docs/joint_training/plans/active/README.md CLAUDE.md AGENTS.md`
- Protected status:
  `git -C /data-1/code/verl status --short`

## Deferred Follow-ups

- A localhost-only read-only Web dashboard over `hubctl status`.
- Win11/Obsidian automatic ingestion and Windows scheduler deployment.
- Event/webhook-driven Feishu sync after polling proves reliable.
- Live deletion-event subscription/consumption and
  `drive.file.trashed_v1`-backed tombstone ingestion.
- Sheets, Base, Slides, Minutes, Wiki, and Whiteboard writer adapters beyond
  individually approved, capability-probed publication needs.
- Semantic/CRDT merge of concurrent prose.
- Public sharing automation, external-tenant permissions, and broad collaborator
  lifecycle management.
- High-frequency W&B telemetry mirroring and full paper-PDF distribution.
