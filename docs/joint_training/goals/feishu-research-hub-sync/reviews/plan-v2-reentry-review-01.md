# Plan v2 Re-entry Review 01

## Review identity

- Reviewer: fresh independent Plan re-entry reviewer (`/root/plan_v2_reentry_review`)
- Requested model: GPT-5.5, medium. The routing layer did not expose that
  override; this review used the available model.
- Review type: Plan Re-entry Review
- Frozen Plan version / SHA-256: `2` /
  `ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba`
- Candidate Hub commit: `90204fc94eb66c3d299fa5ec95b5c49c3c3b0b72`
- Scope: AC-06 and its Scope, Architecture Contract, Milestone 4, Runtime
  Contract, deferred-follow-up, and D-05 through D-07 cross-references.
- External/shared actions: none. This review made no Feishu, GitHub, PM2,
  submodule, permission, or ledger mutation.

## Overall verdict: READY

Plan v2 is a bounded, user-approved correction to the proved external-adapter
contradiction. It preserves the single Hub-sync outcome and all ACs except the
specific live AC-06 deletion-evidence boundary that the user approved at
`D-AC06-01`. It does not weaken safety: every polling ambiguity, including
`970005`, empty metadata, permission/type errors, and fetch/history failures,
must leave entry bytes, content, history, and sync state unchanged. Trusted
confirmed-deletion fixtures still have to prove recoverable tombstones with no
remote create/delete/recreate write. The only deferred capability is live
`drive.file.trashed_v1` subscription/consumption and resulting live deletion
tombstoning; no event consumer, subscription, runtime service, or shared
mutation has been smuggled into the current Goal.

## Per-AC verdict

| AC | Verdict | Reviewer evidence |
| --- | --- | --- |
| AC-06 | PASS | The revised Given/When/Then explicitly covers rename, in-root move, out-of-root detach, and fail-closed polling ambiguity; it retains trusted confirmed-deletion fixture tombstones. It prohibits automatic remote creation/deletion/recreation, excludes live deletion evidence consumption, and limits disposable live evidence to non-destructive structure changes. |

## Commands and evidence

```bash
cd /data-1/code/verl
goal-plan-runtime validate-plan docs/joint_training/goals/feishu-research-hub-sync
# PASS

sha256sum docs/joint_training/goals/feishu-research-hub-sync/plan.md
# ed773e5841c5e9b2a397dfec49eaa25604d46037050e3fe34bfeae1b8e0f3eba
# docs/joint_training/goals/feishu-research-hub-sync/plan.md

rg -n -C 6 'AC-06|D-AC06-01|trashed_v1|deletion|tombstone|970005|Milestone 4|D-0[5-7]|Feasibility Probes|Runtime Contract|Deferred Follow-ups' \\
  docs/joint_training/goals/feishu-research-hub-sync/plan.md
```

The plan has no absolute numeric resource or performance acceptance budget;
its Feasibility Probes section correctly declares `None`. Runtime records show
that `D-AC06-01` was requested with both bounded and event-subsystem options,
then recorded as approved for the bounded option before `PLAN_AMENDED` v2 was
bound to the reviewed SHA. The independent convergence report establishes the
technical premise: `drive.metas.batch_query` `970005` is not deletion-exclusive,
whereas `drive.file.trashed_v1` is deletion-exclusive but introduces a separate
subscription/consumer/runtime capability. The architecture document has the
same boundary: polling absence retains the entry unchanged; trusted fixtures
cover tombstone retention; any event-backed live tombstone belongs to later
work.

## Cross-reference audit

- **Scope and Architecture Contract:** consistently exclude live deletion
  ingestion while preserving fixture tombstones and fail-closed state
  preservation.
- **AC-06:** requires no mutation on every ambiguous/error polling path and
  requires a readable retained fixture tombstone. It no longer claims a live
  deletion proof.
- **Milestone 4:** permits only disposable create/update/move and explicitly
  excludes live delete or deletion-event work; fixture evidence remains
  required.
- **Runtime Contract:** retains D-05 seed publication, D-06 scheduler/PM2, and
  D-07 final live acceptance gates unchanged. It separately requires a future
  decision for any high-risk delete and reserves event-backed live tombstones
  for a later Goal.
- **Deferred Follow-ups:** names exactly `drive.file.trashed_v1` subscription,
  consumption, and tombstone ingestion. It does not defer rename/move/detach,
  ambiguous-absence protection, or the fixture tombstone contract.

## Blocking in-scope defects

None.

## Deferred suggestions

None. The possible future event adapter is already correctly deferred rather
than being a condition for re-entering this Plan.

## Contract contradictions

None in Plan v2. The Plan explicitly resolves, rather than conceals, the prior
Plan v1 contradiction.

## Single most likely weakness in this review

This is a re-entry Plan review, not a rerun of Hub behavioural tests. The next
Milestone 4 review must independently execute the AC-06 fixture and disposable
live non-destructive evidence, including byte-for-byte no-mutation canaries,
before treating the amended requirement as implemented.
