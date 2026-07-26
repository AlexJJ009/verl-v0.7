Convergence review before a third ordinary repair round for F-M7-R01.

The two related review rounds are final-acceptance-01 and
milestone-7-binding-fix-review-01. Determine whether the recurrence is caused
by duplicated sources of truth, experiment-specific policy in a shared layer,
reviewer scope expansion, or a second independently useful outcome. Confirm
whether `.hub/live-bindings.json` is the single runtime-private authority for
live root/object bindings and whether the remaining correction is an IN_SCOPE
architectural fix under AC-10 and AC-12 without a Plan amendment.

Review the current candidate only; do not implement, push, call Feishu, or edit
the append-only ledgers. Require an explicit convergence disposition for each
of these cases in live mode:

- missing binding store;
- empty `entries` mapping;
- malformed binding document or entry shape;
- missing or empty object token;
- fixture-shaped token in the live store;
- binding for an unknown entry ID;
- a configured managed entry with no matching private binding.

The expected safety property is fail closed with stable nonzero binding errors,
no adapter call, no Git commit/push, no cursor advance, and a truthful ERROR
status from `scripts/sync_once.sh`. Do not require object/folder tokens or
private URLs in tracked evidence. Preserve AC-04: revision/editor identifiers
remain allowed and required in structured Git trailers/private audit, while
their concrete values remain redacted from tracked review and acceptance
reports.
