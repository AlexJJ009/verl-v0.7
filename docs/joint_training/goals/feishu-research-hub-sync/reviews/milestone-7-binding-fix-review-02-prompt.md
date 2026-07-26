# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `66d6f5aaf804d17c0f37f312f28277dea88a2c98`
- Candidate commit: `82cabe1c361ba7141762c50b5d4da6a4bd0175f7`
- Applicable ACs: `AC-03, AC-04, AC-09, AC-10, AC-12`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && .venv/bin/python -m pytest -q tests/runtime/test_sync_once.py tests/runtime/test_launchers.py tests/runtime/test_status.py tests/gate/test_live_binding_privacy.py tests/feature_stories/test_fs03_local_publish.py tests/feature_stories/test_fs04_remote_edit_pull.py tests/git/test_attribution.py && .venv/bin/python -m pytest -q && .venv/bin/python -m hubctl check --root . && git diff --check 66d6f5aaf804d17c0f37f312f28277dea88a2c98 82cabe1c361ba7141762c50b5d4da6a4bd0175f7; goal-plan-runtime validate-runtime /data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync

## Additional Task-Specific Focus

Review only the convergence-approved F-M7-R01 correction at candidate
82cabe1c361ba7141762c50b5d4da6a4bd0175f7. The independent convergence
report is `reviews/milestone-7-binding-convergence-01.md`.

Verify `.hub/live-bindings.json` is now the single runtime-private authority
for live object bindings. Independently exercise the full negative matrix:
missing store, empty entries, malformed document/schema/entry, missing or empty
token, fixture-shaped live token, unknown entry binding, and configured entry
without a valid private binding. Each must fail before adapter use, Git commit
or push, or cursor advancement; `sync_once` and the tracked launcher must record
and expose a stable terminal binding ERROR. Also run the positive valid-binding
no-op control and prove direct publish/pull privacy plus AC-04
Feishu-Revision/Feishu-Editor-Ids trailers remain intact.

Review only. Do not implement, push, call Feishu, modify the parent gitlink,
edit deployed PM2 state, or expose any real object/folder token, private URL,
revision, or editor value in tracked evidence.


## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.
9. For a Plan review, reject `READY` when any AC declares an absolute numeric performance or resource budget that has no recorded feasibility probe in the Plan's `Feasibility Probes` section, or whose budget contradicts the probe's measured floor.
10. When your only blocking findings are purely mechanical (formatting, patch context offsets, artifact or directory placement) with no behavioral or contract impact, say so explicitly and offer a light same-reviewer re-verification scoped to those findings instead of demanding a fresh full round.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
