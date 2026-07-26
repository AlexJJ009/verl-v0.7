# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Milestone Review`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/feishu-research-hub-sync`
- Frozen Plan version: `2`
- Current milestone: `Milestone 7`
- Base commit: `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`
- Candidate commit: `e354f950338bed8750e77653395efec4dc473907`
- Applicable ACs: `AC-03, AC-10, AC-12`

## Required Verification

cd /data-1/code/verl/research/feishu-research-hub && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q tests/gate/test_live_binding_privacy.py && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m pytest -q && PYTHONPATH=src /data-1/feishu-research-hub-runtime/repo/.venv/bin/python -m hubctl check --root . && git diff --check 7078fdd..e354f95 && git show -s --format=fuller e354f95

## Additional Task-Specific Focus

Review only the IN_SCOPE repair for `F-M7-R02` at local Hub candidate `e354f950338bed8750e77653395efec4dc473907`, based on `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`.

Required behavior:

- A guarded `binding-refresh` path may update only runtime-private CAS revision/editor metadata after fetching the existing approved object.
- It must require the runtime-private token, run the normal writer/root gate and adapter-version check, and emit no token/revision/editor value.
- It may refresh only when canonical remote content equals the recorded common edition. A remote semantic change must fail `E_BINDING_DIVERGED` without changing binding state, tracked files, common snapshot, Git, or Feishu.
- The command itself performs no Feishu write; it is a read plus protected local metadata refresh.
- Verify the positive and negative tests genuinely exercise those branches and that the complete suite/root gate remain green.
- Verify commit attribution is `Codex Agent <codex-agent@example.invalid>` with `Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>`.
- Do not call Feishu, push, modify ledgers, update the parent gitlink, or implement fixes. Use `PASS` only if the guarded refresh closes the stale-binding readiness defect without weakening CAS or privacy.

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
