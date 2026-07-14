# Milestone 1 Independent Review

## Review Identity

- Reviewer: Codex independent Milestone 1 reviewer, requested model `GPT-5.5`, reasoning effort `medium`
- Review type: Milestone Review
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Reviewed Plan version: `8`
- Reviewed Plan SHA256: `fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67`
- Reviewed commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Review prompt: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-1-review-prompt.md`
- Repository mutation by reviewer: only this report file

## Overall Verdict

`FAIL`.

The frozen validators and protected-asset comparison pass, and Plan v8 is hash-bound to an independent `READY` review. However, I cannot verify the Milestone 1 requirement that the protected baseline predates production repair. The current worktree already contains uncommitted production/test repair changes, and filesystem evidence shows those production/test edits predate the protected baseline file. The append-only ledgers also lack a baseline-capture event that would establish ordering before those edits.

This is not a code-behavior failure in the protected fingerprint comparator: the comparator passes cleanly. The failure is an evidence-ordering defect for Milestone 1.

## Per-AC Verdict Table

| Applicable AC | Verdict | Rationale |
| --- | --- | --- |
| Plan v8 lifecycle gate | `PASS` | `validate-plan` passes; `validate-runtime` reports Plan version `8`, Plan status `READY`, current milestone `Milestone 1`, and latest review `plan-v8-review-01` with matching SHA256 and verdict `READY`. |
| Plan v8 hash binding | `PASS` | `sha256sum plan.md` returns `fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67`, matching runtime seq `32` and seq `33`. |
| Append-only runtime validity | `PASS` | `validate-runtime` exits `0`; runtime seq `34` starts Milestone 1 with GPU requalification excluded pending `USER_DECISION`. |
| Protected baseline compares cleanly | `PASS` | The frozen compare command exits `0` with baseline SHA256 `c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207`; a fresh temp capture produced identical bytes. |
| Protected baseline predates production repair | `FAIL` | Current file mtimes show `scripts/execution_results.py` and related tests were edited around `2026-07-14 00:41-00:42 -0700`, while `protected-assets-baseline.jsonl` was written at `2026-07-14 02:01:20 -0700`; no append-only event proves an earlier canonical baseline capture before the repair edits. |
| No GPU/training/external-service action for Milestone 1 | `PASS` | Runtime records Milestone 4 GPU work as excluded; `tmux ls`, process scan, and Docker container scan found no Stage123/requalification/training execution. Review commands used only the frozen validators and protected fingerprint checks. |

## Commands And Evidence

### Goal validators

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness
```

Output:

```text
PASS
```

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness
```

Relevant output:

```json
{
  "current_milestone": "Milestone 1",
  "goal_status": "ACTIVE",
  "plan_status": "READY",
  "plan_version": 8,
  "latest_review": {
    "candidate_commit": "29089a6c1c63d017384b1ff09eba9821d10a2a7a",
    "event": "PLAN_REVIEWED",
    "plan_sha256": "fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67",
    "plan_version": 8,
    "review_id": "plan-v8-review-01",
    "verdict": "READY"
  },
  "pending_user_decisions": []
}
```

### Protected baseline compare

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Output:

```json
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

Fresh temp capture check:

```bash
tmp=$(mktemp /data-1/tmp/verl_agent_scratch/protected-baseline-review.XXXXXX.jsonl)
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py capture --repo-root /data-1/code/verl --path .claude/skills/experiment-registry --path docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md --path test_data --output "$tmp"
sha256sum "$tmp" docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
cmp -s "$tmp" docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
rm -f "$tmp"
```

Relevant output:

```text
c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207  /data-1/tmp/verl_agent_scratch/protected-baseline-review.vya2XZ.jsonl
c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207  docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
cmp_exit:0
```

### Plan and baseline identity

```bash
sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Output:

```text
fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67  docs/joint_training/goals/stage123-execution-readiness/plan.md
c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207  docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

### Append-only evidence

Runtime seq `32`:

```json
{"actor":"Codex plan author","event":"PLAN_AMENDED","plan_sha256":"fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67","plan_version":8,"reason":"Remove stale Plan v5 authorization contradiction and clarify the current Plan v8 binding; repair pre-Milestone protected-root comparison scaffolding","seq":32,"time":"2026-07-14T09:24:25.664297Z"}
```

Runtime seq `33`:

```json
{"candidate_commit":"29089a6c1c63d017384b1ff09eba9821d10a2a7a","event":"PLAN_REVIEWED","plan_sha256":"fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67","plan_version":8,"prompt":"reviews/plan-v8-review-prompt.md","review_id":"plan-v8-review-01","reviewer":"Socrates fresh independent GPT-5.5 medium","seq":33,"summary":"Plan v8 is an internally consistent and implementable pre-implementation contract; numeric GPU budget is backed by measured calibration evidence; protected baseline compare passes; Milestone 4 remains a separate USER_DECISION gate","time":"2026-07-14T09:32:43.915007Z","verdict":"READY"}
```

Runtime seq `34`:

```json
{"actor":"Codex implementer","authorization":"user-started autonomous Stage123 Execution Readiness CPU Milestones 1-3 envelope; Milestone 4 GPU requalification excluded pending USER_DECISION","event":"MILESTONE_STARTED","milestone":"Milestone 1","seq":34,"time":"2026-07-14T09:37:03.311658Z"}
```

Findings seq `102`:

```json
{"event":"FINDING_CLOSED","finding_id":"F-RD-V7-02","resolution":"Protected-root reconstruction fixed; reviewer-owned explicit-wrapper compare passes with baseline SHA256 c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207","seq":102,"time":"2026-07-14T09:32:44.003044Z"}
```

### Ordering defect evidence

```bash
stat -c '%y %n' docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl scripts/execution_results.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py
```

Output:

```text
2026-07-14 02:01:20.578592676 -0700 docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
2026-07-14 00:41:30.235994428 -0700 scripts/execution_results.py
2026-07-14 00:42:18.655170369 -0700 tests/experiment_workflow/test_execution_results.py
2026-07-14 00:42:02.279449068 -0700 tests/experiment_workflow/test_stage123_admission_bundle.py
```

```bash
git status --short
```

Relevant output:

```text
 M scripts/execution_results.py
 M tests/experiment_workflow/test_execution_results.py
?? tests/experiment_workflow/test_stage123_admission_bundle.py
?? docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
?? docs/joint_training/goals/stage123-execution-readiness/tools/
```

The protected assets themselves are stable and pre-existing:

```text
2026-07-10 02:19:04.700106946 -0700 symbolic link .claude/skills/experiment-registry
2026-07-11 06:45:58.793774260 -0700 regular file docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
2026-07-12 19:19:20.144365575 -0700 directory test_data
2026-07-12 19:19:20.145365558 -0700 regular file test_data/test_video.json
```

But the review prompt requires the protected baseline, not only the protected assets, to predate production repair.

### No GPU/training/external-service action

```bash
tmux ls
ps -eo pid,ppid,stat,etime,cmd | rg -i 'stage123|readiness_requalification|CALIBRATION_STATE_ROOT|render_calibration_probe|experiment_execution_core|on_policy_wdl_sft.*stage123' || true
docker ps --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}} {{.Command}}' | rg -i 'stage123|verl|ray|calibration|qwen|training|readiness' || true
```

Relevant output:

```text
docker-export: 1 windows (created Fri Apr  3 22:14:38 2026)
harness: 3 windows (created Fri Apr  3 09:44:29 2026)
```

The process scan only matched the reviewer command itself, and the Docker container scan returned no matching containers.

## Blocking In-Scope Defects

- `F-M1-01`: The protected baseline ordering is not independently proven and appears late relative to production/test repair edits. The baseline file has no embedded generation timestamp or append-only `BASELINE_CAPTURED` event, and its filesystem mtime is later than the current repair edits in `scripts/execution_results.py` and related tests.

## Deferred Suggestions

- Add an append-only baseline-capture event in future goals, binding command, output path, SHA256, protected paths, actor, and timestamp. This is not a production-code requirement; it is evidence hygiene for lifecycle reviews.

## Contract Contradictions

None. The Plan v8 contract itself is internally consistent and independently reviewed `READY`; the failure is that current Milestone 1 evidence does not satisfy one requested ordering property.

## Single Most Likely Weakness In This Review

The ordering finding relies partly on filesystem mtimes because the ledger does not record a canonical baseline-capture event. Filesystem mtimes can be imperfect evidence, but here they are the only available independent ordering signal and they conflict with the prompt's baseline-before-repair requirement.
