# Milestone 3 Independent Review

## Review Identity

- Review type: Milestone 3 independent review.
- Goal: `stage123-execution-readiness`.
- Frozen Plan version: `8`.
- Plan SHA256: `fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67`.
- Base commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`.
- Candidate commit reviewed: `62f6c5c8d34007d71b5269b795e8f7c72db92752`.
- Recipe gitlink reviewed: `eeadc66e13592708b7870a93312b5ab9eb82c4a6`.
- Expected implementation tree SHA256: `3ff13ef9bffb2a87984cfa68284885a2cb791fafb8e4d60bd331cc07c5cad3f7`.
- Reviewer: independent Codex reviewer; no production or test files modified.

## Overall Verdict

PASS.

Milestone 3 satisfies the committed production-identity gate: `HEAD` equals the candidate commit, covered production roots and recipe are clean, the recipe gitlink matches the expected commit, the implementation tree recomputes byte-for-byte to the expected SHA256, focused behavior tests pass, protected assets match the captured baseline, and I found no active Stage123 training/GPU job or external publication action.

## Per-AC Verdict Table

| AC | Milestone 3 verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Current implementation identity recomputed to `3ff13ef9bffb2a87984cfa68284885a2cb791fafb8e4d60bd331cc07c5cad3f7`; Plan v8 and runtime validators pass. |
| AC-02 | PASS | Focused preflight/admission wrapper tests passed; committed recipe preflight boundary is clean at gitlink `eeadc66e13592708b7870a93312b5ab9eb82c4a6`. |
| AC-03 | PASS | Focused end-to-end tests passed; candidate diff keeps the primary run-set behavior under tested manifest/queue surfaces. |
| AC-04 | PASS | Focused end-to-end tests passed; no active queue/training process was observed after review commands. |
| AC-05 | PASS | Focused end-to-end tests passed; no duplicate live execution authority was observed in this milestone review scope. |
| AC-06 | PASS | `test_execution_results.py`, `test_stage123_admission_bundle.py`, and `test_stage123_end_to_end.py` passed. |
| AC-07 | PASS | Required focused tests passed; no active Stage123 tmux, Ray, W&B sync, HF upload, git push, rsync, or scp process found after checks. |
| AC-08 | PASS | Candidate commit is clean for covered roots; current checkout identity and protected baseline checks passed before launch rendering can be authorized in later milestones. |

## Commands And Evidence

### Repository And Gitlink State

Command:

```bash
git rev-parse HEAD
git rev-parse 62f6c5c8d34007d71b5269b795e8f7c72db92752
git rev-parse 29089a6c1c63d017384b1ff09eba9821d10a2a7a
git status --short -- config/experiment_execution scripts verl recipe
git -C recipe status --short
git ls-tree 62f6c5c8d34007d71b5269b795e8f7c72db92752 recipe
git -C recipe rev-parse HEAD
git status --short -- .claude/skills/experiment-registry docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md test_data
```

Relevant output:

```text
HEAD=62f6c5c8d34007d71b5269b795e8f7c72db92752
candidate=62f6c5c8d34007d71b5269b795e8f7c72db92752
base=29089a6c1c63d017384b1ff09eba9821d10a2a7a

-- git status scoped --

-- recipe status --

-- recipe gitlink in candidate --
160000 commit eeadc66e13592708b7870a93312b5ab9eb82c4a6	recipe

-- recipe HEAD --
eeadc66e13592708b7870a93312b5ab9eb82c4a6

-- protected status --
?? .claude/skills/experiment-registry
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/
```

Interpretation: covered roots `config/experiment_execution`, `scripts`, `verl`, and `recipe` are clean; protected assets remain in their accepted untracked baseline state.

### Plan And Runtime Validation

Command:

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness
sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md
```

Relevant output:

```text
PASS
{
  "current_milestone": "Milestone 3",
  "goal_status": "ACTIVE",
  "pending_user_decisions": [],
  "plan_status": "READY",
  "plan_version": 8
}
fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67  docs/joint_training/goals/stage123-execution-readiness/plan.md
```

### Candidate Diff And Recipe Delta

Command:

```bash
git diff --name-status 29089a6c1c63d017384b1ff09eba9821d10a2a7a 62f6c5c8d34007d71b5269b795e8f7c72db92752 -- config/experiment_execution scripts verl recipe tests docs/joint_training/guides docs/joint_training/goals/stage123-execution-readiness
old_recipe=$(git ls-tree 29089a6c1c63d017384b1ff09eba9821d10a2a7a recipe | awk '{print $3}')
new_recipe=$(git ls-tree 62f6c5c8d34007d71b5269b795e8f7c72db92752 recipe | awk '{print $3}')
git -C recipe diff --name-status "$old_recipe" "$new_recipe"
```

Relevant output:

```text
old_recipe=888d8e1a979070013ffc9ccca401ea17c73f26d6
new_recipe=eeadc66e13592708b7870a93312b5ab9eb82c4a6

-- recipe changed names old..new --
M	on_policy_wdl_sft/code_task/stage123_preflight.py
```

Interpretation: the recipe gitlink delta is explicit and narrow; the superproject candidate binds the expected recipe commit.

### Implementation Identity Byte-For-Byte Check

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
sha256sum docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
cat config/experiment_execution/stage123_implementation_boundary_v1.json
cat docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Relevant output:

```text
{"kind":"git_tree","path":"config/experiment_execution","tree_sha1":"0d793f3418cd167e977b1b377c3ae3bc5dce1035"}
{"gitlink_commit":"eeadc66e13592708b7870a93312b5ab9eb82c4a6","kind":"gitlink","mode":"160000","path":"recipe"}
{"kind":"git_tree","path":"scripts","tree_sha1":"fdf293870863c49c82201a6c4712e67854a011dc"}
{"kind":"git_tree","path":"verl","tree_sha1":"40deac7dc6da65ef470c5e42c75fb2fd35b9335a"}
{"implementation_tree_sha256": "3ff13ef9bffb2a87984cfa68284885a2cb791fafb8e4d60bd331cc07c5cad3f7"}

3ff13ef9bffb2a87984cfa68284885a2cb791fafb8e4d60bd331cc07c5cad3f7  docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Interpretation: the canonical implementation identity matches the expected SHA256 and includes the complete configured boundary: `config/experiment_execution`, `scripts`, `verl`, and the entire `recipe` gitlink commit.

### Focused Behavior Tests

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_stage123_end_to_end.py
```

Relevant output:

```text
17 passed in 33.46s
```

### Protected Assets

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
git status --short -- .claude/skills/experiment-registry docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md test_data
```

Relevant output:

```text
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}

?? .claude/skills/experiment-registry
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/
```

Interpretation: protected content matches the captured baseline; the untracked statuses are the Plan-declared accepted baseline, not new staging.

### No Training, GPU Job, Or External Publication

Command:

```bash
tmux list-sessions 2>/dev/null | grep -Ei 'stage123|readiness|calibration|train' || true
ps -eo pid,ppid,stat,etime,cmd | grep -Ei 'stage123|readiness|calibration|experiment_execution_core|run_calibration|ray::|raylet|wandb sync|huggingface-cli|git push|rsync|scp' | grep -v grep || true
```

Relevant output after verification commands completed:

```text
-- current stage123/ray/train processes after checks --

-- current stage123 tmux sessions after checks --
```

Interpretation: I found no active Stage123 training tmux session, Ray training process, calibration execution-core process, W&B sync, Hugging Face upload, git push, rsync, or scp process after the required checks. The required wrapper command starts a Docker container with the project-standard `--gpus all` flag, but the commands executed here were identity/protected-asset scripts and CPU pytest checks; I did not run calibration, formal training, launch rendering, W&B, HF, GitHub publication, or network publication commands.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

The project wrapper exposes GPUs to the Docker container via `--gpus all` even for CPU-only Python checks. I verified that no Stage123/Ray/training process or tmux session remained active afterward and did not run calibration or formal training, but I did not instrument Docker/NVIDIA internals to prove that no CUDA context was briefly initialized by imported libraries during pytest collection.
