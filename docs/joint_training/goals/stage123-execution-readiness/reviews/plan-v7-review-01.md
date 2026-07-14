# Plan v7 Independent Review 01

## Review Identity

- Reviewer: Codex independent Plan Reviewer, requested model `GPT-5.5`, reasoning effort `medium`
- Review type: Plan Review
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Frozen Plan version reviewed: `7`
- Plan SHA256 reviewed: `4796cce83a89c9c34928dd5735decd823f78c53714d981d0b0ed111bdf7a1faa`
- Candidate commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Candidate worktree: dirty; covered implementation roots include modified `scripts/execution_results.py` and modified `recipe` submodule content.

## Overall Verdict

`NOT_READY`.

Plan v7 is not implementable as frozen. The blockers are behavioral/contractual, not merely mechanical. The largest issues are: missing required producer scripts/tests, host/container preflight boundary not implemented, protected-baseline admission path not implemented and its compare probe fails, admission validation does not use the single current-checkout validator on every path, strict acceptance schema is not implemented, and AC-01 still contains a stale `Plan v5` authorization binding.

## Per-AC Verdict Table

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | `FAIL` | The Plan references missing `scripts/render_calibration_result.py`; the implementation-tree probe fails on the dirty covered tree; AC-01 line 146 requires `Plan v5` hash while v7 progression requires Plan v7. |
| AC-02 | `FAIL` | The required `scripts/stage123_host_facts.sh` does not exist; `stage123_preflight.py` has no `--host-facts` argument and still invokes `tmux` in container code; mandatory wrapper integration test file is missing. |
| AC-03 | `PASS` | Existing focused tests passed, and the reviewed preflight shard constants are exactly the two FRAC25 Stage2/Stage3 files. |
| AC-04 | `PASS` | Existing queue/core lifecycle tests passed in the reviewer-owned run. |
| AC-05 | `PASS` | Existing queue/monitor contract and end-to-end tests passed in the reviewer-owned run. |
| AC-06 | `FAIL` | `scripts/execution_results.py admission validate` lacks `--protected-baseline`; protected-baseline compare fails; `--bundle` and `render-launch` paths do not recompute current implementation identity or compare protected assets. |
| AC-07 | `WEAKENED` | No active Stage123 execution was observed and existing no-training tests passed, but the mandatory wrapper integration test is missing and preflight still contains a container-side `tmux` probe. |
| AC-08 | `FAIL` | Required `render-launch` command fails because no admission bundle exists; the implemented render path lacks protected-baseline and current-checkout revalidation required before printing launch. |
| AC-09 | `FAIL` | No strict schema-v1 acceptance report/bundle exists; runtime still reports Plan v7 as `UNREVIEWED`; implementation does not validate required model/effort, Plan hash, candidate commit, protected baseline, or all AC-01 through AC-08 verdicts. |

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
  "goal_status": "ACTIVE",
  "plan_status": "UNREVIEWED",
  "plan_version": 7,
  "latest_review": {
    "plan_version": 6,
    "verdict": "NOT_READY"
  }
}
```

### Candidate identity and status

```bash
git rev-parse HEAD
sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md
git status --short
```

Relevant output:

```text
29089a6c1c63d017384b1ff09eba9821d10a2a7a
4796cce83a89c9c34928dd5735decd823f78c53714d981d0b0ed111bdf7a1faa  docs/joint_training/goals/stage123-execution-readiness/plan.md
 M docs/joint_training/goals/stage123-execution-readiness/findings.jsonl
 M docs/joint_training/goals/stage123-execution-readiness/plan.md
 M docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl
 M docs/joint_training/guides/training_script_index.md
 m recipe
 M scripts/execution_results.py
 M tests/experiment_workflow/test_execution_results.py
?? docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
?? docs/joint_training/goals/stage123-execution-readiness/tools/
?? tests/experiment_workflow/test_stage123_admission_bundle.py
```

```bash
git diff --name-only -- config/experiment_execution scripts verl recipe
git status --porcelain=v1 -- config/experiment_execution scripts verl recipe
```

Output:

```text
recipe
scripts/execution_results.py
 M recipe
 M scripts/execution_results.py
```

### Required CPU verification tests

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_new_experiment_gate.py
```

Output:

```text
no tests ran in 0.00s
ERROR: file or directory not found: tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py
```

Rerun excluding the missing file:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_stage123_preflight_model_identity.py tests/experiment_workflow/test_operational_calibration_scorer_preflight.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_execution_results.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_new_experiment_gate.py
```

Output:

```text
63 passed in 239.64s (0:03:59)
```

### Missing required files

```bash
ls -l scripts/render_calibration_result.py scripts/render_calibration_probe_command.py scripts/stage123_host_facts.sh 2>&1 || true
```

Output:

```text
ls: cannot access 'scripts/render_calibration_result.py': No such file or directory
ls: cannot access 'scripts/stage123_host_facts.sh': No such file or directory
-rwx------ 1 root root 2157 Jul 13 04:47 scripts/render_calibration_probe_command.py
```

### Implementation-tree probe

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json
```

Output:

```json
{"failure": {"code": "implementation_tree_identity", "context": {}, "message": "covered_superproject_tree_dirty"}, "ok": false}
```

### Protected baseline probe

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Output: exit code `1`; the JSON output begins with `{"actual": [{"path": ".claude", "type": "directory"}, ...], "expected": [{"path": ".claude/skills/experiment-registry", ...}], "ok": false}`. The compare implementation derives requested roots with `item["path"].split("/")[0]`, so the baseline path `.claude/skills/experiment-registry` causes a broad `.claude` scan and cannot match the captured baseline.

### Admission launch rendering

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission render-launch --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json --repo-host /data-1/code/verl
```

Output:

```json
{"authorized": false, "code": "invalid_admission_bundle", "context": {}, "message": "[Errno 2] No such file or directory: 'docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json'"}
```

### Runtime no-training probe

```bash
tmux ls 2>/dev/null || true
pgrep -af 'stage123|qwen3_1p7b_stage123|experiment_execution_core' || true
```

Relevant output:

```text
docker-export: 1 windows (created Fri Apr  3 22:14:38 2026)
harness: 3 windows (created Fri Apr  3 09:44:29 2026)
```

No active Stage123 process was observed by `pgrep`; the only matching process was the reviewer shell command itself.

## Blocking In-Scope Defects

### F-V7-01 — AC-01 producer/result path is not implementable

Classification: `IN_SCOPE_DEFECT`.

Evidence:

- Plan v7 AC-01 requires `scripts/render_calibration_result.py render` and `validate`, but that file does not exist.
- Plan v7 AC-01 line 146 says authorization identity uses the exact `Plan v5` hash; Plan v7 line 350 requires the Milestone 4 decision to name exact Plan v7 hash. This is an internal frozen-contract contradiction.
- The required implementation-tree identity command currently fails with `covered_superproject_tree_dirty`, so the candidate cannot produce the fresh canonical-tree evidence demanded by AC-01.

Blocks: AC-01, AC-06, AC-08, AC-09.

### F-V7-02 — Host-facts ownership contract is absent

Classification: `IN_SCOPE_DEFECT`.

Evidence:

- Plan v7 requires `scripts/stage123_host_facts.sh`; the file does not exist.
- `recipe/on_policy_wdl_sft/code_task/stage123_preflight.py` accepts `--docker-image-id` and `--implementation-tree-sha256`, but not the Plan-required single `--host-facts` argument.
- `stage123_preflight.py` still runs `tmux list-sessions` in container code, contradicting the requirement that container preflight invoke neither Docker nor tmux.
- The mandatory wrapper integration test `tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py` does not exist.

Blocks: AC-02 and weakens AC-07.

### F-V7-03 — Protected-baseline admission path is not wired and its tool fails

Classification: `IN_SCOPE_DEFECT`.

Evidence:

- Plan v7 requires `scripts/execution_results.py admission validate` to consume `--protected-baseline`; the CLI has no such argument.
- The candidate bundle bindings in `scripts/execution_results.py` do not include `protected_baseline_sha256`.
- `validate_admission_bundle`, `admission validate --bundle`, and `render-launch` do not call the protected baseline compare tool.
- The reviewer-owned compare command exits `1` because compare expands `.claude/skills/experiment-registry` to `.claude`, making the current baseline unusable.

Blocks: AC-06, AC-08, AC-09.

### F-V7-04 — There is no single current-checkout admission validator on every path

Classification: `IN_SCOPE_DEFECT`.

Evidence:

- `build_admission_bundle` can call `current_implementation_tree_sha256(repo_root)`, but `admission validate --bundle` and `render-launch` only load the bundle, recompute its hash, and call `validate_admission_bundle`.
- `validate_admission_bundle` checks schema, run IDs, a small binding set, and acceptance equality; it does not recompute current implementation identity, compare current `HEAD` to readiness evidence commit, recompute referenced input hashes, enforce freshness from source inputs, or compare protected assets.

Blocks: AC-06 and AC-08.

### F-V7-05 — Strict acceptance schema v1 is not implemented

Classification: `IN_SCOPE_DEFECT`.

Evidence:

- Plan v7 requires acceptance schema v1 to bind Goal/Plan ID, Plan version/hash, independent reviewer identity, model `GPT-5.5`, reasoning effort `medium`, candidate and Readiness commits, exact run IDs, candidate bundle hash, all input hashes, protected-baseline SHA256, and AC-01 through AC-08 verdicts all exactly `PASS`.
- `scripts/execution_results.py` only checks `result_type`, `decision`, selected hash fields, `readiness_evidence_commit`, `run_ids`, and `bundle_sha256`; it does not validate Plan binding, reviewer/model/effort, candidate commit, protected baseline, acceptance report SHA256, or per-AC verdicts.

Blocks: AC-09 and AC-08.

## Deferred Suggestions

- None. All observations above block frozen ACs.

## Contract Contradictions

- AC-01 contains a direct Plan-version contradiction: line 146 requires the exact `Plan v5` hash, while the same Plan v7 requires Plan v7 review/start conditions and line 350 requires the Milestone 4 decision to name the exact Plan v7 hash.

## Feasibility Probe Assessment

- AC-01 includes a raw timing measurement from the completed calibration result and derives a 5400-second ceiling with approximately 4.92x measured end-to-end duration. That satisfies the numeric-budget probe requirement on paper.
- AC-02 storage/freshness and AC-03 cardinality waivers are described and do not introduce absolute performance budgets.
- However, feasibility is not sufficient for `READY` because required producer scripts and admission paths are missing or contradictory.

## Single Most Likely Weakness In This Review

I did not run the GPU requalification producer command or any live preflight that would create `host_facts.json` / `preflight_result.json`, because Plan v7 explicitly gates GPU requalification behind a later `USER_DECISION` and the required host-facts script is absent. The review therefore emphasizes implementability and reviewer-owned CPU/static evidence rather than live deployability evidence.
