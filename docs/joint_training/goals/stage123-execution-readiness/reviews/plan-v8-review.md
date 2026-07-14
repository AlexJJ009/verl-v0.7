# Plan v8 Independent Review

## Review Identity

- Reviewer: Codex independent Plan Reviewer, requested model `GPT-5.5`, reasoning effort `medium`
- Review type: pre-implementation Plan Review under Goal Plan lifecycle
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`
- Reviewed Plan version: `8`
- Reviewed Plan SHA256: `fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67`
- Reviewed commit: `29089a6c1c63d017384b1ff09eba9821d10a2a7a`
- Review prompt: `docs/joint_training/goals/stage123-execution-readiness/reviews/plan-v8-review-prompt.md`
- Repository mutation by reviewer: only this report file

## Overall Verdict

`READY`.

Plan v8 is implementable as a pre-implementation Goal contract. The former Plan-v5 authorization contradiction is gone: the current Plan references the exact Plan v8 hash requirement, not Plan v5. The pre-Milestone protected baseline compare now passes. I did not require Milestone 2-6 outputs to already exist, because the Plan freezes how those artifacts will be produced and verified before they can authorize admission.

This is not final readiness acceptance. It authorizes implementation to begin under Plan v8, subject to the Milestone 4 `USER_DECISION` GPU gate and the frozen AC evidence requirements.

## Per-AC Verdict Table

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | `PASS` | The Plan now explicitly invalidates predecessor calibration after compatibility repair, freezes a bounded zero-training-step requalification command, freezes render/validate commands, records the 5400s feasibility margin from prior measured calibration evidence, and gates GPU work behind `USER_DECISION`. The exact Plan binding now says Plan v8. |
| AC-02 | `PASS` | The host/container split is specified: host facts are produced outside the container, container preflight consumes exactly `--host-facts`, and wrapper integration evidence is frozen. Missing `scripts/stage123_host_facts.sh` is a Milestone 2 implementation output, not a Plan-review blocker. |
| AC-03 | `PASS` | The run-set contract is exact across manifest, queue dry-run, and monitor inventory: only `frac25-stage2` and `frac25-stage3`; FRAC50/P60 and hidden env filtering are excluded. The numeric-budget waiver is valid because this AC defines identity/cardinality, not a resource budget. |
| AC-04 | `PASS` | Queue lifecycle ownership is stated as Python-owned with fake clock/child/Docker/tmux/GPU/filesystem/checkpoint adapter evidence and shell-policy checks. This is sandbox-verifiable by the frozen tests. |
| AC-05 | `PASS` | Queue and monitor authority is tied to persisted events/state, with explicit prohibition on hard-coded arrays, log scraping policy, duplicated run facts, or independent completion inference. The specified tests are appropriate evidence. |
| AC-06 | `PASS` | The Plan now requires one common admission validator across candidate construction, `--bundle`, accepted-bundle validation, and `render-launch`, including current tree, current `HEAD`, input hashes, freshness, run set, and protected-baseline comparison. This closes the earlier admission-path ambiguity at Plan level. |
| AC-07 | `PASS` | No-training/no-publication evidence is made mandatory, and the wrapper integration test is explicitly required evidence for this AC. The Plan excludes formal checkpoints, optimizer steps, external services, registry mutation, W&B sync, and publication. |
| AC-08 | `PASS` | Launch rendering is frozen behind accepted bundle validation and requires current checkout/commit, candidate bundle hash, acceptance report hash, freshness, input hashes, acceptance bindings, exact repo path, run set, tmux session, and human intent variable. |
| AC-09 | `PASS` | Independent acceptance is bound to schema-v1 `acceptance_report.json`, exact Plan ID/version/hash, model/effort, candidate/evidence commits, all input hashes, protected-baseline SHA256, and AC-01 through AC-08 verdicts exactly `PASS`. |

## Commands And Evidence

### Goal validators

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/stage123-execution-readiness
```

Output:

```text
PASS
EXIT:0
```

```bash
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-execution-readiness
```

Relevant output:

```json
{
  "goal_status": "ACTIVE",
  "plan_status": "UNREVIEWED",
  "plan_version": 8,
  "latest_review": {
    "plan_version": 7,
    "verdict": "NOT_READY",
    "candidate_commit": "29089a6c1c63d017384b1ff09eba9821d10a2a7a"
  },
  "pending_user_decisions": []
}
```

Exit code: `0`. `UNREVIEWED` is expected before this Plan v8 review is recorded.

### Protected baseline compare

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Output:

```json
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

Exit code: `0`.

### Plan identity and candidate status

```bash
sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md
git rev-parse HEAD
git status --short
```

Relevant output:

```text
fc079ef6634aaf8e40f8aa99f81e38755f3e96a611815690ca8ac4eba1750c67  docs/joint_training/goals/stage123-execution-readiness/plan.md
29089a6c1c63d017384b1ff09eba9821d10a2a7a
 M docs/joint_training/goals/stage123-execution-readiness/findings.jsonl
 M docs/joint_training/goals/stage123-execution-readiness/plan.md
 M docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl
 M docs/joint_training/guides/training_script_index.md
 m recipe
 M scripts/execution_results.py
 M tests/experiment_workflow/test_execution_results.py
?? .claude/skills/experiment-registry
?? docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
?? docs/joint_training/goals/stage123-execution-readiness/reviews/plan-v8-review-prompt.md
?? docs/joint_training/goals/stage123-execution-readiness/tools/
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/
?? tests/experiment_workflow/test_stage123_admission_bundle.py
```

The dirty worktree is expected for a Plan review of an uncommitted Plan amendment and pre-Milestone protected baseline. The Plan makes dirty covered implementation roots fail later admission, which is correct.

### Former Plan-version contradiction check

```bash
rg -n "Plan v5|Plan v6|Plan v7|Plan v8 hash|exact Plan" docs/joint_training/goals/stage123-execution-readiness/plan.md
```

Output:

```text
146:  `USER_DECISION_RECORDED.decision_id` plus the exact Plan v8 hash and candidate
270:  candidate and evidence commits, exact Plan ID/version/hash, every AC-01 through
350:- The Milestone 4 decision must name the exact Plan v8 hash, decision id, maximum
```

The stale Plan-v5 binding from the Plan v7 review is removed.

### Relevant interface inspection

```bash
rg -n "validate_admission_bundle|build_admission_bundle|current_implementation|render-launch|host-facts|protected|acceptance" scripts/execution_results.py recipe/on_policy_wdl_sft/code_task/stage123_preflight.py tests/experiment_workflow/test_stage123_admission_bundle.py 2>&1
```

Relevant evidence:

```text
scripts/execution_results.py:176:def current_implementation_tree_sha256(repo_root: Path) -> str:
scripts/execution_results.py:202:def validate_admission_bundle(bundle: dict[str, Any], *, require_accepted: bool = False) -> EvidenceDecision:
scripts/execution_results.py:254:def build_admission_bundle(
scripts/execution_results.py:367:def admission_main(argv: list[str]) -> int:
scripts/execution_results.py:381:    render = sub.add_parser("render-launch")
tests/experiment_workflow/test_stage123_admission_bundle.py:35:def test_preflight_primary_inputs_exclude_frac50() -> None:
tests/experiment_workflow/test_stage123_admission_bundle.py:45:def test_accepted_bundle_requires_complete_report_bindings() -> None:
tests/experiment_workflow/test_stage123_admission_bundle.py:66:def test_accepted_bundle_round_trip_uses_unsigned_bundle_hash() -> None:
```

I also confirmed these future Milestone 2 artifacts are not present yet:

```text
scripts/render_calibration_result.py: missing
scripts/stage123_host_facts.sh: missing
tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py: missing
```

I do not treat those missing files as Plan Review blockers because Plan v8 explicitly schedules the compatibility repair before calibration/preflight/admission and freezes exact commands/evidence that must pass before final acceptance.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

- The Plan could name `scripts/render_calibration_probe_command.py` and `scripts/run_calibration_probe_zero_step.py` explicitly inside the Milestone 2 repair surface for the producer-pointer v2 linkage. I do not block on this because AC-01 already freezes the producer command, pointer requirements, render command, and validation command, and the included repair scope covers AC-01 through AC-06 compatibility.

## Contract Contradictions

None found.

The Plan-v7 contradiction requiring an exact Plan v5 hash is fixed in Plan v8, and the protected baseline compare now passes before Milestone 1.

## Single Most Likely Weakness In This Review

The weakest point is that I judged implementability mostly from the frozen Plan commands, validators, and interface inspection rather than running the full future AC test matrix. That is intentional for this pre-implementation Plan Review: many required artifacts are Milestone 2-6 outputs and should not exist before a `READY` Plan authorizes implementation.
