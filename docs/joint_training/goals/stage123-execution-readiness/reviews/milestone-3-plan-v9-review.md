# Milestone 3 Plan v9 Independent Review

## Review Identity

- Review type: Milestone Review
- Goal: `stage123-execution-readiness`
- Frozen Plan version: `9`
- Plan SHA256: `29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c`
- Base commit: `7099830b3b0beebf8959ba4535f1f68e7b49427b`
- Candidate commit reviewed: `a60ed1f92b1a2541503e23884a33a1ac0a5847a6`
- Reviewer: independent Milestone 3 reviewer, requested GPT-5.5 medium
- Review prompt: `docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-3-plan-v9-review-prompt.md`
- Applicable ACs: `AC-01`, `AC-06`, `AC-08`

## Overall Verdict

PASS.

The committed candidate `a60ed1f9` is suitable for the authorized zero-step requalification. Reviewer-owned evidence recomputes the canonical implementation identity as `903f637ac931a3969f1e3beff09864728887ab66bbd17a4180867ddb0faab0fd`, confirms the expected complete Git-tree identities plus recipe gitlink `6fcccb353a87045a17f9d52b3821f0e20f7f9a9d`, passes the frozen behavioral tests, and confirms protected assets match the frozen baseline. No covered dirty path was present at review time.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Fresh production identity recomputation matches the Plan v9 prompt and `implementation-tree.jsonl`; Plan hash is current; runtime records the Plan v9 authorization and Milestone 3 start. |
| AC-06 | PASS | Frozen `test_execution_results.py` coverage passed as part of the required pytest command; protected-baseline comparison passed; no covered dirty paths were present. |
| AC-08 | PASS | Frozen `test_execution_results.py` coverage passed; the same committed implementation identity and protected-baseline gates required before launch rendering are reproducible from candidate `a60ed1f9`. |

## Candidate Diff Inspection

Command:

```bash
git diff --name-only 7099830b..a60ed1f9 -- config/experiment_execution scripts verl recipe tests/experiment_workflow docs/joint_training/goals/stage123-execution-readiness | sed -n '1,200p'
```

Relevant output:

```text
config/experiment_execution/calibration_policy_v1.json
docs/joint_training/goals/stage123-execution-readiness/findings.jsonl
docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-f-m2-mc-03-rereview-prompt.md
docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-f-m2-mc-03-rereview.md
docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-review-prompt.md
docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-2-plan-v9-review.md
docs/joint_training/goals/stage123-execution-readiness/reviews/plan-v9-matched-control-review-prompt.md
docs/joint_training/goals/stage123-execution-readiness/runtime.jsonl
recipe
scripts/execution_results.py
scripts/experiment_execution_core.py
scripts/experiment_manifest.py
scripts/render_calibration_probe_command.py
scripts/render_calibration_result.py
scripts/run_calibration_probe_zero_step.py
scripts/stage123_phase_adapter.py
tests/experiment_workflow/fixtures/experiment_batch_admission_v1.json
tests/experiment_workflow/fixtures/experiment_batch_v1.json
tests/experiment_workflow/test_calibration_milestone3.py
tests/experiment_workflow/test_evidence_compatibility.py
tests/experiment_workflow/test_execution_results.py
tests/experiment_workflow/test_experiment_batch_core.py
tests/experiment_workflow/test_experiment_manifest.py
tests/experiment_workflow/test_stage123_admission_bundle.py
tests/experiment_workflow/test_stage123_phase_adapter.py
```

Submodule command:

```bash
git diff --submodule=log 7099830b..a60ed1f9 -- recipe && git ls-tree HEAD recipe && git -C recipe rev-parse HEAD && git -C recipe status --short
```

Relevant output:

```text
Submodule recipe 9b83f9f4..6fcccb35:
  > stage123: add matched Stage1 control admission
160000 commit 6fcccb353a87045a17f9d52b3821f0e20f7f9a9d	 recipe
6fcccb353a87045a17f9d52b3821f0e20f7f9a9d
```

`git -C recipe status --short` produced no output.

## Commands And Evidence

### Frozen Verification 1: Implementation Identity Compare

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Exit code: `0`.

Output:

```jsonl
{"implementation_tree_sha256": "903f637ac931a3969f1e3beff09864728887ab66bbd17a4180867ddb0faab0fd"}
{"kind":"git_tree","path":"config/experiment_execution","tree_sha1":"4377c79f59537caf09f47bbc9ef6464e2b078d1f"}
{"gitlink_commit":"6fcccb353a87045a17f9d52b3821f0e20f7f9a9d","kind":"gitlink","mode":"160000","path":"recipe"}
{"kind":"git_tree","path":"scripts","tree_sha1":"a14fa2208cacd87294c2ffe2088752b510164b27"}
{"kind":"git_tree","path":"verl","tree_sha1":"40deac7dc6da65ef470c5e42c75fb2fd35b9335a"}
```

Identity conclusion: matches the required `implementation_tree_sha256` `903f637ac931a3969f1e3beff09864728887ab66bbd17a4180867ddb0faab0fd`, with complete `config/experiment_execution`, `scripts`, and `verl` Git trees plus recipe gitlink `6fcccb353a87045a17f9d52b3821f0e20f7f9a9d`.

### Frozen Verification 2: Focused Pytest Suite

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_core_migration.py tests/experiment_workflow/test_stage123_phase_adapter.py tests/experiment_workflow/test_execution_results.py
```

Exit code: `0`.

Output:

```text
....................                                                     [100%]
20 passed in 47.21s
```

### Frozen Verification 3: Protected Asset Compare

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Exit code: `0`.

Output:

```json
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

## Identity Recomputations

Command:

```bash
sha256sum docs/joint_training/goals/stage123-execution-readiness/plan.md docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl && git rev-parse HEAD && git rev-parse a60ed1f9 && git rev-parse 7099830b
```

Output:

```text
29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c  docs/joint_training/goals/stage123-execution-readiness/plan.md
903f637ac931a3969f1e3beff09864728887ab66bbd17a4180867ddb0faab0fd  docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207  docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
a60ed1f92b1a2541503e23884a33a1ac0a5847a6
a60ed1f92b1a2541503e23884a33a1ac0a5847a6
7099830b3b0beebf8959ba4535f1f68e7b49427b
```

Boundary manifest inspected:

```json
{
  "schema_version": 1,
  "algorithm": "stage123-implementation-boundary-v1",
  "superproject_tree_roots": [
    "config/experiment_execution",
    "scripts",
    "verl"
  ],
  "submodules": [
    {"path": "recipe", "coverage": "entire_gitlink_commit"}
  ]
}
```

## Covered Dirty And Protected Audit

Command:

```bash
git status --short && printf '\n--covered status--\n' && git status --short -- config/experiment_execution scripts verl recipe && printf '\n--protected status--\n' && git status --short -- .claude/skills/experiment-registry docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md test_data && printf '\n--review target status--\n' && git status --short -- docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-3-plan-v9-review.md
```

Output before writing this report:

```text
?? .claude/skills/experiment-registry
?? docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-3-plan-v9-review-prompt.md
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/

--covered status--

--protected status--
?? .claude/skills/experiment-registry
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/

--review target status--
```

Covered implementation roots were clean. The three protected paths remain in their accepted pre-existing untracked baseline status and the protected fingerprint comparison passed. The untracked review prompt is outside the implementation boundary and protected assets. This review report is the only reviewer-owned file intended to be created by this task.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

The review did not rerun Milestone 4's GPU zero-step requalification producer; that is intentionally outside Milestone 3. The conclusion is limited to committed production identity, frozen CPU verification, protected-asset stability, and suitability to proceed to the already authorized zero-step requalification.
