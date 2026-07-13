# Final Acceptance Review — Experiment Execution Core Consolidation

## Review Identity

- Reviewer: Independent final acceptance reviewer (Codex, GPT-5.5 medium requested by prompt; model selection not exposed in this harness)
- Review type: `final-acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Plan version: `1`
- Plan SHA256: `b0929f0b0c910515c14140489d629c6103637c0ceb477c24e34185ddf4887ba8`
- Base commit: `390be933`
- Implementation candidate commit: `97c1c1f596da4d1e977b252a411e9444029d9599`
- Reviewed HEAD: `61f860f2a257d1c7270c97583b504b5684fe3869`
- Later prompt/runtime-only commits reviewed: `535a74d0`, `61f860f2`
- Recipe submodule: `fb49bccdfc7eba1474720338af4c346733a6f783`, clean

## Overall Verdict

`ACCEPTED`

All applicable criteria AC-01 through AC-09 are `PASS` from reviewer-owned command output or, for AC-06, the frozen prompt's required audit of the sole full-suite run plus focused replacement evidence. No AC is `FAIL` or `WEAKENED`.

## Per-AC Verdict Table

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-01 Shared Trainer Is Generic | PASS | Focused validation suites passed `12 passed`; forbidden-name grep on `verl/trainer/ppo/ray_trainer.py` returned zero matches. |
| AC-02 Manifest Has One Owner | PASS | Manifest and queue-monitor contract suites passed `20 passed`; tests exercise changed manifest facts and policy errors through structured validation. |
| AC-03 Queue State Is Python-Owned | PASS | Runner, deadline cleanup, and runtime-isolation suites passed `10 passed`; fake adapters/clock/state cover success, failure, deadline, cleanup, resume, and thin shell behavior. |
| AC-04 Failures Are Structured | PASS | Checker and outcomes suites passed `18 passed`; assertions target stable `code`, `message`, and `context`, including message-format variation. |
| AC-05 Evidence And Compatibility Are Consolidated | PASS | Compatibility/migration/receipt/adoption/evidence selector passed `14 passed, 124 deselected`; legacy evidence classes fail closed and historical fixtures remain copy-checked. |
| AC-06 Accepted Reliability Behavior Is Preserved | PASS | Did not rerun full suite per prompt. Audited `full-cpu-gate-summary.json`: one full run recorded `146 passed`, `3 failed` stale source-text assertions; targeted replacements record `10 passed`, `22 passed`, `23 passed`. Current focused AC suites also pass. |
| AC-07 Operational Policies Remain Enforced | PASS | PM2 lifecycle suite passed `6 passed`; `scripts/check_pm2_ci_keepalive.py --require-no-systemd` returned `{ "ok": true, "failures": [] }`. |
| AC-08 Accretion Is Removed | PASS | Independent line-budget recomputation: production `3763 -> 2259` (<=3000), tests `4694 -> 2164` (<=4000), active reliability docs `2091 -> 0` (<=650); all classes net-negative. `git diff --check 390be933...HEAD` passed. Retained >300-line files have inventory disposition. |
| AC-09 Independent Acceptance Is Bound To This Plan | PASS | Plan hash matches frozen hash; candidate is ancestor of reviewed HEAD; `goal-plan-runtime validate-runtime` passed with active Milestone 6 and no open findings; `git status --short` shows only the three protected assets untracked before this reviewer write. Acceptance is bound to this file and `acceptance_report.json`. |

## Commands And Evidence

```text
sha256sum docs/joint_training/goals/experiment-execution-core-consolidation/plan.md
b0929f0b0c910515c14140489d629c6103637c0ceb477c24e34185ddf4887ba8  docs/joint_training/goals/experiment-execution-core-consolidation/plan.md
```

```text
git merge-base --is-ancestor 97c1c1f596da4d1e977b252a411e9444029d9599 HEAD && echo candidate-is-ancestor
candidate-is-ancestor

git log --oneline 97c1c1f596da4d1e977b252a411e9444029d9599..HEAD
61f860f2 docs: request final acceptance
535a74d0 docs: add final acceptance prompt
```

```text
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/joint_training/regression/test_validation_generation_logging.py tests/joint_training/regression/test_validation_reward_metadata.py
12 passed, 8 warnings in 22.81s

rg -n 'CALIBRATION_|stage123|deployab' verl/trainer/ppo/ray_trainer.py
zero forbidden matches
```

```text
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_experiment_manifest.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py
20 passed in 120.69s
```

```text
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_runner.py tests/experiment_workflow/test_validation_deadline_cleanup.py tests/experiment_workflow/test_operational_calibration_runtime_isolation.py
10 passed in 0.11s
```

```text
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_operational_calibration_checker.py tests/experiment_workflow/test_calibration_outcomes.py
18 passed in 121.05s
```

```text
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow -k 'compat or migration or receipt or adoption or evidence'
14 passed, 124 deselected in 24.62s
```

```text
cat docs/joint_training/goals/experiment-execution-core-consolidation/full-cpu-gate-summary.json
exit_code=1, passed=146, failed=3, run_count=1, targeted_fix_evidence={three_replacement_tests: 10 passed, fast_gate: 22 passed, preservation_focused: 23 passed}
```

```text
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_pm2_ci_keepalive.py
6 passed in 0.71s

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/check_pm2_ci_keepalive.py --contract config/pm2/experiment-reliability-ci.json --repo-root . --require-no-systemd
{"failures": [], "ok": true}
```

```text
git diff --numstat 390be933...HEAD -- scripts tests/experiment_workflow verl/trainer/ppo/ray_trainer.py recipe/on_policy_wdl_sft/code_task docs/joint_training/codereview/active docs/joint_training/manifests docs/joint_training/goals/experiment-execution-core-consolidation
Reviewed file-level additions/deletions; net removals dominate the frozen responsibility classes.

git diff --check 390be933...HEAD
passed with no output
```

```text
Independent AC-08 recomputation:
production: 3763 baseline lines -> 2259 candidate lines, net -1504, ceiling 3000
tests: 4694 baseline lines -> 2164 candidate lines, net -2530, ceiling 4000
active reliability docs: 2091 baseline lines -> 0 candidate lines, net -2091, ceiling 650
retained >300-line files checked: scripts/check_calibration_prediction_contract.py justified in inventory; tests/experiment_workflow/test_dynamic_calibration_interval.py justified as deferred compatibility/serialization coverage in inventory.
```

```text
goal-plan-runtime validate-runtime docs/joint_training/goals/experiment-execution-core-consolidation
passed: current_milestone=Milestone 6, goal_status=ACTIVE, plan_status=READY, plan_version=1, all findings CLOSED

git status --short
?? .claude/skills/experiment-registry
?? docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
?? test_data/

git -C recipe status --short
git -C recipe rev-parse HEAD
fb49bccdfc7eba1474720338af4c346733a6f783
```

## Test Deletion / Weakening Audit

- Deleted tests and fixtures are dispositioned by the Goal inventory as adoption/document-hash receipts, stale preflight qualification, old assembler/source-text assertions, and Stage123 readiness surfaces outside this Goal.
- `rg` found no `pytest.mark.skip`, `pytest.mark.xfail`, `skip(`, or `xfail(` in the retained workflow/regression tests.
- Replacement tests exercise fake executable, fake clock, temporary state, structured failure fields, observer event order, compatibility fail-closed behavior, PM2-only policy, no-systemd policy, dirty-baseline protection, release prohibition, and dependency-failure-before-Ray behavior.
- No real W&B, WxPusher, Hugging Face, GitHub, Docker-daemon, GPU run, or registry mutation was used as acceptance evidence.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

AC-06 is accepted by auditing the recorded single full-suite run and focused replacement evidence, as explicitly required by the final prompt, rather than by rerunning the full AC-06 suite after the replacement commits. That is the intended review mode but remains the weakest evidence boundary.
