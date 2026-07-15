# Milestone 4 Plan v9 Independent Review

## Review Identity

- Reviewer: Independent Milestone 4 reviewer, requested GPT-5.5 medium; executed in Codex environment.
- Review type: Milestone Review.
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/stage123-execution-readiness`.
- Frozen Plan version: 9.
- Plan SHA256: `29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c`.
- Base commit: `9632ebdc`.
- Candidate commit reviewed: `a7ca8ee2f5208f7fccb1be732fc8000ca5f32689`.
- Applicable ACs: AC-01, AC-07.
- Review prompt: `docs/joint_training/goals/stage123-execution-readiness/reviews/milestone-4-plan-v9-review-prompt.md`.

## Overall Verdict

PASS.

The committed candidate and live scratch evidence satisfy AC-01 and AC-07. The first matched-control requalification attempt is correctly archived as producer-only `child_exit`; it did not produce `latest-probe.json` or `probe-report.json`. The second matched-control requalification under `RD-MATCHED-CONTROL-REQUAL-01` is the authoritative probe: it completed `stage1`, `stage2`, and `stage3`, three repetitions each, with zero optimizer steps, empty formal-checkpoint file lists, scratch-only outputs, cleanup recorded, and a passed `calibration_result.json` binding Plan v9, the current three-run manifest, and implementation tree `0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc`.

## Per-AC Verdicts

| AC | Verdict | Reviewer-owned evidence |
| --- | --- | --- |
| AC-01 - Re-Aligned Production Identity Is Freshly Qualified | PASS | `render_calibration_result.py validate` returned `{"ok": true, "sha256": "24647c4c5031ab199e40a2338b18dec02788c355600bed96192bb22faf43f880"}`. `implementation_tree_identity.py --compare` returned implementation tree SHA256 `0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc`. Live `probe-report.json` has status `passed`, phases `stage1/stage2/stage3`, three repetitions each, and prediction comparison metrics `validation_elapsed_seconds`, `phase_elapsed_seconds`, `peak_rss_gib`, and `gpu_wait_fraction`, all qualified. |
| AC-07 - Readiness Does Not Train Or Publish | PASS | Frozen pytest command returned `33 passed in 30.64s`. Live report and calibration result record total optimizer steps `0`, every repetition has `training_steps=0` and `optimizer_enabled=false`, every repetition has empty `formal_checkpoint_files`, and filesystem inspection found `0` files under the scratch checkpoint directories. Protected asset comparison returned `{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}`. |

No AC is WEAKENED.

## Frozen Verification Commands

### Calibration Result Validation

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/render_calibration_result.py validate --input docs/joint_training/goals/stage123-execution-readiness/calibration_result.json --schema config/experiment_execution/calibration_result_schema_v1.json
```

Result:

```text
{"ok": true, "sha256": "24647c4c5031ab199e40a2338b18dec02788c355600bed96192bb22faf43f880"}
```

### Focused Tests

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_calibration_milestone3.py tests/experiment_workflow/test_calibration_outcomes.py tests/experiment_workflow/test_execution_results.py
```

Result:

```text
33 passed in 30.64s
```

### Implementation Tree Identity

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl
```

Result:

```text
{"implementation_tree_sha256": "0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc"}
{"kind":"git_tree","path":"config/experiment_execution","tree_sha1":"4377c79f59537caf09f47bbc9ef6464e2b078d1f"}
{"gitlink_commit":"6fcccb353a87045a17f9d52b3821f0e20f7f9a9d","kind":"gitlink","mode":"160000","path":"recipe"}
{"kind":"git_tree","path":"scripts","tree_sha1":"60a5b6491cc6646bc00d92ccda0a192fe485cbbf"}
{"kind":"git_tree","path":"verl","tree_sha1":"40deac7dc6da65ef470c5e42c75fb2fd35b9335a"}
```

### Protected Asset Fingerprint

Command:

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python docs/joint_training/goals/stage123-execution-readiness/tools/protected_asset_fingerprint.py compare --repo-root /data-1/code/verl --baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl
```

Result:

```text
{"ok": true, "sha256": "c739413ba6c4014f246f53bbaa5d86a4647b41cef640d580dbff1703864c7207"}
```

## Live Evidence Inspection

### Archived First Attempt

Inspected root:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01-failed-child-exit-20260715T0548Z
```

Evidence:

- Execution state status: `failed`.
- Failure code: `child_exit`.
- Return code: `1`.
- Cleanup: `resources_released=true`, `term_sent=false`, `kill_sent=false`.
- No `latest-probe.json` exists.
- No `probe-report.json` exists.
- Files present are only producer candidate/spec/workload plus execution-core state/events.

Verdict: archived producer-only failure; it is not used as authoritative calibration evidence.

### Authoritative Second Attempt

Inspected root:

```text
/data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01
```

Pointer evidence from `latest-probe.json`:

- `schema_version=2`.
- `status=passed`.
- `run_id=stage123_matched_control_requalification_RD-MATCHED-CONTROL-REQUAL-01`.
- `authorization_decision_id=RD-MATCHED-CONTROL-REQUAL-01`.
- `report_sha256=31bf1198ff64ffbc16e8342992c9a3ebbdda88eac1c008bfea223edbf15a6a33`.
- Report path: `/data-1/tmp/verl_agent_scratch/experiment_workflow/readiness-requalification/RD-MATCHED-CONTROL-REQUAL-01/probe-20260715T055350Z/probe-report.json`.

Execution-core state evidence:

- Status: `succeeded`.
- Cleanup: `resources_released=true`, `term_sent=false`, `kill_sent=false`.
- Deadline window: started monotonic `11678933.489180751`, completed `11680336.667784564`, deadline `11684333.489180751`; completed within the 5400-second envelope.

Probe report evidence:

- `status=passed`.
- `optimizer_steps=0`.
- `formal_checkpoints=[]`.
- `probe_spec.training_steps=0`.
- `probe_spec.optimizer_enabled=false`.
- `probe_spec.phases=["stage1", "stage2", "stage3"]`.
- Repetition matrix:
  - `stage1`: 3 passed repetitions; `training_steps=0`; `optimizer_enabled=false`; `formal_checkpoint_files=[]`; cleanup released for every repetition.
  - `stage2`: 3 passed repetitions; `training_steps=0`; `optimizer_enabled=false`; `formal_checkpoint_files=[]`; cleanup released for every repetition.
  - `stage3`: 3 passed repetitions; `training_steps=0`; `optimizer_enabled=false`; `formal_checkpoint_files=[]`; cleanup released for every repetition.
- Checkpoint directory audit: scratch checkpoint directories exist because the val-only launcher creates output directories, but recursive file count under those directories is `0`.

Calibration result evidence:

- `decision=passed`.
- Authorization identity: `decision_id=RD-MATCHED-CONTROL-REQUAL-01`, `plan_version=9`, `plan_sha256=29fd07ea9cbf5a13bae70d0907c4f25652fd8efe90e11dfd911c299ec669f43c`.
- Evidence commit recorded by the result renderer: `8a5402e5a31c7810c7c0c77b4d8dcd1aa6129f56`.
- Reviewed committed candidate: `a7ca8ee2f5208f7fccb1be732fc8000ca5f32689`.
- The only diff from `8a5402e5` to `a7ca8ee2` is `calibration_result.json`; the implementation tree command independently confirms the current covered implementation identity remains `0958211eec8ee0169261b1dba24bc33d0a930249e76f741a9599d7378e8072fc` at the reviewed candidate.
- Manifest SHA256: `323bcc6084c1b01653bdf3cb5b299cab51c76864c8ca24878a9e5b002cb76278`.
- Resource profile SHA256: `d9b6a36dd9fcc4307f7b502e5511989e60c1a257f57c9ac70574acaf12eee2b5`.
- Workload run IDs: `frac25-stage1-control`, `frac25-stage2`, `frac25-stage3`.
- Prediction comparison policy: `stage123-calibration-policy-v1`, policy SHA256 `c8f4df7240a9b4d2fef0ad6adb183b2562604375f5adba7dc396e406252850a1`.
- Prediction comparison uses historical predecessor result as history only, not current authority, and contains qualified comparisons for all four required metrics.

## F-M4-MC-01 Closure Decision

F-M4-MC-01 is CLOSED for this milestone review.

Reason: the original failure condition was an authorized requalification that reached all nine repetition directories but ended as `child_exit` during terminal result production. Candidate `a7ca8ee2` preserves that first attempt as failed producer-only evidence and supplies a second exact three-phase x3 zero-step probe that reached execution-core `succeeded`, generated a matching `latest-probe.json` and `probe-report.json`, rendered a schema-valid `passed` `calibration_result.json`, and passed all frozen verification commands. The compatibility repair is fail-closed: tests now cover internally inconsistent historical policy binding, and the live result binds a complete policy comparison rather than an empty or documentation-only assertion.

## Test-Weakening Audit

Reviewed committed test change in `tests/experiment_workflow/test_calibration_milestone3.py` from `9632ebdc..a7ca8ee2`.

- The changed fixture no longer requires the historical predecessor policy SHA to equal the current local `calibration_policy_v1.json` SHA. That is not a weakening for Plan v9 because F-M4-MC-01 is specifically about accepting internally consistent predecessor history as historical prediction data while allowing the current full policy hash to differ after required phase expansion.
- The same test adds a stricter negative check: if top-level predecessor `policy_sha256` and nested `prediction_comparison.policy_sha256` disagree, `build_prediction_comparison` must raise `policy binding mismatch`.
- Existing fail-closed checks remain: insufficient history and prediction exceedance fail, empty prediction comparison is rejected, and the schema requires all four prediction metrics.
- Frozen focused tests pass in-container: `33 passed in 30.64s`.

Verdict: no skipped, deleted, loosened, or trivialized test was found in the candidate diff for the applicable ACs. The compatibility change narrows acceptance to internally consistent historical policy evidence and adds a regression for inconsistent history.

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Most Likely Weakness In This Review

The review did not rerun the long GPU producer command. It independently inspected the live scratch evidence and reran all frozen verification commands, as required, but it relies on the existing authoritative second probe artifacts for the expensive GPU run.
