# Milestone 1 Disposition Inventory

This inventory freezes the consolidation boundary at baseline `390be933`. It is an
implementation map, not a new authority or receipt. Historical artifacts remain
historical; only `preflight_result.json`, `calibration_result.json`, and
`acceptance_report.json` may authorize the current workflow after consolidation.

## Disposition Vocabulary

- `KEEP`: retained as the single generic implementation.
- `MERGE`: behavior moves into another retained implementation, then the source is deleted.
- `MIGRATE`: data or callers move to the new authority, preserving historical bytes.
- `COMPAT_SHIM`: warning delegate only; it owns no policy or copied facts.
- `FAIL_CLOSED`: unsupported or stale inputs are rejected explicitly.
- `ARCHIVE`: retained only as historical documentation, outside active authority surfaces.
- `DELETE`: redundant active implementation or review artifact is removed.

## Frozen Production Budget Set

The AC-08 production budget is computed over these 21 baseline files. They contain
3,763 lines at `390be933`; the candidate must contain at most 3,000 lines and have
more deletions than additions within this exact set.

| File | Baseline lines | Disposition | Rationale |
|---|---:|---|---|
| `verl/trainer/ppo/ray_trainer.py` | 2409 | MERGE | Remove calibration/Stage123 protocol; retain generic observer and accepted validation behavior. |
| `scripts/assemble_code_task_operational_calibration.py` | 451 | MERGE | Fold result assembly into the retained Python execution core. |
| `scripts/calibration_outcomes.py` | 108 | MERGE | Move stable result classification into the structured result/core model. |
| `scripts/calibration_timing.py` | 48 | MERGE | Retain generic timing arithmetic only where used by the Python executor. |
| `scripts/check_calibration_port_quiet.py` | 47 | MERGE | Fold port isolation check into structured preflight execution. |
| `scripts/check_calibration_prediction_contract.py` | 599 | DELETE | Dynamic prediction qualification is deferred and cannot remain current authority. |
| `scripts/check_code_task_operational_calibration.py` | 737 | MERGE | Replace monolithic free-form checker with structured result validation. |
| `scripts/check_code_task_preflight_budget.py` | 52 | MERGE | Preserve static budget fail-closed behavior in policy validator. |
| `scripts/check_goal_git_isolation.py` | 155 | MERGE | Preserve dirty-baseline protection without adoption/document receipts. |
| `scripts/experiment_failure_classifier.py` | 49 | MERGE | Convert free-form classifications to stable code/message/context results. |
| `scripts/experiment_manifest.py` | 217 | KEEP | Becomes generic canonicalizer; Stage123 invariants move to versioned policy. |
| `scripts/experiment_workflow_baseline.py` | 167 | MERGE | Preserve Git dirty-baseline evidence without receipt authority. |
| `scripts/run_code_task_operational_calibration_queue.sh` | 429 | COMPAT_SHIM | Reduce to environment setup plus one `exec` into Python. |
| `scripts/run_code_task_operational_calibration.sh` | 226 | COMPAT_SHIM | Warning delegate only; no state machine or policy ownership. |
| `scripts/stage123_preflight_receipt.py` | 121 | DELETE | Receipt class is forbidden; migrate authority to `preflight_result.json`. |
| `scripts/stage123_manifest_monitor.py` | 50 | MERGE | Child lifecycle/deadline/cleanup/resume move to Python executor. |
| `scripts/stage123_manifest_release_dispatch.sh` | 17 | COMPAT_SHIM | Delegate to retained release gate without accepting stale evidence. |
| `scripts/training_queue_monitor.sh` | 244 | KEEP | Retain generic tmux queue-monitor contract; remove Stage123-specific policy if present. |
| `scripts/validation_deadline_controller.py` | 106 | MERGE | Deadline and cleanup state move into Python executor. |
| `scripts/verify_manifest_release_provenance.py` | 58 | MERGE | Preserve provenance validation against normalized manifest/results. |
| `recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh` | 75 | COMPAT_SHIM | Thin compatibility gate delegates to canonicalizer/policy validation. |

Files retained above 300 lines require this justification: `ray_trainer.py` is an
existing central trainer whose unrelated behavior is outside this Goal;
`assemble_code_task_operational_calibration.py`,
`check_calibration_prediction_contract.py`,
`check_code_task_operational_calibration.py`, and
`run_code_task_operational_calibration_queue.sh` are not intended to remain above
300 lines after their `MERGE`/`DELETE`/`COMPAT_SHIM` dispositions.

## Legacy Class Map

| Legacy class | Current examples | Disposition | Candidate authority/behavior |
|---|---|---|---|
| Concrete experiment facts | `recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml`, rendered manifest | KEEP | Normalized manifest solely owns identity, dataset, producer, path, hash, TTL. |
| Generic canonicalization | `scripts/experiment_manifest.py` | KEEP | Parse/default/normalize only; emit stable structured failures. |
| Stage123 cross-field rules | checks embedded across manifest/checker/shell | MIGRATE | One versioned policy validator referencing manifest values without copying them. |
| Shell queue state machine | calibration queue/runner shell scripts | MERGE | Python owns states, deadlines, children, cleanup, resume; shell performs one `exec`. |
| Calibration file protocol in trainer | `CALIBRATION_*` handling in `ray_trainer.py` | DELETE | Default no-op generic validation observer; optional recording observer. |
| Free-form checker failures | checker strings and shell exit interpretation | MIGRATE | Stable `{code,message,context}` records and deterministic exit mapping. |
| Preflight receipt | `stage123_preflight_receipt.py` and receipt fixtures | DELETE | `preflight_result.json`. |
| Calibration/adoption receipt | calibration completion/adoption JSON | DELETE | `calibration_result.json`; documentation-only changes need no receipt. |
| Document-hash/dirty adoption receipt | `stage123_dirty_adoption.json`, `goal_contract_dirty_adoption.json`, plan adoption JSON | DELETE | Git commit/content state plus dirty-baseline protection; stale receipt inputs fail closed. |
| Acceptance/review receipt chains | active reliability review series | ARCHIVE | Goal reviewer prompt plus reviewer-owned `acceptance.md` and `acceptance_report.json`. |
| Legacy runnable inventory | `recipe_legacy_runnables.json`, `superproject_legacy_runnables.json` | MIGRATE | Compatibility tests enumerate supported shims; unsupported legacy inputs fail closed. |
| Historical runtime evidence | committed historical result/log fixtures | ARCHIVE | Preserve bytes; never infer current authorization from them. |
| Release provenance | manifest release hook and provenance checker | MERGE | Current normalized manifest + retained result classes + release gate. |
| PM2 persistent CI | `config/pm2/experiment-reliability-ci.json`, checker | KEEP | PM2 only; explicit systemd rejection. |
| Tmux long-run ownership | queue/monitor entrypoints | KEEP | Long-running ownership remains tmux; persistent CI remains PM2. |
| Source-text runtime tests | source grep/substring contract tests | DELETE | Replace with fake executable, fake clock, temp state, observer behavior tests. |

## Test Disposition

The test baseline is exactly `tests/experiment_workflow/**` plus
`tests/joint_training/regression/test_validation_generation_logging.py` (4,694 lines).
Behavior fixtures for scorer dependency, timeout, fatal exit, CUDA OOM, occupied port,
incomplete identity, cleanup failure, incomplete training, provenance, PM2, and tmux
are `KEEP` or `MIGRATE`. Tests whose only assertion is source text, receipt presence,
document hash, duplicated manifest facts, or shell implementation detail are `DELETE`
and replaced only when needed by behavior-level coverage. The candidate budget is at
most 4,000 lines and independently net-negative.

## Active Documentation Disposition

The active reliability review series under `docs/joint_training/codereview/active/`,
legacy adoption manifests under `docs/joint_training/manifests/`, and superseded
`experiment_execution_reliability_goal*` active plan/launch files are `ARCHIVE` or
`DELETE`. The active candidate keeps only concise current workflow documentation and
this Goal's governance artifacts. Historical bytes selected for preservation are moved
without modification. The candidate budget is at most 650 lines and independently
net-negative from the 1,638-line frozen baseline.

## AC-05 Feasibility

AC-05 is feasible without real services or GPU work: compatibility is exercised with
temporary copies of historical artifacts, fake executables, and local result fixtures.
The migration can prove byte identity, reject stale authority, and allow documentation
changes without adoption receipts entirely on CPU.

## AC-08 Deletion Map

- Production: remove prediction/receipt machinery; merge checker, assembly, deadline,
  provenance, and state logic into a smaller Python core; reduce shell to shims.
- Tests: delete duplicated source-text and receipt/document-hash assertions; retain one
  behavior-oriented suite using fakes.
- Active docs: archive/delete the superseded multi-review chain and adoption manifests;
  retain Goal governance and concise current reliability documentation.
- Protected assets are excluded from every budget and remain unstaged.
