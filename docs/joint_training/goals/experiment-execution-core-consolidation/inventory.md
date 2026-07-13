# Milestone 1 Inventory And Disposition Map

Generated against baseline `390be933`. This is the frozen file-class inventory used
by AC-05 and AC-08. Git history preserves deleted implementation and review artifacts;
runtime evidence under `/data-2` is outside these deletion classes and remains intact.

## Baseline Budgets

| Class | Frozen selector | Files | Lines | Candidate ceiling |
| --- | --- | ---: | ---: | ---: |
| Production | The 21 paths in Production Dispositions | 21 | 3,763 | 3,000 |
| Workflow tests | `tests/experiment_workflow/**` plus `test_validation_generation_logging.py` | 54 | 4,694 | 4,000 |
| Active reliability docs | The 15 paths in Documentation Dispositions | 15 | 2,091 | 650 |

Each class must independently be net-negative from `390be933`. Files added to replace
an inventoried responsibility count in that responsibility's class. Generated caches,
archived runtime evidence, and unrelated paths cannot offset a class total.

## Production Dispositions

| Path | Disposition | Target responsibility and compatibility |
| --- | --- | --- |
| `scripts/assemble_code_task_operational_calibration.py` | MERGE | Fold result assembly into the Python execution core; retain CLI shim only if tests prove a live caller. AC-03/05. |
| `scripts/build_stage123_adoption_manifest.py` | DELETE | Git is document/code version evidence; no replacement receipt. AC-05/08. |
| `scripts/calibration_outcomes.py` | KEEP | Typed calibration result/outcome model; absorb structured failure codes if cohesive. AC-04/05. |
| `scripts/calibration_timing.py` | KEEP | Generic lifecycle interval calculation; consume observer events, not trainer file protocol. AC-01/06. |
| `scripts/check_calibration_port_quiet.py` | MERGE | Runtime isolation validator owned by Python execution core. AC-03/06. |
| `scripts/check_calibration_prediction_contract.py` | DEFERRED | Prediction qualification belongs to the next Goal; retain read-only compatibility, remove from current authorization. AC-05. |
| `scripts/check_code_task_operational_calibration.py` | MERGE | Replace God checker with schema/invariant validation, decision policy, and result renderer; remove concrete facts. AC-02/04. |
| `scripts/check_code_task_preflight_budget.py` | DEFERRED | Preserve for later Calibration Qualification; cannot authorize this Goal. AC-05/06. |
| `scripts/check_goal_git_isolation.py` | KEEP | Protected/untracked content gate remains generic. AC-06/09. |
| `scripts/check_new_experiment_gate.py` | MERGE | Retain fail-closed release/preflight policy in consolidated validator; remove receipt coupling. AC-05/06. |
| `scripts/check_pm2_ci_keepalive.py` | KEEP | PM2-only CI policy and fake lifecycle. AC-07. |
| `scripts/compare_code_task_preflight.py` | DEFERRED | Historical comparison stays available but cannot authorize current execution. AC-05. |
| `scripts/experiment_failure_classifier.py` | KEEP | Stable typed failure classification; align output with AC-04. |
| `scripts/experiment_manifest.py` | MERGE | Generic canonicalizer plus explicit versioned Stage123 policy validator; remove embedded current values. AC-02. |
| `scripts/experiment_notification_policy.py` | KEEP | Pure notification decision policy; no real WxPusher in acceptance. AC-06/07. |
| `scripts/run_code_task_operational_calibration.sh` | COMPAT_SHIM | Thin environment setup plus one Python `exec`; warn on legacy use. AC-03/05. |
| `scripts/run_code_task_operational_calibration_queue.sh` | MIGRATE | Replace shell state machine with Python execution core; leave thin fail-closed/delegating shim if callers exist. AC-03/05/08. |
| `scripts/stage123_manifest_monitor.py` | KEEP | Manifest-derived read-only monitor state; remove duplicated policy if found. AC-02/06. |
| `scripts/stage123_manifest_release_dispatch.sh` | COMPAT_SHIM | Delegate to retained release gate; failed/incomplete evidence remains blocked. AC-05/06. |
| `scripts/stage123_preflight_receipt.py` | DELETE | Migrate necessary fields to `preflight_result.json`; legacy receipt is read-only and non-authoritative. AC-05/08. |
| `scripts/validation_deadline_controller.py` | MERGE | Deadline/process cleanup owned by Python execution core. AC-03/06. |

Additional production surfaces discovered outside the frozen 21-file selector:

| Surface | Disposition | Target responsibility and compatibility |
| --- | --- | --- |
| `verl/trainer/ppo/ray_trainer.py` calibration environment/file protocol | MIGRATE | Generic no-op/recording validation observer; no `CALIBRATION_*` in shared trainer. AC-01. |
| `recipe/.../stage123_deployability_receipt.py` | MIGRATE | Necessary result fields move to `acceptance_report.json`; old evidence fails closed. AC-05. |
| `recipe/.../stage123_stage12_producer_receipt.py` | DELETE | Producer identity comes from manifest/provenance; old receipt cannot authorize. AC-02/05. |
| `recipe/.../calibration_workload_descriptor.py` | KEEP | Workload description remains manifest-derived input, not a second policy owner. AC-02/06. |

## Test Dispositions

| Test group | Disposition | Required replacement behavior |
| --- | --- | --- |
| Manifest, release, PM2, failure classifier, git isolation, scorer dependency, deadline, runtime isolation | KEEP/MERGE | Preserve fail-closed behavior with structured assertions. AC-02/04/06/07. |
| `test_operational_calibration_runner.py` | MERGE | Replace shell-string assertions with fake adapters, persisted state, and thin-entrypoint policy. AC-03. |
| `test_operational_calibration_checker.py` | MERGE | Split by validator/decision/result behavior; mutate manifest facts. AC-02/04. |
| `test_operational_calibration_assembler.py` | MERGE | Test consolidated result rendering, not an independent assembly layer. AC-04/05. |
| `test_dynamic_calibration_interval.py` and prediction fixtures | DEFERRED | Keep only compatibility/serialization tests needed by current core; model-quality acceptance moves to next Goal. AC-05/06. |
| `test_stage123_adoption.py` | DELETE | Adoption receipts are removed. AC-05/08. |
| `test_stage12_producer_receipt.py` | DELETE/MIGRATE | Replace only with manifest/provenance fail-closed compatibility cases. AC-02/05. |
| Source-text assertions for env propagation, Docker/tmux cleanup, deadlines, and manifest policy | MIGRATE | Execute fake child/executables/clock/parser APIs. AC-03/07. |
| Static absence of systemd and thin shell entrypoint shape | KEEP | Narrow source policy is appropriate. AC-03/07. |
| Large copied benchmark fixture | MERGE | Replace with minimal generated/parameterized facts where full payload is not behaviorally required. AC-08. |

All 54 baseline test paths remain in the line-count class even when deleted or merged.
New replacement workflow tests also count toward the 4,000-line ceiling.

## Documentation Dispositions

| Path group | Disposition | Compatibility |
| --- | --- | --- |
| Ten `docs/joint_training/codereview/active/experiment_execution_reliability_*.md` files | ARCHIVE | Remove from active context; Git history preserves them. No runtime consumer. |
| `docs/joint_training/manifests/experiment_execution_reliability_plan_adoption.json` | DELETE | Superseded by Goal plan hash and append-only runtime ledger. |
| `docs/joint_training/manifests/goal_contract_dirty_adoption.json` | DELETE | Superseded by Goal runtime plus git-isolation checker. |
| `docs/joint_training/manifests/stage123_dirty_adoption.json` | DELETE | Dirty-baseline protection remains in generic content gate; adoption receipt removed. |
| Superseded active Goal and launch entrypoints | COMPAT_SHIM | Keep compact pointers to the Goal directory; no duplicated contract. |
| Goal `plan.md`, ledgers, prompts, inventory, and final acceptance | KEEP | The only active Goal artifacts. Prompt instances remain bounded lifecycle evidence, not a review chain. |

## Concrete Duplicated Facts To Remove

- `STAGE12_PRODUCER` and `VALIDATION_DATASETS` in
  `check_code_task_operational_calibration.py`.
- Literal total `1379` in checker decisions; derive from manifest dataset records.
- Checker-enforced receipt TTL equality; consume manifest policy TTL.
- Stage2/Stage3 final-step and fixed provenance rules inside generic `normalize()`.
- Queue-side run identity arrays or producer facts already present in normalized manifest.
- `CALIBRATION_VALIDATION_TIMELINE_FILE` and
  `CALIBRATION_VALIDATION_READY_FILE` in `RayPPOTrainer`.

## Compatibility Matrix

| Legacy class | Behavior after consolidation |
| --- | --- |
| Existing manifest schema version still supported | MIGRATE through versioned loader; no value rewriting. |
| Unsupported manifest schema | FAIL_CLOSED with supported-version message. |
| Legacy preflight/deployability/producer receipt | Read-only parser may explain migration; never current authority. |
| Old queue shell entrypoint | COMPAT_SHIM delegates to Python core or fails closed with migration command. |
| Historical calibration artifacts | KEEP byte-identical and outside cleanup roots. |
| Adoption/document-hash receipt | DELETE; Git plus Goal ledger is authoritative. |
| Failed/incomplete training artifact | FAIL_CLOSED at release gate. |

## Milestone Ownership

- Milestone 2 implements Trainer and manifest boundaries.
- Milestone 3 implements Python execution state and structured failures.
- Milestone 4 applies evidence migration, compatibility shims, and deletions.
- Milestone 5 proves preserved behavior and frozen budgets.

Any implementation discovery not represented above must be classified in
`findings.jsonl` before action. A new independently useful subsystem or acceptance
surface is not silently added to this inventory.
