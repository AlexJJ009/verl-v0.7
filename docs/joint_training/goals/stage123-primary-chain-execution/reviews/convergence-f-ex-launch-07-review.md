# F-EX-LAUNCH-07 Convergence Review

- Review type: `Convergence Review`
- Reviewer: independent GPT-5.5 medium convergence reviewer
- Goal directory: `docs/joint_training/goals/stage123-primary-chain-execution`
- Frozen Plan version: `18`
- Reviewed commit: `425f844734607b6e02bcd83a1de702d6e3239a30`
- Plan SHA256: `471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852`
- Verdict: `IN_SCOPE_SHARED_LAYER_REPAIR`

## Overall Verdict

F-EX-LAUNCH-07 is an in-scope shared-layer architectural defect, not a Plan contradiction and not an AC change.

The root cause is that the current launch stack still has full admission validation embedded in phase-child paths after an item has already been admitted:

1. `BatchExecutor` validates a `stage123_admission_bundle` by calling `validate_current_checkout(...)`, which enforces calibration and preflight wall-clock freshness.
2. Each Stage123 wrapper sources `stage123_manifest_gate.sh`, and `stage123_require_formal_admission` calls `execution_results.py admission validate --require-accepted`, which also calls `validate_current_checkout(...)` and re-enforces the same 3600-second preflight TTL.
3. After V18 Control ran for about 98 minutes, the full-bundle preflight was stale. A subsequent Stage2 child therefore re-ran the original 3600-second admission check and failed before training, even though Plan v18 says full admission is required before `item_started` and later phases use the live admitted item plus static binding and host-health checks.

The prior F-EX-M2-24 convergence fix removed host-facts wall-clock age checks from the treatment-reuse path only. It did not split the shared `stage123_admission_bundle` path into one-time fresh item admission versus post-admission phase static validation. That is why the same TTL class recurred in the full V18 batch.

## AC Verdicts

| AC | Verdict | Rationale |
| --- | --- | --- |
| AC-01 | FAIL | The implementation still applies full admission freshness after `item_started`; direct invocation must keep full admission, but already-admitted batch phases must not reapply wall-clock TTL. |
| AC-07 | FAIL | The failed Stage2 child is correctly terminal for the old item, but the failure itself is caused by stale admission freshness after successful Control, which Plan v18 excludes as a phase failure. |
| AC-08 | PASS | Persisted state accurately records Control success, Stage2 child exit, cleanup, skipped Stage3, and `completed_with_failures`; no stale process was observed by `nvidia-smi`. |
| AC-12 | PASS | No evidence shows decoder or V13/V14 reuse drift in this finding; the defect is launch/admission layering. |

## Commands And Evidence

```bash
goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution
goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution
```

Result: plan validation passed; runtime validation passed and reports Plan v18 active with F-EX-LAUNCH-07 open.

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q \
  tests/experiment_workflow/test_manifest_queue_monitor_contract.py \
  tests/experiment_workflow/test_stage123_admission_bundle.py \
  tests/experiment_workflow/test_experiment_batch_routing.py \
  tests/experiment_workflow/test_stage123_control_reuse.py
```

Result: `26 passed in 49.87s`.

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/execution_results.py admission validate \
  --bundle docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json \
  --require-accepted \
  --protected-baseline docs/joint_training/goals/stage123-execution-readiness/protected-assets-baseline.jsonl \
  --repo-root /data-1/code/verl
```

Result: failed closed with `preflight result is stale: age=15336s max=3600s`.

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate \
  --manifest docs/joint_training/goals/stage123-primary-chain-execution/experiment_batch_manifest.json \
  --state-root /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/convergence-f-ex-launch-07-review-state \
  --repo-root /data-1/code/verl
```

Result: failed closed with `Stage123 admission rejected: current_checkout: preflight result is stale: age=15337s max=3600s`.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Result: no rows; no active GPU process observed.

Preserved V18 state under `/data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/primary-v18-20260716T123423Z-portfix/state` shows:

- `frac25-stage1-control.json`: `status=succeeded`, `resume_from_checkpoint=false`.
- `frac25-stage2.json`: `status=failed`, `failure.code=child_exit`, `returncode=1`, `resume_from_checkpoint=false`.
- `events.jsonl`: `item_started` occurred once, Control succeeded, Stage2 then started and failed, `item_failed_fallback` fired, and the batch reached `completed_with_failures`.

## Blocking In-Scope Defect

### F-EX-LAUNCH-07.1: Admission Freshness Is Not Scoped To Item Start

The repair must be in the shared admission/batch/phase boundary, not in a Stage123-only workaround.

Exact repair boundary:

1. Keep direct phase invocation strict: without a verified batch-admitted-item context, `stage123_require_formal_admission` must continue to run full `execution_results.py admission validate --require-accepted`, including calibration/preflight TTL.
2. Add an explicit no-TTL static-binding validation path for already-admitted immutable batch items. It must verify bundle SHA, run ID membership, command binding, implementation tree, protected baseline, manifest/profile/input hashes, acceptance binding, and host-health/profile checks, but must not call `enforce_freshness` for calibration or preflight after `item_started`.
3. Persist the item admission authority at `item_started`: item ID, run IDs, command SHA, admission bundle SHA, implementation tree SHA, plan SHA, batch manifest SHA, and the full-admission validation timestamp/evidence.
4. During phase child launch, pass only consistency inputs from `BatchExecutor` into the child environment, for example admitted item ID, run ID, batch manifest SHA, admission bundle SHA, and a static-validation mode. These values are not a hostile security boundary, but mismatches must fail closed.
5. On batch process restart for an already-started item, reload the persisted admitted-item record and run static binding/host-health validation only. Do not re-run wall-clock TTL solely because Control consumed more than 3600 seconds.
6. Preserve no retry/resume: the failed V18 full item remains terminal diagnostic evidence. A repaired run must use a new execution identity, new state root, and fresh output roots.

Required tests:

1. Direct phase invocation with stale preflight still fails with the stale-preflight error.
2. A full batch item performs full admission once before `item_started`.
3. A two-phase or three-phase batch with a fake clock advanced beyond 3600 seconds between phases still launches the next phase using static post-admission validation.
4. Fabricated or incomplete batch environment is rejected: wrong bundle SHA, wrong item ID, wrong run ID, wrong command SHA, or missing persisted admitted-item record must fail closed.
5. Restart after `item_started` and successful Control must not reapply TTL, must not set `resume_from_checkpoint=true`, and must continue only through the admitted next phase.
6. `batch-validate` for a not-yet-started item remains full admission and can still fail on stale preflight.
7. The phase wrapper/gate tests must distinguish direct full admission from batch admitted-item static validation; the current text-only check that batch env is not authorization is too weak.

## V18 Control Reuse Boundary

The successful V18 Control may be reused only as certified baseline evidence for one new treatment-only execution identity:

- Source evidence: the preserved V18 item state must prove Control succeeded and Stage2 failed before any Stage2 training/checkpoint/provenance/extraction output.
- New identity: create a distinct Stage2 -> extraction -> Stage3 treatment-only manifest/admission/batch with fresh run prefixes, artifact roots, state root, monitor path, and authorization binding.
- No retry/resume: do not resume the failed V18 item, do not edit its state/events, do not convert its terminal `completed_with_failures` record into success, and do not reuse any partially created Stage2 output.
- Control usage: the V18 Control final metrics/checkpoint/provenance are baseline evidence only; the Control final weights are not Stage2 input.
- Admission: treatment-only admission may reuse the certified Control baseline and the current training-plane identity, but it must still bind the new treatment manifest, implementation tree, plan hash, admission hash, and host/profile health.

This is the same outcome surface as AC-01/07 recovery integrity and does not change the scientific estimand or acceptance criteria.

## Deferred Suggestions

None.

## Contract Contradictions

None. Plan v18 already states that full admission is required before `item_started`, later phases use a live single-use capability plus static binding/host-health checks, and wall-clock age after `item_started` is not a phase failure.

## Most Likely Weakness

I did not recover a dedicated Stage2 child stdout/stderr log for the exact stale-preflight message; the preserved state records only `child_exit`. The causal conclusion is still strong because reviewer-owned direct admission and batch validation both currently reproduce the stale preflight failure, and the inspected wrapper/core code shows the same full-admission path is executed by phase children.
