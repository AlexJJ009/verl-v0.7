# Milestone 2 Independent Review — F-EX-M2-10 and F-EX-M2-11

- Reviewer: independent GPT-5.5 medium reviewer
- Plan: v13
- Reviewed baseline commit: `9c736bc029f4da16e5932a16b3f8bdf49dba57f1`, plus the current dirty/untracked worktree
- Verdict: **PASS**

## AC verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | The treatment admission has its own hash-bound authorization and batch validation uses it rather than original three-run preflight. |
| AC-03 | PASS | Certified control evidence remains hash-bound and preserved old evidence matches the certificate. |
| AC-04 | PASS | Treatment Stage2 is distinct and `/data-2` rooted; direct and batch validation admit only Stage2 then Stage3. |
| AC-07 | PASS | Prepared admissions fail closed unless authorized; bad run IDs and prepared-without-allow reject. |
| AC-08 | PASS | Monitor, state, and provenance roots are distinct; old state is not adopted. |
| AC-09 | PASS | Run set and training parameters remain frozen; rendering changes treatment identity/provenance only. |

## Reviewer-owned verification

```bash
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_control_reuse.py tests/experiment_workflow/test_manifest_queue_monitor_contract.py
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_manifest.py render /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260715T154800Z/treatment-manifest.yaml --format json
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/experiment_execution_core.py batch-validate --manifest /data-1/tmp/verl_agent_scratch/experiment_workflow/stage123/treatment-reuse-20260715T154800Z/treatment-batch-manifest.json --repo-root /data-1/code/verl
```

The tests passed (`14 passed in 17.85s`); manifest render and batch validation exited zero.

## Finding verdicts

- `F-EX-M2-10`: PASS. Treatment-only admission uses a dedicated validator/authorizer rather than applying the original control+Stage2+Stage3 preflight topology.
- `F-EX-M2-11`: PASS. `prepare` rejects non-`/data-2` artifact roots and rewrites the Stage3 pending producer path, output path, and provenance path consistently.

No blocking in-scope defect or contract contradiction was found.

## Deferred suggestion

The prepared batch manifest retains a top-level `prepared_not_authorized` / pending authorization label after its referenced admission becomes authorized. This is non-blocking because `batch-validate` binds and validates the authorized admission hash; it should not be changed during this frozen launch.

## Review limitation

This is a prelaunch review and does not replace final execution acceptance.
