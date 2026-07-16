# Milestone 2 Independent Review — F-EX-M2-16 and F-EX-M2-17

The independent GPT-5.5 medium reviewer verified the same-container Ray lifecycle.
The initial review found the cleanup trap was installed after `ray start`; the
implementer moved it before all Ray temp-root creation/start work. The same reviewer
performed a mechanical re-verification.

Commands run by the reviewer:

```bash
bash -n recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh
bash -n recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_manifest_queue_monitor_contract.py
```

Result: **PASS** for the scoped findings; `11 passed in 16.43s`.
Each Stage2/Stage3 wrapper now installs cleanup before `ray start`, uses a short
`/tmp/stage123-ray-${STAGE123_RUN_ID}` root, starts its local head on the calibrated
ports, exports localhost `RAY_ADDRESS`, and force-stops/removes the Ray root on exit.

This review does not replace final AC acceptance or the required committed implementation
identity check before another launch.
