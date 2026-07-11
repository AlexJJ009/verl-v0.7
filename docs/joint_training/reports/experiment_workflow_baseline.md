# Experiment Workflow Baseline

This report is generated from structured local evidence by
`scripts/experiment_workflow_baseline.py`. The initial deterministic fixture records
the Stage123 failure shape that motivated this Goal: full code validation can leave
all GPUs waiting while reward execution times out and RewardLoopWorker host RSS grows.

The canonical live baseline must record unavailable observations as `unknown`; it
must not infer metrics from absence of log lines. Bounded fixture scores and later
preflight scores are infrastructure evidence only, not formal experiment results.

## Deterministic Baseline

The initial fixture models the observed Stage123 scorer-starvation failure shape:

| Field | Value |
| --- | ---: |
| Submitted scoring items | 4 |
| Completed scoring items | 2 |
| Timeout count | 2 |
| Timeout rate | 0.5 |
| Scorer elapsed time | 71 seconds |
| RewardLoopWorker peak RSS | 191.25 GiB |
| GPU idle time while phase active | 60 seconds |
| GPU idle fraction | 0.75 |
| Primary terminal reason | `scorer_timeout` |

The fixture profile hash is
`f1bb99873e16bc2398e40cfe2b7597633f8ced2085827e6ad7cef308c5cae817`.
The fixture Docker ID is deliberately synthetic and cannot be used as live-runtime
provenance.

Verification commands and expected fields are defined by AC-01 and AC-02 in
`docs/joint_training/plans/active/experiment_execution_reliability_goal.md`.
