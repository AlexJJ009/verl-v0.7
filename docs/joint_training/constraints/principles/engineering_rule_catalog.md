# Engineering Rule Catalog

This catalog is the compact canonical form of project-level operating rules selected
for the experiment-execution workflow. Bulk history remains in the cited sources.

## ER-001

- Scope: training result publication
- Enforcement tier: machine-check
- Evidence source: `docs/joint_training/constraints/experiment_tracking/training_result_release_gate_policy.md`
- When: a tool is about to import a training result into SQLite or sync it to W&B.
- Do: run `scripts/training_result_release_gate.py check --run-name <RUN_NAME>` and continue only on exit code 0.
- Otherwise: the release is blocked and the attempt remains local diagnostic evidence.
- Checker: `scripts/check_experiment_workflow_fast.sh`; test `test_manifest_release_gate.py`; failure `release_gate_blocked`; reachability `fast,full`.

## ER-002

- Scope: new experiment queues and monitors
- Enforcement tier: structural
- Evidence source: `docs/joint_training/constraints/experiment_tracking/training_script_index_policy.md`
- When: a runnable queue or monitor is added, changed, or renamed after the recorded baseline.
- Do: make it manifest-native and pass `scripts/check_new_experiment_gate.py`.
- Otherwise: the workflow cutoff gate rejects the change as untraceable legacy structure.
- Checker: `scripts/check_new_experiment_gate.py`; test `test_new_experiment_gate.py`; failure `non_manifest_native_runnable`; reachability `fast,full`.

## ER-003

- Scope: training, monitoring, transfer, and large downloads
- Enforcement tier: judgment-only
- Evidence source: `CLAUDE.md`
- When: a command is expected to outlive the interactive shell or is operationally expensive to restart.
- Do: launch it in a named tmux session and record the session and evidence paths.
- Otherwise: disconnects can destroy work and leave the runtime state untraceable.

## ER-004

- Scope: generated code, tests, dry-runs, and benchmark artifacts
- Enforcement tier: machine-check
- Evidence source: `docs/joint_training/constraints/principles/workspace_artifact_hygiene.md`
- When: a command may create scratch files or runtime artifacts.
- Do: write under `/data-1/tmp/verl_agent_scratch/...` or a declared durable artifact root.
- Otherwise: the working tree becomes polluted and project evidence becomes indistinguishable from disposable output.
- Checker: `scripts/check_experiment_workflow_fast.sh`; test `test_context_size_budget.py`; failure `workspace_artifact_leak`; reachability `fast,full`.

## ER-005

- Scope: Stage1, Stage2, and Stage3 resource semantics
- Enforcement tier: structural
- Evidence source: `docs/joint_training/goals/experiment-execution-core-consolidation/plan.md`
- When: any Stage123 phase is rendered or launched.
- Do: consume the shared profile hash and preserve `MAX_RESPONSE_LENGTH=8192` with the full validation contract.
- Otherwise: the run is not a comparable Stage123 experiment and must stop before trainer startup.
- Checker: `scripts/experiment_manifest.py`; test `test_manifest_queue_monitor_contract.py`; failure `resource_profile_mismatch`; reachability `fast,full`.

## ER-006

- Scope: soft preflight threshold failures
- Enforcement tier: judgment-only
- Evidence source: `docs/joint_training/goals/experiment-execution-core-consolidation/plan.md`
- When: a soft threshold fails during unattended Goal execution.
- Do: stop, send one guarded `user_decision_required` notification with evidence, and wait for an interactive user decision followed by a reviewed manifest or policy commit and a fresh passing preflight.
- Otherwise: an unapproved high-risk run may consume GPU time and no phone action constitutes approval.
