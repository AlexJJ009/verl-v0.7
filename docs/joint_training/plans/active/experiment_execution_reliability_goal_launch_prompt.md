# Experiment Execution Reliability Goal Launch Prompt

Resume the persistent Goal using this single execution contract:

```text
/data-1/code/verl/docs/joint_training/plans/active/experiment_execution_reliability_goal.md
```

Repository and branch:

```text
/data-1/code/verl
codex/experiment-execution-reliability
```

Start from the contract's recorded baseline. Do not restart the superseded 30-AC
workflow and do not resume GPU calibration or the 27-run Stage123 queue.

Execute Milestones 1-6 serially:

1. record a file-level duplication and deletion inventory;
2. remove calibration file protocols from generic `RayPPOTrainer`;
3. make the normalized manifest the only owner of concrete Stage123 facts;
4. move queue state, deadlines, and cleanup from shell into Python;
5. replace source-text runtime assertions with executable fakes;
6. collapse evidence to preflight, calibration, and acceptance results and pass
   one non-duplicative CPU gate.

The branch must have a meaningful negative line count. Combined production, test,
and active reliability-document lines must decrease by at least 25% from the
Milestone 1 baseline. Do not add new checker, receipt, AC, or review-document layers
to satisfy tests.

Preserve these user-owned untracked paths without staging or modifying them:

```text
.claude/skills/experiment-registry
docs/joint_training/plans/active/qwen3_1p7b_code_stage123_plateau_breakthrough.md
test_data/
```

Preserve diagnostic calibration artifacts, but do not treat them as current
authorization. External services must be mocked in CPU acceptance. Long-running
work uses tmux; persistent CI uses PM2, never systemd.

Commit each independently verifiable milestone. Use focused tests during
implementation and one final non-duplicative CPU gate. The implementer must not
self-accept. A fresh GPT-5.5 medium reviewer must run the final commands from
committed state and report every AC as PASS, FAIL, or WEAKENED.

After independent CPU acceptance, stop and ask the user before generating a fresh
GPU preflight or running one bounded sampled-validation probe. Do not automatically
advance to formal calibration or training.
