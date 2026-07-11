# Engineering Rule Migration

This report records only recoverable project evidence. It does not invent incidents or
claim that uncited chat history is durable policy.

| Rule | Disposition | Source evidence | Result |
| --- | --- | --- | --- |
| ER-001 | replaced-by-gate | `docs/joint_training/constraints/experiment_tracking/training_result_release_gate_policy.md` | Canonical behavior is the local release checker and tests. |
| ER-002 | replaced-by-gate | `docs/joint_training/constraints/experiment_tracking/training_script_index_policy.md` | New queue/monitor cutoff is enforced from Git baselines. |
| ER-003 | retained | `CLAUDE.md` | Tmux remains judgment-dependent because runtime duration is contextual. |
| ER-004 | reworded | `docs/joint_training/constraints/principles/workspace_artifact_hygiene.md` | Scratch routing is expressed as a trigger/action/failure rule. |
| ER-005 | project-local | `docs/joint_training/plans/active/experiment_execution_reliability_goal.md` | The 8K shared profile is specific to the current Stage123 experiment. |
| ER-006 | project-local | `docs/joint_training/plans/active/experiment_execution_reliability_goal.md` | Human decision policy is specific to this Goal and guarded notification flow. |

The older prose remains historical or explanatory context. For machine-checkable rules,
the catalog points to the executable gate rather than duplicating detailed normative text.
