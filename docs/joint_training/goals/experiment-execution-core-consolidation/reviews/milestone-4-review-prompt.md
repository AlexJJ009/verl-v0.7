# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `milestone`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/experiment-execution-core-consolidation`
- Frozen Plan version: `1`
- Current milestone: `Milestone 4`
- Base commit: `7f570f4e`
- Candidate commit: `59b802cf`
- Applicable ACs: `AC-05,AC-08`

## Required Verification

REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow -k 'compat or migration or receipt or adoption or evidence'; git diff --numstat 390be933...59b802cf -- scripts tests/experiment_workflow verl/trainer/ppo/ray_trainer.py recipe/on_policy_wdl_sft/code_task docs/joint_training/codereview/active docs/joint_training/manifests docs/joint_training/goals/experiment-execution-core-consolidation; git diff --check 390be933...59b802cf

## Additional Task-Specific Focus

Verify every inventory legacy class has a disposition, only preflight_result/calibration_result/acceptance_report authorize current execution, legacy receipt/adoption evidence fails closed, temporary historical migration is byte-identical, documentation-only changes need no receipt, protected assets remain unstaged, submodule commit is valid, and independently recompute production <=3000, tests <=4000, active docs <=650 with each class net-negative. Audit retained >300 file justification.

## Reviewer Rules

1. Read the frozen Goal contract and inspect the candidate diff.
2. Run the required verification commands yourself.
3. Evaluate only the applicable frozen ACs as `PASS`, `FAIL`, or `WEAKENED`.
4. Classify additional observations as `IN_SCOPE_DEFECT`, `DEFERRED_SUGGESTION`, or `CONTRACT_CONTRADICTION`.
5. Do not turn a suggestion outside the frozen ACs into a blocking requirement.
6. Do not amend the Plan, continue implementation, or accept implementer claims as evidence.
7. Receipt existence, documentation text, and test names are not proof without reviewer-owned behavioral evidence.
8. Report the commands executed, relevant output, reviewed Plan version, and reviewed commit.

## Required Output

- Review identity
- Overall verdict
- Per-AC verdict table
- Commands and evidence
- Blocking in-scope defects
- Deferred suggestions
- Contract contradictions
- The single most likely weakness in this review
