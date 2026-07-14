# Independent Goal Review

Use the stable Goal Plan reviewer protocol. Do not continue implementation.

## Review Context

- Review type: `Final Acceptance`
- Goal directory: `/data-1/code/verl/docs/joint_training/goals/calibration-qualification`
- Frozen Plan version: `1`
- Current milestone: `Milestone 7`
- Base commit: `81fdd6d1f315200981bf89e268089f3c927f366d`
- Candidate commit: `34264e45cf7c054fd92f433b73442944f5da0567`
- Applicable ACs: `AC-01, AC-02, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08`

## Required Verification

goal-plan-runtime validate-plan docs/joint_training/goals/calibration-qualification
goal-plan-runtime validate-runtime docs/joint_training/goals/calibration-qualification
FULL_GATE_SESSION=cq_final_cpu_gate; FULL_GATE_LOG=/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/final-cpu-preservation-gate.log; test ! -e "$FULL_GATE_LOG"; tmux new-session -d -s "$FULL_GATE_SESSION" "cd /data-1/code/verl && bash scripts/check_experiment_workflow_full.sh >'$FULL_GATE_LOG' 2>&1; rc=\$?; echo \$rc >'$FULL_GATE_LOG.rc'"; while tmux has-session -t "$FULL_GATE_SESSION" 2>/dev/null; do sleep 10; done; cat "$FULL_GATE_LOG.rc"; cat "$FULL_GATE_LOG"
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/check_code_task_operational_calibration.py --report /data-1/tmp/verl_agent_scratch/experiment_workflow/calibration/probe-20260714T015835Z/probe-report.json --manifest recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml
REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json
sha256sum docs/joint_training/goals/calibration-qualification/plan.md docs/joint_training/goals/calibration-qualification/calibration_result.json config/experiment_execution/calibration_result_schema_v1.json
git diff --stat 81fdd6d1f315200981bf89e268089f3c927f366d..34264e45cf7c054fd92f433b73442944f5da0567
git status --short

## Additional Task-Specific Focus

Perform independent final acceptance from committed candidate 34264e45cf7c054fd92f433b73442944f5da0567. Evaluate AC-01 through AC-08 individually as PASS, FAIL, or WEAKENED using reviewer-owned evidence. Run the full CPU preservation gate exactly once and only once, in the provided tmux command; do not rerun it even if it fails, and report exact failure instead. Revalidate the sole calibration_result.json, fresh GPU probe, implementation identity, evidence commit, Plan hash, result hash, authorization usage, cleanup, protected assets, and absence of new receipt/adoption/review-document chains. Audit candidate diff for deleted, skipped, loosened, or trivialized tests and confirm no real external service or training step was used. You are the only actor allowed to complete acceptance.md. Modify only docs/joint_training/goals/calibration-qualification/acceptance.md after evidence is complete. Bind it to Plan version 1, Plan hash cac84de536b1b64ac84bb3fe0197b4e2693dfdf953ea9d563a134bf6f9e7427b, candidate commit 34264e45cf7c054fd92f433b73442944f5da0567, implementation identity 453b60bf1b626934a20916a1ca1aa9c90cc9de2b028d19b5ad875471d3c90ead, evidence commit 7cc302aa4ec9ae3efd8729749342a83f441753d7, manifest hash e665049cc67a40c32f0b104058bfe4e20c2529dc22328a485622bed78d3c8f0c, and calibration_result file hash 8bdf646803916e231d7ce684edfa7302a706824e3afc008887c009374986218d. Status is ACCEPTED only if every AC is PASS. Do not modify runtime.jsonl, findings.jsonl, production code, tests, prompts, or protected assets. You are not alone in the codebase; do not revert others' edits.

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
