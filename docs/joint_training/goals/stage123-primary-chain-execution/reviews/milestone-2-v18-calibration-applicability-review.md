# Milestone 2 V18 Calibration Applicability Review

## Review Identity

- Reviewer: independent GPT-5.5 medium Plan/implementation reviewer
- Review type: Milestone Review
- Goal: `stage123-primary-chain-execution`
- Plan version: 18
- Candidate commit: `458aa212067ba04e9bcea441ae6224d902b81836`
- Base commit: `31b1c31b`
- Applicable ACs: `AC-01`, `AC-12`
- Scope: `F-EX-IMPL-02`, `F-EX-IMPL-03`, V18 calibration applicability

## Overall Verdict

`PASS`

The committed candidate satisfies the V18 applicability contract for the reviewed scope. It proves an empty capacity-plane diff against the accepted source calibration reconstructed from recipe gitlink `6fcccb353a87045a17f9d52b3821f0e20f7f9a9d`, derives a current-binding calibration result, preserves fresh V16 output identities, and does not show GPU training/calibration or external-service launch evidence in the required checks.

## Per-AC Verdicts

| AC | Verdict | Evidence |
| --- | --- | --- |
| AC-01 | PASS | Current implementation tree comparison passes with `implementation_tree_sha256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b`; calibration result schema validates as `decision=passed`; applicability artifact binds Plan hash, evidence commit, implementation tree, and empty capacity diff. |
| AC-12 | PASS | Focused Stage123 tests pass; fresh V16 manifest roots/names are present; applicability report shows no V13/V14 capacity or artifact reuse; GPU process query is empty. |

## Commands And Evidence

- `goal-plan-runtime validate-plan docs/joint_training/goals/stage123-primary-chain-execution` -> `PASS`, exit `0`.
- `goal-plan-runtime validate-runtime docs/joint_training/goals/stage123-primary-chain-execution` -> exit `0`; runtime reports Plan v18 `READY`, latest review before this one is V18 Plan review, and `F-EX-IMPL-02` / `F-EX-IMPL-03` remain open pending implementation review.
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -m pytest -q tests/experiment_workflow/test_stage123_calibration_applicability.py tests/experiment_workflow/test_stage123_validation_protocol.py tests/experiment_workflow/test_stage123_admission_bundle.py tests/experiment_workflow/test_stage123_wrapper_preflight_integration.py tests/experiment_workflow/test_stage123_end_to_end.py tests/experiment_workflow/test_stage123_preflight_model_identity.py` -> `20 passed in 95.25s`, exit `0`.
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python scripts/implementation_tree_identity.py --repo-root /data-1/code/verl --boundary-manifest config/experiment_execution/stage123_implementation_boundary_v1.json --format json --compare docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl` -> exit `0`; output includes recipe gitlink `aa972ba489f75b9faebf42ae91307a542749faa3` and `implementation_tree_sha256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b`.
- `REPO_HOST=/data-1/code/verl /data-1/verl07/run_train.sh python -c 'import json; from pathlib import Path; from scripts.calibration_result import validate; value=json.loads(Path("docs/joint_training/goals/stage123-execution-readiness/calibration_result.json").read_text()); schema=json.loads(Path("config/experiment_execution/calibration_result_schema_v1.json").read_text()); result=validate(value,schema); print(json.dumps(result,sort_keys=True)); raise SystemExit(0 if result["ok"] else 1)'` -> `{"decision":"passed","failures":[],"ok":true}`, exit `0`.
- `jq '{decision,capacity_differences,source_capacity_sha256,candidate_capacity_sha256,implementation_tree_sha256,evidence_commit,plan_sha256}' docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json` -> `decision=applicable`, `capacity_differences=[]`, matching source/candidate capacity SHA `7cab911b63caba6c001e29e1b0a7cb7d7bacf04b0e23a642e7638ed0cd91e2f5`, `implementation_tree_sha256=f97f5478f0c32c602d36f3eacea43073ea9b865a3396916aa584af33754fb39b`, `evidence_commit=13f0e2765fe4ac3adab2642cb4cc239147888ee1`, `plan_sha256=471c12f95e1969948105626d25ddb90659bc2e8242d8309fe9576a9145850852`.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` -> no rows, exit `0`.
- Reviewer recompute from base source calibration and source recipe gitlink `6fcccb353a87045a17f9d52b3821f0e20f7f9a9d` -> exit `0`, `decision=applicable`, `capacity_differences=[]`, source/candidate capacity SHA both `7cab911b63caba6c001e29e1b0a7cb7d7bacf04b0e23a642e7638ed0cd91e2f5`, source manifest SHA `323bcc6084c1b01653bdf3cb5b299cab51c76864c8ca24878a9e5b002cb76278`, candidate manifest SHA `1d2c5d55da7d758784b7383a574449cecc9bd67281a2b17752150dcb542d14fc`.
- Reviewer negative probe: mutating source calibration training steps to `1` -> exit `1`, `source calibration contains training or failed repetition`.
- Reviewer negative probe: mutating a capacity field (`MAX_RESPONSE_LENGTH`) -> exit `1`, profile validation fails closed before applicability.

## Implementation Evidence

- `scripts/stage123_calibration_applicability.py` defines capacity fields, reconstructs source profile and manifest from `--source-recipe-gitlink`, rejects missing capacity fields, diffs flattened source/candidate capacity snapshots, requires source calibration `decision=passed`, exact Stage1/Stage2/Stage3 phase set, zero training steps, and released resources, then writes an applicable report only when the diff is empty.
- `tests/experiment_workflow/test_stage123_calibration_applicability.py` covers empty diffs, changed/missing capacity fields, and rejection of training-bearing calibration.
- `recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml` uses V16 artifact root, scratch root, chain label, run prefixes, tmux names, and Stage2/Stage3 model/provenance paths under V16 roots; no V13/V14/treatment-reuse/stage3-handoff roots appear in the fresh manifest.
- `docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json` binds source and candidate manifest/profile hashes, source and candidate capacity hashes, current implementation tree, evidence commit, Plan hash, and authorization decision.

## Files Changed In Candidate

- `docs/joint_training/goals/stage123-execution-readiness/calibration_result.json`
- `docs/joint_training/goals/stage123-execution-readiness/implementation-tree.jsonl`
- `docs/joint_training/goals/stage123-primary-chain-execution/calibration_applicability.json`
- `docs/joint_training/goals/stage123-primary-chain-execution/findings.jsonl`
- `docs/joint_training/goals/stage123-primary-chain-execution/plan.md`
- `docs/joint_training/goals/stage123-primary-chain-execution/reviews/plan-v18-calibration-applicability-review-prompt.md`
- `docs/joint_training/goals/stage123-primary-chain-execution/runtime.jsonl`
- `recipe`
- `scripts/stage123_calibration_applicability.py`
- `tests/experiment_workflow/test_stage123_calibration_applicability.py`

## Blocking In-Scope Defects

None.

## Deferred Suggestions

None.

## Contract Contradictions

None.

## Single Most Likely Weakness In This Review

The committed current calibration result is already the derived V18 result, so the reviewer-owned positive recompute had to extract the accepted source calibration from base commit `31b1c31b`; the recomputed report has a different timestamp/hash from the committed report but matching substantive bindings and empty capacity diff.
