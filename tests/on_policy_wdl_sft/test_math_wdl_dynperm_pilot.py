# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

from scripts.math_wdl_first_step_gate import DYNPERM_METRICS, REQUIRED_METRICS, validate_step_one

ROOT = Path(__file__).resolve().parents[2]
MATH = ROOT / "recipe/on_policy_wdl_sft/math_task"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_dynperm.yaml"
CAUSAL_MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_causal_p60.yaml"


def _read(name: str) -> str:
    return (MATH / name).read_text(encoding="utf-8")


def test_two_variable_dynperm_interface_is_final_in_shared_causal_entry() -> None:
    common = _read("run_math_qwen3_1p7b_wdl_causal_p60_common.sh")
    call = common[common.index('exec bash "${SCRIPT_DIR}/run_s2_math_qwen3_1p7b_stage123_common.sh"') :]
    assert "export DYNPERM_ENABLED=${DYNPERM_ENABLED:-false}" in common
    assert "export DYNPERM_RHO=${DYNPERM_RHO:-0.0}" in common
    assert common.index('"$@"') < common.index('weak_logit_permutation.enabled="$DYNPERM_ENABLED"')
    assert common.index('"$@"') < common.index('"${DYNPERM_FORMAL_OVERRIDES[@]}"')
    assert call.index('"$@"') < call.index('weak_logit_permutation.rho="$DYNPERM_RHO"')
    assert "weak_logit_permutation.seed=42" in call
    assert "weak_logit_permutation.row_chunk_size=16" in call
    assert "weak_logit_permutation.audit_rows=4" in call
    printed = _read("../staged_v1/_run_stage2_model2_rollout_common.sh")
    assert "FREEZE_MODEL1=${FREEZE_MODEL1:-false}" in printed
    assert "DYNPERM_ENABLED=${DYNPERM_ENABLED:-false}" in printed
    assert "DYNPERM_RHO=${DYNPERM_RHO:-0.0}" in printed


def test_standard_c_and_fixed_m1_own_only_the_model1_update_state() -> None:
    arm_c = _read("run_math_qwen3_1p7b_wdl_causal_arm_c.sh")
    fixed = _read("run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh")
    assert "export WDL_ARM_ID=standard-c" in arm_c
    assert "export FREEZE_MODEL1=false" in arm_c
    assert "export WDL_ARM_ID=fixed-m1-stage1" in fixed
    assert "export FREEZE_MODEL1=true" in fixed
    for invariant in ("export FUSION_LAMBDA=0.8", "export FUSION_MODE=mixture"):
        assert invariant in arm_c
        assert invariant in fixed


def test_dynperm_admission_hard_pins_shared_non_treatment_contract() -> None:
    admission = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")
    for exact_pin in (
        "export TOTAL_TRAINING_STEPS=60",
        "export FUSION_LAMBDA=0.8",
        "export FUSION_MODE=mixture",
        "restored_from_causal_p60_joint_20260812/final_model",
        "stage1_control_stage2_then_stage3.parquet",
        "export WDL_SFT_BETA=0.0",
        "export LOSS_MODE=wdl_sft",
        "export LR=1e-6",
        "export DATA_SEED=20260719",
        "export DATA_SHUFFLE=False",
        "export JOINT_TRAINING_ROLLOUT_SOURCE=model2",
        "export TRAIN_PROMPT_BSZ=64",
        "export ROLLOUT_N=8",
        "export MAX_RESPONSE_LENGTH=4096",
        "export VAL_N=3",
        'export LOG_DIR="${CAUSAL_ARTIFACT_ROOT}/logs"',
    ):
        assert exact_pin in admission
    for final_override in (
        "data.seed=20260719",
        "data.shuffle=False",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.fsdp_config.seed=42",
        "actor_rollout_ref.actor.data_loader_seed=42",
        "actor_rollout_ref.actor.ppo_mini_batch_size=512",
        "actor_rollout_ref.actor.policy_loss.loss_mode=wdl_sft",
        "actor_rollout_ref.actor.policy_loss.wdl_sft_beta=0.0",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.submodel_kl.enabled=false",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.seed=0",
        "trainer.total_training_steps=60",
    ):
        assert final_override in admission


def test_p60_launcher_runs_fixed_m1_then_standard_c_with_same_two_variables() -> None:
    launcher = _read("run_math_qwen3_1p7b_wdl_dynperm_p60.sh")
    assert ': "${DYNPERM_ENABLED:?set DYNPERM_ENABLED=true}"' in launcher
    assert ': "${DYNPERM_RHO:?set DYNPERM_RHO to 0, 0.25, 0.5, or 1}"' in launcher
    assert "20|30" not in launcher
    fixed = "run_math_qwen3_1p7b_wdl_dynperm_fixed_m1_p60.sh"
    standard = "run_math_qwen3_1p7b_wdl_dynperm_standard_c_p60.sh"
    assert launcher.index(fixed) < launcher.index(standard)
    arm_common = _read("run_math_qwen3_1p7b_wdl_dynperm_p60_arm_common.sh")
    assert "export TOTAL_TRAINING_STEPS=60" in arm_common
    assert '[ -z "${TMUX:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]' in arm_common
    assert "nvidia-smi --query-gpu=utilization.gpu" in arm_common
    for entry in (fixed, standard):
        text = _read(entry)
        assert 'if [ "$#" -ne 0 ]' in text
        assert "DYNPERM_ENABLED" in text
        assert "DYNPERM_RHO" in text


def test_formal_p60_entries_reject_rho_outside_the_frozen_matrix() -> None:
    env = os.environ | {"DRY_RUN": "1", "DYNPERM_ENABLED": "true", "DYNPERM_RHO": "0.75"}
    for entry in (
        "run_math_qwen3_1p7b_wdl_dynperm_fixed_m1_p60.sh",
        "run_math_qwen3_1p7b_wdl_dynperm_standard_c_p60.sh",
    ):
        result = subprocess.run(
            ["bash", str(MATH / entry)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "formal DynPerm P60 rho must be one of" in result.stderr


def test_formal_dynperm_is_p60_only_and_candidate_bound() -> None:
    causal = _read("run_math_qwen3_1p7b_wdl_causal_p60_common.sh")
    admission = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")
    assert "formal DynPerm experiments are P60-only" in causal
    assert "formal DynPerm P60 requires DYNPERM_LAUNCH_RECEIPT" in admission
    assert "formal DynPerm P60 requires the exact container image id" in admission
    assert '"max_training_steps": 60' in admission
    assert '"parent_candidate_sha": parent_sha' in admission
    assert '"recipe_candidate_sha": recipe_sha' in admission
    assert '"image_id": image_id' in admission
    assert 'arm_id not in receipt.get("arms", [])' in admission
    assert "DYNPERM_PILOT_ADMISSION_RECEIPT" not in admission
    assert "formal DynPerm launch requires clean parent and recipe worktrees" in admission
    assert "--untracked-files=no" not in admission


def test_engineering_receipt_contract_is_hard_pinned() -> None:
    admission = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    evidence = manifest["engineering_evidence"]
    assert evidence["job_id"] == 146
    assert evidence["candidate_sha"] == "686f3ee1f190387581e38847cb0e75f055021caa"
    assert evidence["result"] == "PASS"
    assert evidence["world_size"] == 8
    assert evidence["formal_experiment"] is False
    assert "ancestor core/FSDP engineering evidence only" in evidence["boundary"]
    assert evidence["receipt_sha256"] == "3c757ffe6eaed509019bc8fd1b338ed8ab8244803f1f7d32510d7b7fc1eb89a2"
    assert evidence["receipt_sha256"] in admission
    assert "DYNPERM_ENGINEERING_RECEIPT" not in admission
    assert "686f3ee1f190387581e38847cb0e75f055021caa" in admission
    assert 'receipt.get("candidate_sha") != expected_candidate' in admission


def test_manifest_matches_c_and_models_the_two_p60_arms() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    causal = yaml.safe_load(CAUSAL_MANIFEST.read_text(encoding="utf-8"))
    contract = manifest["training_contract"]
    for key in (
        "beta",
        "loss_mode",
        "kl_enabled",
        "rollout_source",
        "lr",
        "lr_warmup_steps",
        "data_shuffle",
        "randomness",
        "validation_frequency",
        "save_frequency",
        "validation_views",
        "track_counterfactual_submodel_losses",
    ):
        assert contract[key] == causal["training_contract"][key]
    assert manifest["reward_contract"] == causal["reward_contract"]
    assert manifest["validation"] == causal["validation"]
    for key in ("model1", "dataset_receipt", "train_file", "checkpoint_root", "manipulation_receipt"):
        assert manifest["paths"][key] == causal["paths"][key]
    assert contract["final_step"] == 60
    assert contract["protected_checkpoint_steps"] == [20, 40, 45, 50, 60]
    assert manifest["dynperm_contract"]["parameter_interface"] == {
        "enabled": "DYNPERM_ENABLED",
        "rho": "DYNPERM_RHO",
    }
    assert [(run["id"], run["freeze_model1"]) for run in manifest["runs"]] == [
        ("standard-c", False),
        ("fixed-m1-stage1", True),
    ]
    assert manifest["execution"]["rho_values"] == [0.0, 1.0, 0.25, 0.5]
    assert manifest["execution"]["queue_order"] == ["fixed-m1-stage1", "standard-c"]
    assert manifest["execution"]["formal_job_count"] == 8
    assert manifest["execution"]["slurm_node_count"] == 3
    assert manifest["launch_allowed"] is False
    assert manifest["execution"]["formal_runs_started"] is False


def test_three_node_slurm_matrix_prioritizes_fixed_model1_and_is_fail_closed() -> None:
    submitter = _read("submit_math_qwen3_1p7b_wdl_dynperm_p60_matrix.sh")
    assert "RHO_VALUES=(0 1 0.25 0.5)" in submitter
    assert "ARM_IDS=(fixed-m1-stage1 standard-c)" in submitter
    assert "ARM_NICE=(0 1000)" in submitter
    assert "DYNPERM_SUBMIT_AUTHORIZED:-0" in submitter
    assert "PREVIEW ONLY" in submitter
    assert "sinfo -N -h -p l40s" in submitter
    assert "DYNPERM_ENABLED=true,DYNPERM_RHO=${rho}" in submitter
    assert "DYNPERM_PARENT_SHA=${PARENT_SHA}" in submitter
    assert "DYNPERM_RECIPE_SHA=${RECIPE_SHA}" in submitter
    assert "DYNPERM_IMAGE_ID=${IMAGE_ID}" in submitter
    assert "DYNPERM_LAUNCH_RECEIPT=${launch_receipt}" in submitter
    assert "DYNPERM_EVIDENCE_RELAY_HOST=${DYNPERM_EVIDENCE_RELAY_HOST}" in submitter
    assert submitter.index("launch_receipts=()") < submitter.index("sbatch --parsable")
    assert '"status": "authorized"' in submitter
    assert '"max_training_steps": 60' in submitter
    assert 'set(receipt.get("arms", []))' in submitter
    assert "--parsable --hold" in submitter
    assert "rollback_held_jobs" in submitter
    assert 'scontrol release "$job_id_list"' in submitter
    assert "relay_preflight_root=" in submitter
    assert "DYNPERM_NODE_ROOT_MAP" in submitter
    assert "DYNPERM_STAGE_REL" in submitter
    assert 'node_root="$(node_root_for "$node")"' in submitter
    assert 'git -C "$repo/recipe" rev-parse HEAD' in submitter
    assert 'docker image inspect verl-harness:latest' in submitter

    fixed_sbatch = _read("slurm/run_math_qwen3_1p7b_wdl_dynperm_fixed_m1_p60.sbatch")
    standard_sbatch = _read("slurm/run_math_qwen3_1p7b_wdl_dynperm_standard_c_p60.sbatch")
    for sbatch in (fixed_sbatch, standard_sbatch):
        assert "#SBATCH --partition=l40s" in sbatch
        assert "#SBATCH --gres=gpu:L40S:8" in sbatch
        assert "#SBATCH --exclusive" in sbatch
        assert "#SBATCH --no-requeue" in sbatch
        assert "DYNPERM_NODE_ROOT_MAP" in sbatch
        assert "DYNPERM_STAGE_REL" in sbatch
        assert "dispatch-terminal.json" in sbatch
        assert sbatch.index("trap dispatch_cleanup EXIT") < sbatch.index("DYNPERM_NODE_ROOT_MAP:?")
        assert 'workspace="$(realpath -e "${node_root}/${DYNPERM_STAGE_REL}")"' in sbatch
        assert 'job_body="${workspace}/repo/recipe' in sbatch
        assert 'test -r "$job_body"' in sbatch
        assert "trap - EXIT TERM INT" not in sbatch
    assert "fixed-m1-stage1" in fixed_sbatch
    assert "standard-c" in standard_sbatch

    job = _read("slurm/run_math_qwen3_1p7b_wdl_dynperm_p60_job.sh")
    for identity in (
        "DYNPERM_PARENT_SHA",
        "DYNPERM_RECIPE_SHA",
        "DYNPERM_IMAGE_ID",
        "DYNPERM_LAUNCH_RECEIPT",
        "DYNPERM_EVIDENCE_RELAY_HOST",
        "DYNPERM_NODE_ROOT_MAP",
        "DYNPERM_STAGE_REL",
    ):
        assert identity in job
    assert "foreign GPU compute process present" in job
    assert '--slurm-job-id "$SLURM_JOB_ID"' in job
    assert "relay_files admission.json" in job
    assert "relay_files first-step.json first-step.log" in job
    assert "relay_files admission.json first-step.json first-step.log stdout.tail.log stderr.tail.log" in job
    assert "relay_terminal_verified" in job
    assert "bootstrap-terminal.json" in job
    assert '"phase": "pre-admission"' in job
    assert job.index("trap bootstrap_cleanup EXIT") < job.index("for required_name in")
    assert "${bootstrap_relay_root}/bootstrap-terminal.json" in job
    assert "_slurm_bootstrap" not in job
    assert "bootstrap_root" not in job
    assert "bootstrap_receipt" in job
    assert "cat > '${bootstrap_relay_root}/bootstrap-terminal.json'" in job
    assert "formal DynPerm P60 rho must be one of" in job
    assert 'data1_host="$(realpath -e "${workspace}/runtime/data-1")"' in job
    assert 'data2_host="$(realpath -e "${workspace}/runtime/data-2")"' in job
    assert 'export DATA1_HOST="$data1_host"' in job
    assert 'export DATA2_HOST="$data2_host"' in job
    assert "branch --show-current" not in job
    assert "export REPO_MOUNT_MODE=ro" in job
    assert '"training_exit_code"' in job
    assert '"evidence_set_relayed"' in job
    assert "node_local_job_root" in job
    assert "relay_job_root" in job
    assert 'while kill -0 "$training_pid"' in job
    assert '&& kill -0 "$gate_pid"' in job
    assert "training exited before first-step admission completed" in job
    assert "first-step admission failed; stopping only this job's training container" in job
    assert "scancel" not in job


def test_l40s_wrapper_supports_candidate_staged_data_roots() -> None:
    wrapper = (ROOT / "scripts/l40s/run_train.sh").read_text(encoding="utf-8")
    assert "DATA1_HOST=${DATA1_HOST:-/data-1}" in wrapper
    assert "DATA2_HOST=${DATA2_HOST:-/data-2}" in wrapper
    assert "REPO_MOUNT_MODE=${REPO_MOUNT_MODE:-rw}" in wrapper
    assert '-v "${DATA1_HOST}:/data-1"' in wrapper
    assert '-v "${DATA2_HOST}:/data-2"' in wrapper
    assert '-v "${REPO_HOST}:${REPO_CONTAINER}:${REPO_MOUNT_MODE}"' in wrapper


def _valid_dynperm_metrics(rho: float, *, model1_grad: float = 1.0) -> dict:
    data = {key: 1.0 for key in REQUIRED_METRICS | DYNPERM_METRICS}
    data.update(
        {
            "actor/optimizer_step_applied": 1.0,
            "jointTraining/model1_grad_norm": model1_grad,
            "jointTraining/dynperm/requested_rho": rho,
            "jointTraining/dynperm/realized_rho": rho,
            "jointTraining/dynperm/fixed_points": 0.0,
            "jointTraining/dynperm/target_mismatches": 0.0,
            "jointTraining/dynperm/max_entropy_error": 0.0,
            "jointTraining/dynperm/max_multiset_error": 0.0,
            "jointTraining/dynperm/invariant_failures": 0.0,
        }
    )
    if rho == 0.0:
        data["jointTraining/dynperm/selected_coordinates"] = 0.0
        data["jointTraining/dynperm/audited_rows"] = 0.0
    return data


def test_first_step_gate_supports_standard_and_fixed_model1() -> None:
    assert all(
        value
        for key, value in validate_step_one(_valid_dynperm_metrics(1.0), "nonzero", 1.0).items()
        if key != "missing_metrics"
    )
    assert all(
        value
        for key, value in validate_step_one(_valid_dynperm_metrics(1.0, model1_grad=0.0), "zero", 1.0).items()
        if key != "missing_metrics"
    )


def test_first_step_gate_accepts_partial_rho_integer_bin_error() -> None:
    data = _valid_dynperm_metrics(0.25)
    data["jointTraining/dynperm/realized_rho"] = 0.250006
    checks = validate_step_one(data, "nonzero", 0.25)
    assert checks["dynperm_requested_rho_matches"]
    assert checks["dynperm_realized_rho_matches"]


def test_monitor_covers_both_p60_arms_without_mutating_jobs() -> None:
    monitor = _read("monitor_math_qwen3_1p7b_wdl_dynperm.sh")
    assert "ARMS=(fixed-m1-stage1 standard-c)" in monitor
    assert "expected_gradient=nonzero" in monitor
    assert "expected_gradient=zero" in monitor
    assert "monitor requires the same DYNPERM_RHO as the P60 launcher" in monitor
    assert '--dynperm-rho "$DYNPERM_RHO"' in monitor
    assert "tmux kill-session" not in monitor
    assert "No unowned tmux/Slurm job will be mutated" in monitor
