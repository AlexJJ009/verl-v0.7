# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.math_wdl_first_step_gate import DYNPERM_METRICS, REQUIRED_METRICS, validate_step_one

ROOT = Path(__file__).resolve().parents[2]
MATH = ROOT / "recipe/on_policy_wdl_sft/math_task"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_dynperm.yaml"
CAUSAL_MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_causal_p60.yaml"


def _read(name: str) -> str:
    return (MATH / name).read_text(encoding="utf-8")


def test_dynperm_inherits_c_and_fixed_m1_stage1_contract() -> None:
    arm_c = _read("run_math_qwen3_1p7b_wdl_causal_arm_c.sh")
    fixed = _read("run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh")
    dynperm = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")

    for invariant in ("export FUSION_LAMBDA=0.8", "export FUSION_MODE=mixture"):
        assert invariant in arm_c
        assert invariant in fixed
        assert invariant in dynperm
    assert "export FREEZE_MODEL1=false" in dynperm
    assert 'export PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-"[20,40,45,50,60]"}' in _read(
        "run_math_qwen3_1p7b_wdl_causal_p60_common.sh"
    )
    assert "restored_from_causal_p60_joint_20260812/final_model" in fixed
    assert "restored_from_causal_p60_joint_20260812/final_model" in dynperm
    assert 'STAGE1_MODEL2_PROVENANCE_FILE="$MODEL2_PATH/model_input_provenance.json"' in dynperm
    assert "qwen3_1p7b_wdl_causal_p60/admission/manipulation_receipt.json" in dynperm
    for exact_pin in (
        "export TRAIN_FILE=/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719/stage1_control_stage2_then_stage3.parquet",
        "export WDL_SFT_BETA=0.0",
        "export LR=1e-6",
        "export DATA_SEED=20260719",
        "export TRAIN_PROMPT_BSZ=64",
        "export ROLLOUT_N=8",
        "export MAX_RESPONSE_LENGTH=4096",
        "export VAL_N=3",
        "export JOINT_TRAINING_ROLLOUT_SOURCE=model2",
    ):
        assert exact_pin in dynperm


def test_dynperm_treatment_is_final_and_cannot_be_overridden_by_caller() -> None:
    common = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")
    call = common[common.index('exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh"') :]
    assert call.index('"$@"') < call.index("data.seed=20260719")
    for pin in (
        "data.shuffle=False",
        'data.train_files="$TRAIN_FILE"',
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=512",
        "actor_rollout_ref.actor.policy_loss.loss_mode=wdl_sft",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.submodel_kl.enabled=false",
        "actor_rollout_ref.rollout.n=8",
        'trainer.total_training_steps="$TOTAL_TRAINING_STEPS"',
        'trainer.joint_validation_views="[model1,model2]"',
        "actor_rollout_ref.actor.weak_logit_permutation.enabled=true",
        'actor_rollout_ref.actor.weak_logit_permutation.rho="$DYNPERM_RHO"',
        "actor_rollout_ref.actor.weak_logit_permutation.seed=42",
        "actor_rollout_ref.actor.weak_logit_permutation.row_chunk_size=16",
        "actor_rollout_ref.actor.weak_logit_permutation.audit_rows=4",
    ):
        assert pin in call


def test_manifest_freezes_exact_c_joint_contract_and_launch_boundary() -> None:
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
    for key in ("model1_source_sha256", "model1_config_sha256", "model1_weights_sha256", "train_rows", "train_sha256"):
        assert manifest["identity"][key] == causal["identity"][key]
    assert contract["final_step_options"] == [20, 30, 60]
    assert contract["protected_checkpoint_steps_by_horizon"] == {
        20: [20],
        30: [20, 30],
        60: [20, 40, 45, 50, 60],
    }
    causal_c = next(run for run in causal["runs"] if run["id"] == "arm-c-mixture")
    for run in manifest["runs"]:
        assert run["fusion_lambda"] == causal_c["fusion_lambda"]
        assert run["fusion_mode"] == causal_c["fusion_mode"]
        assert run["freeze_model1"] is False
    assert manifest["launch_allowed"] is False
    assert manifest["execution"]["formal_runs_started"] is False
    assert manifest["execution"]["pilot_steps"] == [20, 30]
    assert manifest["execution"]["p60_conditional"] is True


def test_pilot_and_p60_wrappers_fail_closed_on_horizon() -> None:
    pilot = _read("run_math_qwen3_1p7b_wdl_dynperm_pilot.sh")
    p60 = _read("run_math_qwen3_1p7b_wdl_dynperm_p60.sh")
    common = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")
    assert "20|30)" in pilot
    assert "export TOTAL_TRAINING_STEPS=60" in p60
    assert "real DynPerm runs require DYNPERM_LAUNCH_RECEIPT" in common
    assert "P60 DynPerm endpoint requires DYNPERM_PILOT_ADMISSION_RECEIPT" in common
    assert "material_curve_validity" in common
    assert '"max_training_steps": horizon' in common
    assert '"parent_candidate_sha": parent_sha' in common
    assert '"recipe_candidate_sha": recipe_sha' in common
    assert "formal DynPerm launch requires clean parent and recipe worktrees" in common
    assert "--untracked-files=no" not in common


def test_dynperm_first_step_gate_checks_treatment_validity() -> None:
    data = {key: 1.0 for key in REQUIRED_METRICS | DYNPERM_METRICS}
    data.update(
        {
            "actor/optimizer_step_applied": 1.0,
            "jointTraining/dynperm/requested_rho": 1.0,
            "jointTraining/dynperm/realized_rho": 1.0,
            "jointTraining/dynperm/fixed_points": 0.0,
            "jointTraining/dynperm/target_mismatches": 0.0,
            "jointTraining/dynperm/max_entropy_error": 0.0,
            "jointTraining/dynperm/max_multiset_error": 0.0,
            "jointTraining/dynperm/invariant_failures": 0.0,
        }
    )
    checks = validate_step_one(data, "nonzero", 1.0)
    assert all(value for key, value in checks.items() if key != "missing_metrics")
    data["jointTraining/dynperm/target_mismatches"] = 1.0
    assert not validate_step_one(data, "nonzero", 1.0)["dynperm_target_is_unchanged"]


def test_rho_zero_gate_requires_exact_noop_telemetry() -> None:
    data = {key: 1.0 for key in REQUIRED_METRICS | DYNPERM_METRICS}
    data.update(
        {
            "actor/optimizer_step_applied": 1.0,
            "jointTraining/dynperm/requested_rho": 0.0,
            "jointTraining/dynperm/realized_rho": 0.0,
            "jointTraining/dynperm/selected_coordinates": 0.0,
            "jointTraining/dynperm/audited_rows": 0.0,
            "jointTraining/dynperm/fixed_points": 0.0,
            "jointTraining/dynperm/target_mismatches": 0.0,
            "jointTraining/dynperm/max_entropy_error": 0.0,
            "jointTraining/dynperm/max_multiset_error": 0.0,
            "jointTraining/dynperm/invariant_failures": 0.0,
        }
    )
    checks = validate_step_one(data, "nonzero", 0.0)
    assert all(value for key, value in checks.items() if key != "missing_metrics")


def test_monitor_binds_dynperm_first_step_and_format_gates() -> None:
    monitor = _read("monitor_math_qwen3_1p7b_wdl_dynperm.sh")
    assert "--expected-model1-gradient nonzero" in monitor
    assert '--dynperm-rho "$DYNPERM_RHO"' in monitor
    assert "math_wdl_format_gate.py" in monitor
    assert "tmux kill-session" not in monitor
    assert "will not mutate an unowned tmux/Slurm job" in monitor


def test_engineering_receipt_contract_is_candidate_bound() -> None:
    common = _read("run_math_qwen3_1p7b_wdl_dynperm_common.sh")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    evidence = manifest["engineering_evidence"]
    assert evidence["job_id"] == 146
    assert evidence["result"] == "PASS"
    assert evidence["world_size"] == 8
    assert evidence["formal_experiment"] is False
    assert evidence["receipt_sha256"] in common
    assert 'receipt.get("formal_experiment") is not False' in common
    assert "DYNPERM_ENGINEERING_RECEIPT" not in common
    assert "DYNPERM_ENGINEERING_RECEIPT_SHA256" not in common
    assert "686f3ee1f190387581e38847cb0e75f055021caa" in common
    assert 'receipt.get("candidate_sha") != expected_candidate' in common
