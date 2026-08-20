from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "recipe/on_policy_wdl_sft"
MATH = RECIPE / "math_task"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_joint_common_plumbs_and_fail_closes_freeze_model1() -> None:
    common = _read(RECIPE / "_common_wdl_sft_is_joint.sh")
    assert "FREEZE_MODEL1=${FREEZE_MODEL1:-false}" in common
    assert "prepare_joint_args+=(--freeze_model1)" in common
    assert 'expected_freeze_model1 = sys.argv[4].lower() == "true"' in common
    assert "joint model freeze_model1 mismatch" in common
    assert "FREEZE_MODEL1=${FREEZE_MODEL1}" in common
    assert "actor_rollout_ref.actor.fsdp_config.use_orig_params=${actor_use_orig_params}" in common
    assert 'actor_use_orig_params="${ACTOR_FSDP_USE_ORIG_PARAMS:-$FREEZE_MODEL1}"' in common


def test_stage1_fixed_m1_is_c_with_only_freeze_and_identity_changes() -> None:
    arm_c = _read(MATH / "run_math_qwen3_1p7b_wdl_causal_arm_c.sh")
    fixed = _read(MATH / "run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh")

    for invariant in ("export FUSION_LAMBDA=0.8", "export FUSION_MODE=mixture"):
        assert invariant in arm_c
        assert invariant in fixed
    assert "export FREEZE_MODEL1=true" in fixed
    assert "math-wdl-fixed-m1-stage1-p60" in fixed
    assert "restored_from_causal_p60_joint_20260812/final_model" in fixed
    assert "model_input_provenance.json" in fixed
    assert "TRAIN_FILE=" not in fixed
    assert "TOTAL_TRAINING_STEPS=" not in fixed


def test_cold_start_fixed_m1_changes_only_the_matched_source_identity() -> None:
    fixed = _read(MATH / "run_math_qwen3_1p7b_wdl_fixed_m1_cold_start.sh")

    assert "export FREEZE_MODEL1=true" in fixed
    assert "export FUSION_LAMBDA=0.8" in fixed
    assert "export FUSION_MODE=mixture" in fixed
    assert "export MODEL2_PATH=${MODEL2_PATH:-$BASE_MODEL_PATH}" in fixed
    assert "format_cold_start_source.json" in fixed
    assert "TRAIN_FILE=" not in fixed
    assert "TOTAL_TRAINING_STEPS=" not in fixed


def test_fixed_m1_wrappers_use_distinct_joint_caches_and_run_names() -> None:
    cold = _read(MATH / "run_math_qwen3_1p7b_wdl_fixed_m1_cold_start.sh")
    stage1 = _read(MATH / "run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh")

    assert "FIXED-M1-COLD-START-P60" in cold
    assert "FIXED-M1-STAGE1-P60" in stage1
    assert "math-wdl-fixed-m1-cold-start-p60" in cold
    assert "math-wdl-fixed-m1-stage1-p60" in stage1
    assert "OnPolicyWDLSFT-Math-1P7B-Fixed-M1-P60" in cold
    assert "OnPolicyWDLSFT-Math-1P7B-Fixed-M1-P60" in stage1
    assert "qwen3_1p7b_wdl_fixed_m1_p60" in cold
    assert "qwen3_1p7b_wdl_fixed_m1_p60" in stage1
    for wrapper in (cold, stage1):
        assert 'export JOINT_VALIDATION_VIEWS="[model1,model2]"' in wrapper
        assert "export TRACK_JOINT_SUBMODEL_LOSSES=true" in wrapper
        assert "export BEST_CKPT_METRIC_KEY=val-core/model2/math7_macro/acc/mean@3" in wrapper


def test_dual_validation_writes_view_namespaced_metrics_and_generations() -> None:
    trainer = _read(ROOT / "verl/trainer/ppo/ray_trainer.py")
    common = _read(MATH / "run_math_qwen3_1p7b_wdl_causal_p60_common.sh")

    assert 'export JOINT_VALIDATION_VIEWS="[model1,model2]"' in common
    assert 'views = list(self.config.trainer.get("joint_validation_views", ["model2"]))' in trainer
    assert "weight_view=view" in trainer
    assert "metric_view=view if expose_view_namespace else None" in trainer
    assert "val_data_dir = os.path.join(val_data_dir, metric_view)" in trainer
    assert 'view_path = f"/{metric_view}" if metric_view else ""' in trainer


def test_meituan_dispatcher_routes_both_fixed_m1_sources() -> None:
    dispatcher = _read(MATH / "meituan/jupyter.sh")

    assert "fixed-m1-stage1)" in dispatcher
    assert "run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh" in dispatcher
    assert "fixed-m1-cold-start)" in dispatcher
    assert "run_math_qwen3_1p7b_wdl_fixed_m1_cold_start.sh" in dispatcher
    assert 'export MODEL2_PATH="$BASE_MODEL_PATH"' in dispatcher
    assert 'export STAGE1_MODEL2_PROVENANCE_FILE="$BASE_MODEL_PATH/format_cold_start_source.json"' in dispatcher
    assert 'if [[ "${EXPERIMENT,,}" == fixed-m1-* ]]' in dispatcher
    assert "qwen3_1p7b_wdl_fixed_m1_p60" in dispatcher


def test_fixed_m1_queue_and_monitor_cover_both_arms() -> None:
    queue = _read(MATH / "run_math_qwen3_1p7b_wdl_fixed_m1_p60_queue.sh")
    monitor = _read(MATH / "monitor_math_qwen3_1p7b_wdl_fixed_m1_p60.sh")

    cold_script = "run_math_qwen3_1p7b_wdl_fixed_m1_cold_start.sh"
    stage1_script = "run_math_qwen3_1p7b_wdl_fixed_m1_stage1.sh"
    assert queue.index(cold_script) < queue.index(stage1_script)
    assert 'if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${TMUX:-}" ]' in queue
    assert "nvidia-smi --query-gpu=utilization.gpu" in queue
    assert "MATH-WDL-FIXED-M1-COLD-START-P60-QWEN3-1P7B" in monitor
    assert "MATH-WDL-FIXED-M1-STAGE1-P60-QWEN3-1P7B" in monitor
    assert 'for prefix in "${RUN_PREFIXES[@]}"' in monitor
    assert "--expected-model1-gradient zero" in monitor
