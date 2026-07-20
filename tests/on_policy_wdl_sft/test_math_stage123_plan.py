import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from recipe.joint_training.custom_reward_function_latex_verify import compute_format_telemetry
from recipe.joint_training.offline_eval import compute_shared_metrics
from recipe.joint_training.custom_reward_function_latex_verify import compute_score_latex_verify
from verl.trainer.ppo.ray_trainer import _add_validation_macro_average


ROOT = Path(__file__).resolve().parents[2]


def load_data_module():
    path = ROOT / "recipe/on_policy_wdl_sft/math_task/prepare_qwen3_1p7b_math_stage123_data.py"
    spec = importlib.util.spec_from_file_location("math_stage123_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_format_telemetry_requires_one_ordered_pair_per_tag():
    assert compute_format_telemetry("<think>x</think><answer>\\boxed{1}</answer>") == {
        "think_complete": True,
        "answer_complete": True,
    }


def test_complete_format_contract_is_the_intersection():
    result = compute_score_latex_verify(
        "test",
        "<think>work</think><answer>\\boxed{1}</answer>",
        "1",
        {"valid_response_length": 10, "max_resp_len": 20},
    )
    assert result["format_contract_success"] is True
    truncated = compute_score_latex_verify(
        "test",
        "<think>work</think><answer>\\boxed{1}</answer>",
        "1",
        {"valid_response_length": 20, "max_resp_len": 20},
    )
    assert truncated["format_contract_success"] is False
    assert compute_format_telemetry("<think>x</think><answer>missing close") == {
        "think_complete": True,
        "answer_complete": False,
    }


def test_offline_format_metrics_are_response_rates():
    metrics = compute_shared_metrics(
        [
            {
                "results": [
                    {"acc": True, "think_complete": True, "answer_complete": True, "boxed_extraction_success": True, "reward_grader_success": True, "format_contract_success": True, "has_eos": True, "truncated": False},
                    {"acc": False, "think_complete": True, "answer_complete": False, "boxed_extraction_success": False, "reward_grader_success": True, "format_contract_success": False, "has_eos": False, "truncated": True},
                    {"acc": False, "think_complete": False, "answer_complete": False, "boxed_extraction_success": False, "reward_grader_success": False, "format_contract_success": False, "has_eos": True, "truncated": False},
                ]
            }
        ],
        3,
    )
    assert metrics["mean@3"] == pytest.approx(1 / 3)
    assert metrics["think_complete_rate"] == pytest.approx(2 / 3)
    assert metrics["boxed_extraction_success_rate"] == pytest.approx(1 / 3)
    assert metrics["format_contract_success_rate"] == pytest.approx(1 / 3)
    assert metrics["truncation_rate"] == pytest.approx(1 / 3)


def test_math7_macro_average_uses_equal_dataset_weighting():
    sources = ["a", "b", "c"]
    metrics = {f"val-core/{source}/acc/mean@3": value for source, value in zip(sources, [0.3, 0.6, 0.9])}
    config = {
        "validation_macro_average_sources": sources,
        "validation_macro_average_name": "math7_macro",
        "validation_macro_average_metric": "acc/mean@3",
    }
    _add_validation_macro_average(metrics, config)
    assert metrics["val-core/math7_macro/acc/mean@3"] == pytest.approx(0.6)


def test_disjoint_split_receipt_and_control_order(tmp_path):
    module = load_data_module()
    source = tmp_path / "source.parquet"
    output = tmp_path / "output"
    pd.DataFrame({"row": list(range(7500))}).to_parquet(source, index=False)
    old_argv = module.parse_args
    module.parse_args = lambda: type("Args", (), {"source": source, "output_root": output, "seed": 20260719, "verify_only": False})()
    try:
        module.main()
    finally:
        module.parse_args = old_argv
    receipt = module.verify_receipt(output)
    assert [receipt["shards"][name]["rows"] for name in ("cold_start", "stage1", "stage2", "stage3")] == [1100, 2560, 1280, 2560]
    stage2 = pd.read_parquet(output / "stage2.parquet")["stage123_source_index"].tolist()
    stage3 = pd.read_parquet(output / "stage3.parquet")["stage123_source_index"].tolist()
    control = pd.read_parquet(output / "stage1_control_stage2_then_stage3.parquet")["stage123_source_index"].tolist()
    assert control == stage2 + stage3
    assert len(control) == 60 * 64


def test_manifests_freeze_full_math7_and_block_invalidated_launches():
    manifest_root = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest"
    cold = yaml.safe_load((manifest_root / "math_qwen3_1p7b_cold_start_cotmask_v3.yaml").read_text())
    stage = yaml.safe_load((manifest_root / "math_qwen3_1p7b_stage123_cotmask_v3.yaml").read_text())
    assert cold["training"]["step_interval"] == 5
    assert cold["paths"]["source_train_file"] == stage["paths"]["source_train_file"]
    assert len(cold["validation"]["datasets"]) == 7
    assert cold["validation"]["n"] == 1
    assert cold["admission_thresholds"]["format_contract_success_rate"] == 0.95
    assert cold["execution"]["launch_allowed"] is False
    assert cold["execution"]["requires_whole_message_loss_mask"] is True
    assert "loss_mask_preflight_receipt" in cold["paths"]
    assert cold["execution"]["auto_select_first_passing_checkpoint"] is True
    assert stage["launch_allowed"] is False
    assert stage["validation"]["n"] == 3
    assert stage["validation"]["primary_metric"] == "val-core/math7_macro/acc/mean@3"
    assert len(stage["runs"]) == 16
    assert {run["beta"] for run in stage["runs"]} == {0.0, 0.1}

    invalidated = [
        manifest_root / "math_qwen3_1p7b_cold_start.yaml",
        manifest_root / "math_qwen3_1p7b_cold_start_lr5e6_v2.yaml",
        manifest_root / "math_qwen3_1p7b_stage123_lr5e6_v2.yaml",
    ]
    for path in invalidated:
        manifest = yaml.safe_load(path.read_text())
        launch_allowed = manifest.get("launch_allowed", manifest.get("execution", {}).get("launch_allowed"))
        assert "invalidated" in manifest["status"]
        assert launch_allowed is False
