# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_wdl_causal_p60.yaml"


def _load_queue_module():
    spec = importlib.util.spec_from_file_location("math_wdl_causal_queue", ROOT / "scripts/math_wdl_causal_queue.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_freezes_matched_inputs_and_optional_direct_model2_arm():
    manifest = yaml.safe_load(MANIFEST.read_text())
    module = _load_queue_module()
    module.validate_manifest(manifest, require_launch=False)

    runs = {run["id"]: run for run in manifest["runs"]}
    assert runs["arm-c-mixture"]["fusion_lambda"] == 0.8
    assert runs["arm-c-mixture"]["fusion_mode"] == "mixture"
    assert runs["arm-d-strong-only"]["fusion_lambda"] == 1.0
    assert runs["arm-d-strong-only"]["execution"] == "optional"
    assert runs["arm-d-strong-only"]["default_in_queue"] is False
    assert runs["arm-d0-matched-scale-no-weak"]["fusion_lambda"] == 0.8
    assert runs["arm-d0-matched-scale-no-weak"]["fusion_mode"] == "strong_scaled"
    assert manifest["identity"]["train_rows"] == 3840
    assert manifest["training_contract"]["rollout_source"] == "model2"
    assert manifest["reward_contract"]["path"] == "recipe/joint_training/custom_reward_function_latex_verify.py"
    assert manifest["reward_contract"]["missing_answer_tag_reward"] == -1.0


def test_manifest_fails_closed_if_matched_scale_control_is_removed():
    manifest = yaml.safe_load(MANIFEST.read_text())
    manifest["runs"] = [run for run in manifest["runs"] if run["id"] != "arm-d0-matched-scale-no-weak"]
    with pytest.raises(RuntimeError, match="treatment matrix"):
        _load_queue_module().validate_manifest(manifest, require_launch=False)


def test_default_queue_omits_d_only_after_equivalence_probe_passes():
    manifest = yaml.safe_load(MANIFEST.read_text())
    module = _load_queue_module()
    checks = {
        name: True
        for name in next(run for run in manifest["runs"] if run["id"] == "arm-d-strong-only")[
            "omit_if_manipulation_checks"
        ]
    }
    receipt = {"status": "pass", "checks": checks}
    assert module.select_run_ids(manifest, receipt, include_optional_d=False) == [
        "arm-d0-matched-scale-no-weak",
        "arm-c-mixture",
    ]
    assert module.select_run_ids(manifest, receipt, include_optional_d=True) == [
        "arm-d0-matched-scale-no-weak",
        "arm-d-strong-only",
        "arm-c-mixture",
    ]
    checks["D_is_direct_model2"] = False
    with pytest.raises(RuntimeError, match="cannot omit optional D"):
        module.select_run_ids(manifest, receipt, include_optional_d=False)


def test_manifest_fails_closed_on_wrong_train_hash_or_fusion_mode():
    manifest = yaml.safe_load(MANIFEST.read_text())
    module = _load_queue_module()

    bad_hash = copy.deepcopy(manifest)
    bad_hash["identity"]["train_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="row/hash identity mismatch"):
        module.validate_manifest(bad_hash, require_launch=False)

    bad_mode = copy.deepcopy(manifest)
    bad_mode["runs"][2]["fusion_mode"] = "mixture"
    with pytest.raises(RuntimeError, match="treatment matrix"):
        module.validate_manifest(bad_mode, require_launch=False)


def test_joint_launcher_plumbs_dual_validation_and_resumable_anchors():
    common = (ROOT / "recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh").read_text()
    math_wrapper = (ROOT / "recipe/on_policy_wdl_sft/math_task/run_s2_math_qwen3_1p7b_stage123_common.sh").read_text()
    assert 'trainer.joint_validation_views="${JOINT_VALIDATION_VIEWS}"' in common
    assert '+trainer.protected_ckpt_steps="${PROTECTED_CKPT_STEPS}"' in common
    assert "TRACK_JOINT_SUBMODEL_LOSSES=${TRACK_JOINT_SUBMODEL_LOSSES:-true}" in math_wrapper
    assert 'PROTECTED_CKPT_STEPS=${PROTECTED_CKPT_STEPS:-"[20,40,45,50,60]"}' in math_wrapper


def test_causal_launcher_pins_strict_reward_and_runtime_canary():
    causal = (ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_wdl_causal_p60_common.sh").read_text()
    assert 'CUSTOM_REWARD_FN_PATH="${SCRIPT_DIR}/../../joint_training/custom_reward_function_latex_verify.py"' in causal
    assert "check_math_reward_contract.py" in causal


def test_format_gate_rejects_two_consecutive_large_drops(tmp_path):
    module_path = ROOT / "scripts/math_wdl_format_gate.py"
    spec = importlib.util.spec_from_file_location("math_wdl_format_gate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    run_dir = tmp_path / "RUN_1" / "model2"
    run_dir.mkdir(parents=True)
    good = {
        "output": "<think>reasoning</think><answer>\\boxed{42}</answer>",
        "has_eos": True,
    }
    bad = {"output": "<think>reasoning</think>\\boxed{42}", "has_eos": True}
    for step, rows in ((0, [good] * 10), (5, [bad] * 10), (10, [bad] * 10)):
        (run_dir / f"{step}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = module.inspect_run(run_dir.parent, max_drop=0.05, required_consecutive=2)
    assert result["status"] == "fail"


def test_each_arm_wrapper_pins_its_treatment():
    expected = {
        "arm_c": ("0.8", "mixture"),
        "arm_d": ("1.0", "mixture"),
        "arm_d0": ("0.8", "strong_scaled"),
    }
    for suffix, (lambda_value, mode) in expected.items():
        text = (ROOT / f"recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_wdl_causal_{suffix}.sh").read_text()
        assert f"export FUSION_LAMBDA={lambda_value}" in text
        assert f"export FUSION_MODE={mode}" in text


def test_first_step_gate_requires_causal_health_metrics():
    module_path = ROOT / "scripts/math_wdl_first_step_gate.py"
    spec = importlib.util.spec_from_file_location("math_wdl_first_step_gate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    data = {key: 1.0 for key in module.REQUIRED_METRICS}
    data["jointTraining/model1_grad_norm"] = 0.0
    checks = module.validate_step_one(data, "zero")
    assert checks["optimizer_step_applied"]
    assert checks["model1_gradient_matches_arm"]
    data.pop("actor/optimizer_step_applied")
    assert not module.validate_step_one(data, "zero")["all_required_metrics_present"]
