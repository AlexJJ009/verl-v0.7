from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "scripts/stage123_phase_adapter.py"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
SPLIT_MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml"


def module():
    spec = importlib.util.spec_from_file_location("stage123_phase_adapter", ADAPTER)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def dry_run(run_id: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(ADAPTER), "--manifest", str(MANIFEST), "--run-id", run_id, "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def split_dry_run(run_id: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(ADAPTER), "--manifest", str(SPLIT_MANIFEST), "--run-id", run_id, "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def test_three_phase_commands_are_manifest_driven_and_frozen() -> None:
    control = dry_run("frac25-stage1-control")["environment"]
    stage2 = dry_run("frac25-stage2")["environment"]
    stage3 = dry_run("frac25-stage3")["environment"]
    assert control["TOTAL_TRAINING_STEPS"] == "60"
    assert control["INIT_MODEL_PATH"].endswith("/beta01/stage1_model2")
    assert control["LOG_DIR"].startswith("/data-2/model_weights/")
    assert "/recipe/" not in control["LOG_DIR"]
    assert control["WANDB_MODE"] == "offline"
    assert control["LR"] == "1e-6"
    assert control["LR_WARMUP_STEPS"] == "0"
    assert stage2["TOTAL_TRAINING_STEPS"] == "20"
    assert stage2["STAGE2_HANDOFF_STEP"] == "40"
    assert stage2["MERGED_MODEL2_DIR"].endswith("/beta01/stage1_model2")
    assert stage2["LR"] == "1e-6"
    assert stage2["LR_WARMUP_STEPS"] == "0"
    assert stage2["TRACK_JOINT_SUBMODEL_LOSSES"] == "true"
    assert stage3["TOTAL_TRAINING_STEPS"] == "40"
    assert stage3["STAGE2_MODEL2_PATH"].endswith("/stage2_final_model2")
    assert stage3["STAGE2_PROVENANCE_FILE"].endswith("/frac25-stage2.provenance.json")
    assert stage3["LR"] == "1e-6"
    assert stage3["LR_WARMUP_STEPS"] == "0"


def test_provenance_write_is_atomic_and_retry_forbidden(tmp_path: Path) -> None:
    tool = module()
    path = tmp_path / "run.provenance.json"
    tool.write_provenance(path, {"run_id": "one"})
    assert json.loads(path.read_text()) == {"run_id": "one"}
    try:
        tool.write_provenance(path, {"run_id": "two"})
    except RuntimeError as exc:
        assert "retry/resume is forbidden" in str(exc)
    else:
        raise AssertionError("existing provenance did not fail closed")


def test_split_matrix_has_two_kl_arms_and_two_stage3_submodel_branches() -> None:
    no_kl = split_dry_run("frac25-stage2-nokl")["environment"]
    model2_kl = split_dry_run("frac25-stage2-m2kl")["environment"]
    no_kl_model1 = split_dry_run("frac25-stage3-nokl-model1")["environment"]
    no_kl_model2 = split_dry_run("frac25-stage3-nokl-model2")["environment"]

    assert no_kl["VAL_N"] == "3"
    assert no_kl["BASE_MODEL_PATH"].endswith("/qwen3-1p7b-kodcode-format-sft-frac25")
    assert no_kl["EXPECTED_MODEL1_CONFIG_SHA256"] == "a4a451865e8d45a519133031f19cda7d347813159fde1756d63e2beaf67f2288"
    assert no_kl["JOINT_VALIDATION_VIEWS"] == "[model1,model2]"
    assert no_kl["BEST_CKPT_METRIC_KEY"] == "val-core/model2/HumanEval+/acc/mean@3"
    assert no_kl["SUBMODEL_KL_ENABLED"] == "false"
    assert model2_kl["SUBMODEL_KL_ENABLED"] == "true"
    assert model2_kl["SUBMODEL_KL_MODEL1_ENABLED"] == "false"
    assert model2_kl["SUBMODEL_KL_MODEL2_ENABLED"] == "true"
    assert model2_kl["SUBMODEL_KL_MODEL2_COEF"] == "0.01"
    assert no_kl_model1["STAGE2_SUBMODEL"] == "model1"
    assert no_kl_model1["STAGE2_MODEL_PATH"].endswith("/stage2_final_model1")
    assert no_kl_model2["STAGE2_SUBMODEL"] == "model2"
    assert no_kl_model2["STAGE2_MODEL_PATH"].endswith("/stage2_final_model2")
    assert no_kl_model1["BEST_CKPT_METRIC_KEY"] == "val-core/HumanEval+/acc/mean@3"
