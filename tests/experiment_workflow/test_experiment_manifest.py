# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def module():
    spec = importlib.util.spec_from_file_location("experiment_manifest", ROOT / "scripts/experiment_manifest.py")
    result = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(result)
    return result


def test_stage123_manifest_normalizes_primary_run_identity():
    tool = module()
    report = tool.normalize(tool.load(MANIFEST))
    assert report["resource_profile"]["max_response_length"] == 8192
    assert report["semantics"]["validation_datasets"] == ["HumanEval+", "MBPP+", "LiveCodeBench"]
    assert [item["id"] for item in report["runs"]] == ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]
    assert [item["final_step"] for item in report["runs"]] == [60, 20, 40]
    assert report["preflight"]["result_max_age_seconds"] == 3600
    assert report["calibration_policy"]["calibration_result_max_age_seconds"] == 86400
    assert len(report["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    ("section", "legacy", "current"),
    [
        ("preflight", "receipt_max_age_seconds", "result_max_age_seconds"),
        ("calibration_policy", "calibration_receipt_max_age_seconds", "calibration_result_max_age_seconds"),
    ],
)
def test_legacy_receipt_freshness_names_fail_closed(section: str, legacy: str, current: str):
    tool = module()
    data = tool.load(MANIFEST)
    data[section][legacy] = data[section].pop(current)
    with pytest.raises(tool.ManifestPolicyError) as raised:
        tool.normalize(data)
    assert raised.value.as_dict() == {
        "code": "legacy_freshness_field",
        "message": f"legacy freshness field is not current authority: {section}.{legacy}",
        "context": {"section": section, "legacy_field": legacy, "current_field": current},
    }


@pytest.mark.parametrize("field", ["run_prefix", "id", "tmux_name"])
def test_duplicate_identity_is_rejected(field: str):
    tool = module()
    data = tool.load(MANIFEST)
    data["runs"][1][field] = data["runs"][0][field]
    with pytest.raises(ValueError, match="duplicate"):
        tool.normalize(data)


def test_missing_stage3_source_and_wrong_artifact_mount_are_rejected():
    tool = module()
    data = tool.load(MANIFEST)
    data["runs"][1]["source"]["run_id"] = "missing"
    with pytest.raises(ValueError, match="missing source"):
        tool.normalize(data)
    data = tool.load(MANIFEST)
    data["runs"][0]["artifact_dir"] = "/data-1/model_weights/wrong"
    with pytest.raises(ValueError, match="must use /data-2"):
        tool.normalize(data)


def test_hash_changes_when_lifecycle_data_changes():
    tool = module()
    data = tool.load(MANIFEST)
    first = tool.normalize(data)["manifest_sha256"]
    data["runs"][0]["run_prefix"] += "-CHANGED"
    assert tool.normalize(data)["manifest_sha256"] != first


def test_stage3_pending_producer_final_step_drift_is_rejected():
    tool = module()
    data = tool.load(MANIFEST)
    next(item for item in data["runs"] if item["id"] == "frac25-stage2")["final_step"] = 21
    with pytest.raises(ValueError, match="pending producer identity mismatch"):
        tool.normalize(data)


def test_pending_producer_follows_mutated_manifest_final_step_without_source_constant():
    tool = module()
    data = tool.load(MANIFEST)
    producer_run = next(item for item in data["runs"] if item["id"] == "frac25-stage2")
    producer_run["final_step"] = 21
    source = data["calibration_workloads"]["stage3"]["model_sources"][0]
    source["producer"]["final_step"] = 21
    assert next(item for item in tool.normalize(data)["runs"] if item["id"] == "frac25-stage2")["final_step"] == 21


def test_manifest_policy_errors_have_stable_code_message_and_context():
    tool = module()
    data = tool.load(MANIFEST)
    next(item for item in data["runs"] if item["id"] == "frac25-stage3")["source"]["run_id"] = "missing"
    with pytest.raises(tool.ManifestPolicyError) as raised:
        tool.normalize(data)
    assert raised.value.as_dict() == {
        "code": "missing_source_run",
        "message": "missing source run for frac25-stage3",
        "context": {"run_id": "frac25-stage3", "source_run_id": "missing"},
    }


def test_stage1_base_substitution_and_provenance_drift_are_rejected():
    tool = module()
    data = tool.load(MANIFEST)
    source = data["calibration_workloads"]["stage1"]["model_sources"][0]
    source["path"] = data["paths"]["base_model"]
    with pytest.raises(ValueError, match="init model path mismatch"):
        tool.normalize(data)

    data = tool.load(MANIFEST)
    data["calibration_workloads"]["stage1"]["model_sources"][0]["provenance"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model provenance hash mismatch"):
        tool.normalize(data)


def test_legacy_stage2_60_step_model_cannot_replace_pending_stage3_source():
    tool = module()
    data = tool.load(MANIFEST)
    source = data["calibration_workloads"]["stage3"]["model_sources"][0]
    source.clear()
    source.update(
        {
            "role": "rollout",
            "state": "materialized",
            "path": "/data-1/model_weights/code_task/kodcode_qwen3_1p7b_coldstart_fraction_ctx8k_stage2_p40/frac25/beta01/ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA01-V1/step_40_s2steps60",
            "artifact_sha256": "516a69cd83677ca132b0fb6a2885fc092956cb65b8925a89040d03b4b7e0c16a",
            "hash_algorithm": "sorted_relative_path_content_sha256_v1",
        }
    )
    with pytest.raises(ValueError, match="stage3 materialized source requires current producer binding"):
        tool.normalize(data)


def test_current_stage2_20_step_materialized_stage3_source_is_accepted(tmp_path):
    tool = module()
    data = tool.load(MANIFEST)
    model = tmp_path / "stage2_final_model2"
    model.mkdir()
    (model / "config.json").write_text((Path(data["paths"]["base_model"]) / "config.json").read_text())
    (model / "model.safetensors").write_bytes(b"weights")
    provenance_path = tmp_path / "frac25-stage3.provenance.json"
    provenance_path.write_text('{"schema_version":1,"run_id":"frac25-stage2","final_step":20}\n')
    source = data["calibration_workloads"]["stage3"]["model_sources"][0]
    source.update(
        {
            "state": "materialized",
            "path": str(model),
            "artifact_sha256": tool._load_workload_hashing()[0](model),
            "hash_algorithm": "sorted_relative_path_content_sha256_v1",
            "provenance": {
                "path": str(provenance_path),
                "sha256": tool._load_workload_hashing()[1](provenance_path),
                "schema_version": 1,
                "kind": "stage2_model2_source",
            },
        }
    )
    source["producer"]["output_path"] = str(model)
    source["producer"]["provenance_path"] = str(provenance_path)
    assert tool.normalize(data)["calibration_workloads"]["stage3"]["model_sources"][0]["state"] == "materialized"


def test_eligibility_denominator_and_phase_identity_are_enforced():
    tool = module()
    data = tool.load(MANIFEST)
    data["calibration_workloads"]["stage1"]["validation_eligibility"]["submitted_prompt_count"] += 1
    with pytest.raises(ValueError, match="submitted prompt count mismatch"):
        tool.normalize(data)

    data = tool.load(MANIFEST)
    data["calibration_workloads"]["stage2"]["validation_eligibility"]["ordered_eligible_uid_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs across phases"):
        tool.normalize(data)


def test_eligibility_cannot_exceed_full_source_provenance():
    tool = module()
    data = tool.load(MANIFEST)
    eligibility = data["calibration_workloads"]["stage1"]["validation_eligibility"]
    eligibility["per_dataset_eligible_counts"]["LiveCodeBench"] = 881
    eligibility["submitted_prompt_count"] = sum(eligibility["per_dataset_eligible_counts"].values())
    with pytest.raises(ValueError, match="exceeds source row count"):
        tool.normalize(data)
