from __future__ import annotations

import copy
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


def test_stage123_manifest_normalizes_all_run_identity():
    tool = module()
    report = tool.normalize(tool.load(MANIFEST))
    assert report["resource_profile"]["max_response_length"] == 8192
    assert report["semantics"]["validation_datasets"] == ["HumanEval+", "MBPP+", "LiveCodeBench"]
    assert [item["id"] for item in report["runs"]] == ["frac25-stage2", "frac25-stage3", "frac50-stage2", "frac50-stage3"]
    assert [item["final_step"] for item in report["runs"]] == [20, 40, 20, 40]
    assert len(report["manifest_sha256"]) == 64


@pytest.mark.parametrize("field", ["run_prefix", "id", "tmux_name"])
def test_duplicate_identity_is_rejected(field: str):
    tool = module(); data = tool.load(MANIFEST)
    data["runs"][1][field] = data["runs"][0][field]
    with pytest.raises(ValueError, match="duplicate"):
        tool.normalize(data)


def test_missing_stage3_source_and_wrong_artifact_mount_are_rejected():
    tool = module(); data = tool.load(MANIFEST)
    data["runs"][1]["source"]["run_id"] = "missing"
    with pytest.raises(ValueError, match="missing source"):
        tool.normalize(data)
    data = tool.load(MANIFEST); data["runs"][0]["artifact_dir"] = "/data-1/model_weights/wrong"
    with pytest.raises(ValueError, match="must use /data-2"):
        tool.normalize(data)


def test_hash_changes_when_lifecycle_data_changes():
    tool = module(); data = tool.load(MANIFEST)
    first = tool.normalize(data)["manifest_sha256"]
    data["runs"][0]["final_step"] = 21
    assert tool.normalize(data)["manifest_sha256"] != first


def test_eligibility_denominator_and_phase_identity_are_enforced():
    tool = module(); data = tool.load(MANIFEST)
    data["calibration_workloads"]["stage1"]["validation_eligibility"]["submitted_prompt_count"] += 1
    with pytest.raises(ValueError, match="submitted prompt count mismatch"):
        tool.normalize(data)

    data = tool.load(MANIFEST)
    data["calibration_workloads"]["stage2"]["validation_eligibility"]["ordered_eligible_uid_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs across phases"):
        tool.normalize(data)


def test_eligibility_cannot_exceed_full_source_provenance():
    tool = module(); data = tool.load(MANIFEST)
    eligibility = data["calibration_workloads"]["stage1"]["validation_eligibility"]
    eligibility["per_dataset_eligible_counts"]["LiveCodeBench"] = 881
    eligibility["submitted_prompt_count"] = sum(eligibility["per_dataset_eligible_counts"].values())
    with pytest.raises(ValueError, match="exceeds source row count"):
        tool.normalize(data)
