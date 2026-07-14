from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/execution_results.py"


def module():
    spec = importlib.util.spec_from_file_location("execution_results", TOOL)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def bundle(tool, tmp_path: Path) -> tuple[Path, dict]:
    value = {
        "schema_version": 1,
        "bundle_type": "stage123_admission_bundle",
        "bundle_path": str(tmp_path / "admission_bundle.json"),
        "run_ids": ["frac25-stage2", "frac25-stage3"],
        "bindings": {
            "manifest_sha256": "1" * 64,
            "resource_profile_sha256": "2" * 64,
            "implementation_tree_sha256": "3" * 64,
            "calibration_result_sha256": "4" * 64,
            "preflight_result_sha256": "5" * 64,
            "readiness_evidence_commit": "6" * 40,
        },
    }
    value["bundle_sha256"] = tool.result_sha256(value)
    path = tmp_path / "admission_bundle.json"
    path.write_text(json.dumps(value))
    return path, value


def test_admission_primary_pair_and_hash_authorize(tmp_path: Path) -> None:
    tool = module(); _, value = bundle(tool, tmp_path)
    assert tool.validate_admission_bundle(value).as_dict()["authorized"] is True


def test_admission_mutations_fail_closed(tmp_path: Path) -> None:
    tool = module(); _, value = bundle(tool, tmp_path)
    value["run_ids"].append("frac50-stage2")
    decision = tool.validate_admission_bundle(value)
    assert decision.code == "admission_run_set"
    del value["bindings"]["implementation_tree_sha256"]
    assert tool.validate_admission_bundle(value).code == "admission_run_set"


def test_acceptance_is_required_and_bound(tmp_path: Path) -> None:
    tool = module(); _, value = bundle(tool, tmp_path)
    assert tool.validate_admission_bundle(value, require_accepted=True).code == "admission_not_accepted"
    value["acceptance"] = {"decision": "accepted", "bundle_sha256": value["bundle_sha256"]}
    assert tool.validate_admission_bundle(value, require_accepted=True).authorized is True


def test_launch_renderer_is_deterministic_and_contains_no_secrets(tmp_path: Path) -> None:
    tool = module(); _, value = bundle(tool, tmp_path)
    first = tool.admission_launch_command(value, ROOT)
    assert first == tool.admission_launch_command(value, ROOT)
    rendered = " ".join(first)
    assert "ALLOW_QWEN3_1P7B_STAGE123_TRAINING=1" in rendered
    assert "frac50" not in rendered.lower()
    assert "token" not in rendered.lower()


def test_launch_requires_accepted_bundle(tmp_path: Path, capsys) -> None:
    tool = module(); path, value = bundle(tool, tmp_path)
    assert tool.admission_main(["render-launch", "--bundle", str(path), "--repo-host", str(ROOT)]) == 1
    value["acceptance"] = {"decision": "accepted", "bundle_sha256": value["bundle_sha256"]}
    path.write_text(json.dumps(value))
    assert tool.admission_main(["render-launch", "--bundle", str(path), "--repo-host", str(ROOT)]) == 0


def test_admission_builder_never_self_binds_calibration_identity(tmp_path: Path) -> None:
    tool = module(); manifest = tmp_path / "manifest.json"; profile = tmp_path / "profile.sh"; calibration = tmp_path / "calibration.json"; preflight = tmp_path / "preflight.json"
    manifest.write_text(json.dumps({"manifest_sha256": "a" * 64, "resource_profile": {"sha256": "c" * 64}, "runs": [{"id": "frac25-stage2"}, {"id": "frac25-stage3"}]})); profile.write_text("profile")
    calibration.write_text(json.dumps({"schema_version": 1, "result_type": "calibration_result", "decision": "passed"}))
    preflight.write_text(json.dumps({"schema_version": 1, "result_type": "preflight_result", "decision": "passed", "manifest_sha256": "a" * 64}))
    try:
        tool.build_admission_bundle(manifest, profile, calibration, preflight, "b" * 40, tmp_path / "bundle.json")
    except ValueError as exc:
        assert str(exc) == "preflight result lacks calibration expected bindings"
    else:
        raise AssertionError("builder accepted self-bound calibration identity")


def test_admission_uses_manifest_owned_resource_profile_identity() -> None:
    tool = module()
    manifest = {"resource_profile": {"sha256": "d" * 64}}
    assert tool.manifest_resource_profile_sha256(manifest) == "d" * 64
    for invalid in ({}, {"resource_profile": {}}, {"resource_profile": {"sha256": "short"}}):
        try:
            tool.manifest_resource_profile_sha256(invalid)
        except ValueError as exc:
            assert str(exc) == "manifest lacks resource profile identity"
        else:
            raise AssertionError("invalid manifest resource profile identity was accepted")
