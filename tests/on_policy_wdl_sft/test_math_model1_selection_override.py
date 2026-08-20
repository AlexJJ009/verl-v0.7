# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.math_stage123_queue import model1_selection_policy, selected_model_from_receipt

ROOT = Path(__file__).resolve().parents[2]


def load_selector_module():
    path = ROOT / "scripts/select_math_model1.py"
    spec = importlib.util.spec_from_file_location("select_math_model1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_candidate_root(tmp_path: Path, *, passed: bool = False) -> Path:
    model = tmp_path / "candidates/step_20"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"model")
    payload = {
        "schema_version": 1,
        "candidates": [
            {
                "step": 20,
                "model_path": str(model),
                "passed_format_gate": passed,
                "micro_metrics": {"format_contract_success_rate": 0.8963545389563974},
            }
        ],
    }
    (tmp_path / "cold_start_candidates.json").write_text(json.dumps(payload))
    return tmp_path


def run_selector(monkeypatch, artifact_root: Path, *extra_args: str) -> None:
    module = load_selector_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_math_model1.py",
            "--step",
            "20",
            "--artifact-root",
            str(artifact_root),
            "--review-note",
            "User explicitly selected step 20 after reviewing the full cold-start curve.",
            *extra_args,
        ],
    )
    module.main()


def test_below_threshold_selection_requires_explicit_override(tmp_path, monkeypatch):
    artifact_root = make_candidate_root(tmp_path)
    with pytest.raises(ValueError, match="did not pass"):
        run_selector(monkeypatch, artifact_root)


def test_manual_override_is_recorded_and_bound_by_stage123_manifest(tmp_path, monkeypatch):
    artifact_root = make_candidate_root(tmp_path)
    run_selector(monkeypatch, artifact_root, "--allow-below-format-threshold")
    selection_path = artifact_root / "model1_selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["selected_step"] == 20
    assert selection["format_gate_override"] is True
    assert selection["selection_policy"] == "manual_format_gate_override"

    manifest = {"model1_selection_policy": {"selected_step": 20, "allow_below_format_threshold": True}}
    assert selected_model_from_receipt(manifest, selection_path) == artifact_root / "candidates/step_20"

    manifest["model1_selection_policy"]["selected_step"] = 15
    with pytest.raises(RuntimeError, match="selected_step must be 20"):
        selected_model_from_receipt(manifest, selection_path)


def test_stage123_manifest_must_explicitly_allow_override(tmp_path, monkeypatch):
    artifact_root = make_candidate_root(tmp_path)
    run_selector(monkeypatch, artifact_root, "--allow-below-format-threshold")
    manifest = {"model1_selection_policy": {"selected_step": 20, "allow_below_format_threshold": False}}
    with pytest.raises(RuntimeError, match="allow_below_format_threshold must be true"):
        selected_model_from_receipt(manifest, artifact_root / "model1_selection.json")


def test_dry_run_policy_validation_fails_closed_without_step_binding():
    with pytest.raises(RuntimeError, match="must bind model1_selection_policy"):
        model1_selection_policy({})
    with pytest.raises(RuntimeError, match="must bind selected_step"):
        model1_selection_policy({"model1_selection_policy": {}})
    with pytest.raises(RuntimeError, match="must be 20"):
        model1_selection_policy(
            {"model1_selection_policy": {"selected_step": 15, "allow_below_format_threshold": True}}
        )
    with pytest.raises(RuntimeError, match="must be true"):
        model1_selection_policy(
            {"model1_selection_policy": {"selected_step": 20, "allow_below_format_threshold": False}}
        )
    assert (
        model1_selection_policy(
            {"model1_selection_policy": {"selected_step": 20, "allow_below_format_threshold": True}}
        )["selected_step"]
        == 20
    )
