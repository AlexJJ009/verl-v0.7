from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path("/data-1/tmp/verl_agent_scratch/experiment_workflow/git_baseline/recipe.json")


def module():
    spec = importlib.util.spec_from_file_location("adoption", ROOT / "scripts/build_stage123_adoption_manifest.py")
    result = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(result); return result


def test_real_adoption_scope_is_exact():
    tool = module(); result = tool.build(json.loads(BASELINE.read_text()), ROOT / "recipe")
    assert tuple(result["allowed_paths"]) == tool.ALLOWED
    assert len(result["entries"]) == 8
    assert all(item["baseline"]["status"] == "??" for item in result["entries"])
    assert all(len(item["baseline"]["sha256"]) == 64 for item in result["entries"])


def test_unlisted_dirty_path_is_rejected():
    tool = module(); baseline = json.loads(BASELINE.read_text())
    baseline["entries"].append({"path": "unlisted.sh"})
    with pytest.raises(ValueError, match="do not equal"):
        tool.build(baseline, ROOT / "recipe")
