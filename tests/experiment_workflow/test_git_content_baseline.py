from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def tool():
    path = ROOT / "scripts/check_goal_git_isolation.py"
    spec = importlib.util.spec_from_file_location("git_isolation", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module); return module


def baseline(path: Path) -> dict:
    item = tool().describe(path)
    entry = {"path": path.name, **item, "status": "??", "head_blob": None, "index_blob": None}
    aggregate = hashlib.sha256(json.dumps([entry], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"entries": [entry], "aggregate_sha256": aggregate, "head": "abc"}


def test_unchanged_file_passes_and_content_or_mode_change_fails(tmp_path: Path):
    path = tmp_path / "work.txt"; path.write_text("original")
    data = baseline(path); verifier = tool()
    assert verifier.verify_baseline(tmp_path, data, set()) == []
    path.write_text("replacement")
    assert any("changed" in item for item in verifier.verify_baseline(tmp_path, data, set()))
    path.write_text("original"); path.chmod(0o755)
    assert any("mode changed" in item for item in verifier.verify_baseline(tmp_path, data, set()))


def test_deletion_symlink_target_and_manifest_truncation_fail(tmp_path: Path):
    target = tmp_path / "target"; target.write_text("x")
    path = tmp_path / "link"; path.symlink_to("target")
    data = baseline(path); verifier = tool()
    path.unlink(); path.symlink_to("other")
    assert any("changed" in item for item in verifier.verify_baseline(tmp_path, data, set()))
    path.unlink()
    assert any("missing" in item for item in verifier.verify_baseline(tmp_path, data, set()))
    data["entries"] = []
    assert any("aggregate mismatch" in item for item in verifier.verify_baseline(tmp_path, data, set()))


def test_explicit_adoption_allows_only_declared_path(tmp_path: Path):
    path = tmp_path / "adopted.sh"; path.write_text("old")
    data = baseline(path); data["head"] = "base"
    original = data["entries"][0]; path.write_text("new")
    now = tool().describe(path)
    adoption = {"baseline_commit": "base", "baseline_aggregate_sha256": data["aggregate_sha256"], "allowed_paths": [path.name], "entries": [{"path": path.name, "baseline": {key: original.get(key) for key in ("status", "type", "mode", "size", "sha256", "head_blob", "index_blob")}, "result": now, "changed": True}]}
    assert tool().verify_baseline(tmp_path, data, {path.name}) == []
    assert tool().verify_adoption(tmp_path, data, adoption) == []


def test_cli_discovers_standard_adoption_manifests(monkeypatch, tmp_path: Path):
    superproject = tmp_path / "super"; recipe = superproject / "recipe"; baseline_root = tmp_path / "baseline"
    manifests = superproject / "docs/joint_training/manifests"
    recipe.mkdir(parents=True); baseline_root.mkdir(); manifests.mkdir(parents=True)
    super_file = superproject / "goal.md"; recipe_file = recipe / "stage.sh"
    super_file.write_text("before"); recipe_file.write_text("before")
    super_baseline = baseline(super_file); recipe_baseline = baseline(recipe_file)
    super_baseline["head"] = "super-head"; recipe_baseline["head"] = "recipe-head"
    (baseline_root / "superproject.json").write_text(json.dumps(super_baseline))
    (baseline_root / "recipe.json").write_text(json.dumps(recipe_baseline))
    super_original = super_baseline["entries"][0]; recipe_original = recipe_baseline["entries"][0]
    super_file.write_text("after"); recipe_file.write_text("after")
    def adoption(head, aggregate, original, path, current):
        return {"baseline_commit":head,"baseline_aggregate_sha256":aggregate,"allowed_paths":[path.name],"entries":[{"path":path.name,"baseline":{k:original.get(k) for k in ("status","type","mode","size","sha256","head_blob","index_blob")},"result":tool().describe(current),"changed":True}]}
    (manifests / "goal_contract_dirty_adoption.json").write_text(json.dumps(adoption("super-head",super_baseline["aggregate_sha256"],super_original,super_file,super_file)))
    (manifests / "stage123_dirty_adoption.json").write_text(json.dumps(adoption("recipe-head",recipe_baseline["aggregate_sha256"],recipe_original,recipe_file,recipe_file)))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"")
    monkeypatch.setattr("sys.argv", ["check_goal_git_isolation.py","--superproject",str(superproject),"--submodule",str(recipe),"--baseline-root",str(baseline_root)])
    assert tool().main() == 0


def test_unregistered_dirty_paths_are_rejected(monkeypatch, tmp_path: Path):
    known = tmp_path / "known.txt"
    known.write_text("preserved")
    data = baseline(known)
    failures = tool().verify_no_unregistered_dirty(tmp_path, data, current_dirty={"known.txt", "unexpected.py"})
    assert failures == [f"{tmp_path}: unregistered dirty path: unexpected.py"]
