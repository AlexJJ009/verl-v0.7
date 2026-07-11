from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path


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
