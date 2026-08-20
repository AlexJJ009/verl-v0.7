# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tool():
    path = ROOT / "scripts/check_goal_git_isolation.py"
    spec = importlib.util.spec_from_file_location("git_isolation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def baseline(path: Path) -> dict:
    item = tool().describe(path)
    entry = {"path": path.name, **item, "status": "??", "head_blob": None, "index_blob": None}
    aggregate = hashlib.sha256(json.dumps([entry], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"entries": [entry], "aggregate_sha256": aggregate, "head": "abc"}


def test_unchanged_file_passes_and_content_or_mode_change_fails(tmp_path: Path):
    path = tmp_path / "work.txt"
    path.write_text("original")
    data = baseline(path)
    verifier = tool()
    assert verifier.verify_baseline(tmp_path, data, set()) == []
    path.write_text("replacement")
    assert any("changed" in item for item in verifier.verify_baseline(tmp_path, data, set()))
    path.write_text("original")
    path.chmod(0o755)
    assert any("mode changed" in item for item in verifier.verify_baseline(tmp_path, data, set()))


def test_deletion_symlink_target_and_manifest_truncation_fail(tmp_path: Path):
    target = tmp_path / "target"
    target.write_text("x")
    path = tmp_path / "link"
    path.symlink_to("target")
    data = baseline(path)
    verifier = tool()
    path.unlink()
    path.symlink_to("other")
    assert any("changed" in item for item in verifier.verify_baseline(tmp_path, data, set()))
    path.unlink()
    assert any("missing" in item for item in verifier.verify_baseline(tmp_path, data, set()))
    data["entries"] = []
    assert any("aggregate mismatch" in item for item in verifier.verify_baseline(tmp_path, data, set()))


def test_baseline_change_has_no_adoption_bypass(tmp_path: Path):
    path = tmp_path / "adopted.sh"
    path.write_text("old")
    data = baseline(path)
    path.write_text("new")
    failures = tool().verify_baseline(tmp_path, data, set())
    assert failures and "changed" in failures[0]


def test_unregistered_dirty_paths_are_rejected(monkeypatch, tmp_path: Path):
    known = tmp_path / "known.txt"
    known.write_text("preserved")
    data = baseline(known)
    failures = tool().verify_no_unregistered_dirty(tmp_path, data, current_dirty={"known.txt", "unexpected.py"})
    assert failures == [f"{tmp_path}: unregistered dirty path: unexpected.py"]
