from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_tool():
    path = ROOT / "scripts/check_new_experiment_gate.py"
    spec = importlib.util.spec_from_file_location("new_experiment_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    legacy = repo / "legacy_queue.sh"
    legacy.write_text("#!/bin/sh\necho legacy\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    dirty = {"schema_version": 1, "head": head, "entries": []}
    tool = load_tool()
    inventory = tool.inventory_payload(repo, head, dirty)
    return tool, repo, dirty, inventory


def test_unchanged_legacy_and_manifest_native_addition_pass(tmp_path: Path):
    tool, repo, dirty, inventory = fixture(tmp_path)
    (repo / "new_queue.sh").write_text("#!/bin/sh\nSTAGE123_MANIFEST=experiment_manifest/stage123.yaml\n")
    assert tool.check(repo, inventory, dirty) == []


def test_new_legacy_queue_and_rename_fail(tmp_path: Path):
    tool, repo, dirty, inventory = fixture(tmp_path)
    (repo / "bad_queue.sh").write_text("#!/bin/sh\necho bad\n")
    failures = tool.check(repo, inventory, dirty)
    assert "non-manifest-native runnable change: bad_queue.sh" in failures
    (repo / "legacy_queue.sh").rename(repo / "renamed_queue.sh")
    failures = tool.check(repo, inventory, dirty)
    assert "legacy runnable removed or renamed: legacy_queue.sh" in failures
    assert "non-manifest-native runnable change: renamed_queue.sh" in failures


def test_tampered_inventory_fails_closed(tmp_path: Path):
    tool, repo, dirty, inventory = fixture(tmp_path)
    inventory["paths"]["legacy_queue.sh"] = "0" * 64
    assert tool.check(repo, inventory, dirty) == ["baseline inventory content mismatch"]
