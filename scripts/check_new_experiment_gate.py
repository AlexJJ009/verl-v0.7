#!/usr/bin/env python3
"""Reject new or changed runnable workflows that are not manifest-native."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runnable(path: str) -> bool:
    name = Path(path).name
    return name.endswith(".sh") and (name.startswith("monitor") or "queue" in name)


def git_bytes(repo: Path, commit: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True)
    return result.stdout if result.returncode == 0 else None


def snapshot(repo: Path, commit: str, dirty_baseline: dict) -> dict[str, str]:
    names = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit], text=True).splitlines()
    result = {path: digest(data) for path in names if runnable(path) and (data := git_bytes(repo, commit, path)) is not None}
    for entry in dirty_baseline.get("entries", []):
        path = entry["path"]
        if runnable(path):
            result[path] = entry["sha256"]
    return result


def current(repo: Path) -> dict[str, str]:
    paths = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    return {path: digest((repo / path).read_bytes()) for path in paths if runnable(path) and (repo / path).is_file()}


def inventory_payload(repo: Path, commit: str, dirty_baseline: dict) -> dict:
    paths = snapshot(repo, commit, dirty_baseline)
    return {"schema_version": 1, "baseline_commit": commit, "paths": paths}


def manifest_native(repo: Path, path: str) -> bool:
    file_path = repo / path
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if "experiment_manifest" in text or "STAGE123_MANIFEST" in text or "stage123_manifest_gate.sh" in text:
        return True
    return False


def check(repo: Path, inventory: dict, dirty_baseline: dict) -> list[str]:
    failures = []
    if inventory.get("schema_version") != 1:
        failures.append("unsupported inventory schema")
        return failures
    expected = inventory_payload(repo, inventory["baseline_commit"], dirty_baseline)
    if inventory.get("paths") != expected["paths"]:
        failures.append("baseline inventory content mismatch")
        return failures
    before, after = inventory["paths"], current(repo)
    removed = set(before) - set(after)
    added = set(after) - set(before)
    changed = {path for path in set(before) & set(after) if before[path] != after[path]}
    for path in sorted(added | changed):
        if not manifest_native(repo, path):
            failures.append(f"non-manifest-native runnable change: {path}")
    for path in sorted(removed):
        failures.append(f"legacy runnable removed or renamed: {path}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dirty-baseline", type=Path, required=True)
    parser.add_argument("--write-inventory", action="store_true")
    args = parser.parse_args()
    dirty = json.loads(args.dirty_baseline.read_text())
    if args.write_inventory:
        payload = inventory_payload(args.repo, dirty["head"], dirty)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"ok": True, "path_count": len(payload["paths"])}))
        return 0
    failures = check(args.repo, json.loads(args.inventory.read_text()), dirty)
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
