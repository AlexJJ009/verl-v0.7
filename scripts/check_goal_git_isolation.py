#!/usr/bin/env python3
"""Verify content-addressed dirty baselines and the controlled recipe adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


def content(path: Path) -> bytes:
    return os.readlink(path).encode() if path.is_symlink() else path.read_bytes()


def describe(path: Path) -> dict:
    info = path.lstat()
    kind = "symlink" if stat.S_ISLNK(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
    return {
        "type": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "size": len(content(path)),
        "sha256": hashlib.sha256(content(path)).hexdigest(),
    }


def verify_baseline(repo: Path, baseline: dict, allowed: set[str]) -> list[str]:
    failures = []
    entries = baseline.get("entries")
    if not isinstance(entries, list):
        return [f"{repo}: baseline entries missing"]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != baseline.get("aggregate_sha256"):
        failures.append(f"{repo}: baseline aggregate mismatch")
    for entry in entries:
        rel = entry["path"]
        if rel in allowed:
            continue
        path = repo / rel
        if not path.exists() and not path.is_symlink():
            failures.append(f"{repo}: protected path missing: {rel}")
            continue
        now = describe(path)
        for key in ("type", "mode", "size", "sha256"):
            if now[key] != entry[key]:
                failures.append(f"{repo}: protected path {key} changed: {rel}")
    return failures


def dirty_paths(repo: Path) -> set[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo,
    )
    paths: set[str] = set()
    text = output.decode("utf-8", "surrogateescape") if isinstance(output, bytes) else output
    records = text.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status, path = record[:2], record[3:]
        paths.add(path)
        if status[0] in "RC" and index < len(records):
            renamed_from = records[index]
            index += 1
            if renamed_from:
                paths.add(renamed_from)
    return paths


def verify_no_unregistered_dirty(
    repo: Path,
    baseline: dict,
    extra_allowed: set[str] | None = None,
    current_dirty: set[str] | None = None,
) -> list[str]:
    allowed = {entry["path"] for entry in baseline.get("entries", [])} | (extra_allowed or set())
    extra = sorted((dirty_paths(repo) if current_dirty is None else current_dirty) - allowed)
    return [f"{repo}: unregistered dirty path: {path}" for path in extra]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--superproject", type=Path, required=True)
    parser.add_argument("--submodule", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    super_baseline = json.loads((args.baseline_root / "superproject.json").read_text())
    recipe_baseline = json.loads((args.baseline_root / "recipe.json").read_text())
    failures = verify_baseline(args.superproject, super_baseline, {"recipe"})
    failures += verify_baseline(args.submodule, recipe_baseline, set())
    failures += verify_no_unregistered_dirty(args.superproject, super_baseline, {"recipe"})
    failures += verify_no_unregistered_dirty(args.submodule, recipe_baseline)
    pointer = subprocess.check_output(
        ["git", "-C", str(args.superproject), "ls-files", "-s", "recipe"], text=True
    ).strip()
    if pointer and not pointer.startswith("160000 "):
        failures.append("superproject recipe path is not a gitlink")
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
