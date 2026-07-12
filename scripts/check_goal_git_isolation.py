#!/usr/bin/env python3
"""Verify content-addressed dirty baselines and the controlled recipe adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def content(path: Path) -> bytes:
    return os.readlink(path).encode() if path.is_symlink() else path.read_bytes()


def describe(path: Path) -> dict:
    info = path.lstat()
    kind = "symlink" if stat.S_ISLNK(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
    return {"type": kind, "mode": stat.S_IMODE(info.st_mode), "size": len(content(path)), "sha256": hashlib.sha256(content(path)).hexdigest()}


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


def verify_adoption(recipe: Path, baseline: dict, adoption: dict, *, require_full_scope: bool = True) -> list[str]:
    failures = []
    source = {item["path"]: item for item in baseline["entries"]}
    allowed = adoption.get("allowed_paths", [])
    if adoption.get("baseline_commit") != baseline.get("head") or adoption.get("baseline_aggregate_sha256") != baseline.get("aggregate_sha256"):
        failures.append("adoption baseline identity mismatch")
    if require_full_scope and set(allowed) != set(source):
        failures.append("adoption scope differs from recipe dirty baseline")
    if not set(allowed).issubset(source):
        failures.append("adoption contains path outside dirty baseline")
    records = {item["path"]: item for item in adoption.get("entries", [])}
    if set(records) != set(allowed):
        failures.append("adoption entries do not match allowed paths")
    for rel in allowed:
        record = records.get(rel, {})
        original = source.get(rel, {})
        baseline_record = record.get("baseline", {})
        for key in ("status", "type", "mode", "size", "sha256", "head_blob", "index_blob"):
            if baseline_record.get(key) != original.get(key):
                failures.append(f"adoption original {key} mismatch: {rel}")
        path = recipe / rel
        if not path.is_file():
            failures.append(f"adopted path missing: {rel}")
            continue
        now = describe(path); result = record.get("result", {})
        for key in ("type", "mode", "size", "sha256"):
            if result.get(key) != now[key]:
                failures.append(f"adoption result {key} mismatch: {rel}")
        if record.get("changed") != (now["sha256"] != original.get("sha256")):
            failures.append(f"adoption changed flag mismatch: {rel}")
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
    parser.add_argument("--adoption-manifest", type=Path)
    parser.add_argument("--superproject-adoption-manifest", type=Path)
    args = parser.parse_args()
    if args.adoption_manifest is None:
        candidate = args.superproject / "docs/joint_training/manifests/stage123_dirty_adoption.json"
        if candidate.is_file():
            args.adoption_manifest = candidate
    if args.superproject_adoption_manifest is None:
        candidate = args.superproject / "docs/joint_training/manifests/goal_contract_dirty_adoption.json"
        if candidate.is_file():
            args.superproject_adoption_manifest = candidate
    super_baseline = json.loads((args.baseline_root / "superproject.json").read_text())
    recipe_baseline = json.loads((args.baseline_root / "recipe.json").read_text())
    adoption = json.loads(args.adoption_manifest.read_text()) if args.adoption_manifest else None
    super_adoption = json.loads(args.superproject_adoption_manifest.read_text()) if args.superproject_adoption_manifest else None
    recipe_allowed = set(adoption.get("allowed_paths", [])) if adoption else set()
    super_allowed = {"recipe"} | (set(super_adoption.get("allowed_paths", [])) if super_adoption else set())
    failures = verify_baseline(args.superproject, super_baseline, super_allowed)
    failures += verify_baseline(args.submodule, recipe_baseline, recipe_allowed)
    failures += verify_no_unregistered_dirty(args.superproject, super_baseline, {"recipe"})
    failures += verify_no_unregistered_dirty(args.submodule, recipe_baseline)
    if super_adoption:
        failures += verify_adoption(args.superproject, super_baseline, super_adoption, require_full_scope=False)
    if adoption:
        failures += verify_adoption(args.submodule, recipe_baseline, adoption)
    pointer = subprocess.check_output(["git", "-C", str(args.superproject), "ls-files", "-s", "recipe"], text=True).strip()
    if pointer and not pointer.startswith("160000 "):
        failures.append("superproject recipe path is not a gitlink")
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
