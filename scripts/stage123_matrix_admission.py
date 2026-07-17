#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def render_manifest(repo_root: Path, manifest: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, str(repo_root / "scripts/stage123_matrix_manifest.py"), "render", str(manifest)], text=True
    )
    return json.loads(output)


def gpu_facts() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    return [
        {"index": int(index.strip()), "name": name.strip(), "memory_total_mib": int(memory.strip())}
        for index, name, memory in (line.split(",") for line in output.splitlines())
    ]


def create(args: argparse.Namespace) -> int:
    manifest = render_manifest(args.repo_root, args.manifest)
    probe = json.loads(args.memory_probe.read_text())
    if probe.get("status") != "passed" or probe.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("memory probe is not a passing result for this manifest")
    facts = gpu_facts()
    if len(facts) != 8 or any(item["name"] != "NVIDIA L40S" for item in facts):
        raise SystemExit("matrix admission requires exactly 8 NVIDIA L40S GPUs")
    payload = {
        "schema_version": 1,
        "admission_type": "stage123_matrix_training_admission",
        "decision": "accepted",
        "candidate_commit": git_output(args.repo_root, "rev-parse", "HEAD"),
        "recipe_gitlink": git_output(args.repo_root, "rev-parse", "HEAD:recipe"),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "resource_profile_sha256": manifest["resource_profile"]["sha256"],
        "memory_probe": str(args.memory_probe),
        "memory_probe_sha256": file_sha256(args.memory_probe),
        "run_ids": [run["id"] for run in manifest["runs"]],
        "gpu_facts": facts,
        "training_steps_calibrated": probe["training_steps"],
        "optimizer_enabled_during_calibration": probe["optimizer_enabled"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"ok": True, "admission": str(args.output), "sha256": file_sha256(args.output)}, sort_keys=True))
    return 0


def validate(args: argparse.Namespace) -> int:
    admission = json.loads(args.admission.read_text())
    if admission.get("schema_version") != 1 or admission.get("decision") != "accepted":
        raise SystemExit("invalid matrix admission")
    if args.run_id not in admission.get("run_ids", []):
        raise SystemExit("run id is not admitted")
    if git_output(args.repo_root, "rev-parse", "HEAD") != admission["candidate_commit"]:
        raise SystemExit("candidate commit mismatch")
    if git_output(args.repo_root, "rev-parse", "HEAD:recipe") != admission["recipe_gitlink"]:
        raise SystemExit("recipe gitlink mismatch")
    manifest_path = Path(admission["manifest"])
    manifest = render_manifest(args.repo_root, manifest_path)
    if manifest["manifest_sha256"] != admission["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    probe_path = Path(admission["memory_probe"])
    probe = json.loads(probe_path.read_text())
    if file_sha256(probe_path) != admission["memory_probe_sha256"] or probe.get("status") != "passed":
        raise SystemExit("memory probe binding mismatch")
    if probe.get("manifest_sha256") != admission["manifest_sha256"]:
        raise SystemExit("memory probe manifest mismatch")
    if gpu_facts() != admission["gpu_facts"]:
        raise SystemExit("GPU facts mismatch")
    print(json.dumps({"ok": True, "run_id": args.run_id}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--repo-root", type=Path, required=True)
    create_parser.add_argument("--manifest", type=Path, required=True)
    create_parser.add_argument("--memory-probe", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--repo-root", type=Path, required=True)
    validate_parser.add_argument("--admission", type=Path, required=True)
    validate_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    return create(args) if args.command == "create" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
