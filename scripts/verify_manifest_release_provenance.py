#!/usr/bin/env python3
"""Verify immutable manifest/preflight provenance before release actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(manifest: dict, provenance: dict, run_prefix: str, final_step: int, train_file: Path, receipt: Path) -> dict:
    failures = []
    run = next((item for item in manifest["runs"] if item["run_prefix"] == run_prefix), None)
    if run is None:
        return {"ok": False, "failures": ["run prefix absent from manifest"]}
    checks = {
        "manifest_sha256": manifest["manifest_sha256"],
        "profile_sha256": manifest["resource_profile"]["sha256"],
        "train_file_sha256": run["train_file_sha256"],
        "preflight_receipt_sha256": digest(receipt) if receipt.is_file() else None,
    }
    for key, expected in checks.items():
        if provenance.get(key) != expected:
            failures.append(f"{key} mismatch")
    if run["final_step"] != final_step:
        failures.append("final_step mismatch")
    if Path(run["train_file"]) != train_file:
        failures.append("train_file mismatch")
    if not train_file.is_file() or digest(train_file) != run["train_file_sha256"]:
        failures.append("train_file content mismatch")
    if provenance.get("run_prefix") != run_prefix:
        failures.append("provenance run_prefix mismatch")
    if provenance.get("release_eligible") is not True:
        failures.append("run provenance is not release eligible")
    return {"ok": not failures, "failures": failures, "run_id": run["id"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.normalized_manifest.read_text()), json.loads(args.provenance.read_text()), args.run_prefix, args.final_step, args.train_file, args.preflight_receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
