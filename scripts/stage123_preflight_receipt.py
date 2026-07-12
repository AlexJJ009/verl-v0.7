#!/usr/bin/env python3
"""Issue and verify content-addressed Stage123 preflight receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def authorized_runs(manifest: dict) -> tuple[list[str], dict[str, str]]:
    workloads = manifest["calibration_workloads"]
    workload_hashes = {phase: canonical_sha256(workloads[phase]) for phase in ("stage1", "stage2", "stage3")}
    materialized = {
        phase: all(source.get("state") == "materialized" for source in workloads[phase]["model_sources"])
        for phase in workloads
    }
    authorized = []
    for run in manifest["runs"]:
        if run["phase"] == "stage2" and materialized["stage1"] and materialized["stage2"]:
            authorized.append(run["id"])
        elif run["phase"] == "stage3" and materialized["stage3"]:
            authorized.append(run["id"])
    return authorized, workload_hashes


def issue(args) -> dict:
    manifest = json.loads(args.normalized_manifest.read_text())
    report = json.loads(args.report.read_text())
    budget = json.loads(args.budget_result.read_text())
    if budget.get("ok") is not True or budget.get("decision") != "pass":
        raise ValueError("budget result is not passing")
    run_ids, workload_hashes = authorized_runs(manifest)
    return {
        "schema_version": 1,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "normalized_manifest_file_sha256": digest(args.normalized_manifest),
        "report_sha256": digest(args.report),
        "policy_sha256": digest(args.policy),
        "budget_result_sha256": digest(args.budget_result),
        "profile_sha256": manifest["resource_profile"]["sha256"],
        "authorized_run_ids": run_ids,
        "workload_descriptor_sha256": workload_hashes,
    }


def verify(args) -> dict:
    receipt = json.loads(args.receipt.read_text())
    manifest = json.loads(args.normalized_manifest.read_text())
    failures = []
    expected = {
        "status": "pass",
        "manifest_sha256": manifest["manifest_sha256"],
        "normalized_manifest_file_sha256": digest(args.normalized_manifest),
        "report_sha256": digest(args.report),
        "policy_sha256": digest(args.policy),
        "profile_sha256": args.profile_hash,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            failures.append(f"{key} mismatch")
    authorized, workload_hashes = authorized_runs(manifest)
    if receipt.get("authorized_run_ids") != authorized:
        failures.append("authorized_run_ids mismatch")
    if receipt.get("workload_descriptor_sha256") != workload_hashes:
        failures.append("workload_descriptor_sha256 mismatch")
    if args.run_id not in receipt.get("authorized_run_ids", []):
        failures.append("run_id not authorized by receipt")
    age = (datetime.now(timezone.utc) - parse_time(receipt["generated_at"])).total_seconds()
    if age < 0 or age > args.max_age_seconds:
        failures.append("receipt stale")
    return {"ok": not failures, "failures": failures, "age_seconds": round(age, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("issue")
    p.add_argument("--normalized-manifest", type=Path, required=True); p.add_argument("--report", type=Path, required=True); p.add_argument("--policy", type=Path, required=True); p.add_argument("--budget-result", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--receipt", type=Path, required=True); p.add_argument("--normalized-manifest", type=Path, required=True); p.add_argument("--report", type=Path, required=True); p.add_argument("--policy", type=Path, required=True); p.add_argument("--run-id", required=True); p.add_argument("--profile-hash", required=True); p.add_argument("--max-age-seconds", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "issue":
            result = issue(args); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            result = verify(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok", True) else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"receipt error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
