#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_METRICS = {"validation_elapsed_seconds", "phase_elapsed_seconds", "peak_rss_gib", "gpu_wait_fraction"}
REQUIRED_PHASES = ["stage1", "stage2", "stage3"]
TREATMENT_ONLY_PHASES = ["stage2", "stage3"]
ALLOWED_PHASE_SETS = (REQUIRED_PHASES, TREATMENT_ONLY_PHASES)
PRIMARY_RUN_IDS = ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def iso_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def implementation_hash(path: Path) -> str:
    if not path.is_file():
        raise ValueError("implementation tree evidence is missing")
    return sha256(path)


def matching_authorization(ledger_path: Path, decision_id: str) -> dict:
    decisions = []
    if ledger_path.is_file():
        for line in ledger_path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get("event") == "USER_DECISION_RECORDED" and value.get("decision_id") == decision_id:
                    decisions.append(value)
    if not decisions:
        raise ValueError("matching USER_DECISION_RECORDED event is missing")
    decision = decisions[-1]
    plan_version = decision.get("plan_version")
    plan_sha256 = decision.get("plan_sha256")
    if not isinstance(plan_version, int) or plan_version < 1:
        raise ValueError("authorization decision lacks a valid plan version")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("authorization decision lacks a valid plan hash")
    plan_path = ledger_path.parent / "plan.md"
    if not plan_path.is_file() or sha256(plan_path) != plan_sha256:
        raise ValueError("authorization decision does not match the current plan")
    return decision


def validate_prediction_comparison(value: dict) -> None:
    policy_path = ROOT / "config/experiment_execution/calibration_policy_v1.json"
    policy = load(policy_path)
    if value.get("policy_id") != policy.get("policy_id") or value.get("policy_sha256") != sha256(policy_path):
        raise ValueError("prediction comparison policy binding mismatch")
    comparisons = value.get("comparisons")
    if value.get("qualified") is not True or not isinstance(comparisons, list):
        raise ValueError("prediction comparison is incomplete")
    if {item.get("metric") for item in comparisons if isinstance(item, dict)} != PREDICTION_METRICS:
        raise ValueError("prediction comparison metric set mismatch")
    for item in comparisons:
        decision = item.get("decision", {})
        if item.get("history_count") != len(item.get("history", [])) or item.get("history_count", 0) < 3:
            raise ValueError("prediction comparison history is insufficient")
        if not isinstance(item.get("predicted_bound"), int | float) or not isinstance(
            item.get("observed_maximum"), int | float
        ):
            raise ValueError("prediction comparison values are missing")
        predicted = float(item["predicted_bound"])
        observed = float(item["observed_maximum"])
        ratio = observed / predicted if predicted > 0 else float("inf")
        if predicted != max(float(value) for value in item["history"]):
            raise ValueError("prediction bound does not match history")
        if ratio > policy["prediction"]["maximum_observed_to_predicted_ratio"] or not math.isclose(
            decision.get("context", {}).get("ratio", -1), ratio, rel_tol=1e-12
        ):
            raise ValueError("prediction comparison ratio mismatch")
        if decision.get("qualified") is not True or decision.get("code") != "qualified":
            raise ValueError("prediction comparison is not qualified")


def validate_pointer(pointer_path: Path, *, run_id: str, decision_id: str, scratch_root: Path) -> tuple[dict, dict]:
    pointer = load(pointer_path)
    required = (
        "schema_version",
        "run_id",
        "authorization_decision_id",
        "report_sha256",
        "generated_at_utc",
        "report_started_at_utc",
        "report_completed_at_utc",
        "run_root",
        "report",
        "status",
    )
    missing = [key for key in required if key not in pointer]
    if missing or pointer.get("schema_version") != 2:
        raise ValueError(f"invalid producer pointer: missing={missing}")
    if pointer["run_id"] != run_id or pointer["authorization_decision_id"] != decision_id:
        raise ValueError("producer pointer decision or run mismatch")
    run_root = Path(pointer["run_root"]).resolve()
    report_path = Path(pointer["report"]).resolve()
    if scratch_root.resolve() not in run_root.parents or scratch_root.resolve() not in report_path.parents:
        raise ValueError("producer report is outside decision-specific scratch root")
    if pointer.get("status") != "passed" or not report_path.is_file():
        raise ValueError("producer report is not terminal passed")
    if sha256(report_path) != pointer["report_sha256"]:
        raise ValueError("producer report hash mismatch")
    report = load(report_path)
    if (
        report.get("status") != "passed"
        or report.get("run_id") != run_id
        or report.get("authorization_decision_id") != decision_id
    ):
        raise ValueError("producer report identity or status mismatch")
    phases = report.get("phases")
    phase_names = [item.get("phase") for item in phases] if isinstance(phases, list) else []
    if phase_names not in ALLOWED_PHASE_SETS:
        raise ValueError("producer report must contain stage1,stage2,stage3 or treatment-only stage2,stage3")
    if any(
        not isinstance(item.get("repetitions"), list)
        or len(item["repetitions"]) != 3
        or any(rep.get("status") != "passed" for rep in item["repetitions"])
        for item in phases
    ):
        raise ValueError("producer report repetition evidence is incomplete")
    if report.get("optimizer_steps") != 0 or report.get("formal_checkpoints") != []:
        raise ValueError("producer report is not zero-step and checkpoint-free")
    validate_prediction_comparison(report.get("prediction_comparison", {}))
    if report.get("cleanup", {}).get("resources_released") is not True:
        raise ValueError("producer report lacks qualified prediction or cleanup evidence")
    return pointer, report


def render(args: argparse.Namespace) -> int:
    ledger_path = Path(args.runtime_ledger)
    decision = matching_authorization(ledger_path, args.decision_id)
    decision_time = datetime.fromisoformat(decision["time"].replace("Z", "+00:00")).timestamp()
    if Path(args.state_root).stat().st_mtime <= decision_time:
        raise ValueError("decision-specific state root predates authorization decision")
    pointer, report = validate_pointer(
        Path(args.latest_probe),
        run_id=args.run_id,
        decision_id=args.decision_id,
        scratch_root=Path(args.state_root).parent,
    )
    state_path = Path(args.state_root) / f"{args.run_id}.json"
    state = load(state_path)
    if state.get("status") != "succeeded" or state.get("run_id") != args.run_id:
        raise ValueError("execution core run is not succeeded")
    manifest = json.loads(
        subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", args.manifest, "--format", "json"],
            text=True,
        )
    )
    profile_hash = subprocess.check_output(
        ["bash", "-lc", f"source {args.resource_profile}; stage123_profile_hash"], text=True
    ).strip()
    if profile_hash != manifest.get("resource_profile", {}).get("sha256"):
        raise ValueError("resource profile does not match manifest")
    tree_hash = implementation_hash(Path(args.implementation_tree))
    report_manifest = report.get("manifest_sha256")
    if report_manifest != manifest.get("manifest_sha256"):
        raise ValueError("producer report does not match manifest")
    completed = report.get("report_completed_at_utc") or pointer["report_completed_at_utc"]
    started = report.get("report_started_at_utc") or pointer["report_started_at_utc"]
    policy_path = ROOT / "config/experiment_execution/calibration_policy_v1.json"
    result = {
        "schema_version": 1,
        "result_type": "calibration_result",
        "decision": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "resource_profile_sha256": profile_hash,
        "implementation_tree_sha256": tree_hash,
        "evidence_commit": args.evidence_commit,
        "workload_identity": {"run_ids": PRIMARY_RUN_IDS, "producer_report_sha256": pointer["report_sha256"]},
        "policy_id": load(policy_path)["policy_id"],
        "policy_sha256": sha256(policy_path),
        "authorization_identity": {
            "decision_id": args.decision_id,
            "run_id": args.run_id,
            "plan_version": decision["plan_version"],
            "plan_sha256": decision["plan_sha256"],
        },
        "started_at": started,
        "completed_at": completed,
        "phase_evidence": [
            {"phase": phase, "status": "passed", "repetitions": data.get("repetitions", [])}
            for phase, data in ((item["phase"], item) for item in report.get("phases", []))
        ],
        "prediction_comparison": report["prediction_comparison"],
        "cleanup": {
            "resources_released": report["cleanup"]["resources_released"],
            "execution_state": state.get("cleanup"),
        },
        "failures": [],
        "probe_pointer": pointer,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def validate(args: argparse.Namespace) -> int:
    value = load(Path(args.input))
    schema = load(Path(args.schema))
    missing = [key for key in schema["required"] if key not in value]
    if value.get("schema_version") != 1 or value.get("result_type") != "calibration_result" or missing:
        raise ValueError(f"invalid calibration result schema: {missing}")
    if value.get("decision") != "passed":
        raise ValueError("calibration result is not passed")
    if value.get("workload_identity", {}).get("run_ids") != PRIMARY_RUN_IDS:
        raise ValueError("calibration result run set mismatch")
    validate_prediction_comparison(value.get("prediction_comparison", {}))
    print(json.dumps({"ok": True, "sha256": sha256(Path(args.input))}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--run-id", required=True)
    render_parser.add_argument("--state-root", required=True)
    render_parser.add_argument("--latest-probe", required=True)
    render_parser.add_argument("--manifest", required=True)
    render_parser.add_argument("--resource-profile", required=True)
    render_parser.add_argument("--implementation-tree", required=True)
    render_parser.add_argument("--evidence-commit", required=True)
    render_parser.add_argument("--runtime-ledger", required=True)
    render_parser.add_argument("--decision-id", required=True)
    render_parser.add_argument("--output", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    try:
        return render(args) if args.action == "render" else validate(args)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
