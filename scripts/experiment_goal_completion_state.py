#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def state(calibration: dict | None, reviewer: dict | None, calibration_sha256: str | None = None) -> str:
    required_hashes = {
        "report_sha256",
        "manifest_sha256",
        "rendered_manifest_sha256",
        "policy_sha256",
        "history_index_sha256",
        "prediction_contract_sha256",
        "preflight_receipt_sha256",
    }
    if (
        not calibration
        or calibration.get("receipt_type") != "code_task_operational_calibration_deployability"
        or calibration.get("decision") != "deployable"
        or set(calibration.get("hashes", {})) != required_hashes
        or not isinstance(calibration.get("profile"), dict)
    ):
        return "PENDING OPERATIONAL CALIBRATION"
    if (
        not reviewer
        or reviewer.get("verdict") != "ACCEPTED"
        or not reviewer.get("all_acceptance_criteria_pass")
        or not calibration_sha256
        or reviewer.get("calibration_receipt_sha256") != calibration_sha256
    ):
        return "PENDING INDEPENDENT ACCEPTANCE"
    return "GOAL COMPLETE"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=Path)
    p.add_argument("--reviewer", type=Path)
    a = p.parse_args()
    load = lambda x: json.loads(x.read_text()) if x and x.is_file() else None
    calibration = load(a.calibration)
    if calibration and calibration.get("receipt_type") == "code_task_operational_calibration_stage12_producer":
        print("limited_receipt_scope_mismatch")
        return 1
    receipt_hash = (
        hashlib.sha256(a.calibration.read_bytes()).hexdigest() if a.calibration and a.calibration.is_file() else None
    )
    result = state(calibration, load(a.reviewer), receipt_hash)
    print(result)
    return 0 if result == "GOAL COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
