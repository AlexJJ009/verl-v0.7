#!/usr/bin/env python3
"""Create fail-closed evaluator and launch-review receipts for Code A/D0/C."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCES = ("HumanEval+", "MBPP+", "LiveCodeBench")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_evaluator_results(results: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for source in OFFICIAL_SOURCES:
        source_result = results.get(source, {})
        known_pass = source_result.get("known_pass", {})
        known_fail = source_result.get("known_fail", {})
        checks[f"{source}_known_pass"] = (
            float(known_pass.get("score", -1.0)) == 1.0 and int(known_pass.get("code_reward_dependency_error", 1)) == 0
        )
        checks[f"{source}_known_fail"] = (
            float(known_fail.get("score", 1.0)) == -1.0 and int(known_fail.get("code_reward_dependency_error", 1)) == 0
        )
    fail_closed = results.get("fail_closed", {})
    malformed = fail_closed.get("malformed", {})
    missing_eos = fail_closed.get("missing_eos", {})
    checks["malformed_rejected_before_execution"] = (
        float(malformed.get("score", 1.0)) == -1.0 and malformed.get("code_reward_status") == "format_error"
    )
    checks["missing_eos_rejected"] = (
        float(missing_eos.get("score", 1.0)) == -1.0
        and missing_eos.get("truncated") is True
        and missing_eos.get("has_eos") is False
    )
    return checks


def _load_positive_examples(validation_jsonl: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with validation_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source = row.get("data_source")
            if (
                source in OFFICIAL_SOURCES
                and source not in selected
                and float(row.get("score", -1.0)) == 1.0
                and row.get("format_contract_success") is True
                and row.get("response_eos_present") is True
            ):
                selected[source] = row
    missing = sorted(set(OFFICIAL_SOURCES) - set(selected))
    if missing:
        raise RuntimeError(f"post-fix validation has no admitted positive example for: {missing}")
    return selected


def evaluator_probe(manifest_path: Path, validation_jsonl: Path, output: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    os.environ["LCB_INPUT_OUTPUT_INDEX"] = manifest["evaluator_contract"]["lcb_index"]
    evaluator_paths = (
        Path(os.environ.get("CODE_EVAL_OFFICIAL_SITE", "/data-1/code_eval_envs/official_site")),
        Path(os.environ.get("LCB_REPO_DIR", "/data-1/code_eval_envs/LiveCodeBench")),
    )
    for path in reversed(evaluator_paths):
        if not path.is_dir():
            raise RuntimeError(f"official evaluator path missing: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in evaluator_paths), os.environ.get("PYTHONPATH", "")]
    )
    from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import (
        compute_score_code_official_aligned,
    )

    examples = _load_positive_examples(validation_jsonl)
    results: dict[str, Any] = {}
    extra = {"response_eos_present": True, "valid_response_length": 1, "max_resp_len": 8192}
    for source, row in examples.items():
        ground_truth = json.loads(row["gts"]) if isinstance(row["gts"], str) else row["gts"]
        known_pass = compute_score_code_official_aligned(source, row["output"], ground_truth, extra_info=extra)
        known_fail = compute_score_code_official_aligned(
            source,
            "<think>Return an intentionally wrong implementation.</think>\n<answer>\n```python\npass\n```\n</answer>",
            ground_truth,
            extra_info=extra,
        )
        results[source] = {
            "uid": row.get("uid"),
            "known_pass": _jsonable(known_pass),
            "known_fail": _jsonable(known_fail),
        }

    first = examples[OFFICIAL_SOURCES[0]]
    first_gt = json.loads(first["gts"]) if isinstance(first["gts"], str) else first["gts"]
    malformed = compute_score_code_official_aligned(
        OFFICIAL_SOURCES[0],
        "<answer>\n```python\npass\n```\n</answer>",
        first_gt,
        extra_info=extra,
    )
    missing_eos = compute_score_code_official_aligned(
        OFFICIAL_SOURCES[0],
        first["output"],
        first_gt,
        extra_info={"response_eos_present": False, "valid_response_length": 8192, "max_resp_len": 8192},
    )
    results["fail_closed"] = {
        "malformed": _jsonable(malformed),
        "missing_eos": _jsonable(missing_eos),
    }
    checks = validate_evaluator_results(results)
    lcb_index = Path(manifest["evaluator_contract"]["lcb_index"])
    reward_path = ROOT / manifest["evaluator_contract"]["reward_path"]
    receipt = {
        "schema_version": 1,
        "receipt_type": "code_wdl_acd0_official_evaluator_admission",
        "status": "pass" if checks and all(checks.values()) else "fail",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "validation_jsonl": str(validation_jsonl),
        "validation_jsonl_sha256": sha256(validation_jsonl),
        "reward_path": str(reward_path),
        "reward_sha256": sha256(reward_path),
        "lcb_index": str(lcb_index),
        "lcb_index_sha256": sha256(lcb_index),
        "evaluator_python_paths": [str(path) for path in evaluator_paths],
        "checks": checks,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt["status"] != "pass":
        raise RuntimeError("official evaluator admission failed")
    return receipt


def _load_queue_module():
    path = ROOT / "scripts/code_wdl_acd0_queue.py"
    spec = importlib.util.spec_from_file_location("code_wdl_acd0_queue_review", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def review(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    queue = _load_queue_module()
    queue.validate_manifest(manifest, require_launch=False)
    required = ("stage1_reuse_receipt", "eos_regression_receipt", "evaluator_receipt", "gpu_probe_receipt")
    receipt_checks = {}
    for name in required:
        path = Path(manifest["paths"][name])
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipt_checks[name] = payload.get("status") == "pass"
    target_checks = {}
    for run in manifest["runs"]:
        try:
            queue.require_clean_targets(manifest, run)
        except RuntimeError:
            target_checks[run["id"]] = False
        else:
            target_checks[run["id"]] = True
    wrapper_checks = {run_id: path.is_file() for run_id, path in queue.WRAPPERS.items()}
    disk = {}
    for mount, minimum_gib in (("/data-1", 180), ("/data-2", 300)):
        free = shutil.disk_usage(mount).free // (1024**3)
        disk[mount] = {"free_gib": free, "minimum_gib": minimum_gib, "pass": free >= minimum_gib}
    dry_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/code_wdl_acd0_queue.py"), "--manifest", str(manifest_path), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    checks = {
        "all_required_receipts_pass": all(receipt_checks.values()),
        "all_targets_clean": all(target_checks.values()),
        "all_wrappers_present": all(wrapper_checks.values()),
        "disk_headroom": all(item["pass"] for item in disk.values()),
        "queue_dry_run": dry_run.returncode == 0,
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": "code_wdl_acd0_independent_launch_review",
        "status": "pass" if all(checks.values()) else "fail",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checks": checks,
        "receipt_checks": receipt_checks,
        "target_checks": target_checks,
        "wrapper_checks": wrapper_checks,
        "disk": disk,
        "queue_dry_run": {"returncode": dry_run.returncode, "stdout": dry_run.stdout, "stderr": dry_run.stderr},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt["status"] != "pass":
        raise RuntimeError("independent launch review failed")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluator = sub.add_parser("evaluator")
    evaluator.add_argument("--manifest", type=Path, required=True)
    evaluator.add_argument("--validation-jsonl", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    launch_review = sub.add_parser("review")
    launch_review.add_argument("--manifest", type=Path, required=True)
    launch_review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluator":
        payload = evaluator_probe(args.manifest, args.validation_jsonl, args.output)
    else:
        payload = review(args.manifest, args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
