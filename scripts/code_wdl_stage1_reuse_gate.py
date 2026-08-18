#!/usr/bin/env python3
"""Audit whether a frozen Code Stage1 model is safe to reuse for A/C/D0.

Native generation telemetry is authoritative for EOS/truncation.  Historical
reward telemetry before the DAPO plumbing fix can incorrectly report EOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def native_truncated(row: dict) -> bool:
    reason = row.get("response_finish_reason")
    if reason is not None:
        return reason == "length"
    eos = row.get("response_eos_present")
    if eos is not None:
        return not bool(eos)
    return bool(row.get("truncated"))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("validation file is empty")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("data_source", "unknown"))].append(row)

    def one(group: list[dict]) -> dict:
        n = len(group)
        fmt = sum(bool(row.get("format_contract_success")) for row in group)
        correct = sum(float(row.get("acc") or 0.0) > 0 for row in group)
        truncated = sum(native_truncated(row) for row in group)
        completed = [row for row in group if not native_truncated(row)]
        completed_fmt = sum(bool(row.get("format_contract_success")) for row in completed)
        format_bad = n - fmt
        truncated_positive = sum(
            native_truncated(row) and float(row.get("score") or 0.0) > 0 for row in group
        )
        format_bad_positive = sum(
            not bool(row.get("format_contract_success")) and float(row.get("score") or 0.0) > 0
            for row in group
        )
        return {
            "responses": n,
            "strict_format_rate": fmt / n,
            "completed_responses": len(completed),
            "strict_format_given_not_truncated_rate": completed_fmt / len(completed) if completed else 0.0,
            "correct_rate": correct / n,
            "usable_positive_rate": correct / n,
            "correct_given_format_rate": correct / fmt if fmt else 0.0,
            "native_truncation_rate": truncated / n,
            "format_failures_due_to_truncation_rate": truncated / format_bad if format_bad else 0.0,
            "dependency_error_rate": sum(bool(row.get("code_reward_dependency_error")) for row in group) / n,
            "timeout_rate": sum(bool(row.get("code_reward_timeout")) for row in group) / n,
            "format_bad_reward_positive": format_bad_positive,
            "truncated_reward_positive": truncated_positive,
        }

    micro = one(rows)
    sources = {source: one(group) for source, group in sorted(by_source.items())}
    macro_keys = ["strict_format_rate", "correct_rate", "native_truncation_rate"]
    macro = {key: sum(source[key] for source in sources.values()) / len(sources) for key in macro_keys}
    # At beta=0, only correct formatted responses contribute positive SFT loss.
    # This counterfactual holds conditional correctness fixed and exposes the
    # response-throughput loss caused by format/truncation; it is not a causal
    # estimate of final benchmark accuracy.
    retained = micro["strict_format_rate"]
    micro["estimated_positive_signal_loss_fraction"] = 1.0 - retained
    micro["expected_positive_responses_per_group_n8"] = 8.0 * micro["usable_positive_rate"]
    micro["estimated_all_nonpositive_group_probability_n8"] = math.pow(
        1.0 - micro["usable_positive_rate"], 8
    )
    return {"micro": micro, "macro": macro, "sources": sources}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--current-file", type=Path)
    parser.add_argument("--baseline-file", type=Path)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluation-kind",
        choices=("historical_audit", "post_fix_reevaluation"),
        default="historical_audit",
    )
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--min-completed-format", type=float, default=0.99)
    parser.add_argument("--max-native-truncation", type=float, default=0.50)
    parser.add_argument("--min-usable-positive", type=float, default=0.30)
    args = parser.parse_args()

    if args.current_file or args.baseline_file:
        if not args.current_file or not args.baseline_file:
            parser.error("--current-file and --baseline-file must be provided together")
        if args.validation_dir:
            parser.error("use either --validation-dir or explicit current/baseline files")
        current_path = args.current_file
        baseline_path = args.baseline_file
    else:
        if not args.validation_dir:
            parser.error("--validation-dir or explicit current/baseline files are required")
        current_path = args.validation_dir / f"{args.step}.jsonl"
        baseline_path = args.validation_dir / f"{args.baseline_step}.jsonl"
    provenance_path = None
    provenance_sha256 = None
    if args.evaluation_kind == "post_fix_reevaluation":
        if not args.provenance or not args.provenance.is_file():
            parser.error("post_fix_reevaluation requires --provenance")
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        if provenance.get("evaluation_kind") != "post_fix_reevaluation":
            parser.error("provenance evaluation_kind mismatch")
        provenance_path = str(args.provenance.resolve())
        provenance_sha256 = hashlib.sha256(args.provenance.read_bytes()).hexdigest()
    current = summarize(load_rows(current_path))
    baseline = summarize(load_rows(baseline_path))
    delta = current["micro"]["strict_format_rate"] - baseline["micro"]["strict_format_rate"]

    checks = {
        "micro_completed_strict_format": current["micro"]["strict_format_given_not_truncated_rate"]
        >= args.min_completed_format,
        "each_source_completed_strict_format": all(
            item["strict_format_given_not_truncated_rate"] >= args.min_completed_format
            for item in current["sources"].values()
        ),
        "native_truncation": current["micro"]["native_truncation_rate"] <= args.max_native_truncation,
        "usable_positive_signal": current["micro"]["usable_positive_rate"] >= args.min_usable_positive,
        "no_dependency_errors": current["micro"]["dependency_error_rate"] == 0.0,
        "no_format_bad_reward_positive": current["micro"]["format_bad_reward_positive"] == 0,
        "no_truncated_reward_positive": current["micro"]["truncated_reward_positive"] == 0,
    }
    hard_pass = all(checks.values())
    launch_evidence_pass = hard_pass and args.evaluation_kind == "post_fix_reevaluation"
    quality_targets = {
        "raw_micro_format_at_least_0p85": current["micro"]["strict_format_rate"] >= 0.85,
        "raw_each_source_format_at_least_0p80": all(
            item["strict_format_rate"] >= 0.80 for item in current["sources"].values()
        ),
        "native_truncation_at_most_0p25": current["micro"]["native_truncation_rate"] <= 0.25,
    }
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if launch_evidence_pass else ("review" if hard_pass else "fail"),
        "decision": (
            "reuse_allowed"
            if launch_evidence_pass
            else ("provisional_reuse_allowed_pending_post_fix_reevaluation" if hard_pass else "reuse_blocked")
        ),
        "evaluation_kind": args.evaluation_kind,
        "validation_file": str(current_path.resolve()),
        "baseline_file": str(baseline_path.resolve()),
        "provenance_file": provenance_path,
        "provenance_sha256": provenance_sha256,
        "step": args.step,
        "baseline_step": args.baseline_step,
        "thresholds": {
            "min_completed_format": args.min_completed_format,
            "max_native_truncation": args.max_native_truncation,
            "min_usable_positive": args.min_usable_positive,
        },
        "checks": checks,
        "quality_targets_not_used_as_launch_checks": quality_targets,
        "stage1_strict_format_delta": delta,
        "current": current,
        "baseline": baseline,
        "evidence_boundary": (
            "Signal-loss fields quantify discarded beta=0 response throughput under fixed conditional "
            "correctness; final benchmark degradation requires a matched format/truncation intervention."
        ),
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
