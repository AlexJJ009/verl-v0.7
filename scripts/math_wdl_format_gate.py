#!/usr/bin/env python3
"""Check the causal-P60 validation stream for a sustained format-contract collapse."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def row_format_success(row: dict) -> bool:
    if isinstance(row.get("format_contract_success"), bool):
        return row["format_contract_success"]
    text = row.get("output") or row.get("response_text") or ""
    thinks = THINK_RE.findall(text)
    answers = ANSWER_RE.findall(text)
    if text.count("<think>") != 1 or text.count("</think>") != 1 or len(thinks) != 1:
        return False
    if text.count("<answer>") != 1 or text.count("</answer>") != 1 or len(answers) != 1:
        return False
    if not thinks[0].strip() or "\\boxed{" not in answers[0]:
        return False
    if text.index("</think>") > text.index("<answer>"):
        return False
    return bool(row.get("has_eos", row.get("response_eos_present", True)))


def format_rate(path: Path) -> float:
    total = 0
    passed = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            passed += int(row_format_success(json.loads(line)))
    if total == 0:
        raise RuntimeError(f"empty validation artifact: {path}")
    return passed / total


def inspect_run(run_dir: Path, max_drop: float, required_consecutive: int) -> dict:
    model_dir = run_dir / "model2"
    step_files = sorted(
        ((int(path.stem), path) for path in model_dir.glob("*.jsonl") if path.stem.isdigit()),
        key=lambda item: item[0],
    )
    if not step_files or step_files[0][0] != 0:
        return {"run": run_dir.name, "status": "pending", "reason": "P0 validation missing"}
    rates = [(step, format_rate(path)) for step, path in step_files]
    baseline = rates[0][1]
    evaluated = [(step, rate, baseline - rate) for step, rate in rates[1:]]
    violations = [item for item in evaluated if item[2] > max_drop]
    latest = evaluated[-required_consecutive:]
    failed = len(latest) == required_consecutive and all(item[2] > max_drop for item in latest)
    return {
        "run": run_dir.name,
        "status": "fail" if failed else "pass",
        "baseline_rate": baseline,
        "latest": latest,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--run-prefix", action="append", required=True)
    parser.add_argument("--max-drop", type=float, default=0.05)
    parser.add_argument("--required-consecutive", type=int, default=2)
    args = parser.parse_args()

    results = []
    for prefix in args.run_prefix:
        candidates = sorted(path for path in args.validation_root.glob(f"{prefix}_*") if path.is_dir())
        if not candidates:
            results.append({"run_prefix": prefix, "status": "pending", "reason": "run directory missing"})
            continue
        results.append(inspect_run(candidates[-1], args.max_drop, args.required_consecutive))
    payload = {"status": "fail" if any(item["status"] == "fail" for item in results) else "pass", "runs": results}
    print(json.dumps(payload, sort_keys=True))
    return 2 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
