#!/usr/bin/env python3
"""Fail closed unless a math reward module enforces the structured format contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("causal_math_reward_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-path", type=Path, required=True)
    parser.add_argument("--function", default="compute_score_latex_verify")
    args = parser.parse_args()

    module = load_module(args.reward_path)
    score_fn = getattr(module, args.function)
    common = {
        "data_source": "contract-canary",
        "ground_truth": "42",
        "extra_info": {"valid_response_length": 32, "max_resp_len": 1024},
    }
    valid = score_fn(
        solution_str="<think>Compute the requested value.</think><answer>\\boxed{42}</answer>",
        **common,
    )
    missing_answer = score_fn(
        solution_str="<think>Compute the requested value.</think>\\boxed{42}",
        **common,
    )
    misordered = score_fn(
        solution_str="<answer>\\boxed{42}</answer><think>Compute the requested value.</think>",
        **common,
    )
    duplicate_answer = score_fn(
        solution_str=(
            "<think>Compute the requested value.</think>"
            "<answer>\\boxed{42}</answer><answer>\\boxed{42}</answer>"
        ),
        **common,
    )
    missing_think = score_fn(
        solution_str="<answer>\\boxed{42}</answer>",
        **common,
    )
    truncated = score_fn(
        solution_str="<think>Compute the requested value.</think><answer>\\boxed{42}</answer>",
        data_source=common["data_source"],
        ground_truth=common["ground_truth"],
        extra_info={"valid_response_length": 1024, "max_resp_len": 1024},
    )
    required_keys = {
        "score",
        "acc",
        "answer_correct",
        "format_contract_success",
        "answer_complete",
        "format_ordered",
        "boxed_extraction_success",
    }
    missing_keys = sorted(required_keys - set(valid))
    checks = {
        "valid_formatted_answer_is_positive": valid.get("score") == 1.0 and valid.get("acc") is True,
        "missing_answer_tag_is_negative": missing_answer.get("score") == -1.0
        and missing_answer.get("acc") is False,
        "misordered_tags_are_negative": misordered.get("score") == -1.0 and misordered.get("acc") is False,
        "duplicate_answer_tags_are_negative": duplicate_answer.get("score") == -1.0
        and duplicate_answer.get("acc") is False,
        "missing_think_tag_is_negative": missing_think.get("score") == -1.0
        and missing_think.get("acc") is False,
        "truncated_or_no_eos_is_negative": truncated.get("score") == -1.0
        and truncated.get("acc") is False
        and truncated.get("has_eos") is False,
        "telemetry_keys_present": not missing_keys,
    }
    payload = {
        "reward_path": str(args.reward_path.resolve()),
        "function": args.function,
        "checks": checks,
        "missing_keys": missing_keys,
        "status": "pass" if all(checks.values()) else "fail",
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
