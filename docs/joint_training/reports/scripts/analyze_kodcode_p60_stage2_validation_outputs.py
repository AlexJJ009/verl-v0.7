#!/usr/bin/env python3
"""Analyze raw validation outputs for KodCode Instruct2507 CTX8K P60 Stage2."""

from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VAL_ROOT = ROOT / "recipe/on_policy_wdl_sft/staged_v1/validation"
OUT_DIR = ROOT / "docs/joint_training/reports/data"

RUNS = {
    "stage2_beta0": VAL_ROOT / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0-V1_1782469996",
    "stage2_beta01": VAL_ROOT / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA01-BETA01-V1_1782476261",
}


def longest_repeated_line_run(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    best = 1
    current = 1
    prev = None
    for line in lines:
        if line == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
            prev = line
    return best


def max_ngram_repeat(text: str, n: int = 16) -> int:
    tokens = re.findall(r"\S+", text)
    if len(tokens) < n:
        return 1
    counts = collections.Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(counts.values(), default=1)


def has_basic_code_markers(text: str) -> bool:
    return "```python" in text or "def " in text or "class " in text


def strict_format(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text and "```python" in text


def non_ascii_ratio(text: str) -> float:
    return sum(1 for ch in text if ord(ch) > 127) / max(1, len(text))


def summarize_file(run: str, path: Path) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    n = len(rows)
    statuses = collections.Counter(r.get("code_reward_status", "") for r in rows)
    preds = collections.Counter(r.get("pred", "") for r in rows)
    data_sources = collections.Counter(r.get("data_source", "") for r in rows)
    lengths = [len(r.get("output", "")) for r in rows]
    word_lengths = [len(re.findall(r"\S+", r.get("output", ""))) for r in rows]
    rep16 = [max_ngram_repeat(r.get("output", ""), 16) for r in rows]
    rep_line = [longest_repeated_line_run(r.get("output", "")) for r in rows]
    code_marker = [has_basic_code_markers(r.get("output", "")) for r in rows]
    strict = [strict_format(r.get("output", "")) for r in rows]
    non_ascii = [non_ascii_ratio(r.get("output", "")) for r in rows]

    return {
        "run": run,
        "step": int(path.stem),
        "n": n,
        "acc": sum(float(r.get("acc", 0.0)) for r in rows) / n,
        "extraction_fail": statuses["extraction_fail"] / n,
        "compile_error": statuses["compile_error"] / n,
        "runtime_error": statuses["runtime_error"] / n,
        "timeout": statuses["timeout"] / n,
        "passed": statuses["passed"] / n,
        "no_code": preds["[NO_CODE]"] / n,
        "no_answer": preds["[NO_ANSWER]"] / n,
        "mean_chars": sum(lengths) / n,
        "p95_chars": sorted(lengths)[int(0.95 * (n - 1))],
        "max_chars": max(lengths),
        "mean_words": sum(word_lengths) / n,
        "long_2k_chars": sum(x >= 2000 for x in lengths) / n,
        "long_4k_chars": sum(x >= 4000 for x in lengths) / n,
        "max_rep16": max(rep16),
        "repeat16_ge3": sum(x >= 3 for x in rep16) / n,
        "repeat_line_ge3": sum(x >= 3 for x in rep_line) / n,
        "has_code_marker": sum(code_marker) / n,
        "strict_format": sum(strict) / n,
        "non_ascii_or_unicode": sum(x > 0 for x in non_ascii) / n,
        "top_statuses": json.dumps(statuses.most_common(5), ensure_ascii=True),
        "top_preds": json.dumps(preds.most_common(5), ensure_ascii=True),
        "data_sources": json.dumps(data_sources.most_common(), ensure_ascii=True),
    }


def dump_bad_examples(run: str, step: int, rows: list[dict], out_path: Path) -> None:
    selected = []
    for r in rows:
        text = r.get("output", "")
        badness = (
            int(r.get("code_reward_status") == "extraction_fail")
            + int(not has_basic_code_markers(text))
            + int(len(text) >= 2000)
            + int(max_ngram_repeat(text, 16) >= 3)
        )
        if badness > 0:
            selected.append(
                {
                    "run": run,
                    "step": step,
                    "data_source": r.get("data_source"),
                    "status": r.get("code_reward_status"),
                    "pred": r.get("pred"),
                    "acc": r.get("acc"),
                    "chars": len(text),
                    "max_rep16": max_ngram_repeat(text, 16),
                    "has_code_marker": has_basic_code_markers(text),
                    "strict_format": strict_format(text),
                    "input_head": r.get("input", "")[:500],
                    "output_head": text[:2000],
                }
            )
        if len(selected) >= 12:
            break
    with out_path.open("a") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    examples_path = OUT_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_bad_validation_examples.jsonl"
    examples_path.write_text("")

    for run, run_dir in RUNS.items():
        for path in sorted(run_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
            summary_rows.append(summarize_file(run, path))
            if int(path.stem) in {30, 35, 40}:
                rows = [json.loads(line) for line in path.open() if line.strip()]
                dump_bad_examples(run, int(path.stem), rows, examples_path)

    summary_path = OUT_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_validation_output_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path)
    print(examples_path)
    for row in summary_rows:
        if row["step"] in {0, 20, 30, 35, 40}:
            print(
                row["run"],
                row["step"],
                "acc",
                f"{100 * row['acc']:.2f}",
                "extract_fail",
                f"{100 * row['extraction_fail']:.2f}",
                "no_code",
                f"{100 * row['no_code']:.2f}",
                "mean_chars",
                f"{row['mean_chars']:.0f}",
                "long2k",
                f"{100 * row['long_2k_chars']:.2f}",
                "strict",
                f"{100 * row['strict_format']:.2f}",
                "rep16>=3",
                f"{100 * row['repeat16_ge3']:.2f}",
            )


if __name__ == "__main__":
    main()
