#!/usr/bin/env python3
"""Summarize Qwen3-1.7B raw-base vs code format cold-start SFT fractions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
EVAL_ROOT = Path("/data-1/eval_outputs/code_task/qwen3_1p7b_coldstart_sft_fraction")
OUT_DATA = ROOT / "docs/joint_training/reports/data"
OUT_FIG = ROOT / "docs/joint_training/reports/figures"

RUNS = [
    ("raw_base", 0, "Raw base"),
    ("frac25", 25, "SFT 25%"),
    ("frac50", 50, "SFT 50%"),
    ("frac100", 100, "SFT 100%"),
]

BENCHMARKS = [
    ("humaneval", "HumanEval+", "plus"),
    ("mbpp", "MBPP+", "plus"),
    ("livecodebench", "LiveCodeBench", "pass@1"),
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_fail_stats(report: dict | None) -> tuple[int | None, int | None, float | None]:
    if report is None:
        return None, None, None
    extraction = report.get("extraction")
    if isinstance(extraction, list):
        counts = Counter(item.get("status") for item in extraction if isinstance(item, dict))
    else:
        counts = Counter(report.get("extraction_status_counts", {}))
    total = int(sum(counts.values()) or report.get("num_rows") or 0)
    fail = int(total - counts.get("ok", 0))
    rate = fail / total if total else None
    return fail, total, rate


def metric_values(benchmark: str, official: dict | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "pass1": None,
        "base_pass1": None,
        "plus_pass1": None,
    }
    if official is None:
        return out
    summary = official.get("summary", official)
    if benchmark in {"humaneval", "mbpp"}:
        out["base_pass1"] = summary.get("base_pass_rate")
        out["plus_pass1"] = summary.get("plus_pass_rate")
        out["pass1"] = out["plus_pass1"]
        return out
    metrics = summary.get("metrics")
    if isinstance(metrics, list) and metrics:
        out["pass1"] = metrics[0].get("pass@1")
    elif isinstance(metrics, dict):
        out["pass1"] = metrics.get("pass@1")
    else:
        out["pass1"] = summary.get("pass@1")
    return out


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for label, fraction_pct, display in RUNS:
        for benchmark, benchmark_display, metric_name in BENCHMARKS:
            case_dir = EVAL_ROOT / label / benchmark
            official = load_json(case_dir / "official_summary.json")
            conversion = load_json(case_dir / "conversion_report.json")
            fail, total, fail_rate = extract_fail_stats(conversion)
            metrics = metric_values(benchmark, official)
            rows.append(
                {
                    "run_label": label,
                    "run_display": display,
                    "fraction_pct": fraction_pct,
                    "benchmark": benchmark,
                    "benchmark_display": benchmark_display,
                    "primary_metric": metric_name,
                    "pass1": metrics["pass1"],
                    "base_pass1": metrics["base_pass1"],
                    "plus_pass1": metrics["plus_pass1"],
                    "extraction_fail": fail,
                    "extraction_total": total,
                    "extraction_fail_rate": fail_rate,
                    "official_complete": official is not None,
                    "conversion_complete": conversion is not None,
                }
            )
    return rows


def write_csv(rows: list[dict]) -> Path:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    path = OUT_DATA / "qwen3_1p7b_coldstart_sft_fraction_raw_comparison.csv"
    fields = [
        "run_label",
        "run_display",
        "fraction_pct",
        "benchmark",
        "benchmark_display",
        "primary_metric",
        "pass1",
        "base_pass1",
        "plus_pass1",
        "extraction_fail",
        "extraction_total",
        "extraction_fail_rate",
        "official_complete",
        "conversion_complete",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fmt_pct(value: float | None) -> str:
    return "pending" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(rows: list[dict]) -> Path:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    path = OUT_DATA / "qwen3_1p7b_coldstart_sft_fraction_raw_comparison.md"
    lines = [
        "# Qwen3-1.7B Code Format Cold-start SFT Fraction Comparison",
        "",
        "| run | benchmark | pass@1 | base pass@1 | plus pass@1 | extract fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fail = (
            "pending"
            if row["extraction_fail_rate"] is None
            else f"{row['extraction_fail']}/{row['extraction_total']} ({100.0 * row['extraction_fail_rate']:.2f}%)"
        )
        lines.append(
            "| {run} | {bench} | {pass1} | {base} | {plus} | {fail} |".format(
                run=row["run_display"],
                bench=row["benchmark_display"],
                pass1=fmt_pct(row["pass1"]),
                base=fmt_pct(row["base_pass1"]),
                plus=fmt_pct(row["plus_pass1"]),
                fail=fail,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def rows_for_benchmark(rows: list[dict], benchmark: str) -> list[dict]:
    out = [row for row in rows if row["benchmark"] == benchmark]
    out.sort(key=lambda row: row["fraction_pct"])
    return out


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    png = OUT_FIG / f"{stem}.png"
    pdf = OUT_FIG / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=240)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_metric(rows: list[dict], key: str, ylabel: str, stem: str) -> tuple[Path, Path]:
    colors = {
        "humaneval": "#2563eb",
        "mbpp": "#16a34a",
        "livecodebench": "#dc2626",
    }
    markers = {"humaneval": "o", "mbpp": "s", "livecodebench": "^"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for benchmark, display, _ in BENCHMARKS:
        points = rows_for_benchmark(rows, benchmark)
        xs = [row["fraction_pct"] for row in points if row[key] is not None]
        ys = [100.0 * row[key] for row in points if row[key] is not None]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=display,
            color=colors[benchmark],
            marker=markers[benchmark],
            linewidth=2.0,
            markersize=5.0,
        )
    ax.set_xlabel("Code format cold-start SFT data fraction (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 25, 50, 100], ["raw", "25", "50", "100"])
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, stem)


def main() -> None:
    rows = collect_rows()
    csv_path = write_csv(rows)
    md_path = write_markdown(rows)
    pass_png, pass_pdf = plot_metric(
        rows,
        "pass1",
        "official pass@1 (%)",
        "qwen3_1p7b_coldstart_sft_fraction_vs_raw_pass1",
    )
    fail_png, fail_pdf = plot_metric(
        rows,
        "extraction_fail_rate",
        "strict extraction fail (%)",
        "qwen3_1p7b_coldstart_sft_fraction_vs_raw_extract_fail",
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote figures: {pass_png}, {pass_pdf}, {fail_png}, {fail_pdf}")


if __name__ == "__main__":
    main()
