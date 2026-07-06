#!/usr/bin/env python3
"""Plot raw Qwen3-1.7B KodCode Stage1 online validation curves."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask"
OUT_DIR = ROOT / "docs/joint_training/reports/figures"
DATA_DIR = ROOT / "docs/joint_training/reports/data"
OVERLEAF_IMAGE_DIR = ROOT / "docs/joint_training/courses/on-policy-wdl-overleaf/images"

RUNS = [
    {
        "label": r"Raw 1.7B Stage1 $\beta=0.0$",
        "short": "beta0",
        "path": METRICS_DIR
        / "ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA0-V1_1783242212.jsonl",
        "color": "#2563eb",
        "marker": "o",
    },
    {
        "label": r"Raw 1.7B Stage1 $\beta=0.1$",
        "short": "beta01",
        "path": METRICS_DIR
        / "ONPOLICY-SFT-Qwen3-1P7B-INSTRUCT-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783258331.jsonl",
        "color": "#dc2626",
        "marker": "s",
    },
]

METRICS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1", (0, 76)),
    ("MBPP+", "val-core/MBPP+/acc/pass@1", (0, 72)),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1", (0, 56)),
]


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append((int(record["step"]), record.get("data", {})))
    return rows


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for run in RUNS:
        for step, data in read_jsonl(run["path"]):
            for metric_name, key, _ylim in METRICS:
                value = data.get(key)
                if value is None:
                    continue
                rows.append(
                    {
                        "run": run["label"],
                        "run_short": run["short"],
                        "step": step,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    return rows


def write_rows(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "qwen3_1p7b_raw_kodcode_stage1_online_validation.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "run_short", "step", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=240)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(png_path, OVERLEAF_IMAGE_DIR / png_path.name)
    shutil.copy2(pdf_path, OVERLEAF_IMAGE_DIR / pdf_path.name)


def plot(rows: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (metric_name, _key, ylim) in zip(axes, METRICS, strict=True):
        for run in RUNS:
            series = [
                row
                for row in rows
                if row["run_short"] == run["short"] and row["metric"] == metric_name
            ]
            series.sort(key=lambda row: row["step"])
            ax.plot(
                [row["step"] for row in series],
                [100.0 * row["value"] for row in series],
                label=run["label"],
                color=run["color"],
                marker=run["marker"],
                linewidth=2.0,
                markersize=3.8,
            )
        ax.axvspan(70, 80, color="#f59e0b", alpha=0.16, label="format transition" if ax is axes[0] else None)
        ax.axvline(75, color="#92400e", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(metric_name)
        ax.set_xlim(0, 150)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Stage1 training step")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("online validation pass@1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Raw Qwen3-1.7B KodCode Stage1: Format-Learning Transition", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save_figure(fig, "qwen3_1p7b_raw_kodcode_stage1_format_jump")


def print_summary(rows: list[dict]) -> None:
    for run in RUNS:
        print(run["label"])
        for metric_name, _key, _ylim in METRICS:
            series = [r for r in rows if r["run_short"] == run["short"] and r["metric"] == metric_name]
            series.sort(key=lambda row: row["step"])
            best = max(series, key=lambda row: row["value"])
            lookup = {row["step"]: row["value"] for row in series}
            points = {step: round(100.0 * lookup[step], 2) for step in [70, 75, 80, 100, 150] if step in lookup}
            print(" ", metric_name, "points", points, "best", best["step"], round(100.0 * best["value"], 2))


def main() -> None:
    rows = collect_rows()
    write_rows(rows)
    plot(rows)
    print_summary(rows)
    print("Wrote raw 1.7B Stage1 figure/data.")


if __name__ == "__main__":
    main()
