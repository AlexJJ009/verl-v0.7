#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plot Qwen3-1.7B cold-start KodCode Stage1 online validation curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask"
OUT_DIR = ROOT / "docs/joint_training/reports/figures"
DATA_DIR = ROOT / "docs/joint_training/reports/data"

RUNS = [
    {
        "label": r"Cold-start 1.7B Stage1 $\beta=0.0$",
        "short": "beta0",
        "path": METRICS_DIR / "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA0-V1_1783319854.jsonl",
        "color": "#2563eb",
        "marker": "o",
    },
    {
        "label": r"Cold-start 1.7B Stage1 $\beta=0.1$",
        "short": "beta01",
        "path": METRICS_DIR / "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783329189.jsonl",
        "color": "#dc2626",
        "marker": "s",
    },
]

VAL_METRICS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1", (45, 55)),
    ("MBPP+", "val-core/MBPP+/acc/pass@1", (48, 55)),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1", (10, 22)),
]

DIAG_METRICS = [
    ("Rollout correct ratio", "wdl_sft/correct_ratio", (30, 55), "percent"),
    ("HumanEval+ extraction fail", "val-aux/HumanEval+/code_reward_extraction_fail/mean@1", (0, 6), "percent"),
    ("Gradient norm", "actor/grad_norm", None, "raw"),
]


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            data = record.get("data", {})
            rows.append((step, data))
    return rows


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for run in RUNS:
        for step, data in read_jsonl(run["path"]):
            for metric_name, key, *_rest in [*VAL_METRICS, *DIAG_METRICS]:
                value = data.get(key)
                if value is None:
                    continue
                rows.append(
                    {
                        "run": run["label"],
                        "run_short": run["short"],
                        "step": step,
                        "metric": metric_name,
                        "key": key,
                        "value": float(value),
                    }
                )
    return rows


def write_rows(rows: list[dict]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "qwen3_1p7b_coldstart_kodcode_stage1_curves.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "run_short", "step", "metric", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=240)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def series(rows: list[dict], run_short: str, metric_name: str) -> list[dict]:
    out = [row for row in rows if row["run_short"] == run_short and row["metric"] == metric_name]
    out.sort(key=lambda row: row["step"])
    return out


def plot_validation(rows: list[dict]) -> tuple[Path, Path]:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (metric_name, _key, ylim) in zip(axes, VAL_METRICS, strict=True):
        for run in RUNS:
            points = series(rows, run["short"], metric_name)
            ax.plot(
                [row["step"] for row in points],
                [100.0 * row["value"] for row in points],
                label=run["label"],
                color=run["color"],
                marker=run["marker"],
                linewidth=2.0,
                markersize=3.4,
                markevery=5,
            )
        ax.set_title(metric_name)
        ax.set_xlim(0, 150)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Stage1 training step")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("online validation pass@1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Qwen3-1.7B Cold-start KodCode Stage1: Online Validation", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return save_figure(fig, "qwen3_1p7b_coldstart_kodcode_stage1_online_validation")


def plot_diagnostics(rows: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (metric_name, _key, ylim, scale) in zip(axes, DIAG_METRICS, strict=True):
        for run in RUNS:
            points = series(rows, run["short"], metric_name)
            values = [row["value"] for row in points]
            if scale == "percent":
                values = [100.0 * value for value in values]
            ax.plot(
                [row["step"] for row in points],
                values,
                label=run["label"],
                color=run["color"],
                marker=run["marker"],
                linewidth=2.0,
                markersize=3.0,
                markevery=5,
            )
        ax.set_title(metric_name)
        ax.set_xlim(0, 150)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Stage1 training step")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("percent / raw value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Qwen3-1.7B Cold-start KodCode Stage1: Training Diagnostics", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return save_figure(fig, "qwen3_1p7b_coldstart_kodcode_stage1_diagnostics")


def print_summary(rows: list[dict]) -> None:
    for run in RUNS:
        print(run["label"])
        for metric_name, _key, *_rest in VAL_METRICS:
            points = series(rows, run["short"], metric_name)
            best = max(points, key=lambda row: row["value"])
            final = points[-1]
            print(
                f"  {metric_name}: best step {best['step']} = {100.0 * best['value']:.2f}%; "
                f"final step {final['step']} = {100.0 * final['value']:.2f}%"
            )


def main() -> None:
    rows = collect_rows()
    csv_path = write_rows(rows)
    val_png, val_pdf = plot_validation(rows)
    diag_png, diag_pdf = plot_diagnostics(rows)
    print_summary(rows)
    print(f"Wrote data: {csv_path}")
    print(f"Wrote figures: {val_png}, {val_pdf}, {diag_png}, {diag_pdf}")


if __name__ == "__main__":
    main()
