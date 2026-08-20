#!/usr/bin/env python3
"""Plot all available Qwen3-1.7B code Stage1 pass@1 curves together."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "docs/joint_training/reports/data"
FIGURE_DIR = ROOT / "docs/joint_training/reports/figures"

RAW_DATA = DATA_DIR / "qwen3_1p7b_raw_kodcode_stage1_online_validation.csv"
COLDSTART_DATA = DATA_DIR / "qwen3_1p7b_coldstart_fraction_stage1_curves.csv"
COMBINED_DATA = DATA_DIR / "qwen3_1p7b_code_stage1_all_pass1_curves.csv"
FIGURE_STEM = "qwen3_1p7b_code_stage1_all_pass1_curves"

METRICS = ["HumanEval+", "MBPP+", "LiveCodeBench"]
COLORS = {
    "Raw": "#4b5563",
    "Cold-start 25%": "#2563eb",
    "Cold-start 50%": "#7c3aed",
    "Cold-start 100%": "#dc2626",
}
LINESTYLES = {"0.0": "-", "0.1": (0, (5, 2))}
MARKERS = {"0.0": "o", "0.1": "s"}


def read_raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with RAW_DATA.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            beta = "0.1" if row["run_short"] == "beta01" else "0.0"
            rows.append(
                {
                    "family": "Raw",
                    "beta": beta,
                    "step": int(row["step"]),
                    "metric": row["metric"],
                    "value": float(row["value"]),
                    "source_run": row["run"],
                }
            )
    return rows


def read_coldstart_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with COLDSTART_DATA.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["metric"].endswith(" pass@1"):
                continue
            rows.append(
                {
                    "family": f"Cold-start {row['fraction']}",
                    "beta": "0.1" if row["beta_key"] == "beta01" else "0.0",
                    "step": int(row["step"]),
                    "metric": row["metric"].removesuffix(" pass@1"),
                    "value": float(row["value"]),
                    "source_run": row["run_name"],
                }
            )
    return rows


def write_combined_rows(rows: list[dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["family", "beta", "step", "metric", "value", "source_run"]
    with COMBINED_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.8, 6.2), sharex=True)
    figure.patch.set_facecolor("#f8fafc")

    family_order = ["Raw", "Cold-start 25%", "Cold-start 50%", "Cold-start 100%"]
    for axis, metric in zip(axes, METRICS, strict=True):
        axis.set_facecolor("#ffffff")
        for family in family_order:
            for beta in ("0.0", "0.1"):
                points = [
                    row for row in rows if row["family"] == family and row["beta"] == beta and row["metric"] == metric
                ]
                points.sort(key=lambda row: int(row["step"]))
                axis.plot(
                    [int(row["step"]) for row in points],
                    [100.0 * float(row["value"]) for row in points],
                    color=COLORS[family],
                    linestyle=LINESTYLES[beta],
                    marker=MARKERS[beta],
                    markevery=3,
                    linewidth=2.0 if family != "Raw" else 1.7,
                    markersize=3.8,
                    alpha=0.96 if family != "Raw" else 0.78,
                    zorder=3 if family != "Raw" else 2,
                )

        metric_rows = [row for row in rows if row["metric"] == metric]
        values = [100.0 * float(row["value"]) for row in metric_rows]
        padding = max(2.0, (max(values) - min(values)) * 0.10)
        axis.set_ylim(max(0.0, min(values) - padding), min(100.0, max(values) + padding))
        axis.set_xlim(0, 150)
        axis.set_xticks(range(0, 151, 25))
        axis.set_title(metric, loc="left", fontweight="bold", color="#0f172a")
        axis.set_xlabel("Stage1 training step")
        axis.grid(axis="y", color="#dbe3ee", linewidth=0.8)
        axis.grid(axis="x", color="#eef2f7", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#94a3b8")
        axis.spines["bottom"].set_color("#94a3b8")
        axis.tick_params(colors="#475569")

    axes[0].set_ylabel("Online validation pass@1 (%)")

    family_handles = [Line2D([0], [0], color=COLORS[family], linewidth=3, label=family) for family in family_order]
    beta_handles = [
        Line2D(
            [0],
            [0],
            color="#111827",
            linestyle=LINESTYLES[beta],
            marker=MARKERS[beta],
            linewidth=2,
            markersize=5,
            label=rf"$\beta={beta}$",
        )
        for beta in ("0.0", "0.1")
    ]
    figure.legend(
        handles=[*family_handles, *beta_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.8,
        handlelength=2.6,
    )
    figure.suptitle(
        "Qwen3-1.7B Code Stage1 — Combined Training Curves",
        x=0.055,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    figure.text(
        0.055,
        0.90,
        "KodCode CTX8K · raw initialization vs. cold-start SFT fractions · solid β=0.0 / dashed β=0.1",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.99, 0.84), w_pad=2.0)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{FIGURE_STEM}.png"
    pdf_path = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    rows = [*read_raw_rows(), *read_coldstart_rows()]
    rows.sort(key=lambda row: (str(row["metric"]), str(row["family"]), str(row["beta"]), int(row["step"])))
    write_combined_rows(rows)
    png_path, pdf_path = plot(rows)
    print(f"Wrote {COMBINED_DATA}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
