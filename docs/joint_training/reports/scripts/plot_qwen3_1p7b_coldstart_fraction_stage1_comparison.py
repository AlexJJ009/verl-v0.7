#!/usr/bin/env python3
"""Compare Qwen3-1.7B KodCode Stage1 curves across cold-start SFT fractions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask"
DATA_DIR = ROOT / "docs/joint_training/reports/data"
FIG_DIR = ROOT / "docs/joint_training/reports/figures"

RUNS = [
    ("25%", 25, "beta0", "beta=0.0", "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA0-V1_1783416222"),
    ("25%", 25, "beta01", "beta=0.1", "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC25-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783425947"),
    ("50%", 50, "beta0", "beta=0.0", "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC50-CODE-KODCODE-CTX8K-S1-BETA0-V1_1783435522"),
    ("50%", 50, "beta01", "beta=0.1", "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-FRAC50-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783444888"),
    ("100%", 100, "beta0", "beta=0.0", "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA0-V1_1783319854"),
    ("100%", 100, "beta01", "beta=0.1", "ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-CODE-KODCODE-CTX8K-S1-BETA01-V1_1783329189"),
]

METRICS = {
    "HumanEval+ pass@1": "val-core/HumanEval+/acc/pass@1",
    "MBPP+ pass@1": "val-core/MBPP+/acc/pass@1",
    "LiveCodeBench pass@1": "val-core/LiveCodeBench/acc/pass@1",
    "HumanEval+ extract fail": "val-aux/HumanEval+/code_reward_extraction_fail/mean@1",
    "MBPP+ extract fail": "val-aux/MBPP+/code_reward_extraction_fail/mean@1",
    "LiveCodeBench extract fail": "val-aux/LiveCodeBench/code_reward_extraction_fail/mean@1",
    "Rollout correct ratio": "wdl_sft/correct_ratio",
    "Gradient norm": "actor/grad_norm",
    "Train score mean": "critic/score/mean",
    "Response length mean": "response_length/mean",
    "Time per step": "perf/time_per_step",
}

PASS_METRICS = ["HumanEval+ pass@1", "MBPP+ pass@1", "LiveCodeBench pass@1"]
FAIL_METRICS = ["HumanEval+ extract fail", "MBPP+ extract fail", "LiveCodeBench extract fail"]
DIAG_METRICS = ["Rollout correct ratio", "Gradient norm", "Train score mean", "Response length mean", "Time per step"]

COLORS = {25: "#2563eb", 50: "#16a34a", 100: "#dc2626"}
MARKERS = {"beta0": "o", "beta01": "s"}
LINESTYLES = {"beta0": "-", "beta01": "--"}


def read_records(run_name: str) -> list[tuple[int, dict]]:
    path = METRICS_DIR / f"{run_name}.jsonl"
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
    for fraction_label, fraction_pct, beta_key, beta_label, run_name in RUNS:
        for step, data in read_records(run_name):
            for metric_name, key in METRICS.items():
                value = data.get(key)
                if value is None:
                    continue
                rows.append(
                    {
                        "run_name": run_name,
                        "fraction": fraction_label,
                        "fraction_pct": fraction_pct,
                        "beta": beta_label,
                        "beta_key": beta_key,
                        "step": step,
                        "metric": metric_name,
                        "key": key,
                        "value": float(value),
                    }
                )
    return rows


def write_csv(rows: list[dict]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "qwen3_1p7b_coldstart_fraction_stage1_curves.csv"
    fields = ["run_name", "fraction", "fraction_pct", "beta", "beta_key", "step", "metric", "key", "value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def series(rows: list[dict], run_name: str, metric: str) -> list[dict]:
    out = [row for row in rows if row["run_name"] == run_name and row["metric"] == metric]
    out.sort(key=lambda row: row["step"])
    return out


def save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=240)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_grid(rows: list[dict], metrics: list[str], stem: str, ylabel: str, percent: bool) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.7), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        for fraction_label, fraction_pct, beta_key, beta_label, run_name in RUNS:
            points = series(rows, run_name, metric)
            values = [row["value"] for row in points]
            if percent:
                values = [100.0 * value for value in values]
            ax.plot(
                [row["step"] for row in points],
                values,
                label=f"{fraction_label} {beta_label}",
                color=COLORS[fraction_pct],
                linestyle=LINESTYLES[beta_key],
                marker=MARKERS[beta_key],
                markevery=5,
                linewidth=1.9,
                markersize=3.2,
            )
        ax.set_title(metric)
        ax.set_xlabel("Stage1 training step")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return save(fig, stem)


def summarize(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for fraction_label, fraction_pct, beta_key, beta_label, run_name in RUNS:
        for metric in PASS_METRICS:
            points = series(rows, run_name, metric)
            best = max(points, key=lambda row: row["value"])
            final = [row for row in points if row["step"] == 150][0]
            summary.append(
                {
                    "run_name": run_name,
                    "fraction": fraction_label,
                    "fraction_pct": fraction_pct,
                    "beta": beta_label,
                    "metric": metric,
                    "best_step": best["step"],
                    "best_value": best["value"],
                    "final_value": final["value"],
                    "delta_final_minus_best": final["value"] - best["value"],
                }
            )
    return summary


def write_summary(summary: list[dict]) -> Path:
    path = DATA_DIR / "qwen3_1p7b_coldstart_fraction_stage1_summary.csv"
    fields = ["run_name", "fraction", "fraction_pct", "beta", "metric", "best_step", "best_value", "final_value", "delta_final_minus_best"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return path


def print_summary(summary: list[dict]) -> None:
    for row in summary:
        print(
            f"{row['fraction']} {row['beta']} {row['metric']}: "
            f"best step {row['best_step']}={100*row['best_value']:.2f}%, "
            f"final={100*row['final_value']:.2f}%"
        )


def main() -> None:
    rows = collect_rows()
    csv_path = write_csv(rows)
    summary = summarize(rows)
    summary_path = write_summary(summary)
    pass_fig = plot_grid(
        rows,
        PASS_METRICS,
        "qwen3_1p7b_coldstart_fraction_stage1_pass1_curves",
        "online validation pass@1 (%)",
        True,
    )
    fail_fig = plot_grid(
        rows,
        FAIL_METRICS,
        "qwen3_1p7b_coldstart_fraction_stage1_extract_fail_curves",
        "strict extraction fail (%)",
        True,
    )
    diag_fig = plot_grid(
        rows,
        DIAG_METRICS,
        "qwen3_1p7b_coldstart_fraction_stage1_training_diagnostics",
        "percent / raw value",
        False,
    )
    print_summary(summary)
    print(f"Wrote data: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figures: {pass_fig}, {fail_fig}, {diag_fig}")


if __name__ == "__main__":
    main()
