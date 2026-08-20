#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plot fresh KodCode Instruct2507 CTX8K P40 Stage2 lambda sweep curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
STAGE1_METRICS = (
    ROOT
    / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/"
    / "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871.jsonl"
)
STAGE2_METRICS_DIR = ROOT / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask"
OUT_DIR = ROOT / "docs/joint_training/reports/figures"
DATA_DIR = ROOT / "docs/joint_training/reports/data"

METRICS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1", (0, 82)),
    ("MBPP+", "val-core/MBPP+/acc/pass@1", (0, 74)),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1", (0, 62)),
]

DIAG_METRICS = [
    ("HumanEval+ extraction fail", "val-aux/HumanEval+/code_reward_extraction_fail/mean@1", (0, 1)),
    ("MBPP+ extraction fail", "val-aux/MBPP+/code_reward_extraction_fail/mean@1", (0, 1)),
    ("LiveCodeBench extraction fail", "val-aux/LiveCodeBench/code_reward_extraction_fail/mean@1", (0, 1)),
    ("rollout correct ratio", "wdl_sft/correct_ratio", (0, 0.75)),
    ("response length mean", "response_length/mean", (0, 4200)),
]

FRESH_RUNS = [
    {
        "label": "Stage2 fresh P40 lambda=0.6",
        "short": "fresh-lambda0.6",
        "lambda": "0.6",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA06-FRESH100-V1_1782785388.jsonl",
        "color": "#f97316",
        "marker": "D",
        "linestyle": "--",
    },
    {
        "label": "Stage2 fresh P40 lambda=0.7",
        "short": "fresh-lambda0.7",
        "lambda": "0.7",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA07-FRESH100-V1_1782797935.jsonl",
        "color": "#2563eb",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "label": "Stage2 fresh P40 lambda=0.8",
        "short": "fresh-lambda0.8",
        "lambda": "0.8",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-FRESH100-V1_1782807271.jsonl",
        "color": "#059669",
        "marker": "^",
        "linestyle": "--",
    },
]

OLD_RUNS = [
    {
        "label": "Stage2 old/resume P40 lambda=0.6",
        "short": "old-lambda0.6",
        "lambda": "0.6",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA06-V1_1782753372.jsonl",
        "color": "#f97316",
        "marker": "D",
        "linestyle": ":",
    },
    {
        "label": "Stage2 old/resume P40 lambda=0.7",
        "short": "old-lambda0.7",
        "lambda": "0.7",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA07-V1_1782735326.jsonl",
        "color": "#2563eb",
        "marker": "s",
        "linestyle": ":",
    },
    {
        "label": "Stage2 old/resume P40 lambda=0.8",
        "short": "old-lambda0.8",
        "lambda": "0.8",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-V1_1782741149.jsonl",
        "color": "#059669",
        "marker": "^",
        "linestyle": ":",
    },
    {
        "label": "Stage2 old/resume P40 lambda=0.9",
        "short": "old-lambda0.9",
        "lambda": "0.9",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA09-V1_1782770242.jsonl",
        "color": "#7c3aed",
        "marker": "v",
        "linestyle": ":",
    },
]

STAGE1_RUN = {
    "label": "Pure Stage1 beta=0.1",
    "short": "stage1-beta0.1",
    "lambda": "",
    "path": STAGE1_METRICS,
    "color": "#111827",
    "marker": None,
    "linestyle": "-",
}


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            data = record.get("data", {})
            raw_step = int(data.get("training/global_step") or record.get("step"))
            rows.append((raw_step, data))
    return rows


def collect_metric_rows(runs: list[dict], metric_defs: list[tuple[str, str, tuple[float, float]]]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        if not run["path"].exists():
            continue
        offset = 0 if run is STAGE1_RUN else 40
        for raw_step, data in read_jsonl(run["path"]):
            effective_step = raw_step + offset
            for metric_name, key, _ylim in metric_defs:
                value = data.get(key)
                if value is None:
                    continue
                rows.append(
                    {
                        "run": run["label"],
                        "run_short": run["short"],
                        "lambda": run["lambda"],
                        "raw_step": raw_step,
                        "effective_step": effective_step,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "run_short", "lambda", "raw_step", "effective_step", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_pass_curves(rows: list[dict], runs: list[dict], filename: str, title: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (dataset, _key, ylim) in zip(axes, METRICS, strict=True):
        for run in runs:
            series = [
                row
                for row in rows
                if row["run"] == run["label"] and row["metric"] == dataset and 35 <= row["effective_step"] <= 100
            ]
            series.sort(key=lambda row: row["effective_step"])
            if not series:
                continue
            ax.plot(
                [row["effective_step"] for row in series],
                [100.0 * row["value"] for row in series],
                label=run["label"],
                color=run["color"],
                marker=run["marker"],
                linestyle=run["linestyle"],
                linewidth=2.0,
                markersize=4.0,
            )
        ax.axvline(40, color="#6b7280", linewidth=1.0, alpha=0.7)
        ax.text(40.6, ylim[0] + 1.0, "P40 handoff", rotation=90, color="#4b5563")
        ax.set_title(dataset)
        ax.set_xlim(35, 100)
        ax.set_ylim(*ylim)
        ax.set_xlabel("effective training step")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("online validation pass@1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{filename}.png", bbox_inches="tight", dpi=240)
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(rows: list[dict], runs: list[dict], filename: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
    flat_axes = axes.flatten()
    for ax, (metric_name, _key, ylim) in zip(flat_axes, DIAG_METRICS, strict=False):
        for run in runs:
            series = [
                row
                for row in rows
                if row["run"] == run["label"] and row["metric"] == metric_name and 35 <= row["effective_step"] <= 100
            ]
            series.sort(key=lambda row: row["effective_step"])
            if not series:
                continue
            ax.plot(
                [row["effective_step"] for row in series],
                [row["value"] for row in series],
                label=run["label"],
                color=run["color"],
                marker=run["marker"],
                linestyle=run["linestyle"],
                linewidth=1.8,
                markersize=3.5,
            )
        ax.axvline(40, color="#6b7280", linewidth=1.0, alpha=0.7)
        ax.set_title(metric_name)
        ax.set_xlim(35, 100)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    flat_axes[-1].axis("off")
    flat_axes[3].set_xlabel("effective training step")
    flat_axes[4].set_xlabel("effective training step")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Fresh P40 Stage2 Diagnostics", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{filename}.png", bbox_inches="tight", dpi=240)
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def print_final_table(rows: list[dict], runs: list[dict]) -> None:
    print("Final effective-step 100 pass@1 (%):")
    for run in runs:
        vals = {}
        for dataset, _key, _ylim in METRICS:
            series = [
                row
                for row in rows
                if row["run"] == run["label"] and row["metric"] == dataset and row["effective_step"] <= 100
            ]
            if series:
                vals[dataset] = 100.0 * sorted(series, key=lambda row: row["effective_step"])[-1]["value"]
        if vals:
            print(run["label"], {key: round(value, 2) for key, value in vals.items()})


def main() -> None:
    fresh_pass_rows = collect_metric_rows([STAGE1_RUN, *FRESH_RUNS], METRICS)
    fresh_diag_rows = collect_metric_rows(FRESH_RUNS, DIAG_METRICS)
    compare_pass_rows = collect_metric_rows([STAGE1_RUN, *OLD_RUNS, *FRESH_RUNS], METRICS)

    write_rows(DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_fresh100_online_validation.csv", fresh_pass_rows)
    write_rows(DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_fresh100_diagnostics.csv", fresh_diag_rows)
    write_rows(
        DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_old_vs_fresh_online_validation.csv", compare_pass_rows
    )

    plot_pass_curves(
        fresh_pass_rows,
        [STAGE1_RUN, *FRESH_RUNS],
        "kodcode_instruct2507_ctx8k_p40_lambda_sweep_fresh100_online_validation",
        "KodCode Instruct2507 CTX8K Fresh P40 Stage2 Lambda Sweep vs Pure Stage1",
    )
    plot_pass_curves(
        compare_pass_rows,
        [STAGE1_RUN, *OLD_RUNS, *FRESH_RUNS],
        "kodcode_instruct2507_ctx8k_p40_lambda_sweep_old_vs_fresh_online_validation",
        "KodCode Instruct2507 CTX8K P40 Lambda Sweep: Old Resume vs Fresh",
    )
    plot_diagnostics(
        fresh_diag_rows,
        FRESH_RUNS,
        "kodcode_instruct2507_ctx8k_p40_lambda_sweep_fresh100_diagnostics",
    )

    print_final_table(fresh_pass_rows, [STAGE1_RUN, *FRESH_RUNS])
    print("Wrote fresh and comparison figures under", OUT_DIR)


if __name__ == "__main__":
    main()
