#!/usr/bin/env python3
"""Plot KodCode Instruct2507 CTX8K P40 Stage2 lambda sweep curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "docs/joint_training/reports/figures"
DATA_DIR = ROOT / "docs/joint_training/reports/data"

METRICS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1", (0, 82)),
    ("MBPP+", "val-core/MBPP+/acc/pass@1", (30, 74)),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1", (0, 62)),
]

RUNS = [
    {
        "label": "Pure Stage1 beta=0.1",
        "kind": "stage1",
        "lambda": "",
        "offset": 0,
        "path": ROOT
        / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/"
        / "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871.jsonl",
        "color": "#111827",
        "linestyle": "-",
        "marker": None,
    },
    {
        "label": "Stage2 P40 lambda=0.5",
        "kind": "stage2",
        "lambda": "0.5",
        "offset": 40,
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-V1_1782562814.jsonl",
        "color": "#ef4444",
        "linestyle": "--",
        "marker": "o",
    },
    {
        "label": "Stage2 P40 lambda=0.6",
        "kind": "stage2",
        "lambda": "0.6",
        "offset": 40,
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA06-V1_1782753372.jsonl",
        "color": "#f97316",
        "linestyle": "--",
        "marker": "D",
    },
    {
        "label": "Stage2 P40 lambda=0.7",
        "kind": "stage2",
        "lambda": "0.7",
        "offset": 40,
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA07-V1_1782735326.jsonl",
        "color": "#2563eb",
        "linestyle": "--",
        "marker": "s",
    },
    {
        "label": "Stage2 P40 lambda=0.8",
        "kind": "stage2",
        "lambda": "0.8",
        "offset": 40,
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-V1_1782741149.jsonl",
        "color": "#059669",
        "linestyle": "--",
        "marker": "^",
    },
    {
        "label": "Stage2 P40 lambda=0.9",
        "kind": "stage2",
        "lambda": "0.9",
        "offset": 40,
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA09-V1_1782770242.jsonl",
        "color": "#7c3aed",
        "linestyle": "--",
        "marker": "v",
    },
]


def load_points(run: dict) -> list[dict]:
    rows: list[dict] = []
    with run["path"].open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            data = record.get("data", {})
            raw_step = int(data.get("training/global_step") or record["step"])
            effective_step = raw_step + int(run["offset"])
            for dataset, key, _ylim in METRICS:
                value = data.get(key)
                if value is None:
                    continue
                rows.append(
                    {
                        "run": run["label"],
                        "kind": run["kind"],
                        "lambda": run["lambda"],
                        "raw_step": raw_step,
                        "effective_step": effective_step,
                        "dataset": dataset,
                        "pass_at_1": float(value),
                    }
                )
    return rows


def write_csv(rows: list[dict], summary: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_online_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "kind",
                "lambda",
                "raw_step",
                "effective_step",
                "dataset",
                "pass_at_1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "dataset",
                "final_effective_step",
                "final_pass_at_1",
                "best_effective_step",
                "best_pass_at_1",
                "delta_final_vs_lambda05_pp",
                "delta_final_vs_stage1_step80_pp",
                "delta_best_vs_lambda05_best_pp",
                "delta_best_vs_stage1_40_80_best_pp",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def summarize(rows: list[dict]) -> list[dict]:
    by_run_dataset: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_run_dataset.setdefault((row["run"], row["dataset"]), []).append(row)
    for series in by_run_dataset.values():
        series.sort(key=lambda item: item["effective_step"])

    summary: list[dict] = []
    stage1_label = "Pure Stage1 beta=0.1"
    lambda05_label = "Stage2 P40 lambda=0.5"
    for run in RUNS:
        label = run["label"]
        for dataset, _key, _ylim in METRICS:
            series = by_run_dataset.get((label, dataset), [])
            if not series:
                continue
            final = series[-1]
            best = max(series, key=lambda item: item["pass_at_1"])

            lambda05_series = by_run_dataset.get((lambda05_label, dataset), [])
            stage1_series = by_run_dataset.get((stage1_label, dataset), [])
            lambda05_final = lambda05_series[-1]["pass_at_1"] if lambda05_series else None
            lambda05_best = (
                max(lambda05_series, key=lambda item: item["pass_at_1"])["pass_at_1"] if lambda05_series else None
            )
            stage1_step80 = next(
                (item["pass_at_1"] for item in stage1_series if item["effective_step"] == 80),
                None,
            )
            stage1_40_80 = [item for item in stage1_series if 40 <= item["effective_step"] <= 80]
            stage1_40_80_best = (
                max(stage1_40_80, key=lambda item: item["pass_at_1"])["pass_at_1"] if stage1_40_80 else None
            )

            def delta(value: float, baseline: float | None) -> float | None:
                if baseline is None:
                    return None
                return 100.0 * (value - baseline)

            summary.append(
                {
                    "run": label,
                    "dataset": dataset,
                    "final_effective_step": final["effective_step"],
                    "final_pass_at_1": 100.0 * final["pass_at_1"],
                    "best_effective_step": best["effective_step"],
                    "best_pass_at_1": 100.0 * best["pass_at_1"],
                    "delta_final_vs_lambda05_pp": delta(final["pass_at_1"], lambda05_final),
                    "delta_final_vs_stage1_step80_pp": delta(final["pass_at_1"], stage1_step80),
                    "delta_best_vs_lambda05_best_pp": delta(best["pass_at_1"], lambda05_best),
                    "delta_best_vs_stage1_40_80_best_pp": delta(best["pass_at_1"], stage1_40_80_best),
                }
            )
    return summary


def plot(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (dataset, _key, ylim) in zip(axes, METRICS, strict=True):
        for run in RUNS:
            series = [
                row
                for row in rows
                if row["run"] == run["label"] and row["dataset"] == dataset and 35 <= row["effective_step"] <= 100
            ]
            series.sort(key=lambda item: item["effective_step"])
            if not series:
                continue
            ax.plot(
                [item["effective_step"] for item in series],
                [100.0 * item["pass_at_1"] for item in series],
                label=run["label"],
                color=run["color"],
                linestyle=run["linestyle"],
                marker=run["marker"],
                linewidth=2.0,
                markersize=4.0,
            )
        ax.axvline(40, color="#6b7280", linewidth=1.0, alpha=0.7)
        ax.text(40.7, ylim[0] + 0.8, "P40 handoff", rotation=90, color="#4b5563")
        ax.set_title(dataset)
        ax.set_xlabel("effective training step")
        ax.set_ylim(*ylim)
        ax.set_xlim(35, 100)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("online validation pass@1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "KodCode Instruct2507 CTX8K P40 Stage2 Lambda Sweep vs Pure Stage1",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"kodcode_instruct2507_ctx8k_p40_lambda_sweep_online_validation.{suffix}",
            bbox_inches="tight",
            dpi=240,
        )
    plt.close(fig)


def main() -> None:
    rows: list[dict] = []
    for run in RUNS:
        if not run["path"].exists():
            raise FileNotFoundError(run["path"])
        rows.extend(load_points(run))
    summary = summarize(rows)
    write_csv(rows, summary)
    plot(rows)
    print("Wrote:")
    print(OUT_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_online_validation.png")
    print(OUT_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_online_validation.pdf")
    print(DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_online_validation.csv")
    print(DATA_DIR / "kodcode_instruct2507_ctx8k_p40_lambda_sweep_summary.csv")


if __name__ == "__main__":
    main()
