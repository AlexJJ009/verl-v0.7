#!/usr/bin/env python3
"""Plot KodCode Instruct2507 CTX8K Stage1 vs P60 Stage2 online validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "docs/joint_training/reports/figures"
DATA_DIR = ROOT / "docs/joint_training/reports/data"

RUNS = [
    {
        "label": "Stage1 beta=0.0",
        "beta": "0.0",
        "phase": "Stage1",
        "path": ROOT
        / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/"
        / "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1_1782371396.jsonl",
        "offset": 0,
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "label": "Stage2 P60 beta=0.0",
        "beta": "0.0",
        "phase": "Stage2",
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0-V1_1782469996.jsonl",
        "offset": 60,
        "color": "#1f77b4",
        "linestyle": "--",
    },
    {
        "label": "Stage1 beta=0.1",
        "beta": "0.1",
        "phase": "Stage1",
        "path": ROOT
        / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/"
        / "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871.jsonl",
        "offset": 0,
        "color": "#d62728",
        "linestyle": "-",
    },
    {
        "label": "Stage2 P60 beta=0.1",
        "beta": "0.1",
        "phase": "Stage2",
        "path": ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA01-BETA01-V1_1782476261.jsonl",
        "offset": 60,
        "color": "#d62728",
        "linestyle": "--",
    },
]

DATASETS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1"),
    ("MBPP+", "val-core/MBPP+/acc/pass@1"),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1"),
]


def read_rows(run: dict) -> list[dict]:
    rows: list[dict] = []
    with run["path"].open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            data = record.get("data", {})
            if not any(key in data for _, key in DATASETS):
                continue
            step = int(record["step"])
            for dataset, key in DATASETS:
                if key in data:
                    rows.append(
                        {
                            "run": run["label"],
                            "beta": run["beta"],
                            "phase": run["phase"],
                            "raw_step": step,
                            "effective_step": step + run["offset"],
                            "dataset": dataset,
                            "pass_at_1": float(data[key]),
                        }
                    )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "beta",
                "phase",
                "raw_step",
                "effective_step",
                "dataset",
                "pass_at_1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], out_png: Path, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for ax, (dataset, _key) in zip(axes, DATASETS, strict=False):
        for run in RUNS:
            series = [
                r
                for r in rows
                if r["dataset"] == dataset and r["run"] == run["label"] and 0 <= r["effective_step"] <= 150
            ]
            series.sort(key=lambda r: r["effective_step"])
            if not series:
                continue
            ax.plot(
                [r["effective_step"] for r in series],
                [100 * r["pass_at_1"] for r in series],
                label=run["label"],
                color=run["color"],
                linestyle=run["linestyle"],
                marker="o" if run["phase"] == "Stage2" else None,
                markersize=3.5,
                linewidth=1.8,
            )
        ax.axvline(60, color="#555555", linewidth=1, alpha=0.6)
        ax.text(61, 2, "P60 handoff", rotation=90, va="bottom", color="#555555")
        ax.set_title(dataset)
        ax.set_xlabel("Effective training step")
        ax.set_ylim(0, 80)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Online validation pass@1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("KodCode Instruct2507 CTX8K: Stage1 Continuation vs P60 Stage2", y=0.98)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def print_key_table(rows: list[dict]) -> None:
    checkpoints = [
        ("Stage1 beta=0.0", 60),
        ("Stage1 beta=0.0", 100),
        ("Stage2 P60 beta=0.0", 60),
        ("Stage2 P60 beta=0.0", 80),
        ("Stage2 P60 beta=0.0", 90),
        ("Stage2 P60 beta=0.0", 100),
        ("Stage1 beta=0.1", 60),
        ("Stage1 beta=0.1", 100),
        ("Stage2 P60 beta=0.1", 60),
        ("Stage2 P60 beta=0.1", 80),
        ("Stage2 P60 beta=0.1", 90),
        ("Stage2 P60 beta=0.1", 100),
    ]
    by_key = {(r["run"], r["effective_step"], r["dataset"]): r["pass_at_1"] for r in rows}
    print("run,effective_step,HumanEval+,MBPP+,LiveCodeBench")
    for run, step in checkpoints:
        vals = [by_key.get((run, step, ds)) for ds, _ in DATASETS]
        if all(v is None for v in vals):
            continue
        pct = ["" if v is None else f"{100 * v:.2f}" for v in vals]
        print(f"{run},{step},{pct[0]},{pct[1]},{pct[2]}")


def main() -> None:
    rows: list[dict] = []
    for run in RUNS:
        if not run["path"].exists():
            raise FileNotFoundError(run["path"])
        rows.extend(read_rows(run))

    csv_path = DATA_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_online_validation.csv"
    png_path = OUT_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_online_validation.png"
    pdf_path = OUT_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_online_validation.pdf"
    write_csv(rows, csv_path)
    plot(rows, png_path, pdf_path)
    print_key_table(rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
