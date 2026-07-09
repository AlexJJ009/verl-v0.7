#!/usr/bin/env python3
"""Plot raw Stage2 diagnostics for KodCode Instruct2507 CTX8K P60 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "docs/joint_training/reports/figures"

RUNS = [
    (
        "Stage2 beta=0.0",
        ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0-V1_1782469996.jsonl",
        "#1f77b4",
    ),
    (
        "Stage2 beta=0.1",
        ROOT
        / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask/"
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA01-BETA01-V1_1782476261.jsonl",
        "#d62728",
    ),
]

VAL_KEYS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1"),
    ("MBPP+", "val-core/MBPP+/acc/pass@1"),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1"),
]


def read_run(path: Path) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            data = record.get("data", {})
            if "wdl_sft/correct_ratio" in data:
                train_rows.append({"step": step, **data})
            if any(key in data for _, key in VAL_KEYS):
                val_rows.append({"step": step, **data})
    return train_rows, val_rows


def plot() -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 160})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
    axes = axes.ravel()

    for label, path, color in RUNS:
        train, val = read_run(path)
        steps = [r["step"] for r in train]
        axes[0].plot(
            steps,
            [100 * r["wdl_sft/correct_ratio"] for r in train],
            label=label,
            color=color,
            linewidth=1.8,
        )
        axes[1].plot(
            steps,
            [r["actor/wdl_sft_loss_positive"] for r in train],
            label=f"{label} positive",
            color=color,
            linewidth=1.7,
        )
        axes[1].plot(
            steps,
            [r["actor/wdl_sft_loss_total"] for r in train],
            label=f"{label} total",
            color=color,
            linestyle="--",
            linewidth=1.4,
        )
        axes[2].plot(
            steps,
            [r["actor/grad_norm"] for r in train],
            label=label,
            color=color,
            linewidth=1.8,
        )
        axes[3].plot(
            steps,
            [r["response_length/mean"] for r in train],
            label=f"{label} mean",
            color=color,
            linewidth=1.8,
        )
        axes[3].plot(
            steps,
            [4096 * r["response_length/clip_ratio"] for r in train],
            label=f"{label} clip ratio x4096",
            color=color,
            linestyle="--",
            linewidth=1.4,
        )
        axes[4].plot(
            steps,
            [r["critic/score/mean"] for r in train],
            label=label,
            color=color,
            linewidth=1.8,
        )
        for dataset, key in VAL_KEYS:
            marker = {"HumanEval+": "o", "MBPP+": "s", "LiveCodeBench": "^"}[dataset]
            axes[5].plot(
                [r["step"] for r in val],
                [100 * r[key] for r in val],
                label=f"{label} {dataset}",
                color=color,
                marker=marker,
                linestyle={"HumanEval+": "-", "MBPP+": "--", "LiveCodeBench": ":"}[dataset],
                linewidth=1.4,
                markersize=3.5,
            )

    titles = [
        "Rollout correct ratio",
        "WDL raw loss",
        "Actor grad norm",
        "Response length",
        "Critic score mean",
        "Online validation pass@1",
    ]
    ylabels = ["correct (%)", "loss", "norm", "tokens", "score", "pass@1 (%)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Stage2 raw step")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylim(0, 75)
    axes[4].set_ylim(-1, 0.4)
    axes[5].set_ylim(0, 80)
    for ax in axes:
        ax.axvline(30, color="#444444", alpha=0.35, linewidth=1)
        ax.axvline(35, color="#444444", alpha=0.35, linewidth=1)
    for ax in axes[:5]:
        ax.legend(frameon=False, loc="best")
    axes[5].legend(frameon=False, loc="lower left", ncol=2)
    fig.suptitle("KodCode Instruct2507 CTX8K P60 Stage2 Raw Diagnostics", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_raw_diagnostics.png")
    fig.savefig(OUT_DIR / "kodcode_instruct2507_ctx8k_p60_stage2_raw_diagnostics.pdf")


if __name__ == "__main__":
    plot()
