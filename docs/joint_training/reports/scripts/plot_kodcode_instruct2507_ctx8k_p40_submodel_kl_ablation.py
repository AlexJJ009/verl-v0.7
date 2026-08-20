#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plot KodCode Instruct2507 CTX8K P40 submodel-KL ablation curves."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
STAGE1_METRICS = (
    ROOT
    / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask/"
    / "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871.jsonl"
)
STAGE1_VALIDATION_DIR = (
    ROOT
    / "recipe/on_policy_wdl_sft/code_task/validation/"
    / "ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA01-V1_1782398871"
)
STAGE2_METRICS_DIR = ROOT / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicyWDLSFT-CodeTask"
STAGE2_VALIDATION_DIR = ROOT / "recipe/on_policy_wdl_sft/staged_v1/validation"
OUT_DIR = ROOT / "docs/joint_training/reports/figures"
DATA_DIR = ROOT / "docs/joint_training/reports/data"
OVERLEAF_IMAGE_DIR = ROOT / "docs/joint_training/courses/on-policy-wdl-overleaf/images"

PASS_METRICS = [
    ("HumanEval+", "val-core/HumanEval+/acc/pass@1", (0, 82)),
    ("MBPP+", "val-core/MBPP+/acc/pass@1", (0, 74)),
    ("LiveCodeBench", "val-core/LiveCodeBench/acc/pass@1", (0, 62)),
]

DIAG_METRICS = [
    ("response length mean", "response_length/mean", (0, 2300)),
    ("response clip ratio", "response_length/clip_ratio", (0, 0.22)),
    ("rollout correct ratio", "wdl_sft/correct_ratio", (0, 0.75)),
    ("actor grad norm", "actor/grad_norm", (0, 230)),
    ("model1 KL", "actor/submodel_kl/model1_loss", (0, 130)),
    ("model2 KL", "actor/submodel_kl/model2_loss", (0, 160)),
]

RUNS = [
    {
        "label": "Pure Stage1 beta=0.1",
        "short": "stage1",
        "path": STAGE1_METRICS,
        "validation_dir": STAGE1_VALIDATION_DIR,
        "offset": 0,
        "color": "#111827",
        "marker": None,
        "linestyle": "-",
    },
    {
        "label": "Stage2 no KL lambda=0.8",
        "short": "no-kl-l0.8",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-FRESH100-V1_1782807271.jsonl",
        "validation_dir": STAGE2_VALIDATION_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-LAMBDA08-FRESH100-V1_1782807271",
        "offset": 40,
        "color": "#6b7280",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "label": "Stage2 M1-only KL lambda=0.8",
        "short": "m1-kl-l0.8",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA08-V1_1782896807.jsonl",
        "validation_dir": STAGE2_VALIDATION_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M1-LAMBDA08-V1_1782896807",
        "offset": 40,
        "color": "#dc2626",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "label": "Stage2 M2-only KL lambda=0.8",
        "short": "m2-kl-l0.8",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA08-V1_1782906337.jsonl",
        "validation_dir": STAGE2_VALIDATION_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-M2-LAMBDA08-V1_1782906337",
        "offset": 40,
        "color": "#059669",
        "marker": "^",
        "linestyle": "--",
    },
    {
        "label": "Stage2 both KL lambda=0.8",
        "short": "both-kl-l0.8",
        "path": STAGE2_METRICS_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA08-V1_1782924238.jsonl",
        "validation_dir": STAGE2_VALIDATION_DIR
        / "CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P40-BETA01-BETA01-SUBKL-BOTH-LAMBDA08-V1_1782924238",
        "offset": 40,
        "color": "#7c3aed",
        "marker": "D",
        "linestyle": "--",
    },
]


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


def collect_metric_rows(metric_defs: list[tuple[str, str, tuple[float, float]]]) -> list[dict]:
    rows: list[dict] = []
    for run in RUNS:
        if not run["path"].exists():
            continue
        for raw_step, data in read_jsonl(run["path"]):
            effective_step = raw_step + run["offset"]
            for metric_name, key, _ylim in metric_defs:
                value = data.get(key)
                if value is None:
                    continue
                rows.append(
                    {
                        "run": run["label"],
                        "run_short": run["short"],
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
            fieldnames=["run", "run_short", "raw_step", "effective_step", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_pass_curves(rows: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (dataset, _key, ylim) in zip(axes, PASS_METRICS, strict=True):
        for run in RUNS:
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
    fig.suptitle("KodCode Instruct2507 CTX8K P40 Submodel-KL Ablation", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save_figure(fig, "kodcode_instruct2507_ctx8k_p40_submodel_kl_online_validation")


def plot_diagnostics(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
    for ax, (metric_name, _key, ylim) in zip(axes.flatten(), DIAG_METRICS, strict=True):
        for run in RUNS:
            if run["short"] == "stage1" and metric_name.startswith("model"):
                continue
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
    axes[1][0].set_xlabel("effective training step")
    axes[1][1].set_xlabel("effective training step")
    axes[1][2].set_xlabel("effective training step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("P40 Submodel-KL Training Diagnostics", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    save_figure(fig, "kodcode_instruct2507_ctx8k_p40_submodel_kl_diagnostics")


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


def response_text(record: dict) -> str:
    return str(record.get("response_text") or record.get("output") or record.get("response") or "")


def format_stats_for_validation(run: dict, raw_step: int) -> dict:
    path = run["validation_dir"] / f"{raw_step}.jsonl"
    texts: list[str] = []
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            texts.append(response_text(json.loads(line)))
    if not texts:
        return {}
    long_count = sum(len(text) > 6000 for text in texts)
    repeat_like = sum(is_repeat_like(text) for text in texts)
    multi_answer = sum(text.count("<answer>") > 1 for text in texts)
    missing_fence = sum("```" not in text for text in texts)
    non_ascii = sum(any(ord(ch) > 127 for ch in text) for text in texts)
    return {
        "run": run["label"],
        "run_short": run["short"],
        "raw_step": raw_step,
        "effective_step": raw_step + run["offset"],
        "n": len(texts),
        "mean_chars": round(sum(len(text) for text in texts) / len(texts), 1),
        "long_gt_6000": long_count,
        "repeat_like": repeat_like,
        "multi_answer": multi_answer,
        "missing_fence": missing_fence,
        "non_ascii": non_ascii,
    }


def is_repeat_like(text: str) -> bool:
    if len(text) > 3500 and len(set(text.split())) < 160:
        return True
    answer_count = text.count("<answer>")
    if answer_count > 1:
        return True
    normalized = re.sub(r"\s+", " ", text)
    chunks = [normalized[i : i + 120] for i in range(0, max(0, len(normalized) - 120), 120)]
    return len(chunks) >= 8 and len(set(chunks)) <= len(chunks) * 0.65


def collect_validation_stats() -> list[dict]:
    rows: list[dict] = []
    for run in RUNS:
        raw_step = 150 if run["short"] == "stage1" else 60
        stats = format_stats_for_validation(run, raw_step)
        if stats:
            rows.append(stats)
    return rows


def write_validation_stats(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "run_short",
        "raw_step",
        "effective_step",
        "n",
        "mean_chars",
        "long_gt_6000",
        "repeat_like",
        "multi_answer",
        "missing_fence",
        "non_ascii",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_final_table(rows: list[dict]) -> None:
    print("Final pass@1 (%):")
    for run in RUNS:
        vals = {}
        for dataset, _key, _ylim in PASS_METRICS:
            series = [
                row
                for row in rows
                if row["run"] == run["label"] and row["metric"] == dataset and row["effective_step"] <= 150
            ]
            if series:
                vals[dataset] = 100.0 * sorted(series, key=lambda row: row["effective_step"])[-1]["value"]
        print(run["label"], {key: round(value, 2) for key, value in vals.items()})


def main() -> None:
    pass_rows = collect_metric_rows(PASS_METRICS)
    diag_rows = collect_metric_rows(DIAG_METRICS)
    validation_rows = collect_validation_stats()
    write_rows(DATA_DIR / "kodcode_instruct2507_ctx8k_p40_submodel_kl_online_validation.csv", pass_rows)
    write_rows(DATA_DIR / "kodcode_instruct2507_ctx8k_p40_submodel_kl_diagnostics.csv", diag_rows)
    write_validation_stats(
        DATA_DIR / "kodcode_instruct2507_ctx8k_p40_submodel_kl_validation_format_stats.csv", validation_rows
    )
    plot_pass_curves(pass_rows)
    plot_diagnostics(diag_rows)
    print_final_table(pass_rows)
    print("Validation format stats:")
    for row in validation_rows:
        print(row)
    print("Wrote figures under", OUT_DIR, "and copied them to", OVERLEAF_IMAGE_DIR)


if __name__ == "__main__":
    main()
