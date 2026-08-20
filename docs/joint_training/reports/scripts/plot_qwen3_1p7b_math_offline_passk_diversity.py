#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build the shared Math n=256 pass@k/diversity table and figures.

The script deliberately uses one input contract for CS0, A, C, D0, and GRPO:
each arm points to the merged ``eval_metrics.json`` produced by the common
8-shard offline-evaluation pipeline.  Missing arms are retained as pending in
the summary CSV, so later results only require adding/repointing an arm and
rerunning this same entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "docs/joint_training/reports/data"
FIGURE_DIR = ROOT / "docs/joint_training/reports/figures"
SUMMARY_CSV = DATA_DIR / "qwen3_1p7b_math_offline_passk_summary.csv"
BY_DATASET_CSV = DATA_DIR / "qwen3_1p7b_math_offline_passk_by_dataset.csv"
PASSK_FIGURE = FIGURE_DIR / "qwen3_1p7b_math_offline_passk_macro_curve"
DIVERSITY_FIGURE = FIGURE_DIR / "qwen3_1p7b_math_offline_diversity_tradeoff"
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
STAGED_RESULT_ROOT = Path(
    os.environ.get(
        "MATH_OFFLINE_RESULT_ROOT",
        "/data-1/model_weights/offline_eval/math_passk_results_20260815",
    )
)


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    metrics_path: Path
    color: str


DEFAULT_ARMS = (
    Arm(
        "cs0",
        "Cold Start (CS0)",
        Path("/data-1/l40s-slurm-mvp/checkpoints/jobs/53/MATH-CS0-PASSK-N256-R2-20260812/merged/eval_metrics.json"),
        "#64748b",
    ),
    Arm(
        "a_p60",
        "A P60",
        Path("/data-1/l40s-slurm-mvp/checkpoints/jobs/77/MATH-A-P60-PASSK-N256-R1-20260813/merged/eval_metrics.json"),
        "#2563eb",
    ),
    Arm(
        "c_p60",
        "C P60",
        Path("/data-1/l40s-slurm-mvp/checkpoints/jobs/75/MATH-C-P60-PASSK-N256-R1-20260813/merged/eval_metrics.json"),
        "#dc2626",
    ),
    Arm(
        "d0_p60",
        "D0 P60",
        Path("/data-1/l40s-slurm-mvp/checkpoints/jobs/76/MATH-D0-P60-PASSK-N256-R1-20260813/merged/eval_metrics.json"),
        "#16a34a",
    ),
    Arm(
        "grpo_cold_p200",
        "GRPO Cold Start LR=1e-6, P195 (~P200)",
        STAGED_RESULT_ROOT / "grpo_cold_p195" / "eval_metrics.json",
        "#9333ea",
    ),
    Arm(
        "grpo_stage1_p200",
        "GRPO Stage1 LR=1e-6, effective P200",
        STAGED_RESULT_ROOT / "grpo_stage1_effective_p200" / "eval_metrics.json",
        "#f59e0b",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="KEY=LABEL=PATH",
        help="override/add one arm; may be repeated",
    )
    return parser.parse_args()


def arms_from_args(values: list[str]) -> tuple[Arm, ...]:
    if not values:
        return DEFAULT_ARMS
    colors = ("#64748b", "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b")
    arms: list[Arm] = []
    for index, value in enumerate(values):
        try:
            key, label, path = value.split("=", 2)
        except ValueError as exc:
            raise SystemExit(f"invalid --arm {value!r}; expected KEY=LABEL=PATH") from exc
        arms.append(Arm(key, label, Path(path), colors[index % len(colors)]))
    return tuple(arms)


def load_arm(arm: Arm) -> dict[str, object] | None:
    if not arm.metrics_path.is_file():
        return None
    with arm.metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("n") != 256 or tuple(payload.get("k_values", ())) != K_VALUES:
        raise ValueError(f"{arm.key}: expected the frozen n=256 k grid: {arm.metrics_path}")
    return payload


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_rows(arms: tuple[Arm, ...]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    for arm in arms:
        payload = load_arm(arm)
        if payload is None:
            summary_rows.append(
                {
                    "arm": arm.key,
                    "label": arm.label,
                    "status": "pending",
                    "dataset_count": 0,
                    "pass_at_1_macro": "",
                    "pass_at_128_macro": "",
                    "pass_at_256_macro": "",
                    "maj_at_256_macro": "",
                    "truncation_rate_macro": "",
                    "mean_unique_prediction_count_macro": "",
                    "mean_unique_response_count": "",
                    "mean_distinct_response_rate": "",
                    "metrics_path": str(arm.metrics_path),
                }
            )
            continue
        metrics = payload["metrics"]
        assert isinstance(metrics, dict)
        datasets = sorted(metrics)
        for dataset in datasets:
            item = metrics[dataset]
            assert isinstance(item, dict)
            row: dict[str, object] = {
                "arm": arm.key,
                "label": arm.label,
                "dataset": dataset,
                "n_prompts": item["n_prompts"],
                "truncation_rate": item["truncation_rate"],
                "format_contract_success_rate": item["format_contract_success_rate"],
                "mean_unique_prediction_count": item["mean_unique_prediction_count"],
            }
            for k in K_VALUES:
                row[f"pass_at_{k}"] = item[f"pass@{k}"]
                row[f"maj_at_{k}"] = item[f"maj@{k}"]
            dataset_rows.append(row)
        diversity = payload["diversity"]
        assert isinstance(diversity, dict)
        summary: dict[str, object] = {
            "arm": arm.key,
            "label": arm.label,
            "status": "complete",
            "dataset_count": len(datasets),
            "pass_at_1_macro": mean([float(metrics[d]["pass@1"]) for d in datasets]),
            "pass_at_128_macro": mean([float(metrics[d]["pass@128"]) for d in datasets]),
            "pass_at_256_macro": mean([float(metrics[d]["pass@256"]) for d in datasets]),
            "maj_at_256_macro": mean([float(metrics[d]["maj@256"]) for d in datasets]),
            "truncation_rate_macro": mean([float(metrics[d]["truncation_rate"]) for d in datasets]),
            "mean_unique_prediction_count_macro": mean(
                [float(metrics[d]["mean_unique_prediction_count"]) for d in datasets]
            ),
            "mean_unique_response_count": diversity["mean_unique_response_count"],
            "mean_distinct_response_rate": diversity["mean_distinct_response_rate"],
            "metrics_path": str(arm.metrics_path),
        }
        for k in K_VALUES:
            summary[f"pass_at_{k}_macro"] = mean([float(metrics[d][f"pass@{k}"]) for d in datasets])
        summary_rows.append(summary)
    return summary_rows, dataset_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(summary_rows: list[dict[str, object]], arms: tuple[Arm, ...]) -> None:
    complete = {str(row["arm"]): row for row in summary_rows if row["status"] == "complete"}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for arm in arms:
        row = complete.get(arm.key)
        if row is None:
            continue
        values = [float(row[f"pass_at_{k}_macro"]) * 100 for k in K_VALUES]
        ax.plot(K_VALUES, values, marker="o", linewidth=2.2, label=arm.label, color=arm.color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.set_xlabel("k")
    ax.set_ylabel("Math-7 macro pass@k (%)")
    ax.set_title("Qwen3-1.7B Math offline pass@k (T=0.6, n=256)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(PASSK_FIGURE.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)

    if len(complete) < 2:
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for arm in arms:
        row = complete.get(arm.key)
        if row is None:
            continue
        x = float(row["pass_at_1_macro"]) * 100
        y = float(row["pass_at_256_macro"]) * 100
        size = 45 + float(row["mean_distinct_response_rate"]) * 90
        ax.scatter(x, y, s=size, color=arm.color)
        ax.annotate(arm.label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Math-7 macro pass@1 (%)")
    ax.set_ylabel("Math-7 macro pass@256 (%)")
    ax.set_title("Accuracy–coverage trade-off (marker size: response distinct rate)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(DIVERSITY_FIGURE.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> None:
    arms = arms_from_args(parse_args().arm)
    summary_rows, dataset_rows = build_rows(arms)
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(BY_DATASET_CSV, dataset_rows)
    plot(summary_rows, arms)
    completed = [row["arm"] for row in summary_rows if row["status"] == "complete"]
    print(f"completed arms: {', '.join(str(value) for value in completed)}")
    print(SUMMARY_CSV)
    print(BY_DATASET_CSV)
    print(PASSK_FIGURE.with_suffix(".png"))


if __name__ == "__main__":
    main()
