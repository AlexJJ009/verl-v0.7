#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plot Model1/Model2 online-validation dynamics for the 1.7B Math/Code C and D0 arms."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "docs/joint_training/reports/data"
FIGURE_DIR = ROOT / "docs/joint_training/reports/figures"
EXPECTED_STEPS = list(range(0, 61, 5))

TASKS: dict[str, dict[str, Any]] = {
    "math": {
        "title": "Qwen3-1.7B Math C/D0 — Model1 and Model2 dynamics",
        "metric_label": "Math-7 macro mean@3 (%)",
        "sources": {
            "aime25",
            "HuggingFaceH4/MATH-500",
            "zwhe99/amc23",
            "deepmind/aqua_rat",
            "openai/gsm8k",
            "mwpt5/MAWPS",
            "ChilleD/SVAMP",
        },
        "validation_root": Path("/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/logs/validation"),
        "runs": {
            "C": "MATH-WDL-CAUSAL-P60-ARM-C-QWEN3-1P7B_1785247036",
            "D0": "MATH-WDL-CAUSAL-P60-ARM-D0-QWEN3-1P7B_1785213811",
        },
    },
    "code": {
        "title": "Qwen3-1.7B Code C/D0 — Model1 and Model2 dynamics",
        "metric_label": "Code-3 macro mean@3 (%)",
        "sources": {"HumanEval+", "MBPP+", "LiveCodeBench"},
        "validation_root": Path("/data-2/model_weights/code_task/qwen3_1p7b_wdl_acd0_p60/logs/validation"),
        "runs": {
            "C": "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B_1785746593",
            "D0": "CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B_1785430935",
        },
    },
}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def aggregate_file(path: Path, expected_sources: set[str]) -> dict[str, float | int]:
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    response_count = 0
    native_truncation_count = 0
    format_success_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row["data_source"])
            by_source[source]["accuracy"].append(float(bool(row.get("acc", False))))
            by_source[source]["native_truncation"].append(float(row.get("response_finish_reason") == "length"))
            by_source[source]["format_success"].append(float(bool(row.get("format_contract_success", False))))
            native_truncation_count += int(row.get("response_finish_reason") == "length")
            format_success_count += int(bool(row.get("format_contract_success", False)))
            response_count += 1

    if set(by_source) != expected_sources:
        raise ValueError(f"{path}: source mismatch: expected={sorted(expected_sources)} observed={sorted(by_source)}")

    source_accuracy = [_mean(values["accuracy"]) for values in by_source.values()]
    return {
        "response_count": response_count,
        "source_count": len(by_source),
        "macro_mean_at_3": _mean(source_accuracy),
        # Accuracy is macro-averaged across benchmark sources.  Truncation and
        # format compliance are response-level telemetry and therefore use the
        # pooled response denominator, matching the experiment reports.
        "native_truncation": native_truncation_count / response_count,
        "format_success": format_success_count / response_count,
    }


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, task_spec in TASKS.items():
        for arm, run_name in task_spec["runs"].items():
            for model_view in ("model1", "model2"):
                observed_steps: list[int] = []
                for step in EXPECTED_STEPS:
                    path = task_spec["validation_root"] / run_name / model_view / f"{step}.jsonl"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    values = aggregate_file(path, task_spec["sources"])
                    rows.append(
                        {
                            "task": task,
                            "arm": arm,
                            "model_view": model_view,
                            "trainable": arm == "C" or model_view == "model2",
                            "run_name": run_name,
                            "step": step,
                            **values,
                        }
                    )
                    observed_steps.append(step)
                if observed_steps != EXPECTED_STEPS:
                    raise AssertionError((task, arm, model_view, observed_steps))
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "qwen3_1p7b_acd0_submodel_online_validation.csv"
    fields = [
        "task",
        "arm",
        "model_view",
        "trainable",
        "run_name",
        "step",
        "response_count",
        "source_count",
        "macro_mean_at_3",
        "native_truncation",
        "format_success",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_task(rows: list[dict[str, Any]], task: str) -> tuple[Path, Path]:
    task_spec = TASKS[task]
    task_rows = [row for row in rows if row["task"] == task]
    colors = {"model1": "#7c3aed", "model2": "#059669"}
    labels = {"model1": "Model 1", "model2": "Model 2"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), sharex=True, sharey=True)
    figure.patch.set_facecolor("#f8fafc")

    for axis, arm in zip(axes, ("C", "D0"), strict=True):
        axis.set_facecolor("#ffffff")
        for model_view in ("model1", "model2"):
            points = [row for row in task_rows if row["arm"] == arm and row["model_view"] == model_view]
            points.sort(key=lambda row: int(row["step"]))
            frozen = arm == "D0" and model_view == "model1"
            axis.plot(
                [row["step"] for row in points],
                [100.0 * float(row["macro_mean_at_3"]) for row in points],
                label=f"{labels[model_view]}{' (frozen)' if frozen else ''}",
                color=colors[model_view],
                linestyle="--" if frozen else "-",
                marker="o",
                markersize=4,
                linewidth=2.2,
            )
            endpoint = points[-1]
            other_view = "model2" if model_view == "model1" else "model1"
            other_endpoint = next(
                row
                for row in task_rows
                if row["arm"] == arm and row["model_view"] == other_view and int(row["step"]) == EXPECTED_STEPS[-1]
            )
            endpoint_y = 100.0 * float(endpoint["macro_mean_at_3"])
            other_y = 100.0 * float(other_endpoint["macro_mean_at_3"])
            axis.annotate(
                f"{endpoint_y:.2f}%",
                (endpoint["step"], endpoint_y),
                xytext=(-5, 10 if endpoint_y >= other_y else -18),
                textcoords="offset points",
                ha="right",
                color=colors[model_view],
                fontsize=9,
            )

        axis.set_title(
            "C: weak-logit mixture, both trainable" if arm == "C" else "D0: matched-scale no-weak, Model 1 frozen",
            loc="left",
            weight="bold",
        )
        axis.set_xlabel("Post-Stage1 training step")
        axis.set_xticks(EXPECTED_STEPS)
        axis.grid(axis="y", color="#dbe3ee", linewidth=0.8)
        axis.grid(axis="x", color="#eef2f7", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="best")

    axes[0].set_ylabel(task_spec["metric_label"])
    figure.suptitle(task_spec["title"], x=0.06, ha="left", weight="bold", fontsize=14)
    figure.text(
        0.06,
        0.91,
        "C updates both submodels; D0 intentionally keeps Model 1 unchanged.",
        color="#475569",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"qwen3_1p7b_{task}_acd0_p60_submodel_dynamics"
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def main() -> None:
    rows = read_rows()
    csv_path = write_csv(rows)
    print(csv_path)
    for task in TASKS:
        for path in plot_task(rows, task):
            print(path)


if __name__ == "__main__":
    main()
