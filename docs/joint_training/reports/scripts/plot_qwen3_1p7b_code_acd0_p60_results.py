#!/usr/bin/env python3
"""Export and plot completed Qwen3-1.7B Code beta=0 A/C/D0 P60 results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "docs/joint_training/reports/data"
FIGURE_DIR = ROOT / "docs/joint_training/reports/figures"

RUNS = {
    "A": {
        "label": "A: single-model continuation",
        "run_name": "CODE-WDL-ACD0-P60-ARM-A-QWEN3-1P7B_1785319831",
        "metrics": Path(
            "/data-2/model_weights/code_task/qwen3_1p7b_wdl_acd0_p60/logs/metrics/"
            "OnPolicyWDLSFT-CodeTask/CODE-WDL-ACD0-P60-ARM-A-QWEN3-1P7B_1785319831.jsonl"
        ),
        "view": "",
        "color": "#2563eb",
        "linestyle": "-",
    },
    "C_model2": {
        "label": "C model2: mixture 0.2 weak + 0.8 strong",
        "run_name": "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B_1785746593",
        "metrics": Path(
            "/data-2/model_weights/code_task/qwen3_1p7b_wdl_acd0_p60/logs/metrics/"
            "OnPolicyWDLSFT-CodeTask/CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B_1785746593.jsonl"
        ),
        "view": "model2",
        "color": "#16a34a",
        "linestyle": "-",
    },
    "D0_model2": {
        "label": "D0 model2: 0.8 * strong logits",
        "run_name": "CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B_1785430935",
        "metrics": Path(
            "/data-2/model_weights/code_task/qwen3_1p7b_wdl_acd0_p60/logs/metrics/"
            "OnPolicyWDLSFT-CodeTask/CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B_1785430935.jsonl"
        ),
        "view": "model2",
        "color": "#dc2626",
        "linestyle": "-",
    },
    "D0_model1": {
        "label": "D0 model1: frozen weak view",
        "run_name": "CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B_1785430935",
        "metrics": Path(
            "/data-2/model_weights/code_task/qwen3_1p7b_wdl_acd0_p60/logs/metrics/"
            "OnPolicyWDLSFT-CodeTask/CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B_1785430935.jsonl"
        ),
        "view": "model1",
        "color": "#64748b",
        "linestyle": (0, (4, 2)),
    },
}

SOURCES = ["HumanEval+", "MBPP+", "LiveCodeBench"]
ONLINE_CSV = DATA_DIR / "qwen3_1p7b_code_acd0_p60_online_validation.csv"
SUMMARY_CSV = DATA_DIR / "qwen3_1p7b_code_acd0_p60_summary.csv"
CODE3_FIGURE_STEM = "qwen3_1p7b_code_acd0_p60_code3_curve"
D0_QUALITY_FIGURE_STEM = "qwen3_1p7b_code_acd0_p60_d0_quality_curve"


def metric_key(prefix: str, view: str, tail: str) -> str:
    if view:
        return f"{prefix}/{view}/{tail}"
    return f"{prefix}/{tail}"


def get(data: dict[str, Any], prefix: str, view: str, tail: str) -> float | None:
    key = metric_key(prefix, view, tail)
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, spec in RUNS.items():
        metrics_path = spec["metrics"]
        view = str(spec["view"])
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                step = int(item["step"])
                data = item["data"]
                code3 = get(data, "val-core", view, "code3_macro/acc/mean@3")
                if code3 is None:
                    continue
                row: dict[str, Any] = {
                    "arm": arm,
                    "label": spec["label"],
                    "run_name": spec["run_name"],
                    "step": step,
                    "code3_macro_mean_at_3": code3,
                    "optimizer_step_applied": data.get("actor/optimizer_step_applied"),
                    "actor_grad_norm": data.get("actor/grad_norm"),
                    "model1_grad_norm": data.get("jointTraining/model1_grad_norm"),
                    "model2_grad_norm": data.get("jointTraining/model2_grad_norm"),
                    "grad_clip_event": data.get("actor/grad_clip_event"),
                    "response_aborted_ratio": data.get("response/aborted_ratio"),
                    "response_length_clip_ratio": data.get("response_length/clip_ratio"),
                    "wdl_correct_ratio": data.get("wdl_sft/correct_ratio"),
                }
                for source in SOURCES:
                    row[f"{source}_mean_at_3"] = get(data, "val-core", view, f"{source}/acc/mean@3")
                    row[f"{source}_truncated_mean_at_3"] = get(data, "val-aux", view, f"{source}/truncated/mean@3")
                    row[f"{source}_format_contract_success_mean_at_3"] = get(
                        data, "val-aux", view, f"{source}/format_contract_success/mean@3"
                    )
                    row[f"{source}_has_eos_mean_at_3"] = get(data, "val-aux", view, f"{source}/has_eos/mean@3")
                rows.append(row)
    rows.sort(key=lambda row: (row["arm"], row["step"]))
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "label",
        "run_name",
        "step",
        "code3_macro_mean_at_3",
        "HumanEval+_mean_at_3",
        "MBPP+_mean_at_3",
        "LiveCodeBench_mean_at_3",
        "HumanEval+_truncated_mean_at_3",
        "MBPP+_truncated_mean_at_3",
        "LiveCodeBench_truncated_mean_at_3",
        "HumanEval+_format_contract_success_mean_at_3",
        "MBPP+_format_contract_success_mean_at_3",
        "LiveCodeBench_format_contract_success_mean_at_3",
        "HumanEval+_has_eos_mean_at_3",
        "MBPP+_has_eos_mean_at_3",
        "LiveCodeBench_has_eos_mean_at_3",
        "optimizer_step_applied",
        "actor_grad_norm",
        "model1_grad_norm",
        "model2_grad_norm",
        "grad_clip_event",
        "response_aborted_ratio",
        "response_length_clip_ratio",
        "wdl_correct_ratio",
    ]
    with ONLINE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = [
        "arm",
        "label",
        "run_name",
        "status",
        "best_step",
        "best_code3_macro_mean_at_3",
        "latest_validation_step",
        "latest_code3_macro_mean_at_3",
        "latest_HumanEval+_mean_at_3",
        "latest_MBPP+_mean_at_3",
        "latest_LiveCodeBench_mean_at_3",
        "latest_LiveCodeBench_truncated_mean_at_3",
        "latest_response_length_clip_ratio",
    ]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for arm, arm_rows in by_arm.items():
            best = max(arm_rows, key=lambda row: float(row["code3_macro_mean_at_3"]))
            latest = max(arm_rows, key=lambda row: int(row["step"]))
            status = "complete" if int(latest["step"]) == 60 else f"partial_step_{latest['step']}"
            writer.writerow(
                {
                    "arm": arm,
                    "label": latest["label"],
                    "run_name": latest["run_name"],
                    "status": status,
                    "best_step": best["step"],
                    "best_code3_macro_mean_at_3": best["code3_macro_mean_at_3"],
                    "latest_validation_step": latest["step"],
                    "latest_code3_macro_mean_at_3": latest["code3_macro_mean_at_3"],
                    "latest_HumanEval+_mean_at_3": latest["HumanEval+_mean_at_3"],
                    "latest_MBPP+_mean_at_3": latest["MBPP+_mean_at_3"],
                    "latest_LiveCodeBench_mean_at_3": latest["LiveCodeBench_mean_at_3"],
                    "latest_LiveCodeBench_truncated_mean_at_3": latest["LiveCodeBench_truncated_mean_at_3"],
                    "latest_response_length_clip_ratio": latest["response_length_clip_ratio"],
                }
            )


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * value


def plot_code3(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#ffffff")

    for arm, spec in RUNS.items():
        points = [row for row in rows if row["arm"] == arm]
        points.sort(key=lambda row: int(row["step"]))
        axis.plot(
            [row["step"] for row in points],
            [pct(float(row["code3_macro_mean_at_3"])) for row in points],
            label=spec["label"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            marker="o",
            markersize=4.2,
            linewidth=2.4,
        )
        best = max(points, key=lambda row: float(row["code3_macro_mean_at_3"]))
        axis.scatter(
            [best["step"]],
            [pct(float(best["code3_macro_mean_at_3"]))],
            s=78,
            color=spec["color"],
            edgecolor="#0f172a",
            linewidth=0.8,
            zorder=4,
        )

    axis.set_title("Qwen3-1.7B Code beta=0 A/C/D0 P60 — Code-3 mean@3", loc="left", weight="bold")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Code-3 macro mean@3 (%)")
    axis.set_xticks(range(0, 61, 5))
    axis.grid(axis="y", color="#dbe3ee", linewidth=0.8)
    axis.grid(axis="x", color="#eef2f7", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.text(
        0.125,
        0.91,
        "All arms are complete at P60; C-D0 is the matched-scale weak-logit comparison.",
        color="#475569",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{CODE3_FIGURE_STEM}.png"
    pdf = FIGURE_DIR / f"{CODE3_FIGURE_STEM}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def plot_d0_quality(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    d0 = [row for row in rows if row["arm"] == "D0_model2"]
    d0.sort(key=lambda row: int(row["step"]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True)
    figure.patch.set_facecolor("#f8fafc")

    colors = {"HumanEval+": "#2563eb", "MBPP+": "#7c3aed", "LiveCodeBench": "#dc2626"}
    panels = [
        (axes[0], "truncated", "Native truncation / length-stop rate"),
        (axes[1], "format_contract_success", "Format contract success rate"),
    ]
    for axis, metric, title in panels:
        axis.set_facecolor("#ffffff")
        for source in SOURCES:
            axis.plot(
                [row["step"] for row in d0],
                [pct(row[f"{source}_{metric}_mean_at_3"]) for row in d0],
                color=colors[source],
                marker="o",
                linewidth=2.2,
                markersize=4,
                label=source,
            )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("Training step")
        axis.set_xticks(range(0, 61, 10))
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#dbe3ee", linewidth=0.8)
        axis.grid(axis="x", color="#eef2f7", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Rate (%)")
    axes[1].legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24))
    figure.suptitle(
        "D0 model2 completion quality",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.91,
        "LiveCodeBench remains the dominant length-stop failure mode at P60.",
        color="#475569",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.02, 0.04, 1, 0.86))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{D0_QUALITY_FIGURE_STEM}.png"
    pdf = FIGURE_DIR / f"{D0_QUALITY_FIGURE_STEM}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def main() -> None:
    rows = read_rows()
    write_csv(rows)
    code3_png, code3_pdf = plot_code3(rows)
    quality_png, quality_pdf = plot_d0_quality(rows)
    print(f"Wrote {ONLINE_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {code3_png}")
    print(f"Wrote {code3_pdf}")
    print(f"Wrote {quality_png}")
    print(f"Wrote {quality_pdf}")


if __name__ == "__main__":
    main()
