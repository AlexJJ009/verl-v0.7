#!/usr/bin/env python3
"""Plot Stage 1 -> Stage 2 intervention curves for the staged-v1 WDL-SFT report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRIC_MATH_MEAN3 = "val-core/HuggingFaceH4/MATH-500/acc/mean@3"
METRIC_MATH_BEST3 = "val-core/HuggingFaceH4/MATH-500/acc/best@3/mean"
METRIC_AIME_MEAN3 = "val-core/aime25/acc/mean@3"
METRIC_EXTRACT_FAIL = "jointTraining/answer_extraction_failure_rate"


@dataclass(frozen=True)
class RunSpec:
    beta_label: str
    stage1_jsonl: str
    stage2_jsonl: str
    stage1_source_step: int
    stage2_best_step: int
    color: str
    title: str | None = None
    stage1_continuation_jsonl: str | None = None


PRESETS = {
    "historical": [
        RunSpec(
            beta_label="beta=0.0",
            stage1_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803.jsonl",
            stage2_jsonl="WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162.jsonl",
            stage1_source_step=85,
            stage2_best_step=35,
            color="#1f77b4",
        ),
        RunSpec(
            beta_label="beta=0.1",
            stage1_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295.jsonl",
            stage2_jsonl="WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269.jsonl",
            stage1_source_step=150,
            stage2_best_step=20,
            color="#d62728",
        ),
    ],
    "boxed": [
        RunSpec(
            beta_label="boxed beta=0.0",
            stage1_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1_1780230447.jsonl",
            stage2_jsonl="WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA0-BETA0_1780249087.jsonl",
            stage1_source_step=135,
            stage2_best_step=15,
            color="#0f766e",
            title="Boxed matched chain, beta=0.0",
        ),
    ],
    "plateau_p60": [
        RunSpec(
            beta_label="plateau P60 beta=0.0",
            stage1_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA0-V1_1780381684.jsonl",
            stage2_jsonl="WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA0-BETA0_1780389822.jsonl",
            stage1_source_step=60,
            stage2_best_step=20,
            color="#0f766e",
            title="Plateau handoff vs Stage 1-only, beta=0.0",
            stage1_continuation_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1_1780230447.jsonl",
        ),
        RunSpec(
            beta_label="plateau P60 beta=0.1",
            stage1_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA01-V1_1780395823.jsonl",
            stage2_jsonl="WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA01-BETA01_1780460682.jsonl",
            stage1_source_step=60,
            stage2_best_step=35,
            color="#1d4ed8",
            title="Plateau handoff vs Stage 1-only, beta=0.1",
            stage1_continuation_jsonl="ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1_1780298630.jsonl",
        ),
    ],
}

PRESET_TITLES = {
    "historical": "Stage 2 intervention gives an early MATH-500 lift, but the current recipe is not yet stable",
    "boxed": "Boxed-prompt rerun: Stage 2 beats same-budget Stage 1 continuation, but still collapses late",
    "plateau_p60": "Plateau handoff P60: Stage 2 outperforms Stage 1-only continuation for the stable beta=0.1 chain",
}


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for path in [current, *current.parents]:
        if (path / ".git").exists() and (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


def read_metric_series(path: Path, metric: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            data = payload.get("data", {})
            if metric not in data:
                continue
            step = int(data.get("training/global_step", payload["step"]))
            value = float(data[metric])
            if math.isfinite(value):
                rows.append((step, value * 100.0))
    if not rows:
        raise ValueError(f"No metric {metric!r} found in {path}")
    return rows


def value_at(series: list[tuple[int, float]], step: int) -> float:
    for s, value in series:
        if s == step:
            return value
    raise ValueError(f"Step {step} not found in series")


def best_point(series: list[tuple[int, float]]) -> tuple[int, float]:
    return max(series, key=lambda row: row[1])


def interpolate_baseline(series: list[tuple[int, float]], step: int) -> float | None:
    exact = {s: v for s, v in series}
    if step in exact:
        return exact[step]
    before = [(s, v) for s, v in series if s < step]
    after = [(s, v) for s, v in series if s > step]
    if not before or not after:
        return None
    s0, v0 = max(before)
    s1, v1 = min(after)
    ratio = (step - s0) / (s1 - s0)
    return v0 + ratio * (v1 - v0)


def write_summary(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "beta",
        "stage1_source_step",
        "stage1_source_math500_mean3_pct",
        "stage1_best_step",
        "stage1_best_math500_mean3_pct",
        "stage2_best_step",
        "stage2_best_effective_step",
        "stage2_best_math500_mean3_pct",
        "stage1_continuation_at_stage2_best_pct",
        "delta_vs_source_pp",
        "delta_vs_stage1_continuation_pp",
        "stage2_final_step",
        "stage2_final_math500_mean3_pct",
        "stage2_final_extraction_failure_pct",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_intervention(
    repo_root: Path,
    metrics_root: Path,
    output_stem: Path,
    overleaf_images: Path | None,
    runs: list[RunSpec],
    title: str,
) -> list[dict[str, str | int | float]]:
    fig_width = 7.4 if len(runs) == 1 else 13.2
    fig, axes = plt.subplots(1, len(runs), figsize=(fig_width, 5.4), sharey=False)
    axes = np.atleast_1d(axes)
    summary_rows: list[dict[str, str | int | float]] = []

    for ax, spec in zip(axes, runs, strict=True):
        stage1_path = metrics_root / spec.stage1_jsonl
        continuation_path = metrics_root / (spec.stage1_continuation_jsonl or spec.stage1_jsonl)
        stage2_path = metrics_root / spec.stage2_jsonl
        stage1_math = read_metric_series(stage1_path, METRIC_MATH_MEAN3)
        continuation_math = read_metric_series(continuation_path, METRIC_MATH_MEAN3)
        stage2_math = read_metric_series(stage2_path, METRIC_MATH_MEAN3)
        stage2_extract = read_metric_series(stage2_path, METRIC_EXTRACT_FAIL)

        source_value = value_at(stage1_math, spec.stage1_source_step)
        stage1_best_step, stage1_best_value = best_point(continuation_math)
        stage2_best_value = value_at(stage2_math, spec.stage2_best_step)
        stage2_best_effective_step = spec.stage1_source_step + spec.stage2_best_step
        baseline_at_stage2_best = interpolate_baseline(continuation_math, stage2_best_effective_step)
        final_stage2_step, final_stage2_value = stage2_math[-1]
        final_extract = stage2_extract[-1][1] if stage2_extract else float("nan")

        stage1_x = np.array([s for s, _ in continuation_math])
        stage1_y = np.array([v for _, v in continuation_math])
        stage2_x = np.array([spec.stage1_source_step + s for s, _ in stage2_math])
        stage2_y = np.array([v for _, v in stage2_math])

        ax.plot(stage1_x, stage1_y, color="#6b7280", lw=2.0, marker="o", ms=3, label="Stage 1-only baseline")
        ax.plot(stage2_x, stage2_y, color=spec.color, lw=2.4, marker="o", ms=3.5, label="Stage 2 intervention")
        ax.axvline(spec.stage1_source_step, color="#111827", ls="--", lw=1.3)
        ax.scatter([spec.stage1_source_step], [source_value], color="#111827", s=42, zorder=5)
        ax.scatter([stage2_best_effective_step], [stage2_best_value], color=spec.color, s=64, zorder=5)

        ax.annotate(
            f"insert Stage 2\nfrom S1 step {spec.stage1_source_step}",
            xy=(spec.stage1_source_step, source_value),
            xytext=(spec.stage1_source_step + 8, source_value - 7),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#111827"},
            fontsize=9,
            color="#111827",
        )
        if baseline_at_stage2_best is None:
            delta_text = f"Stage 2 effect: {stage2_best_value - source_value:+.2f} pp vs source"
        else:
            delta_text = f"Stage 2 effect: {stage2_best_value - baseline_at_stage2_best:+.2f} pp vs S1 cont."
        ax.annotate(
            f"{delta_text}\nat S2 step {spec.stage2_best_step}",
            xy=(stage2_best_effective_step, stage2_best_value),
            xytext=(stage2_best_effective_step + 8, stage2_best_value + 3),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": spec.color},
            fontsize=9,
            color=spec.color,
        )

        if final_stage2_value < 20.0:
            ax.annotate(
                "late collapse",
                xy=(spec.stage1_source_step + final_stage2_step, final_stage2_value),
                xytext=(spec.stage1_source_step + final_stage2_step - 48, final_stage2_value + 10),
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#7c2d12"},
                fontsize=9,
                color="#7c2d12",
            )
        elif stage2_best_value - final_stage2_value > 5.0:
            ax.annotate(
                "late degradation",
                xy=(spec.stage1_source_step + final_stage2_step, final_stage2_value),
                xytext=(spec.stage1_source_step + final_stage2_step - 28, final_stage2_value - 10),
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#7c2d12"},
                fontsize=9,
                color="#7c2d12",
            )

        ax.set_title(spec.title or f"Matched chain, {spec.beta_label}", fontsize=12, weight="bold")
        ax.set_xlabel("Effective training step")
        ax.set_ylabel("MATH-500 mean@3 (%)")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_ylim(0, 82)
        ax.legend(loc="lower left", frameon=False, fontsize=9)

        summary_rows.append(
            {
                "beta": spec.beta_label,
                "stage1_source_step": spec.stage1_source_step,
                "stage1_source_math500_mean3_pct": round(source_value, 4),
                "stage1_best_step": stage1_best_step,
                "stage1_best_math500_mean3_pct": round(stage1_best_value, 4),
                "stage2_best_step": spec.stage2_best_step,
                "stage2_best_effective_step": stage2_best_effective_step,
                "stage2_best_math500_mean3_pct": round(stage2_best_value, 4),
                "stage1_continuation_at_stage2_best_pct": ""
                if baseline_at_stage2_best is None
                else round(baseline_at_stage2_best, 4),
                "delta_vs_source_pp": round(stage2_best_value - source_value, 4),
                "delta_vs_stage1_continuation_pp": ""
                if baseline_at_stage2_best is None
                else round(stage2_best_value - baseline_at_stage2_best, 4),
                "stage2_final_step": final_stage2_step,
                "stage2_final_math500_mean3_pct": round(final_stage2_value, 4),
                "stage2_final_extraction_failure_pct": round(final_extract, 4),
            }
        )

    fig.suptitle(title, fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.01,
        "Stage 2 x-axis is shifted by the selected Stage 1 checkpoint step; metrics are online validation mean@3.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(output_stem.with_suffix(suffix), dpi=220, bbox_inches="tight")
        if overleaf_images is not None:
            overleaf_images.mkdir(parents=True, exist_ok=True)
            fig.savefig(overleaf_images / output_stem.with_suffix(suffix).name, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return summary_rows


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__))
    default_metrics = repo_root / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicySFT-Then-WDLSFT-StagedV1"
    default_output_stem = Path(__file__).resolve().parent / "outputs/stage_intervention_math500"
    default_overleaf_images = repo_root / "docs/joint_training/courses/on-policy-wdl-overleaf/images"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="historical")
    parser.add_argument("--metrics-root", type=Path, default=default_metrics)
    parser.add_argument("--output-stem", type=Path, default=default_output_stem)
    parser.add_argument(
        "--summary-csv", type=Path, default=default_output_stem.parent / "stage_intervention_summary.csv"
    )
    parser.add_argument("--overleaf-images", type=Path, default=default_overleaf_images)
    parser.add_argument("--no-overleaf-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__))
    overleaf_images = None if args.no_overleaf_copy else args.overleaf_images
    summary_rows = plot_intervention(
        repo_root,
        args.metrics_root,
        args.output_stem,
        overleaf_images,
        PRESETS[args.preset],
        PRESET_TITLES[args.preset],
    )
    write_summary(args.summary_csv, summary_rows)
    print(f"Wrote {args.output_stem.with_suffix('.png')}")
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"Wrote {args.summary_csv}")
    if overleaf_images is not None:
        print(f"Copied figure to {overleaf_images}")


if __name__ == "__main__":
    main()
