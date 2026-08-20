#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Export and plot Qwen3-1.7B Math GRPO and WDL online validation curves.

The raw GRPO metrics may live outside the repository.  Set
``MATH_GRPO_METRICS_DIR`` (or pass ``--metrics-dir``) to a directory containing
the JSONL files listed in ``GRPO_PIPELINES``.  The derived CSV is repository
portable and is the only input needed for ``--plot-only`` rerenders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "docs/joint_training/reports/data"
FIGURE_DIR = ROOT / "docs/joint_training/reports/figures"
ASSET_DIR = ROOT / "docs/joint_training/reports/assets"
DERIVED_CSV = DATA_DIR / "qwen3_1p7b_math_grpo_online_validation.csv"
BUDGET_CSV = DATA_DIR / "qwen3_1p7b_math_grpo_wdl_budget_estimate.csv"
WDL_CSV = ASSET_DIR / "qwen3_1p7b_math_wdl_p60_ablation_curves.csv"
WDL_METRICS_ROOT = Path(
    "/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/logs/metrics/OnPolicyWDLSFT-Math-1P7B-Causal-P60"
)
WDL_VALIDATION_ROOT = Path("/data-2/model_weights/math_task/qwen3_1p7b_wdl_causal_p60/logs/validation")
WDL_RUNS = (
    {
        "experiment": "WDL C P60 weak-logit treatment",
        "metrics": WDL_METRICS_ROOT / "MATH-WDL-CAUSAL-P60-ARM-C-QWEN3-1P7B_1785247036.jsonl",
        "validation_dir": WDL_VALIDATION_ROOT / "MATH-WDL-CAUSAL-P60-ARM-C-QWEN3-1P7B_1785247036/model2",
        # The causal P60 contract generates from Model2 only.  Model1 enters
        # the fused training logits, not rollout generation.
        "rollout_models": 1,
        "old_logprob_models": 2,
        "training_models": 2,
        "notes": "fusion_mode=mixture; rollout source=model2-only; actor_training_model=joint; freeze_model1=false",
    },
    {
        "experiment": "WDL D0 P60 matched no-weak control",
        "metrics": WDL_METRICS_ROOT / "MATH-WDL-CAUSAL-P60-ARM-D0-QWEN3-1P7B_1785213811.jsonl",
        "validation_dir": WDL_VALIDATION_ROOT / "MATH-WDL-CAUSAL-P60-ARM-D0-QWEN3-1P7B_1785213811/model2",
        "rollout_models": 1,
        "old_logprob_models": 2,
        "training_models": 2,
        "notes": "fusion_mode=strong_scaled; rollout source=model2-only; actor_training_model=joint; freeze_model1=false",
    },
)
QWEN3_1P7B_PARAM_COUNT = 1_720_574_976
PROMPTS_PER_STEP = 64
ROLLOUTS_PER_PROMPT = 8
VALIDATION_OUTPUT_COUNTS = {
    "aime25": 78,
    "HuggingFaceH4/MATH-500": 1_488,
    "zwhe99/amc23": 120,
    "deepmind/aqua_rat": 762,
    "openai/gsm8k": 3_957,
    "mwpt5/MAWPS": 1_065,
    "ChilleD/SVAMP": 900,
}


@dataclass(frozen=True)
class Phase:
    filename: str
    name: str
    order: int


@dataclass(frozen=True)
class Pipeline:
    key: str
    label: str
    initialization: str
    learning_rate: str
    effective_offset: int
    color: str
    linestyle: str
    phases: tuple[Phase, ...]


GRPO_PIPELINES = (
    Pipeline(
        key="cold_lr5e7",
        label="Cold Start + GRPO, LR 5e-7",
        initialization="cold_start",
        learning_rate="5e-7",
        effective_offset=0,
        color="#64748b",
        linestyle="--",
        phases=(
            Phase(
                "MATH-QWEN3-1P7B-COLD-START-GRPO-GON12-FINAL-20260812T071058Z_fresh.jsonl",
                "epoch1",
                0,
            ),
            Phase(
                "MATH-QWEN3-1P7B-COLD-START-GRPO-LR5E7-P200-RESUME-R3_20260813.jsonl",
                "epoch2",
                1,
            ),
        ),
    ),
    Pipeline(
        key="cold_lr1e6",
        label="Cold Start + GRPO, LR 1e-6",
        initialization="cold_start",
        learning_rate="1e-6",
        effective_offset=0,
        color="#2563eb",
        linestyle="-",
        phases=(
            Phase("MATH-QWEN3-1P7B-COLD-START-GRPO-LR1E6-20260812.jsonl", "epoch1", 0),
            Phase(
                "MATH-QWEN3-1P7B-COLD-START-GRPO-LR1E6-P200-RESUME-R2_20260813.jsonl",
                "epoch2",
                1,
            ),
        ),
    ),
    Pipeline(
        key="stage1_lr5e7",
        label="Stage1 + GRPO, LR 5e-7",
        initialization="stage1",
        learning_rate="5e-7",
        effective_offset=40,
        color="#f59e0b",
        linestyle="--",
        phases=(
            Phase("MATH-QWEN3-1P7B-STAGE1-GRPO-LR5E7-20260812.jsonl", "epoch1", 0),
            Phase(
                "MATH-QWEN3-1P7B-STAGE1-GRPO-LR5E7-P200-RESUME-R3_20260813.jsonl",
                "epoch2",
                1,
            ),
        ),
    ),
    Pipeline(
        key="stage1_lr1e6",
        label="Stage1 + GRPO, LR 1e-6",
        initialization="stage1",
        learning_rate="1e-6",
        effective_offset=40,
        color="#16a34a",
        linestyle="-",
        phases=(
            Phase("MATH-QWEN3-1P7B-STAGE1-GRPO-LR1E6-20260812.jsonl", "epoch1", 0),
            Phase(
                "MATH-QWEN3-1P7B-STAGE1-GRPO-LR1E6-P200-RESUME-R3_20260813.jsonl",
                "epoch2",
                1,
            ),
        ),
    ),
)


def resolve_file(search_roots: Sequence[Path], filename: str) -> Path:
    for root in search_roots:
        direct = root / filename
        if direct.is_file():
            return direct
    matches: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            matches.extend(root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"missing metrics file {filename}; searched: {', '.join(str(root) for root in search_roots)}"
        )
    rendered = "\n".join(str(match) for match in sorted(matches))
    raise RuntimeError(f"ambiguous metrics file {filename}:\n{rendered}")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validation_value(row: dict[str, object]) -> float | None:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return None
    value = data.get("val-core/math7_macro/acc/mean@3")
    if value is None:
        value = data.get("val-core/model2/math7_macro/acc/mean@3")
    return None if value is None else float(value)


def read_grpo_rows(metrics_roots: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pipeline in GRPO_PIPELINES:
        for phase in pipeline.phases:
            path = resolve_file(metrics_roots, phase.filename)
            for item in read_jsonl(path):
                value = validation_value(item)
                if value is None:
                    continue
                local_step = int(item["step"])
                rows.append(
                    {
                        "pipeline": pipeline.key,
                        "label": pipeline.label,
                        "initialization": pipeline.initialization,
                        "learning_rate": pipeline.learning_rate,
                        "phase": phase.name,
                        "phase_order": phase.order,
                        "local_step": local_step,
                        "effective_step": local_step + pipeline.effective_offset,
                        "math7_mean_at_3": value * 100.0,
                        "source_metrics_file": path.name,
                    }
                )
    rows.sort(key=lambda row: (str(row["pipeline"]), int(row["effective_step"]), int(row["phase_order"])))
    return rows


def write_rows(rows: Iterable[dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "pipeline",
        "label",
        "initialization",
        "learning_rate",
        "phase",
        "phase_order",
        "local_step",
        "effective_step",
        "math7_mean_at_3",
        "source_metrics_file",
    ]
    with DERIVED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict[str, object]]:
    with DERIVED_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["phase_order"] = int(row["phase_order"])
        row["local_step"] = int(row["local_step"])
        row["effective_step"] = int(row["effective_step"])
        row["math7_mean_at_3"] = float(row["math7_mean_at_3"])
    return rows


def validation_generated_tokens(validation_root: Path, step_names: set[str] | None = None) -> int:
    if not validation_root.is_dir():
        return 0
    total = 0
    for path in validation_root.rglob("*.jsonl"):
        if step_names is not None and path.stem not in step_names:
            continue
        for row in read_jsonl(path):
            total += int(row.get("response_token_count", 0))
    return total


def resolve_validation_roots(search_roots: Sequence[Path], run_name: str) -> list[Path]:
    roots: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        direct_matches = [path for path in root.rglob(run_name) if path.is_dir() and path.parent.name == "validation"]
        roots.extend(direct_matches)
    return sorted(set(roots))


def summarize_metrics(
    rows: list[dict[str, object]],
    *,
    rollout_models: int,
    old_logprob_models: int,
    reference_models: int,
    training_models: int,
) -> dict[str, float]:
    train_rows = [row for row in rows if "training/global_step" in row.get("data", {})]
    train_sequence_tokens = sum(float(row["data"].get("perf/total_num_tokens", 0)) for row in train_rows)  # type: ignore[index]
    train_generated_tokens = sum(
        float(row["data"].get("response_length/mean", 0)) * PROMPTS_PER_STEP * ROLLOUTS_PER_PROMPT  # type: ignore[index]
        for row in train_rows
    )
    training_gpu_hours = sum(float(row["data"].get("timing_s/step", 0)) for row in train_rows) * 8 / 3600  # type: ignore[index]
    validation_gpu_hours = sum(float(row["data"].get("timing_s/testing", 0)) for row in rows) * 8 / 3600  # type: ignore[index]
    rollout_pf = 2 * QWEN3_1P7B_PARAM_COUNT * rollout_models * train_generated_tokens / 1e15
    old_logprob_pf = 2 * QWEN3_1P7B_PARAM_COUNT * old_logprob_models * train_sequence_tokens / 1e15
    reference_pf = 2 * QWEN3_1P7B_PARAM_COUNT * reference_models * train_sequence_tokens / 1e15
    train_pf = 6 * QWEN3_1P7B_PARAM_COUNT * training_models * train_sequence_tokens / 1e15
    val_points = [(int(row["step"]), validation_value(row)) for row in rows if validation_value(row) is not None]
    best_step, best_value = max(val_points, key=lambda item: item[1] or -1)
    last_step, last_value = val_points[-1]
    return {
        "train_rows": len(train_rows),
        "first_train_step": min(int(row["step"]) for row in train_rows),
        "last_train_step": max(int(row["step"]) for row in train_rows),
        "last_validation_step": last_step,
        "last_math7_mean_at_3_pct": (last_value or 0.0) * 100.0,
        "best_validation_step": best_step,
        "best_math7_mean_at_3_pct": (best_value or 0.0) * 100.0,
        "train_generated_tokens": train_generated_tokens,
        "train_sequence_tokens": train_sequence_tokens,
        "training_gpu_hours_from_metrics": training_gpu_hours,
        "validation_gpu_hours_from_metrics": validation_gpu_hours,
        "trainer_total_gpu_hours_from_metrics": training_gpu_hours + validation_gpu_hours,
        "rollout_forward_pflops": rollout_pf,
        "old_logprob_forward_pflops": old_logprob_pf,
        "reference_forward_pflops": reference_pf,
        "training_forward_backward_pflops": train_pf,
        "total_model_pflops_estimate": rollout_pf + old_logprob_pf + reference_pf + train_pf,
    }


def validation_tokens_from_metrics(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    """Recover exact validation output-token totals from per-dataset means.

    The output counts are fixed by the frozen Math-7 n=3 evaluator.  This
    reproduces raw JSONL token sums while also working for copied metrics-only
    GRPO receipts.
    """
    totals: dict[str, float] = {}
    suffix = "/response_token_count/mean@3"
    for row in rows:
        data = row.get("data", {})
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if not key.startswith("val-aux/") or not key.endswith(suffix):
                continue
            metric = key[len("val-aux/") : -len(suffix)]
            model = "single"
            if metric.startswith("model1/") or metric.startswith("model2/"):
                model, metric = metric.split("/", 1)
            output_count = VALIDATION_OUTPUT_COUNTS.get(metric)
            if output_count is not None:
                totals[model] = totals.get(model, 0.0) + float(value) * output_count
    return {model: round(value) for model, value in totals.items()}


def write_budget_rows(metrics_roots: Sequence[Path]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for pipeline in GRPO_PIPELINES:
        phase_paths = [resolve_file(metrics_roots, phase.filename) for phase in pipeline.phases]
        metrics_rows: list[dict[str, object]] = []
        for path in phase_paths:
            metrics_rows.extend(read_jsonl(path))
        summary = summarize_metrics(
            metrics_rows,
            rollout_models=1,
            old_logprob_models=1,
            reference_models=1,
            training_models=1,
        )
        validation_tokens = validation_tokens_from_metrics(metrics_rows).get("single", 0)
        validation_forward_pf = 2 * QWEN3_1P7B_PARAM_COUNT * validation_tokens / 1e15
        status = "running_partial" if pipeline.key == "stage1_lr5e7" else "training_complete_release_pending"
        rows.append(
            {
                "experiment": pipeline.label,
                "method": "GRPO",
                "status": status,
                "effective_step_offset": pipeline.effective_offset,
                "effective_last_validation_step": int(summary["last_validation_step"]) + pipeline.effective_offset,
                "effective_best_validation_step": int(summary["best_validation_step"]) + pipeline.effective_offset,
                "online_last_math7_mean_at_3_pct": summary["last_math7_mean_at_3_pct"],
                "online_best_math7_mean_at_3_pct": summary["best_math7_mean_at_3_pct"],
                "train_rows": summary["train_rows"],
                "local_train_span": f"{int(summary['first_train_step'])}-{int(summary['last_train_step'])}",
                "training_gpu_hours_from_metrics": summary["training_gpu_hours_from_metrics"],
                "validation_gpu_hours_from_metrics": summary["validation_gpu_hours_from_metrics"],
                "trainer_total_gpu_hours_from_metrics": summary["trainer_total_gpu_hours_from_metrics"],
                "train_generated_tokens": int(summary["train_generated_tokens"]),
                "online_validation_generated_tokens_target": validation_tokens,
                "online_validation_generated_tokens_all_models": validation_tokens,
                "train_sequence_tokens": int(summary["train_sequence_tokens"]),
                "rollout_forward_pflops": summary["rollout_forward_pflops"],
                "old_logprob_forward_pflops": summary["old_logprob_forward_pflops"],
                "reference_forward_pflops": summary["reference_forward_pflops"],
                "training_forward_backward_pflops": summary["training_forward_backward_pflops"],
                "total_model_pflops_estimate": summary["total_model_pflops_estimate"],
                "online_validation_forward_pflops": validation_forward_pf,
                "total_model_pflops_including_validation": summary["total_model_pflops_estimate"]
                + validation_forward_pf,
                "source_metrics_files": ";".join(path.name for path in phase_paths),
                "flops_notes": "single-model rollout; single-model old_log_prob; single-model KL reference; single-model actor update",
            }
        )

    for wdl_run in WDL_RUNS:
        metrics_path = Path(wdl_run["metrics"])
        if not metrics_path.is_file():
            continue
        summary = summarize_metrics(
            read_jsonl(metrics_path),
            rollout_models=int(wdl_run["rollout_models"]),
            old_logprob_models=int(wdl_run["old_logprob_models"]),
            reference_models=0,
            training_models=int(wdl_run["training_models"]),
        )
        validation_tokens = validation_tokens_from_metrics(read_jsonl(metrics_path))
        all_validation_tokens = sum(validation_tokens.values())
        validation_forward_pf = 2 * QWEN3_1P7B_PARAM_COUNT * all_validation_tokens / 1e15
        rows.append(
            {
                "experiment": wdl_run["experiment"],
                "method": "WDL",
                "status": "release_allowed",
                "effective_step_offset": 40,
                "effective_last_validation_step": int(summary["last_validation_step"]) + 40,
                "effective_best_validation_step": int(summary["best_validation_step"]) + 40,
                "online_last_math7_mean_at_3_pct": summary["last_math7_mean_at_3_pct"],
                "online_best_math7_mean_at_3_pct": summary["best_math7_mean_at_3_pct"],
                "train_rows": summary["train_rows"],
                "local_train_span": f"{int(summary['first_train_step'])}-{int(summary['last_train_step'])}",
                "training_gpu_hours_from_metrics": summary["training_gpu_hours_from_metrics"],
                "validation_gpu_hours_from_metrics": summary["validation_gpu_hours_from_metrics"],
                "trainer_total_gpu_hours_from_metrics": summary["trainer_total_gpu_hours_from_metrics"],
                "train_generated_tokens": int(summary["train_generated_tokens"]),
                "online_validation_generated_tokens_target": validation_tokens.get("model2", 0),
                "online_validation_generated_tokens_all_models": all_validation_tokens,
                "train_sequence_tokens": int(summary["train_sequence_tokens"]),
                "rollout_forward_pflops": summary["rollout_forward_pflops"],
                "old_logprob_forward_pflops": summary["old_logprob_forward_pflops"],
                "reference_forward_pflops": summary["reference_forward_pflops"],
                "training_forward_backward_pflops": summary["training_forward_backward_pflops"],
                "total_model_pflops_estimate": summary["total_model_pflops_estimate"],
                "online_validation_forward_pflops": validation_forward_pf,
                "total_model_pflops_including_validation": summary["total_model_pflops_estimate"]
                + validation_forward_pf,
                "source_metrics_files": str(metrics_path),
                "flops_notes": wdl_run["notes"],
            }
        )

    fields = [
        "experiment",
        "method",
        "status",
        "effective_step_offset",
        "effective_last_validation_step",
        "effective_best_validation_step",
        "online_last_math7_mean_at_3_pct",
        "online_best_math7_mean_at_3_pct",
        "train_rows",
        "local_train_span",
        "training_gpu_hours_from_metrics",
        "validation_gpu_hours_from_metrics",
        "trainer_total_gpu_hours_from_metrics",
        "train_generated_tokens",
        "online_validation_generated_tokens_target",
        "online_validation_generated_tokens_all_models",
        "train_sequence_tokens",
        "rollout_forward_pflops",
        "old_logprob_forward_pflops",
        "reference_forward_pflops",
        "training_forward_backward_pflops",
        "total_model_pflops_estimate",
        "online_validation_forward_pflops",
        "total_model_pflops_including_validation",
        "source_metrics_files",
        "flops_notes",
    ]
    with BUDGET_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows_for_pipeline(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    """Return one point per effective step, preferring the resumed reevaluation."""
    selected: dict[int, dict[str, object]] = {}
    for row in rows:
        if row["pipeline"] != key:
            continue
        step = int(row["effective_step"])
        previous = selected.get(step)
        if previous is None or int(row["phase_order"]) >= int(previous["phase_order"]):
            selected[step] = row
    return [selected[step] for step in sorted(selected)]


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Effective pipeline optimizer step")
    ax.set_ylabel("Math-7 macro mean@3 (%)")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_xlim(left=0)
    ax.set_ylim(35, 74)


def plot_grpo_internal(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for pipeline in GRPO_PIPELINES:
        points = plot_rows_for_pipeline(rows, pipeline.key)
        ax.plot(
            [int(row["effective_step"]) for row in points],
            [float(row["math7_mean_at_3"]) for row in points],
            marker="o",
            markersize=3.6,
            linewidth=2.0,
            linestyle=pipeline.linestyle,
            color=pipeline.color,
            label=pipeline.label,
        )
    style_axis(ax, "Qwen3-1.7B Math: canonical GRPO learning curves")
    ax.axvline(100, color="#94a3b8", linestyle=":", linewidth=1.3)
    ax.text(102, 36.2, "first-epoch budget", color="#64748b", fontsize=9)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "qwen3_1p7b_math_grpo_internal_curve")


def read_wdl_rows() -> dict[str, list[tuple[int, float]]]:
    columns = {
        "A_stage1_continuation": [],
        "C_continuous_wdl60": [],
        "D0_no_weak_control": [],
    }
    with WDL_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            effective_step = int(row["post_stage1_step"]) + 40
            for column in columns:
                columns[column].append((effective_step, float(row[column])))
    return columns


def plot_grpo_vs_wdl(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    selected_grpo = {"cold_lr1e6", "stage1_lr1e6"}
    for pipeline in GRPO_PIPELINES:
        if pipeline.key not in selected_grpo:
            continue
        points = plot_rows_for_pipeline(rows, pipeline.key)
        ax.plot(
            [int(row["effective_step"]) for row in points],
            [float(row["math7_mean_at_3"]) for row in points],
            marker="o",
            markersize=3.4,
            linewidth=1.9,
            color=pipeline.color,
            label=pipeline.label,
        )

    wdl = read_wdl_rows()
    wdl_specs = {
        "C_continuous_wdl60": ("WDL C: weak-logit treatment", "#dc2626", "-", 2.7),
        "A_stage1_continuation": ("A: standard on-policy SFT", "#7c3aed", "--", 1.5),
        "D0_no_weak_control": ("D0: matched no-weak", "#0f766e", "--", 1.5),
    }
    for key, points in wdl.items():
        label, color, linestyle, width = wdl_specs[key]
        ax.plot(
            [step for step, _ in points],
            [value for _, value in points],
            marker="o",
            markersize=3.8,
            linewidth=width,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    style_axis(ax, "Qwen3-1.7B Math: GRPO versus WDL at effective-step budgets")
    ax.axvline(100, color="#94a3b8", linestyle=":", linewidth=1.3)
    ax.text(102, 36.2, "WDL C terminal budget", color="#64748b", fontsize=9)
    ax.legend(loc="lower right", frameon=True, fontsize=8.8)
    fig.tight_layout()
    save_figure(fig, "qwen3_1p7b_math_grpo_vs_wdl_curve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_metrics = os.environ.get(
        "MATH_GRPO_METRICS_DIR", "/data-1/tmp/verl_agent_scratch/math_grpo_budget_20260814/metrics"
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        action="append",
        default=[
            Path(default_metrics),
            Path("/data-1/tmp/verl_agent_scratch/grpo_budget_audit_20260814/worker_a_jobs"),
        ],
        help="Raw GRPO metrics root; may be passed more than once.",
    )
    parser.add_argument("--plot-only", action="store_true", help="reuse the derived CSV without raw JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plot_only:
        write_rows(read_grpo_rows(args.metrics_dir))
        write_budget_rows(args.metrics_dir)
    rows = load_rows()
    plot_grpo_internal(rows)
    plot_grpo_vs_wdl(rows)
    print(DERIVED_CSV)
    print(BUDGET_CSV)
    print(FIGURE_DIR / "qwen3_1p7b_math_grpo_internal_curve.png")
    print(FIGURE_DIR / "qwen3_1p7b_math_grpo_vs_wdl_curve.png")


if __name__ == "__main__":
    main()
