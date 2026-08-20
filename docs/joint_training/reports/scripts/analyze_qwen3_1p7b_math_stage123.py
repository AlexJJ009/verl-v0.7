#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("/data-1/tmp/verl_agent_scratch/math_stage123_analysis_20260723")
VALIDATION_ROOT = Path(
    "/data-1/tmp/verl_agent_scratch/math_stage123_resume_b01_stage3_retryfix_20260721T121155Z/"
    "recipe/on_policy_wdl_sft/staged_v1/validation"
)

DATASET_ORDER = [
    "aime25",
    "HuggingFaceH4/MATH-500",
    "openai/gsm8k",
    "deepmind/aqua_rat",
    "ChilleD/SVAMP",
    "mwpt5/MAWPS",
    "zwhe99/amc23",
]
DATASET_SHORT = {
    "aime25": "AIME-2025",
    "HuggingFaceH4/MATH-500": "MATH-500",
    "openai/gsm8k": "GSM8K",
    "deepmind/aqua_rat": "AQUA-RAT",
    "ChilleD/SVAMP": "SVAMP",
    "mwpt5/MAWPS": "MAWPS",
    "zwhe99/amc23": "AMC23",
}

RUNS = [
    {
        "id": "b0-stage1",
        "run": "MATH-B0_STAGE1-QWEN3-1P7B-V1_1784539170",
        "phase": "stage1",
        "beta": 0.0,
        "kl": "na",
        "view": "model2-init",
        "offset": 0,
        "log": "/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B0_STAGE1-QWEN3-1P7B-V1_1784539170.log",
    },
    {
        "id": "b01-stage1",
        "run": "MATH-B01_STAGE1-QWEN3-1P7B-V1_1784549110",
        "phase": "stage1",
        "beta": 0.1,
        "kl": "na",
        "view": "model2-init",
        "offset": 0,
        "log": "/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B01_STAGE1-QWEN3-1P7B-V1_1784549110.log",
    },
    {
        "id": "b0-stage1-control",
        "run": "MATH-B0_STAGE1_CONTROL-QWEN3-1P7B-V1_1784558992",
        "phase": "stage1_control",
        "beta": 0.0,
        "kl": "control",
        "view": "single",
        "offset": 40,
        "log": "/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B0_STAGE1_CONTROL-QWEN3-1P7B-V1_1784558992.log",
    },
    {
        "id": "b01-stage1-control",
        "run": "MATH-B01_STAGE1_CONTROL-QWEN3-1P7B-V1_1784573526",
        "phase": "stage1_control",
        "beta": 0.1,
        "kl": "control",
        "view": "single",
        "offset": 40,
        "log": "/data-1/tmp/verl_agent_scratch/math_stage123_step20_lr1e6_mem055_entropyoff_launch_20260720T091917Z/recipe/on_policy_wdl_sft/staged_v1/MATH-B01_STAGE1_CONTROL-QWEN3-1P7B-V1_1784573526.log",
    },
    {
        "id": "b0-stage2-nokl",
        "run": "MATH-B0_STAGE2_NOKL-QWEN3-1P7B-V1_1784598681",
        "phase": "stage2",
        "beta": 0.0,
        "kl": "nokl",
        "view": "default",
        "offset": 40,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T013450Z-resume-b0-stage2/stage123_queue.log",
    },
    {
        "id": "b0-stage2-m2kl",
        "run": "MATH-B0_STAGE2_M2KL-QWEN3-1P7B-V1_1784615452",
        "phase": "stage2",
        "beta": 0.0,
        "kl": "m2kl",
        "view": "default",
        "offset": 40,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T062805Z-resume-b0-stage2-m2kl/run_attempt_logs/b0-stage2-m2kl.attempt-1.log",
    },
    {
        "id": "b01-stage2-nokl",
        "run": "MATH-B01_STAGE2_NOKL-QWEN3-1P7B-V1_1784621388",
        "phase": "stage2",
        "beta": 0.1,
        "kl": "nokl",
        "view": "default",
        "offset": 40,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T062805Z-resume-b0-stage2-m2kl/run_attempt_logs/b01-stage2-nokl.attempt-1.log",
    },
    {
        "id": "b01-stage2-m2kl",
        "run": "MATH-B01_STAGE2_M2KL-QWEN3-1P7B-V1_1784627013",
        "phase": "stage2",
        "beta": 0.1,
        "kl": "m2kl",
        "view": "default",
        "offset": 40,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T062805Z-resume-b0-stage2-m2kl/run_attempt_logs/b01-stage2-m2kl.attempt-2.log",
    },
    {
        "id": "b0-stage3-nokl-model1",
        "run": "MATH-B0_STAGE3_NOKL_MODEL1-QWEN3-1P7B-V1_1784636513",
        "phase": "stage3",
        "beta": 0.0,
        "kl": "nokl",
        "view": "model1",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-nokl-model1.attempt-1.log",
    },
    {
        "id": "b0-stage3-nokl-model2",
        "run": "MATH-B0_STAGE3_NOKL_MODEL2-QWEN3-1P7B-V1_1784646024",
        "phase": "stage3",
        "beta": 0.0,
        "kl": "nokl",
        "view": "model2",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-nokl-model2.attempt-1.log",
    },
    {
        "id": "b0-stage3-m2kl-model1",
        "run": "MATH-B0_STAGE3_M2KL_MODEL1-QWEN3-1P7B-V1_1784655901",
        "phase": "stage3",
        "beta": 0.0,
        "kl": "m2kl",
        "view": "model1",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-m2kl-model1.attempt-1.log",
    },
    {
        "id": "b0-stage3-m2kl-model2",
        "run": "MATH-B0_STAGE3_M2KL_MODEL2-QWEN3-1P7B-V1_1784665430",
        "phase": "stage3",
        "beta": 0.0,
        "kl": "m2kl",
        "view": "model2",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260721T121155Z-resume-b01-stage3/run_attempt_logs/b0-stage3-m2kl-model2.attempt-1.log",
    },
    {
        "id": "b01-stage3-nokl-model1",
        "run": "MATH-B01_STAGE3_NOKL_MODEL1-QWEN3-1P7B-V1_1784684296",
        "phase": "stage3",
        "beta": 0.1,
        "kl": "nokl",
        "view": "model1",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-nokl-model1.attempt-1.log",
    },
    {
        "id": "b01-stage3-nokl-model2",
        "run": "MATH-B01_STAGE3_NOKL_MODEL2-QWEN3-1P7B-V1_1784693698",
        "phase": "stage3",
        "beta": 0.1,
        "kl": "nokl",
        "view": "model2",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-nokl-model2.attempt-1.log",
    },
    {
        "id": "b01-stage3-m2kl-model1",
        "run": "MATH-B01_STAGE3_M2KL_MODEL1-QWEN3-1P7B-V1_1784703723",
        "phase": "stage3",
        "beta": 0.1,
        "kl": "m2kl",
        "view": "model1",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-m2kl-model1.attempt-1.log",
    },
    {
        "id": "b01-stage3-m2kl-model2",
        "run": "MATH-B01_STAGE3_M2KL_MODEL2-QWEN3-1P7B-V1_1784713149",
        "phase": "stage3",
        "beta": 0.1,
        "kl": "m2kl",
        "view": "model2",
        "offset": 60,
        "log": "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260722T013739Z-resume-b01-stage3-after-cleanup/run_attempt_logs/b01-stage3-m2kl-model2.attempt-1.log",
    },
]


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def calculate_metrics(rows):
    grouped = defaultdict(list)
    sources = defaultdict(list)
    for row in rows:
        key = (row["data_source"], row["uid"])
        grouped[key].append(bool(row["acc"]))
        sources[row["data_source"]].append(bool(row["acc"]))

    per_dataset = {}
    for source in DATASET_ORDER:
        source_groups = [values for (data_source, _), values in grouped.items() if data_source == source]
        assert source_groups, source
        assert all(len(values) == 3 for values in source_groups), (source, {len(values) for values in source_groups})
        mean3 = sum(sum(values) for values in source_groups) / (3 * len(source_groups))
        pass3 = sum(any(values) for values in source_groups) / len(source_groups)
        per_dataset[source] = {"mean@3": mean3, "pass@3": pass3, "samples": len(source_groups)}

    macro_mean3 = sum(item["mean@3"] for item in per_dataset.values()) / len(per_dataset)
    macro_pass3 = sum(item["pass@3"] for item in per_dataset.values()) / len(per_dataset)
    micro_mean3 = sum(bool(row["acc"]) for row in rows) / len(rows)
    return per_dataset, macro_mean3, macro_pass3, micro_mean3


def parse_log_metrics(path):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi.sub("", line) for line in Path(path).read_text(errors="replace").splitlines()]
    starts = []
    for index, line in enumerate(lines):
        if "Initial validation metrics:" in line:
            starts.append((index, 0))
        match = re.search(r"Validation and training metrics at step (\d+):", line)
        if match:
            starts.append((index, int(match.group(1))))
    result = {}
    for position, (start, step) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        text = " ".join(lines[start:end]).replace('"', "")
        text = re.sub(r"\(TaskRunner pid=\d+\)\s*", "", text)
        text = re.sub(r"Training Progress:.*?\d+\.\d+s/it\]", "", text)
        per_dataset = {}
        for source in DATASET_ORDER:
            source_pattern = re.escape(source)
            mean_match = re.search(rf"'val-core/{source_pattern}/acc/mean@3'\s*:\s*([0-9.eE+-]+)", text)
            best_match = re.search(rf"'val-core/{source_pattern}/acc/best@3/mean'\s*:\s*([0-9.eE+-]+)", text)
            if not mean_match or not best_match:
                raise RuntimeError(f"Missing metrics for {source} at step {step} in {path}")
            per_dataset[source] = {"mean@3": float(mean_match.group(1)), "pass@3_bootstrap": float(best_match.group(1))}
        macro_match = re.search(r"'val-core/math7_macro/acc/mean@3'\s*:\s*([0-9.eE+-]+)", text)
        if not macro_match:
            raise RuntimeError(f"Missing macro at step {step} in {path}")
        result[step] = {
            "per_dataset": per_dataset,
            "macro_mean3": float(macro_match.group(1)),
            "macro_pass3_bootstrap": sum(item["pass@3_bootstrap"] for item in per_dataset.values()) / len(per_dataset),
        }
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(frame, metric, output_name):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    panels = [
        (0.0, "model1", axes[0, 0]),
        (0.0, "model2", axes[0, 1]),
        (0.1, "model1", axes[1, 0]),
        (0.1, "model2", axes[1, 1]),
    ]
    colors = {"control": "#111827", "nokl": "#2563eb", "m2kl": "#dc2626"}
    styles = {"control": "--", "nokl": "-", "m2kl": "-"}

    stage1_by_beta = {
        beta: frame[(frame.beta == beta) & (frame.phase == "stage1")].sort_values("effective_step")
        for beta in (0.0, 0.1)
    }
    control_by_beta = {
        beta: frame[(frame.beta == beta) & (frame.phase == "stage1_control")].sort_values("effective_step")
        for beta in (0.0, 0.1)
    }

    for beta, view, axis in panels:
        s1 = stage1_by_beta[beta]
        axis.plot(s1.effective_step, s1[metric] * 100, color="#6b7280", marker="o", linewidth=2, label="Stage1")
        control = control_by_beta[beta]
        axis.plot(
            control.effective_step,
            control[metric] * 100,
            color=colors["control"],
            linestyle=styles["control"],
            marker="o",
            linewidth=2,
            label="Stage1 control",
        )
        for kl in ("nokl", "m2kl"):
            s2 = frame[(frame.beta == beta) & (frame.phase == "stage2") & (frame.kl == kl)].sort_values(
                "effective_step"
            )
            s3 = frame[
                (frame.beta == beta) & (frame.phase == "stage3") & (frame.kl == kl) & (frame.view == view)
            ].sort_values("effective_step")
            axis.plot(
                s2.effective_step,
                s2[metric] * 100,
                color=colors[kl],
                marker="s",
                linewidth=2,
                alpha=0.65,
                label=f"Stage2 {kl}",
            )
            axis.plot(
                s3.effective_step,
                s3[metric] * 100,
                color=colors[kl],
                marker="o",
                linewidth=2.5,
                label=f"Stage3 {kl} {view}",
            )
        axis.axvline(40, color="#9ca3af", linestyle=":", linewidth=1.5)
        axis.axvline(60, color="#9ca3af", linestyle=":", linewidth=1.5)
        axis.text(20, 25.5, "Stage1", ha="center", color="#6b7280")
        axis.text(50, 25.5, "Stage2", ha="center", color="#6b7280")
        axis.text(80, 25.5, "Stage3", ha="center", color="#6b7280")
        axis.set_title(f"beta={beta}, Stage3 {view}")
        axis.set_xlim(0, 100)
        axis.set_ylim(24, 92)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    fig.supxlabel("Effective training step")
    fig.supylabel(f"Math-7 macro {metric} (%)")
    fig.suptitle(f"Qwen3-1.7B Math Stage1-Stage3 validation: {metric}", fontsize=16)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.96))
    fig.savefig(OUTPUT_DIR / output_name, dpi=180)
    plt.close(fig)


def plot_finals(summary):
    selected = summary[(summary.phase == "stage3") | (summary.phase == "stage1_control")].copy()
    selected["label"] = (
        selected["id"].str.replace("-stage1-control", " control", regex=False).str.replace("-stage3-", " ", regex=False)
    )
    selected = selected.sort_values(["beta", "view", "kl"])
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for beta, axis in zip((0.0, 0.1), axes, strict=False):
        part = selected[selected.beta == beta]
        axis.barh(
            part.label,
            part.final_macro_mean3 * 100,
            color=["#111827" if x == "control" else "#dc2626" if x == "m2kl" else "#2563eb" for x in part.kl],
        )
        for index, value in enumerate(part.final_macro_mean3 * 100):
            axis.text(value + 0.4, index, f"{value:.2f}%", va="center")
        axis.set_title(f"beta={beta}")
        axis.set_xlim(35, 75)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Math-7 macro mean@3 at effective step 100")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "math_stage123_final_mean3_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    dataset_rows = []

    for metadata in RUNS:
        run_dir = VALIDATION_ROOT / metadata["run"]
        steps = sorted(int(path.stem) for path in run_dir.glob("*.jsonl") if path.stem.isdigit())
        if not steps:
            raise RuntimeError(f"No validation files for {metadata['run']}")
        for step in steps:
            rows = read_rows(run_dir / f"{step}.jsonl")
            per_dataset, macro_mean3, macro_pass3, micro_mean3 = calculate_metrics(rows)
            common = {
                **metadata,
                "local_step": step,
                "effective_step": metadata["offset"] + step,
                "macro_mean@3": macro_mean3,
                "macro_pass@3_exact": macro_pass3,
                "micro_mean@3": micro_mean3,
                "validation_rows": len(rows),
                "unique_prompts": len(rows) // 3,
            }
            validation_rows.append(common)
            for source, values in per_dataset.items():
                dataset_rows.append({**common, "dataset": DATASET_SHORT[source], "data_source": source, **values})

    fields = list(validation_rows[0])
    write_csv(OUTPUT_DIR / "all_validation_steps.csv", validation_rows, fields)
    write_csv(OUTPUT_DIR / "all_validation_steps_by_dataset.csv", dataset_rows, list(dataset_rows[0]))

    frame = pd.DataFrame(validation_rows)
    frame = frame.rename(
        columns={"macro_mean@3": "macro_mean3", "macro_pass@3_exact": "macro_pass3", "micro_mean@3": "micro_mean3"}
    )
    summary_rows = []
    for metadata in RUNS:
        part = frame[frame.id == metadata["id"]].sort_values("local_step")
        first = part.iloc[0]
        final = part.iloc[-1]
        best = part.loc[part.macro_mean3.idxmax()]
        summary_rows.append(
            {
                **metadata,
                "first_local_step": int(first.local_step),
                "final_local_step": int(final.local_step),
                "first_macro_mean3": first.macro_mean3,
                "final_macro_mean3": final.macro_mean3,
                "delta_mean3": final.macro_mean3 - first.macro_mean3,
                "best_local_step": int(best.local_step),
                "best_effective_step": int(best.effective_step),
                "best_macro_mean3": best.macro_mean3,
                "first_macro_pass3": first.macro_pass3,
                "final_macro_pass3": final.macro_pass3,
                "delta_pass3": final.macro_pass3 - first.macro_pass3,
                "best_macro_pass3": part.macro_pass3.max(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    frame.to_csv(OUTPUT_DIR / "effective_step_curves.csv", index=False)

    plot_metric(frame, "macro_mean3", "math_stage123_effective_step_mean3.png")
    plot_metric(frame, "macro_pass3", "math_stage123_effective_step_pass3.png")
    plot_finals(summary)

    print(
        summary[
            [
                "id",
                "first_macro_mean3",
                "final_macro_mean3",
                "delta_mean3",
                "best_local_step",
                "best_macro_mean3",
                "first_macro_pass3",
                "final_macro_pass3",
                "delta_pass3",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
