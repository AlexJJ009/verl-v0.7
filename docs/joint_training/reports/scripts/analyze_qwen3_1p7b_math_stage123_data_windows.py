#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data-1/tmp/verl_agent_scratch/math_stage123_analysis_20260723")
DATA = Path("/data-1/dataset/math/qwen3_1p7b_stage123_seed20260719")
BATCH = 64


def text_content(prompt):
    return "\n".join(x.get("content", "") for x in prompt if isinstance(x, dict))


def extra(x, k, default=None):
    return x.get(k, default) if isinstance(x, dict) else default


def features(path, phase, offset):
    df = pd.read_parquet(path).copy()
    df["local_step"] = df["stage123_order"] // BATCH + 1
    df["effective_step"] = offset + df["local_step"]
    df["subject"] = df.extra_info.map(lambda x: extra(x, "subject", "unknown"))
    df["level"] = pd.to_numeric(df.extra_info.map(lambda x: extra(x, "level", np.nan)), errors="coerce")
    df["question"] = df.extra_info.map(lambda x: extra(x, "question", ""))
    df["solution"] = df.extra_info.map(lambda x: extra(x, "solution", ""))
    df["question_chars"] = df.question.str.len()
    df["solution_chars"] = df.solution.str.len()
    df["question_tokens_proxy"] = df.question.str.findall(r"\w+|[^\w\s]").str.len()
    df["solution_tokens_proxy"] = df.solution.str.findall(r"\w+|[^\w\s]").str.len()
    df["display_math_count"] = df.question.str.count(r"\$\$") // 2
    df["boxed_count"] = df.solution.str.count(r"\\boxed")
    df["phase"] = phase
    return df


stage2 = features(DATA / "stage2.parquet", "stage2", 40)
stage3 = features(DATA / "stage3.parquet", "stage3", 60)
control = pd.concat(
    [stage2.assign(control_local_step=stage2.local_step), stage3.assign(control_local_step=20 + stage3.local_step)],
    ignore_index=True,
)
control["effective_step"] = 40 + control.control_local_step
all_df = pd.concat([stage2, stage3], ignore_index=True)
all_df.to_pickle(ROOT / "training_data_rows.pkl")

step_rows = []
for phase, df, step_col in [
    ("stage2", stage2, "local_step"),
    ("stage3", stage3, "local_step"),
    ("control", control, "control_local_step"),
]:
    for step, g in df.groupby(step_col):
        counts = g.subject.value_counts(normalize=True)
        entropy = -sum(p * math.log(p + 1e-12) for p in counts)
        row = {
            "phase": phase,
            "local_step": int(step),
            "effective_step": int(g.effective_step.iloc[0]),
            "rows": len(g),
            "mean_level": g.level.mean(),
            "level4plus_rate": (g.level >= 4).mean(),
            "level5_rate": (g.level >= 5).mean(),
            "mean_question_chars": g.question_chars.mean(),
            "mean_solution_chars": g.solution_chars.mean(),
            "mean_question_tokens_proxy": g.question_tokens_proxy.mean(),
            "mean_solution_tokens_proxy": g.solution_tokens_proxy.mean(),
            "subject_entropy": entropy,
        }
        for subject, p in counts.items():
            row["subject_" + re.sub(r"\W+", "_", subject.lower()).strip("_") + "_rate"] = p
        step_rows.append(row)
step_df = pd.DataFrame(step_rows).fillna(0)
step_df.to_csv(ROOT / "training_data_step_features.csv", index=False)

# Window summary around observed transition.
windows = []
for phase, df, step_col, ranges in [
    ("stage2", stage2, "local_step", [(1, 10), (11, 20)]),
    ("stage3", stage3, "local_step", [(1, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40)]),
    (
        "control",
        control,
        "control_local_step",
        [(1, 20), (21, 30), (31, 35), (36, 40), (41, 45), (46, 50), (51, 55), (56, 60)],
    ),
]:
    for a, b in ranges:
        g = df[df[step_col].between(a, b)]
        counts = g.subject.value_counts(normalize=True)
        row = {
            "phase": phase,
            "window": f"{a}-{b}",
            "effective_window": f"{int(g.effective_step.min())}-{int(g.effective_step.max())}",
            "rows": len(g),
            "mean_level": g.level.mean(),
            "level4plus_rate": (g.level >= 4).mean(),
            "level5_rate": (g.level >= 5).mean(),
            "mean_question_chars": g.question_chars.mean(),
            "mean_solution_chars": g.solution_chars.mean(),
            "mean_question_tokens_proxy": g.question_tokens_proxy.mean(),
            "mean_solution_tokens_proxy": g.solution_tokens_proxy.mean(),
            "subject_entropy": -sum(p * math.log(p + 1e-12) for p in counts),
        }
        for s, p in counts.items():
            row["subject_" + re.sub(r"\W+", "_", s.lower()).strip("_") + "_rate"] = p
        windows.append(row)
win = pd.DataFrame(windows).fillna(0)
win.to_csv(ROOT / "training_data_window_features.csv", index=False)

# Subject/level distribution by shard.
parts = []
for phase, df in [("stage2", stage2), ("stage3", stage3)]:
    x = df.groupby(["subject", "level"], dropna=False).size().reset_index(name="rows")
    x.insert(0, "phase", phase)
    parts.append(x)
pd.concat(parts).to_csv(ROOT / "training_data_subject_level_distribution.csv", index=False)
print(win.to_string(index=False))
