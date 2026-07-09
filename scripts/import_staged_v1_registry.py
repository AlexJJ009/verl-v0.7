#!/usr/bin/env python3
"""Import staged-v1 On-Policy SFT and Stage-2 WDL-SFT runs into the registry.

The importer is source-driven and idempotent. It parses local JSONL metric
files, checkpoint metadata, launcher-derived W&B ids, and dataset metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REGISTRY_TOOLS = Path("/data-1/agent-tools/experiment_registry")
if str(REGISTRY_TOOLS) not in sys.path:
    sys.path.insert(0, str(REGISTRY_TOOLS))

from registry_core import connect, init_db, slug, utc_now  # noqa: E402


PROJECT_NAME = "verl:feature/on-policy-wdl-sft"
PROJECT_KEY = "verl_feature_on_policy_wdl_sft"
BRANCH = "feature/on-policy-wdl-sft"
IMPORTER = "import_staged_v1_on_policy_sft_v1"
WANDB_ENTITY = "gongxunli-beihang-universally"
WANDB_PROJECT = "OnPolicySFT-Then-WDLSFT-StagedV1"

REPO = Path("/data-1/verl07/verl")
METRICS_DIR = REPO / "recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicySFT-Then-WDLSFT-StagedV1"
CHECKPOINT_ROOT = Path("/data-1/checkpoints")
MODEL_WEIGHT_ROOT = Path("/data-1/model_weights/staged_v1")
RELEASE_GATE_SCRIPT = REPO / "scripts/training_result_release_gate.py"

TRAIN_STAGE1 = Path("/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet")
TRAIN_STAGE1_BOXED = Path("/data-1/dataset/EnsembleLLM-data-processed/staged_v1/train_rl_format_boxed_prompt.parquet")
TRAIN_STAGE1_BOXED_MANIFEST = Path("/data-1/dataset/EnsembleLLM-data-processed/staged_v1/train_rl_format_boxed_prompt.manifest.json")
TRAIN_STAGE2 = Path("/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet")
TRAIN_STAGE2_MANIFEST = Path("/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.manifest.json")
TRAIN_STAGE2_BOXED = Path("/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_boxed_after_s1_150steps_seed20260528_75steps.parquet")
TRAIN_STAGE2_BOXED_MANIFEST = Path("/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_boxed_after_s1_150steps_seed20260528_75steps.manifest.json")
MATH500 = Path("/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet")
AIME2025 = Path("/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet")

VAL_METRIC_KEYS = [
    "val-core/HuggingFaceH4/MATH-500/acc/mean@3",
    "val-core/HuggingFaceH4/MATH-500/acc/best@3/mean",
    "val-core/HuggingFaceH4/MATH-500/acc/maj@3/mean",
    "val-core/aime25/acc/mean@3",
    "val-core/aime25/acc/best@3/mean",
    "val-core/aime25/acc/maj@3/mean",
]

TRAIN_METRIC_KEYS = [
    "actor/wdl_sft_loss_total",
    "actor/wdl_sft_loss_positive",
    "actor/wdl_sft_loss_negative",
    "actor/grad_norm",
    "actor/wdl_sft_beta",
    "wdl_sft/correct_ratio",
    "wdl_sft/n_correct",
    "wdl_sft/n_incorrect",
    "response_length/clip_ratio",
    "jointTraining/model1_grad_norm",
    "jointTraining/model2_grad_norm",
    "jointTraining/model_grad_norm_ratio",
    "jointTraining/model_grad_cosine_similarity",
    "jointTraining/answer_extraction_failure_rate",
]


RUNS = [
    {
        "stage": 1,
        "beta": 0.0,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803",
        "wandb_id": "tlg5b94n",
        "status": "completed",
        "trust_level": "trusted",
    },
    {
        "stage": 1,
        "beta": 0.1,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295",
        "wandb_id": "4qjnywm0",
        "status": "completed",
        "trust_level": "trusted",
    },
    {
        "stage": 1,
        "beta": 0.2,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA02-V1_1779999847",
        "wandb_id": "dqbd7zq3",
        "status": "completed",
        "trust_level": "trusted",
    },
    {
        "stage": 1,
        "beta": 0.3,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA03-V1_1780019059",
        "wandb_id": "9oicims3",
        "status": "completed",
        "trust_level": "trusted",
    },
    {
        "stage": 1,
        "beta": 0.4,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA04-V1_1780038811",
        "wandb_id": "z5987ls0",
        "status": "aborted",
        "trust_level": "usable_with_caution",
        "notes": "Interrupted Stage-1 beta-grid run; latest checkpoint marker is step 65 and metrics JSONL has 66 rows.",
    },
    {
        "stage": 2,
        "beta": 0.0,
        "stage1_beta": 0.0,
        "stage1_run": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803",
        "stage1_step": 85,
        "run_name": "WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162",
        "wandb_id": "j2qmgtqi",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "notes": "Completed 75 steps; online validation peaked early and collapsed by final step.",
    },
    {
        "stage": 2,
        "beta": 0.1,
        "stage1_beta": 0.1,
        "stage1_run": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295",
        "stage1_step": 150,
        "run_name": "WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269",
        "wandb_id": "0ywgguty",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "notes": "Completed 75 steps; best online validation improved over Stage 1 but final validation collapsed.",
    },
    {
        "stage": 1,
        "beta": 0.0,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1_1780230447",
        "wandb_id": "t0xmcx0y",
        "status": "completed",
        "trust_level": "trusted",
        "train_dataset": "stage1_boxed",
        "chain_group": "boxed_matched_beta0",
        "notes": "Boxed-prompt matched Stage-1 run; use best checkpoint step 135.",
    },
    {
        "stage": 2,
        "beta": 0.0,
        "stage1_beta": 0.0,
        "stage1_run": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1_1780230447",
        "stage1_step": 135,
        "run_name": "WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA0-BETA0_1780249087",
        "wandb_id": "hah6m6o3",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "train_dataset": "stage2_boxed",
        "chain_group": "boxed_matched_beta0",
        "stage1_source_json": "/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta0-best/stage1_source.json",
        "extra_logs": [
            "recipe/on_policy_wdl_sft/staged_v1/WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA0-BETA0_1780249087_resumed_1780278457.log"
        ],
        "notes": "Completed 75 steps after resumed launch; online validation peaked at step 15 and collapsed by final step.",
    },
    {
        "stage": 1,
        "beta": 0.1,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1_1780298630",
        "wandb_id": "vvjrd1ct",
        "status": "completed",
        "trust_level": "trusted",
        "train_dataset": "stage1_boxed",
        "chain_group": "boxed_matched_beta01",
        "notes": "Boxed-prompt matched Stage-1 run; use best checkpoint step 115.",
    },
    {
        "stage": 2,
        "beta": 0.1,
        "stage1_beta": 0.1,
        "stage1_run": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1_1780298630",
        "stage1_step": 115,
        "run_name": "WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA01-BETA01_1780317270",
        "wandb_id": "k4vf25id",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "train_dataset": "stage2_boxed",
        "chain_group": "boxed_matched_beta01",
        "stage1_source_json": "/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta01-best/stage1_source.json",
        "notes": "Completed 75 steps; online validation peaked at step 30 and collapsed by final step.",
    },
    {
        "stage": 1,
        "beta": 0.0,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA0-V1_1780381684",
        "wandb_id": "vcxq990i",
        "status": "completed",
        "trust_level": "trusted",
        "train_dataset": "stage1_boxed",
        "chain_group": "plateau_p60_beta0",
        "notes": "Plateau-handoff P60 Stage-1 source run; fixed handoff checkpoint is step 60.",
    },
    {
        "stage": 2,
        "beta": 0.0,
        "stage1_beta": 0.0,
        "stage1_run": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA0-V1_1780381684",
        "stage1_step": 60,
        "run_name": "WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA0-BETA0_1780389822",
        "wandb_id": "8kei5dy6",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "train_dataset": "stage2_boxed",
        "chain_group": "plateau_p60_beta0",
        "stage1_source_json": "/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta0-step60/stage1_source.json",
        "notes": "Plateau-handoff P60 matched beta=0.0 Stage-2 run; peak improved over Stage-1 source but final step dropped below peak.",
    },
    {
        "stage": 1,
        "beta": 0.1,
        "run_name": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA01-V1_1780395823",
        "wandb_id": "vuzmnho2",
        "status": "completed",
        "trust_level": "trusted",
        "train_dataset": "stage1_boxed",
        "chain_group": "plateau_p60_beta01",
        "notes": "Plateau-handoff P60 Stage-1 source run; fixed handoff checkpoint is step 60.",
    },
    {
        "stage": 2,
        "beta": 0.1,
        "stage1_beta": 0.1,
        "stage1_run": "ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA01-V1_1780395823",
        "stage1_step": 60,
        "run_name": "WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA01-BETA01_1780460682",
        "wandb_id": "uye2wcno",
        "status": "completed",
        "trust_level": "trusted",
        "train_dataset": "stage2_boxed",
        "chain_group": "plateau_p60_beta01",
        "stage1_source_json": "/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta01-step60/stage1_source.json",
        "notes": "Plateau-handoff P60 matched beta=0.1 Stage-2 run; peak and final online MATH-500 mean@3 are both stable.",
    },
]


def check_release_gate(run_names: list[str]) -> None:
    for run_name in run_names:
        subprocess.check_call([sys.executable, str(RELEASE_GATE_SCRIPT), "check", "--run-name", run_name])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_rows(path: Path) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_meta(path: Path) -> tuple[str | None, int | None, str | None]:
    if not path.exists():
        return None, None, None
    stat = path.stat()
    mtime = dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).replace(microsecond=0).isoformat()
    return mtime, stat.st_size, file_sha256(path)


def latest_step(run_name: str) -> int | None:
    path = CHECKPOINT_ROOT / run_name / "latest_checkpointed_iteration.txt"
    if not path.exists():
        return None
    return int(path.read_text(encoding="utf-8").strip())


def validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if "val-core/HuggingFaceH4/MATH-500/acc/mean@3" in r.get("data", {})]


def best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    candidates = [r for r in rows if metric in r.get("data", {})]
    if not candidates:
        raise ValueError(f"metric not found: {metric}")
    return max(candidates, key=lambda r: r["data"][metric])


def final_validation_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    vals = validation_rows(rows)
    return vals[-1] if vals else None


def chain_family_suffix(run: dict[str, Any]) -> str:
    chain_group = run.get("chain_group") or ""
    if chain_group.startswith("plateau_p60"):
        return "_plateau_p60"
    if run.get("train_dataset") in {"stage1_boxed", "stage2_boxed"}:
        return "_boxed_matched"
    return ""


def training_run_key(run: dict[str, Any]) -> str:
    family = "stage1_on_policy_sft_beta" if run["stage"] == 1 else "stage2_model2_rollout_fused_loss_beta"
    family += chain_family_suffix(run)
    beta_tag = f"{run['beta']:.1f}".replace(".", "p")
    return f"feature_on_policy_wdl_sft.staged_v1.{family}_{beta_tag}.{run['run_name']}.training_metrics_jsonl"


def experiment_key(run: dict[str, Any]) -> str:
    family = "stage1_on_policy_sft_beta" if run["stage"] == 1 else "stage2_model2_rollout_fused_loss_beta"
    family += chain_family_suffix(run)
    beta_tag = f"{run['beta']:.1f}".replace(".", "p")
    return f"verl.branch.feature_on_policy_wdl_sft.staged_v1.{family}_{beta_tag}.{run['run_name']}"


def dataset_for_run(run: dict[str, Any], dataset_ids: dict[str, int]) -> int:
    if run.get("train_dataset") == "stage1_boxed":
        return dataset_ids["stage1_boxed"]
    if run.get("train_dataset") == "stage2_boxed":
        return dataset_ids["stage2_boxed"]
    return dataset_ids["stage1"] if run["stage"] == 1 else dataset_ids["stage2"]


def manifest_for_run(run: dict[str, Any]) -> Path | None:
    if run.get("train_dataset") == "stage1_boxed":
        return TRAIN_STAGE1_BOXED_MANIFEST
    if run.get("train_dataset") == "stage2_boxed":
        return TRAIN_STAGE2_BOXED_MANIFEST
    if run["stage"] == 2:
        return TRAIN_STAGE2_MANIFEST
    return None


def upsert_project(conn: sqlite3.Connection) -> int:
    conn.execute(
        """
        insert into projects(project_key, name, repo_path, default_branch, notes)
        values (?, ?, ?, ?, ?)
        on conflict(name) do update set
          project_key=excluded.project_key,
          repo_path=excluded.repo_path,
          default_branch=excluded.default_branch,
          notes=coalesce(excluded.notes, projects.notes)
        """,
        (PROJECT_KEY, PROJECT_NAME, str(REPO), BRANCH, "Branch-scoped registry form for feature/on-policy-wdl-sft runs."),
    )
    return int(conn.execute("select id from projects where name=?", (PROJECT_NAME,)).fetchone()["id"])


def upsert_dataset(conn: sqlite3.Connection, key: str, name: str, path: Path, split: str, notes: str | None = None) -> int:
    conn.execute(
        """
        insert into datasets(dataset_key, name, domain, path, split, row_count, format, notes)
        values (?, ?, 'math', ?, ?, ?, 'parquet', ?)
        on conflict(dataset_key) do update set
          name=excluded.name,
          domain=excluded.domain,
          path=excluded.path,
          split=excluded.split,
          row_count=excluded.row_count,
          format=excluded.format,
          notes=coalesce(excluded.notes, datasets.notes)
        """,
        (key, name, str(path), split, parquet_rows(path), notes),
    )
    return int(conn.execute("select id from datasets where dataset_key=?", (key,)).fetchone()["id"])


def dataset_row_count(conn: sqlite3.Connection, dataset_id: int) -> int | None:
    row = conn.execute("select row_count from datasets where id=?", (dataset_id,)).fetchone()
    return int(row["row_count"]) if row and row["row_count"] is not None else None


def link_training_dataset(conn: sqlite3.Connection, tr_id: int, dataset_id: int, role: str, notes: str | None = None) -> None:
    conn.execute(
        """
        insert into training_run_datasets(training_run_id, dataset_id, role, row_count, notes)
        values (?, ?, ?, ?, ?)
        on conflict(training_run_id, dataset_id, role) do update set
          row_count=excluded.row_count,
          notes=coalesce(excluded.notes, training_run_datasets.notes)
        """,
        (tr_id, dataset_id, role, dataset_row_count(conn, dataset_id), notes),
    )


def link_experiments(conn: sqlite3.Connection, from_exp_id: int, to_exp_id: int, link_type: str, notes: str | None = None) -> None:
    conn.execute(
        """
        insert into experiment_links(from_experiment_id, to_experiment_id, link_type, notes)
        values (?, ?, ?, ?)
        on conflict(from_experiment_id, to_experiment_id, link_type) do update set
          notes=coalesce(excluded.notes, experiment_links.notes)
        """,
        (from_exp_id, to_exp_id, link_type, notes),
    )


def upsert_model(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    model_key: str,
    display_name: str,
    path: str,
    role: str,
    step: int | None,
    is_best: int | None,
    is_latest: int | None,
    selection_metric: str | None,
    selection_value: float | None,
    extra: dict[str, Any] | None = None,
    notes: str | None = None,
) -> int:
    conn.execute(
        """
        insert into models(
          model_key, display_name, base_model, model_path, checkpoint_step,
          global_step, checkpoint_kind, model_role, is_best, is_latest,
          selection_metric_name, selection_metric_value, project_id,
          git_branch, git_commit, extra_json, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(model_key) do update set
          display_name=excluded.display_name,
          model_path=excluded.model_path,
          checkpoint_step=coalesce(excluded.checkpoint_step, models.checkpoint_step),
          global_step=coalesce(excluded.global_step, models.global_step),
          checkpoint_kind=coalesce(excluded.checkpoint_kind, models.checkpoint_kind),
          model_role=coalesce(excluded.model_role, models.model_role),
          is_best=coalesce(excluded.is_best, models.is_best),
          is_latest=coalesce(excluded.is_latest, models.is_latest),
          selection_metric_name=coalesce(excluded.selection_metric_name, models.selection_metric_name),
          selection_metric_value=coalesce(excluded.selection_metric_value, models.selection_metric_value),
          project_id=coalesce(excluded.project_id, models.project_id),
          git_branch=excluded.git_branch,
          git_commit=excluded.git_commit,
          extra_json=coalesce(excluded.extra_json, models.extra_json),
          notes=coalesce(excluded.notes, models.notes)
        """,
        (
            model_key,
            display_name,
            "Qwen/Qwen3-4B-Base",
            path,
            step,
            step,
            "best" if is_best else ("latest" if is_latest else "checkpoint"),
            role,
            is_best,
            is_latest,
            selection_metric,
            selection_value,
            project_id,
            BRANCH,
            GIT_COMMIT,
            json.dumps(extra, ensure_ascii=False) if extra else None,
            notes,
        ),
    )
    return int(conn.execute("select id from models where model_key=?", (model_key,)).fetchone()["id"])


def upsert_experiment(conn: sqlite3.Connection, project_id: int, run: dict[str, Any], best: dict[str, Any], latest: int | None) -> int:
    stage = run["stage"]
    method_variant = "wdl_sft"
    method_version = "staged_v1_stage1" if stage == 1 else "staged_v1_stage2_model2_rollout"
    family = "on_policy_sft" if stage == 1 else "on_policy_wdl_sft"
    display = (
        f"feature/on-policy-wdl-sft / staged v1 Stage 1 beta={run['beta']:.1f} / {run['run_name']}"
        if stage == 1
        else f"feature/on-policy-wdl-sft / staged v1 Stage 2 s1_beta={run['stage1_beta']:.1f} s2_beta={run['beta']:.1f} / {run['run_name']}"
    )
    trust_reason = (
        "Completed source-backed training run with metrics/checkpoint/W&B artifacts present."
        if run["status"] == "completed"
        else "Interrupted source-backed run; retain for beta-grid provenance but use with caution."
    )
    extra = {
        "run_name": run["run_name"],
        "stage": stage,
        "stage1_beta": run.get("stage1_beta"),
        "stage2_beta": run["beta"] if stage == 2 else None,
        "loss_mode": "wdl_sft",
        "train_dataset": run.get("train_dataset"),
        "chain_group": run.get("chain_group"),
        "best_checkpoint": best,
        "latest_step": latest,
        "wandb_cloud": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run['wandb_id']}",
    }
    now = utc_now()
    conn.execute(
        """
        insert into experiments(
          project_id, experiment_key, display_name, method, method_family,
          method_variant, method_version, domain, variant, status,
          trust_level, trust_reason, git_branch, git_commit, created_at,
          updated_at, extra_json, notes
        )
        values (?, ?, ?, 'sft', ?, ?, ?, 'math', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(experiment_key) do update set
          project_id=excluded.project_id,
          display_name=excluded.display_name,
          method=excluded.method,
          method_family=excluded.method_family,
          method_variant=excluded.method_variant,
          method_version=excluded.method_version,
          domain=excluded.domain,
          variant=excluded.variant,
          status=excluded.status,
          trust_level=excluded.trust_level,
          trust_reason=excluded.trust_reason,
          git_branch=excluded.git_branch,
          git_commit=excluded.git_commit,
          updated_at=excluded.updated_at,
          extra_json=excluded.extra_json,
          notes=coalesce(excluded.notes, experiments.notes)
        """,
        (
            project_id,
            experiment_key(run),
            display,
            family,
            method_variant,
            method_version,
            f"stage{stage}_beta_{run['beta']:.1f}",
            run["status"],
            run["trust_level"],
            trust_reason,
            BRANCH,
            GIT_COMMIT,
            "2026-05-28T00:00:00+08:00" if stage == 1 else "2026-05-29T00:00:00+08:00",
            now,
            json.dumps(extra, ensure_ascii=False),
            run.get("notes"),
        ),
    )
    return int(conn.execute("select id from experiments where experiment_key=?", (experiment_key(run),)).fetchone()["id"])


def add_tag(conn: sqlite3.Connection, entity_type: str, entity_id: int, tag: str) -> None:
    conn.execute("insert or ignore into entity_tags(entity_type, entity_id, tag) values (?, ?, ?)", (entity_type, entity_id, tag))


def add_training_metric(conn: sqlite3.Connection, tr_id: int, name: str, value: Any, step: int | None, scope: str = "training", notes: str | None = None) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    conn.execute(
        """
        insert into training_metrics(training_run_id, metric_name, metric_value, step, metric_scope, notes)
        values (?, ?, ?, ?, ?, ?)
        on conflict(training_run_id, metric_name, step, metric_scope) do update set
          metric_value=excluded.metric_value,
          notes=coalesce(excluded.notes, training_metrics.notes)
        """,
        (tr_id, name, float(value), step, scope, notes),
    )


def add_artifact(conn: sqlite3.Connection, exp_id: int, tr_id: int | None, model_id: int | None, kind: str, path: str, desc: str, notes: str | None = None) -> None:
    key = slug(f"{kind}.{path}.{exp_id}.{tr_id}.{model_id}")
    p = Path(path)
    stat = p.stat() if p.exists() else None
    mtime = dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).replace(microsecond=0).isoformat() if stat else None
    conn.execute(
        """
        insert into artifacts(
          artifact_key, experiment_id, training_run_id, model_id, artifact_kind,
          path, description, sha256, exists_checked_at, artifact_exists,
          size_bytes, mtime, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(artifact_key) do update set
          path=excluded.path,
          description=excluded.description,
          sha256=coalesce(excluded.sha256, artifacts.sha256),
          exists_checked_at=excluded.exists_checked_at,
          artifact_exists=excluded.artifact_exists,
          size_bytes=excluded.size_bytes,
          mtime=excluded.mtime,
          notes=coalesce(excluded.notes, artifacts.notes)
        """,
        (
            key,
            exp_id,
            tr_id,
            model_id,
            kind,
            path,
            desc,
            file_sha256(p) if p.exists() else None,
            utc_now(),
            1 if p.exists() else 0,
            stat.st_size if stat else None,
            mtime,
            notes,
        ),
    )


def add_source_record(conn: sqlite3.Connection, path: Path, section: str, kind: str, record_id: int, entity_key: str, notes: str | None = None) -> None:
    mtime, size, sha = source_meta(path)
    conn.execute(
        """
        insert into source_records(
          source_path, source_type, source_section, source_mtime, source_size,
          source_sha256, imported_at, importer, extractor_version,
          record_kind, record_id, entity_table, entity_key, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, 'v1', ?, ?, ?, ?, ?)
        on conflict(importer, source_path, source_section, record_kind, entity_key) do update set
          source_mtime=excluded.source_mtime,
          source_size=excluded.source_size,
          source_sha256=coalesce(excluded.source_sha256, source_records.source_sha256),
          imported_at=excluded.imported_at,
          record_id=excluded.record_id,
          notes=coalesce(excluded.notes, source_records.notes)
        """,
        (str(path), "jsonl" if path.suffix == ".jsonl" else "json", section, mtime, size, sha, utc_now(), IMPORTER, kind, record_id, kind, entity_key, notes),
    )


def upsert_training_run(
    conn: sqlite3.Connection,
    exp_id: int,
    run: dict[str, Any],
    train_dataset_id: int,
    output_model_id: int,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    latest: int | None,
    metrics_path: Path,
) -> int:
    final = rows[-1]["data"]
    val_final = final_validation_row(rows)
    hyper = {
        "stage": run["stage"],
        "loss_mode": "wdl_sft",
        "rollout_n": 8,
        "train_prompt_batch_size": 64,
        "actor_ppo_epochs": 1,
        "ppo_mini_batch_size": 512 if run["stage"] == 2 else 8,
        "shuffle": False if run["stage"] == 2 else None,
        "rollout_source": "model2" if run["stage"] == 2 else "actor",
        "stage1_run": run.get("stage1_run"),
        "stage1_step": run.get("stage1_step"),
        "train_dataset": run.get("train_dataset"),
        "chain_group": run.get("chain_group"),
        "best_checkpoint": best,
    }
    conn.execute(
        """
        insert into training_runs(
          training_run_key, experiment_id, output_model_id, train_dataset_id,
          method, framework, beta, learning_rate, num_epochs,
          effective_batch_size, max_length, warmup_ratio, weight_decay,
          distributed_backend, hyperparams_json, num_gpus, total_steps,
          final_train_loss, raw_summary_path, wandb_run, git_branch,
          git_commit, extra_json, notes
        )
        values (?, ?, ?, ?, 'sft', 'verl', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(training_run_key) do update set
          experiment_id=excluded.experiment_id,
          output_model_id=excluded.output_model_id,
          train_dataset_id=excluded.train_dataset_id,
          beta=excluded.beta,
          learning_rate=excluded.learning_rate,
          num_epochs=excluded.num_epochs,
          effective_batch_size=excluded.effective_batch_size,
          max_length=excluded.max_length,
          warmup_ratio=excluded.warmup_ratio,
          weight_decay=excluded.weight_decay,
          distributed_backend=excluded.distributed_backend,
          hyperparams_json=excluded.hyperparams_json,
          num_gpus=excluded.num_gpus,
          total_steps=excluded.total_steps,
          final_train_loss=excluded.final_train_loss,
          raw_summary_path=excluded.raw_summary_path,
          wandb_run=excluded.wandb_run,
          git_branch=excluded.git_branch,
          git_commit=excluded.git_commit,
          extra_json=excluded.extra_json,
          notes=coalesce(excluded.notes, training_runs.notes)
        """,
        (
            training_run_key(run),
            exp_id,
            output_model_id,
            train_dataset_id,
            run["beta"],
            5e-7,
            2 if run["stage"] == 2 else 1,
            512,
            4596,
            5 / max(len(rows), 1),
            0.1,
            "fsdp+ray+vllm",
            json.dumps(hyper, ensure_ascii=False),
            8,
            latest or rows[-1]["step"],
            final.get("actor/wdl_sft_loss_total"),
            str(metrics_path),
            f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run['wandb_id']}",
            BRANCH,
            GIT_COMMIT,
            json.dumps({"final_validation_step": val_final["step"] if val_final else None}, ensure_ascii=False),
            run.get("notes"),
        ),
    )
    return int(conn.execute("select id from training_runs where training_run_key=?", (training_run_key(run),)).fetchone()["id"])


def add_quality_flag(conn: sqlite3.Connection, entity_type: str, entity_id: int, flag: str, severity: str, reason: str, notes: str | None = None) -> None:
    conn.execute(
        """
        insert into quality_flags(entity_type, entity_id, flag, severity, reason, notes)
        values (?, ?, ?, ?, ?, ?)
        on conflict(entity_type, entity_id, flag) do update set
          severity=excluded.severity,
          reason=excluded.reason,
          notes=coalesce(excluded.notes, quality_flags.notes)
        """,
        (entity_type, entity_id, flag, severity, reason, notes),
    )


def add_validation_check(
    conn: sqlite3.Connection,
    check_name: str,
    source_path: Path,
    source_value: Any,
    database_value: Any,
    notes: str | None = None,
) -> None:
    try:
        passed = abs(float(source_value) - float(database_value)) < 1e-9
    except (TypeError, ValueError):
        passed = str(source_value) == str(database_value)
    conn.execute(
        """
        insert into validation_checks(check_name, source_path, source_value, database_value, passed, checked_at, notes)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (check_name, str(source_path), str(source_value), str(database_value), int(passed), utc_now(), notes),
    )


def db_metric(conn: sqlite3.Connection, tr_id: int, metric_name: str, scope: str, step: int | None) -> float | None:
    row = conn.execute(
        """
        select metric_value
        from training_metrics
        where training_run_id=? and metric_name=? and metric_scope=? and step is ?
        """,
        (tr_id, metric_name, scope, step),
    ).fetchone()
    return float(row["metric_value"]) if row else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/data-1/experiment_registry/experiment_registry.sqlite")
    args = parser.parse_args()
    db = args.db
    check_release_gate([run["run_name"] for run in RUNS])
    init_db(db)
    git_commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    global GIT_COMMIT
    GIT_COMMIT = git_commit

    with connect(db) as conn:
        project_id = upsert_project(conn)
        ds_stage1 = upsert_dataset(conn, "math.ensemblellm_train_rl_format", "EnsembleLLM train RL format", TRAIN_STAGE1, "train")
        ds_stage1_boxed = upsert_dataset(conn, "math.ensemblellm_staged_v1_train_rl_format_boxed_prompt", "EnsembleLLM staged-v1 boxed-prompt train RL format", TRAIN_STAGE1_BOXED, "train", "Train prompts aligned to validation boxed final-answer instruction.")
        ds_stage2 = upsert_dataset(conn, "math.ensemblellm_staged_v1_stage2_nonoverlap_seed20260528", "EnsembleLLM staged-v1 Stage-2 non-overlap shard", TRAIN_STAGE2, "train", "Skip first 9600 eligible prompts and use next 4800 prompts.")
        ds_stage2_boxed = upsert_dataset(conn, "math.ensemblellm_staged_v1_stage2_boxed_nonoverlap_seed20260528", "EnsembleLLM staged-v1 boxed Stage-2 non-overlap shard", TRAIN_STAGE2_BOXED, "train", "Boxed-prompt Stage-2 shard; skip first 9600 eligible prompts and use next 4800 prompts.")
        ds_math500 = upsert_dataset(conn, "math.math500_test_with_system_prompt", "MATH-500 with system prompt", MATH500, "validation")
        ds_aime2025 = upsert_dataset(conn, "math.aime2025_with_system_prompt", "AIME-2025 with system prompt", AIME2025, "validation")
        train_dataset_ids = {
            "stage1": ds_stage1,
            "stage1_boxed": ds_stage1_boxed,
            "stage2": ds_stage2,
            "stage2_boxed": ds_stage2_boxed,
        }

        for run in RUNS:
            metrics_path = METRICS_DIR / f"{run['run_name']}.jsonl"
            best_path = CHECKPOINT_ROOT / run["run_name"] / "best_checkpoint.json"
            if not metrics_path.exists() or not best_path.exists():
                raise FileNotFoundError((metrics_path, best_path))
            rows = read_jsonl(metrics_path)
            best = load_json(best_path)
            latest = latest_step(run["run_name"])
            exp_id = upsert_experiment(conn, project_id, run, best, latest)
            for tag in ["staged_v1", f"stage{run['stage']}", f"beta_{run['beta']:.1f}", "wdl_sft"]:
                add_tag(conn, "experiment", exp_id, tag)
            if run.get("chain_group"):
                add_tag(conn, "experiment", exp_id, run["chain_group"])
            if run.get("train_dataset") in {"stage1_boxed", "stage2_boxed"}:
                add_tag(conn, "experiment", exp_id, "boxed_matched")
            if run["stage"] == 2:
                add_tag(conn, "experiment", exp_id, "model2_rollout_fused_loss")
                add_tag(conn, "experiment", exp_id, "nonoverlap_stage2_shard")

            best_model_path = str(Path(best["checkpoint_dir"]) / "actor")
            if run["stage"] == 1:
                if run.get("chain_group") == "boxed_matched_beta0":
                    merged_path = "/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta0-best"
                elif run.get("chain_group") == "boxed_matched_beta01":
                    merged_path = "/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta01-best"
                elif run.get("chain_group") == "plateau_p60_beta0":
                    merged_path = "/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta0-step60"
                elif run.get("chain_group") == "plateau_p60_beta01":
                    merged_path = "/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta01-step60"
                else:
                    merged_path = str(MODEL_WEIGHT_ROOT / run["run_name"] / f"step_{best['step']}")
                role = "stage1_model2_candidate"
                model_path = merged_path if Path(merged_path).exists() else best_model_path
            else:
                role = "stage2_joint_checkpoint"
                model_path = best_model_path
            model_id = upsert_model(
                conn,
                project_id=project_id,
                model_key=slug(f"{run['run_name']}.best.step_{best['step']}"),
                display_name=f"{run['run_name']} best step {best['step']}",
                path=model_path,
                role=role,
                step=int(best["step"]),
                is_best=1,
                is_latest=1 if latest == best["step"] else 0,
                selection_metric=best["metric_key"],
                selection_value=float(best["metric_value"]),
                extra={"checkpoint_dir": best["checkpoint_dir"], "raw_actor_path": best_model_path},
            )
            tr_id = upsert_training_run(
                conn,
                exp_id,
                run,
                dataset_for_run(run, train_dataset_ids),
                model_id,
                rows,
                best,
                latest,
                metrics_path,
            )
            link_training_dataset(conn, tr_id, dataset_for_run(run, train_dataset_ids), "train")
            link_training_dataset(conn, tr_id, ds_math500, "validation")
            link_training_dataset(conn, tr_id, ds_aime2025, "validation")
            if run["stage"] == 2:
                parent_key = experiment_key({
                    "stage": 1,
                    "beta": run["stage1_beta"],
                    "run_name": run["stage1_run"],
                    "train_dataset": "stage1_boxed" if run.get("train_dataset") == "stage2_boxed" else None,
                    "chain_group": run.get("chain_group"),
                })
                parent = conn.execute("select id from experiments where experiment_key=?", (parent_key,)).fetchone()
                if parent:
                    link_experiments(
                        conn,
                        exp_id,
                        int(parent["id"]),
                        "stage2_from_stage1",
                        f"Stage 2 initializes Model2 from Stage 1 beta={run['stage1_beta']:.1f} step {run['stage1_step']}.",
                    )
            final = rows[-1]["data"]
            best_math = best_row(rows, "val-core/HuggingFaceH4/MATH-500/acc/mean@3")
            best_aime = best_row(rows, "val-core/aime25/acc/mean@3")
            final_val = final_validation_row(rows)
            for key in TRAIN_METRIC_KEYS:
                add_training_metric(conn, tr_id, f"final/{key}", final.get(key), rows[-1]["step"], "final")
            for key in VAL_METRIC_KEYS:
                peak = best_row(rows, key)
                add_training_metric(
                    conn,
                    tr_id,
                    f"peak_metric/{key}",
                    peak["data"].get(key),
                    peak["step"],
                    "peak_metric",
                    "Metric-specific maximum over validation rows.",
                )
                add_training_metric(conn, tr_id, f"best_math_step/{key}", best_math["data"].get(key), best_math["step"], "best_math_step")
                add_training_metric(conn, tr_id, f"best_aime_step/{key}", best_aime["data"].get(key), best_aime["step"], "best_aime_step")
                if final_val:
                    add_training_metric(conn, tr_id, f"final_validation/{key}", final_val["data"].get(key), final_val["step"], "final_validation")
            add_training_metric(conn, tr_id, "best_checkpoint/selection_metric", best["metric_value"], best["step"], "best_checkpoint")
            add_training_metric(conn, tr_id, "latest_checkpoint/step", latest, latest, "checkpoint")

            add_validation_check(
                conn,
                f"staged_v1_{run['run_name']}_best_checkpoint_selection_metric",
                best_path,
                best["metric_value"],
                db_metric(conn, tr_id, "best_checkpoint/selection_metric", "best_checkpoint", best["step"]),
                "Source best_checkpoint.json metric_value equals registry training_metrics.",
            )
            add_validation_check(
                conn,
                f"staged_v1_{run['run_name']}_latest_checkpoint_step",
                CHECKPOINT_ROOT / run["run_name"] / "latest_checkpointed_iteration.txt",
                latest,
                db_metric(conn, tr_id, "latest_checkpoint/step", "checkpoint", latest),
                "Source latest checkpoint marker equals registry training_metrics.",
            )
            if final_val:
                final_math_key = "val-core/HuggingFaceH4/MATH-500/acc/mean@3"
                add_validation_check(
                    conn,
                    f"staged_v1_{run['run_name']}_final_math500_mean3",
                    metrics_path,
                    final_val["data"].get(final_math_key),
                    db_metric(conn, tr_id, f"final_validation/{final_math_key}", "final_validation", final_val["step"]),
                    "Final validation MATH-500 mean@3 in metrics JSONL equals registry training_metrics.",
                )
            peak_math_key = "val-core/HuggingFaceH4/MATH-500/acc/mean@3"
            add_validation_check(
                conn,
                f"staged_v1_{run['run_name']}_peak_math500_mean3",
                metrics_path,
                best_math["data"].get(peak_math_key),
                db_metric(conn, tr_id, f"peak_metric/{peak_math_key}", "peak_metric", best_math["step"]),
                "Peak validation MATH-500 mean@3 in metrics JSONL equals registry training_metrics.",
            )

            add_artifact(conn, exp_id, tr_id, model_id, "training_metrics_jsonl", str(metrics_path), "Training metrics JSONL.")
            add_artifact(conn, exp_id, tr_id, model_id, "best_checkpoint_json", str(best_path), "Best checkpoint metadata.")
            add_artifact(conn, exp_id, tr_id, model_id, "checkpoint_dir", best["checkpoint_dir"], "Best checkpoint directory.")
            log_path = REPO / f"recipe/on_policy_wdl_sft/staged_v1/{run['run_name']}.log"
            add_artifact(conn, exp_id, tr_id, model_id, "training_log", str(log_path), "Training stdout/stderr log.")
            for extra_log in run.get("extra_logs", []):
                add_artifact(conn, exp_id, tr_id, model_id, "training_log", str(REPO / extra_log), "Additional training stdout/stderr log.")
            manifest = manifest_for_run(run)
            if manifest:
                add_artifact(conn, exp_id, tr_id, None, "dataset_manifest", str(manifest), "Training dataset manifest.")
            if run.get("stage1_source_json"):
                add_artifact(conn, exp_id, tr_id, None, "stage1_source_json", run["stage1_source_json"], "Fixed merged Model2 provenance from Stage 1.")

            add_source_record(conn, metrics_path, "full", "training_runs", tr_id, training_run_key(run))
            add_source_record(conn, best_path, "best_checkpoint", "models", model_id, slug(f"{run['run_name']}.best.step_{best['step']}"))
            if run.get("stage1_source_json"):
                add_source_record(conn, Path(run["stage1_source_json"]), "stage1_source", "training_runs", tr_id, training_run_key(run))

            if run["stage"] == 2:
                final_math = final_val["data"].get("val-core/HuggingFaceH4/MATH-500/acc/mean@3") if final_val else None
                best_math_value = best_math["data"].get("val-core/HuggingFaceH4/MATH-500/acc/mean@3")
                if isinstance(final_math, (int, float)) and isinstance(best_math_value, (int, float)) and final_math < best_math_value - 0.3:
                    add_quality_flag(conn, "training_run", tr_id, "online_validation_collapse", "warning", "Final MATH-500 mean@3 is far below early best checkpoint.")
            if run["status"] != "completed":
                add_quality_flag(conn, "training_run", tr_id, "interrupted_run", "warning", "Run did not complete the intended beta-grid schedule.")

        conn.commit()
    print("imported", len(RUNS), "staged-v1 runs")


if __name__ == "__main__":
    main()
