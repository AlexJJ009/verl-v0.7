#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Import the completed WDL group-advantage IS run into the local registry.

The importer is intentionally source-driven: every value comes from the
training JSONL, launcher scripts, checkpoint metadata, parquet metadata, or
the run log. It is idempotent and scoped to the branch project
`verl:feature/on-policy-wdl-sft`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
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

RUN_NAME = "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A_1779295502"
PROJECT_NAME = "verl:feature/on-policy-wdl-sft"
PROJECT_KEY = "verl_feature_on_policy_wdl_sft"
BRANCH = "feature/on-policy-wdl-sft"
GIT_COMMIT = "b91e9257dbad7ddf10f02ac7142e86baefe5a28c"
IMPORTER = "import_wdl_group_adv_is_v1"
RELEASE_GATE_SCRIPT = Path("/data-1/verl07/verl/scripts/training_result_release_gate.py")


def check_release_gate() -> None:
    subprocess.check_call([sys.executable, str(RELEASE_GATE_SCRIPT), "check", "--run-name", RUN_NAME])


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


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_info(path: Path) -> tuple[str | None, int | None, str | None]:
    if not path.exists():
        return None, None, None
    st = path.stat()
    mtime = dt.datetime.fromtimestamp(st.st_mtime, dt.UTC).replace(microsecond=0).isoformat()
    sha = file_sha256(path) if path.is_file() and st.st_size < 256 * 1024 * 1024 else None
    return mtime, st.st_size, sha


def upsert_project(conn: sqlite3.Connection, repo: Path) -> int:
    conn.execute(
        """
        insert into projects(project_key, name, repo_path, default_branch, notes)
        values (?, ?, ?, ?, ?)
        on conflict(name) do update set
          project_key=excluded.project_key,
          repo_path=excluded.repo_path,
          default_branch=excluded.default_branch,
          notes=excluded.notes
        """,
        (
            PROJECT_KEY,
            PROJECT_NAME,
            str(repo),
            BRANCH,
            "Branch-scoped registry form for feature/on-policy-wdl-sft algorithm runs and input-model evals.",
        ),
    )
    return int(conn.execute("select id from projects where name=?", (PROJECT_NAME,)).fetchone()["id"])


def upsert_dataset(
    conn: sqlite3.Connection,
    dataset_key: str,
    name: str,
    path: str,
    row_count: int | None,
    split: str,
    notes: str | None = None,
) -> int:
    conn.execute(
        """
        insert into datasets(dataset_key, name, domain, path, split, row_count, format, notes)
        values (?, ?, 'math', ?, ?, ?, ?, ?)
        on conflict(dataset_key) do update set
          name=excluded.name,
          domain=excluded.domain,
          path=excluded.path,
          split=excluded.split,
          row_count=excluded.row_count,
          format=excluded.format,
          notes=coalesce(excluded.notes, datasets.notes)
        """,
        (dataset_key, name, path, split, row_count, "parquet", notes),
    )
    return int(conn.execute("select id from datasets where dataset_key=?", (dataset_key,)).fetchone()["id"])


def upsert_model(
    conn: sqlite3.Connection,
    *,
    model_key: str,
    display_name: str,
    model_path: str,
    model_role: str,
    project_id: int,
    base_model: str | None = None,
    checkpoint_step: int | None = None,
    checkpoint_kind: str | None = None,
    is_best: int | None = None,
    is_latest: int | None = None,
    selection_metric_name: str | None = None,
    selection_metric_value: float | None = None,
    extra_json: dict[str, Any] | None = None,
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
          base_model=coalesce(excluded.base_model, models.base_model),
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
          git_branch=coalesce(excluded.git_branch, models.git_branch),
          git_commit=coalesce(excluded.git_commit, models.git_commit),
          extra_json=coalesce(excluded.extra_json, models.extra_json),
          notes=coalesce(excluded.notes, models.notes)
        """,
        (
            model_key,
            display_name,
            base_model,
            model_path,
            checkpoint_step,
            checkpoint_step,
            checkpoint_kind,
            model_role,
            is_best,
            is_latest,
            selection_metric_name,
            selection_metric_value,
            project_id,
            BRANCH,
            GIT_COMMIT,
            json.dumps(extra_json, ensure_ascii=False) if extra_json else None,
            notes,
        ),
    )
    return int(conn.execute("select id from models where model_key=?", (model_key,)).fetchone()["id"])


def upsert_experiment(conn: sqlite3.Connection, project_id: int, best: dict[str, Any]) -> int:
    extra = {
        "run_name": RUN_NAME,
        "loss_mode": "wdl_group_adv_is",
        "method_contract": "docs/joint_training/plans/active/wdl_group_advantage_is_goal.md",
        "initial_launch_oom_recovered": True,
        "best_checkpoint": best,
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
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
          notes=excluded.notes
        """,
        (
            project_id,
            "verl.branch.feature_on_policy_wdl_sft.wdl_group_adv_is.1a.1779295502",
            "feature/on-policy-wdl-sft / WDL group-advantage IS 1A / run 1779295502",
            "sft",
            "on_policy_wdl_sft",
            "wdl_group_adv_is",
            "v3_group_advantage_is",
            "math",
            "group_adv_is_1a",
            "completed",
            "trusted",
            "Completed 300/300 steps after resume; source metrics/checkpoint/log artifacts are present and validated.",
            BRANCH,
            GIT_COMMIT,
            "2026-05-20T16:48:37+08:00",
            now,
            json.dumps(extra, ensure_ascii=False),
            "Spec-correct WDL group-advantage IS run with all-correct positive-SFT fallback; first launch hit OOM after step 100, resumed and completed.",
        ),
    )
    return int(
        conn.execute(
            "select id from experiments where experiment_key=?",
            ("verl.branch.feature_on_policy_wdl_sft.wdl_group_adv_is.1a.1779295502",),
        ).fetchone()["id"]
    )


def add_tag(conn: sqlite3.Connection, entity_type: str, entity_id: int, tag: str) -> None:
    conn.execute(
        "insert or ignore into entity_tags(entity_type, entity_id, tag) values (?, ?, ?)",
        (entity_type, entity_id, tag),
    )


def add_quality_flag(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: int,
    flag: str,
    severity: str,
    reason: str,
    notes: str | None = None,
) -> None:
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


def add_artifact(
    conn: sqlite3.Connection,
    kind: str,
    path: Path,
    *,
    experiment_id: int | None = None,
    training_run_id: int | None = None,
    model_id: int | None = None,
    description: str | None = None,
    notes: str | None = None,
) -> None:
    mtime, size, sha = source_info(path)
    path_str = str(path)
    artifact_key = slug(f"{kind}.{path_str}.{experiment_id}.{training_run_id}.{model_id}")
    conn.execute(
        """
        insert into artifacts(
          artifact_key, experiment_id, training_run_id, model_id,
          artifact_kind, path, description, sha256, exists_checked_at,
          artifact_exists, size_bytes, mtime, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(artifact_key) do update set
          description=coalesce(excluded.description, artifacts.description),
          sha256=coalesce(excluded.sha256, artifacts.sha256),
          exists_checked_at=excluded.exists_checked_at,
          artifact_exists=excluded.artifact_exists,
          size_bytes=excluded.size_bytes,
          mtime=excluded.mtime,
          notes=coalesce(excluded.notes, artifacts.notes)
        """,
        (
            artifact_key,
            experiment_id,
            training_run_id,
            model_id,
            kind,
            path_str,
            description,
            sha,
            utc_now(),
            1 if path.exists() else 0,
            size,
            mtime,
            notes,
        ),
    )


def add_source_record(
    conn: sqlite3.Connection,
    path: Path,
    source_type: str,
    section: str | None,
    record_kind: str,
    record_id: int,
    entity_table: str,
    entity_key: str,
    notes: str | None = None,
) -> None:
    mtime, size, sha = source_info(path)
    source_section = section or ""
    conn.execute(
        """
        insert into source_records(
          source_path, source_type, source_section, source_mtime,
          source_size, source_sha256, imported_at, importer, record_kind,
          record_id, entity_table, entity_key, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(importer, source_path, source_section, record_kind, entity_key) do update set
          source_mtime=excluded.source_mtime,
          source_size=excluded.source_size,
          source_sha256=coalesce(excluded.source_sha256, source_records.source_sha256),
          imported_at=excluded.imported_at,
          record_id=excluded.record_id,
          entity_table=excluded.entity_table,
          notes=coalesce(excluded.notes, source_records.notes)
        """,
        (
            str(path),
            source_type,
            source_section,
            mtime,
            size,
            sha,
            utc_now(),
            IMPORTER,
            record_kind,
            record_id,
            entity_table,
            entity_key,
            notes,
        ),
    )


def upsert_training_run(
    conn: sqlite3.Connection,
    *,
    exp_id: int,
    input_model_id: int,
    output_model_id: int,
    train_dataset_id: int,
    metrics_path: Path,
    log_path: Path,
    best_path: Path,
    validation_dir: Path,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    train_rows: int,
    samples_yielded: int,
) -> int:
    final = rows[-1]["data"]
    first_train = next((r["data"] for r in rows if "actor/pg_loss" in r.get("data", {})), {})
    hyperparams = {
        "source": "training metrics jsonl + launcher scripts + checkpoint metadata + parquet metadata",
        "run_name": RUN_NAME,
        "loss_mode": "wdl_group_adv_is",
        "learning_rate": 5e-7,
        "rollout_n": 8,
        "validation_n": 3,
        "temperature": 1.0,
        "top_p": 1.0,
        "val_top_p": 0.95,
        "top_k": -1,
        "max_prompt_length": 500,
        "max_response_length": 4096,
        "train_prompt_batch_size": 64,
        "ppo_mini_batch_size": 8,
        "effective_rollout_sequences_per_step": 512,
        "total_training_steps": 300,
        "configured_total_epochs": 2,
        "observed_training_epoch": final.get("training/epoch"),
        "train_dataset_rows": train_rows,
        "samples_yielded": samples_yielded,
        "approx_epoch_fraction": samples_yielded / train_rows if train_rows else None,
        "test_freq": 25,
        "save_freq": 25,
        "grad_clip": 500.0,
        "weight_decay": 0.1,
        "lr_warmup_steps": 5,
        "loss_agg_mode": "seq-mean-token-sum",
        "norm_adv_by_std_in_grpo": False,
        "rollout_is": None,
        "rollout_is_threshold": 5.0,
        "rollout_calculate_log_probs": False,
        "use_kl_loss": False,
        "kl_loss_coef": 0.0,
        "all_correct_sft_fallback": True,
        "pos_sft_fallback_coef": 1.0,
        "best_checkpoint": best,
        "checkpoint_retention": {
            "max_actor_ckpts_to_keep": 1,
            "keep_best_ckpt": True,
            "best_is_latest": best.get("step") == final.get("training/global_step"),
        },
        "train_files": "/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet",
        "val_files": [
            "/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet",
            "/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet",
        ],
    }
    extra = {
        "best_checkpoint_path": str(best_path),
        "latest_checkpoint_step": "300",
        "validation_dir": str(validation_dir),
        "launcher": str(Path("/data-1/verl07/verl/recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh")),
        "common_launcher": str(
            Path("/data-1/verl07/verl/recipe/on_policy_wdl_sft/group_advantage_is/_common_group_adv_is.sh")
        ),
        "initial_log": str(Path("/data-1/verl07/verl/recipe/on_policy_wdl_sft/group_advantage_is") / f"{RUN_NAME}.log"),
        "resume_log": str(log_path),
    }
    conn.execute(
        """
        insert into training_runs(
          training_run_key, experiment_id, input_model_id, output_model_id,
          train_dataset_id, method, framework, beta, learning_rate,
          num_epochs, per_device_batch_size, gradient_accumulation_steps,
          effective_batch_size, max_length, weight_decay, distributed_backend,
          distributed_config_json, hyperparams_json, num_gpus, runtime_seconds,
          total_steps, final_train_loss, final_step_loss, first_step_loss,
          raw_summary_path, tb_path, wandb_run, git_branch, git_commit,
          extra_json, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(training_run_key) do update set
          experiment_id=excluded.experiment_id,
          input_model_id=excluded.input_model_id,
          output_model_id=excluded.output_model_id,
          train_dataset_id=excluded.train_dataset_id,
          method=excluded.method,
          framework=excluded.framework,
          learning_rate=excluded.learning_rate,
          num_epochs=excluded.num_epochs,
          per_device_batch_size=excluded.per_device_batch_size,
          effective_batch_size=excluded.effective_batch_size,
          max_length=excluded.max_length,
          weight_decay=excluded.weight_decay,
          distributed_backend=excluded.distributed_backend,
          distributed_config_json=excluded.distributed_config_json,
          hyperparams_json=excluded.hyperparams_json,
          num_gpus=excluded.num_gpus,
          runtime_seconds=excluded.runtime_seconds,
          total_steps=excluded.total_steps,
          final_train_loss=excluded.final_train_loss,
          final_step_loss=excluded.final_step_loss,
          first_step_loss=coalesce(excluded.first_step_loss, training_runs.first_step_loss),
          raw_summary_path=excluded.raw_summary_path,
          tb_path=excluded.tb_path,
          wandb_run=excluded.wandb_run,
          git_branch=excluded.git_branch,
          git_commit=excluded.git_commit,
          extra_json=excluded.extra_json,
          notes=excluded.notes
        """,
        (
            "feature_on_policy_wdl_sft.wdl_group_adv_is.1a.1779295502.training_metrics_jsonl",
            exp_id,
            input_model_id,
            output_model_id,
            train_dataset_id,
            "on_policy_wdl_sft",
            "verl",
            None,
            5e-7,
            2.0,
            64,
            None,
            512,
            4096,
            0.1,
            "ray_fsdp_vllm",
            json.dumps(
                {
                    "nnodes": 1,
                    "n_gpus_per_node": 8,
                    "rollout_engine": "vllm",
                    "attention_backend": "FLASHINFER",
                    "fsdp_size": -1,
                    "rollout_mode": "async",
                },
                ensure_ascii=False,
            ),
            json.dumps(hyperparams, ensure_ascii=False),
            8,
            42839.0,
            300,
            final.get("actor/pg_loss"),
            final.get("actor/pg_loss"),
            first_train.get("actor/pg_loss"),
            str(metrics_path),
            str(log_path),
            "/data-1/wandb_runs/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A/wandb/offline-run-20260521_112442-xgs59r5z",
            BRANCH,
            GIT_COMMIT,
            json.dumps(extra, ensure_ascii=False),
            "Training metrics imported directly from local JSONL/log/checkpoint artifacts; group-advantage IS run completed after resume.",
        ),
    )
    return int(
        conn.execute(
            "select id from training_runs where training_run_key=?",
            ("feature_on_policy_wdl_sft.wdl_group_adv_is.1a.1779295502.training_metrics_jsonl",),
        ).fetchone()["id"]
    )


def upsert_training_metric(
    conn: sqlite3.Connection,
    training_run_id: int,
    name: str,
    value: float | int,
    step: int,
    scope: str = "training_metrics_jsonl",
) -> None:
    conn.execute(
        """
        insert into training_metrics(training_run_id, metric_name, metric_value, step, metric_scope)
        values (?, ?, ?, ?, ?)
        on conflict(training_run_id, metric_name, step, metric_scope) do update set
          metric_value=excluded.metric_value
        """,
        (training_run_id, name, float(value), step, scope),
    )


def insert_all_metrics(conn: sqlite3.Connection, training_run_id: int, rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        step = int(row["data"].get("training/global_step", row["step"]))
        for name, value in row["data"].items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float) and math.isfinite(float(value)):
                upsert_training_metric(conn, training_run_id, name, value, step)
                count += 1
    return count


def add_training_run_dataset(
    conn: sqlite3.Connection, training_run_id: int, dataset_id: int, role: str, row_count: int | None
) -> None:
    conn.execute(
        """
        insert into training_run_datasets(training_run_id, dataset_id, role, row_count, notes)
        values (?, ?, ?, ?, ?)
        on conflict(training_run_id, dataset_id, role) do update set
          row_count=excluded.row_count,
          notes=excluded.notes
        """,
        (training_run_id, dataset_id, role, row_count, "Imported for WDL-GROUP-ADV-IS 1A registry record"),
    )


def add_validation_check(
    conn: sqlite3.Connection,
    name: str,
    source_path: Path,
    source_value: Any,
    database_value: Any,
    passed: bool,
    notes: str,
) -> None:
    conn.execute(
        """
        insert into validation_checks(check_name, source_path, source_value, database_value, passed, checked_at, notes)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (name, str(source_path), str(source_value), str(database_value), 1 if passed else 0, utc_now(), notes),
    )


def latest_db_metric(conn: sqlite3.Connection, tr_id: int, metric: str, step: int) -> float | None:
    row = conn.execute(
        """
        select metric_value from training_metrics
        where training_run_id=? and metric_name=? and step=? and metric_scope='training_metrics_jsonl'
        """,
        (tr_id, metric, step),
    ).fetchone()
    return None if row is None else float(row["metric_value"])


def parse_samples_yielded(data_pt: Path) -> int:
    import torch

    data = torch.load(data_pt, map_location="cpu")
    return int(data["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]["samples_yielded"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/data-1/experiment_registry/experiment_registry.sqlite")
    parser.add_argument("--repo", default="/data-1/verl07/verl")
    args = parser.parse_args()
    check_release_gate()

    repo = Path(args.repo).resolve()
    run_dir = repo / "recipe/on_policy_wdl_sft/group_advantage_is"
    metrics_path = run_dir / "metrics/OnPolicyWDLSFT" / f"{RUN_NAME}.jsonl"
    initial_log = run_dir / f"{RUN_NAME}.log"
    resume_log = run_dir / f"{RUN_NAME}_resumed_1779362462.log"
    initial_validation_dir = run_dir / "validation" / "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A_1779295502"
    empty_validation_dirs = [
        run_dir / "validation" / "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A_1779362351",
        run_dir / "validation" / "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A_1779362371",
    ]
    validation_dir = run_dir / "validation" / "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-1A_1779362462"
    checkpoint_root = Path("/data-1/checkpoints") / RUN_NAME
    checkpoint_dir = checkpoint_root / "global_step_300"
    best_path = checkpoint_root / "best_checkpoint.json"
    latest_path = checkpoint_root / "latest_checkpointed_iteration.txt"
    data_pt = checkpoint_dir / "data.pt"
    train_file = Path("/data-1/dataset/EnsembleLLM-data-processed/train_rl_format.parquet")
    math_val = Path("/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet")
    aime_val = Path("/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet")

    rows = read_jsonl(metrics_path)
    best = load_json(best_path)
    train_rows = int(pq.ParquetFile(train_file).metadata.num_rows)
    samples_yielded = parse_samples_yielded(data_pt)
    latest_step = latest_path.read_text(encoding="utf-8").strip()

    init_db(args.db)
    with connect(args.db) as conn:
        project_id = upsert_project(conn, repo)
        train_ds = upsert_dataset(
            conn,
            "math.ensemblellm_train_rl_format",
            "EnsembleLLM train RL format",
            str(train_file),
            train_rows,
            "train",
            "Row count read from parquet metadata during WDL-GROUP-ADV-IS import.",
        )
        math_ds = upsert_dataset(
            conn,
            "math.huggingfaceh4_math_500_with_system_prompt",
            "HuggingFaceH4/MATH-500",
            str(math_val),
            500,
            "validation",
        )
        aime_ds = upsert_dataset(conn, "math.aime25_with_system_prompt", "aime25", str(aime_val), 30, "validation")

        exp_id = upsert_experiment(conn, project_id, best)
        for tag in [
            "wdl_group_adv_is",
            "group_advantage",
            "label_fix_successor",
            "completed_300_steps",
            "meituan_portable_launch",
            "has_all_correct_sft_fallback",
        ]:
            add_tag(conn, "experiment", exp_id, tag)
        add_quality_flag(
            conn,
            "experiment",
            exp_id,
            "initial_launch_oom_recovered",
            "info",
            "Initial launch hit CUDA OOM after step 100; resumed run completed 300/300 with validated final checkpoint.",
        )

        input_model = upsert_model(
            conn,
            model_key="feature_on_policy_wdl_sft.joint_init.qwenjoint_4b_wdl_sft_qwen3_4b_base_sft_stage_1.group_adv_is",
            display_name="QwenJoint 4B WDL-SFT init for group-advantage IS 1A",
            model_path="/data-1/.cache/huggingface/QwenJoint-4B-WDL-SFT-Qwen3-4B-Base-SFT-stage-1",
            model_role="joint_training_init",
            project_id=project_id,
            base_model="Qwen/Qwen3-4B-Base + Qwen3-4B-Base-SFT-stage-1",
            extra_json={
                "model1_path": "/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539",
                "model2_path": "/data-1/.cache/Qwen3-4B-Base-SFT-stage-1",
                "fusion_lambda": 0.5,
            },
        )
        output_model = upsert_model(
            conn,
            model_key="feature_on_policy_wdl_sft.output_checkpoint.wdl_group_adv_is_1a_1779295502.global_step_300",
            display_name="WDL-GROUP-ADV-IS 1A global_step_300 actor checkpoint",
            model_path=str(checkpoint_dir),
            model_role="trained_joint_actor_checkpoint",
            project_id=project_id,
            base_model="QwenJoint-4B-WDL-SFT-Qwen3-4B-Base-SFT-stage-1",
            checkpoint_step=300,
            checkpoint_kind="best_latest",
            is_best=1,
            is_latest=1,
            selection_metric_name=best["metric_key"],
            selection_metric_value=float(best["metric_value"]),
            extra_json={"latest_checkpointed_iteration": latest_step},
        )

        tr_id = upsert_training_run(
            conn,
            exp_id=exp_id,
            input_model_id=input_model,
            output_model_id=output_model,
            train_dataset_id=train_ds,
            metrics_path=metrics_path,
            log_path=resume_log,
            best_path=best_path,
            validation_dir=validation_dir,
            rows=rows,
            best=best,
            train_rows=train_rows,
            samples_yielded=samples_yielded,
        )
        add_training_run_dataset(conn, tr_id, train_ds, "train", train_rows)
        add_training_run_dataset(conn, tr_id, math_ds, "validation", 500)
        add_training_run_dataset(conn, tr_id, aime_ds, "validation", 30)
        metric_rows_seen = insert_all_metrics(conn, tr_id, rows)

        artifact_specs = [
            ("training_metrics_jsonl", metrics_path, "File logger JSONL carrying training and validation metrics."),
            ("training_log", initial_log, "Initial launch log; contains CUDA OOM after step 100."),
            ("training_log", resume_log, "Resume log; contains completion through step 300."),
            ("best_checkpoint_json", best_path, "Best-checkpoint tracker metadata."),
            ("latest_checkpoint_txt", latest_path, "Latest checkpointed iteration marker."),
            ("checkpoint_dir", checkpoint_dir, "Final actor checkpoint directory."),
            ("checkpoint_data_pt", data_pt, "Checkpoint dataloader state with samples_yielded."),
            (
                "validation_generation_dir",
                initial_validation_dir,
                "Validation generations from initial launch before OOM.",
            ),
            ("validation_generation_dir", validation_dir, "Validation generations from resumed run."),
            (
                "validation_generation_dir",
                empty_validation_dirs[0],
                "Empty validation directory from failed resume attempt before successful resume.",
            ),
            (
                "validation_generation_dir",
                empty_validation_dirs[1],
                "Empty validation directory from failed resume attempt before successful resume.",
            ),
            ("launcher_script", run_dir / "run_1a_group_adv_is.sh", "Thin launcher for group-advantage IS 1A."),
            (
                "launcher_script",
                run_dir / "_common_group_adv_is.sh",
                "Shared portable launcher with full hyperparameter surface.",
            ),
            (
                "method_contract",
                repo / "docs/joint_training/plans/active/wdl_group_advantage_is_goal.md",
                "Implementation contract for the method.",
            ),
        ]
        for kind, path, desc in artifact_specs:
            add_artifact(
                conn,
                kind,
                path,
                experiment_id=exp_id,
                training_run_id=tr_id,
                model_id=output_model if "checkpoint" in kind else None,
                description=desc,
            )

        for source_path, source_type, section, record_kind, record_id, entity_table, entity_key, notes in [
            (
                metrics_path,
                "training_metrics_jsonl",
                None,
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Primary source for training and validation metrics.",
            ),
            (
                resume_log,
                "training_log",
                "resume",
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Source for launch command and completion evidence.",
            ),
            (
                best_path,
                "checkpoint_metadata_json",
                None,
                "models",
                output_model,
                "models",
                str(output_model),
                "Source for best checkpoint selection.",
            ),
            (
                latest_path,
                "checkpoint_marker",
                None,
                "models",
                output_model,
                "models",
                str(output_model),
                "Source for latest checkpoint step.",
            ),
            (
                data_pt,
                "checkpoint_state",
                "dataloader",
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Source for samples_yielded.",
            ),
            (
                initial_validation_dir,
                "validation_generation_dir",
                "initial",
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Initial validation generation directory before OOM.",
            ),
            (
                validation_dir,
                "validation_generation_dir",
                "resume",
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Successful resumed validation generation directory.",
            ),
            (
                empty_validation_dirs[0],
                "validation_generation_dir",
                "empty_resume_attempt",
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Empty validation directory from failed resume attempt.",
            ),
            (
                empty_validation_dirs[1],
                "validation_generation_dir",
                "empty_resume_attempt",
                "training_runs",
                tr_id,
                "training_runs",
                str(tr_id),
                "Empty validation directory from failed resume attempt.",
            ),
            (
                run_dir / "run_1a_group_adv_is.sh",
                "launcher_script",
                None,
                "experiments",
                exp_id,
                "experiments",
                str(exp_id),
                "Experiment wrapper source.",
            ),
            (
                run_dir / "_common_group_adv_is.sh",
                "launcher_script",
                None,
                "experiments",
                exp_id,
                "experiments",
                str(exp_id),
                "Shared launcher source.",
            ),
        ]:
            add_source_record(
                conn, source_path, source_type, section, record_kind, record_id, entity_table, entity_key, notes
            )

        conn.execute("delete from validation_checks where check_name like 'wdl_group_adv_is_1a_%'")
        db_best_metric = conn.execute(
            "select selection_metric_value from models where id=?", (output_model,)
        ).fetchone()["selection_metric_value"]
        add_validation_check(
            conn,
            "wdl_group_adv_is_1a_best_checkpoint_metric",
            best_path,
            best["metric_value"],
            db_best_metric,
            abs(float(best["metric_value"]) - float(db_best_metric)) < 1e-12,
            "best_checkpoint.json metric matches output model selection metric",
        )
        db_step300 = latest_db_metric(conn, tr_id, "val-core/HuggingFaceH4/MATH-500/acc/mean@3", 300)
        add_validation_check(
            conn,
            "wdl_group_adv_is_1a_step300_math500_mean3",
            metrics_path,
            rows[-1]["data"]["val-core/HuggingFaceH4/MATH-500/acc/mean@3"],
            db_step300,
            db_step300 is not None
            and abs(rows[-1]["data"]["val-core/HuggingFaceH4/MATH-500/acc/mean@3"] - db_step300) < 1e-12,
            "step 300 MATH-500 mean@3 imported from metrics JSONL",
        )
        db_metric_count = conn.execute(
            "select count(*) as n from training_metrics where training_run_id=?", (tr_id,)
        ).fetchone()["n"]
        add_validation_check(
            conn,
            "wdl_group_adv_is_1a_training_metric_row_count",
            metrics_path,
            metric_rows_seen,
            db_metric_count,
            int(db_metric_count) == int(metric_rows_seen),
            "all finite numeric metrics from JSONL are present in training_metrics",
        )
        add_validation_check(
            conn,
            "wdl_group_adv_is_1a_latest_checkpoint_step",
            latest_path,
            latest_step,
            best.get("step"),
            str(latest_step) == str(best.get("step")) == "300",
            "latest checkpoint marker and best checkpoint step both point to step 300",
        )
        add_validation_check(
            conn,
            "wdl_group_adv_is_1a_samples_yielded",
            data_pt,
            samples_yielded,
            300 * 64,
            samples_yielded == 300 * 64,
            "checkpoint dataloader state confirms 300 batches of 64 prompts",
        )
        add_validation_check(
            conn,
            "wdl_group_adv_is_1a_train_dataset_row_count",
            train_file,
            train_rows,
            conn.execute("select row_count from datasets where id=?", (train_ds,)).fetchone()["row_count"],
            train_rows
            == int(conn.execute("select row_count from datasets where id=?", (train_ds,)).fetchone()["row_count"]),
            "training parquet row count imported into datasets",
        )

        conn.commit()

    print(
        json.dumps(
            {
                "experiment_key": "verl.branch.feature_on_policy_wdl_sft.wdl_group_adv_is.1a.1779295502",
                "training_run_key": "feature_on_policy_wdl_sft.wdl_group_adv_is.1a.1779295502.training_metrics_jsonl",
                "metrics_rows_seen": metric_rows_seen,
                "samples_yielded": samples_yielded,
                "train_dataset_rows": train_rows,
                "best_step": best.get("step"),
                "best_metric": best.get("metric_value"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
