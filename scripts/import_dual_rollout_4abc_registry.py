#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_TOOLS = Path("/data-1/agent-tools/experiment_registry")
sys.path.insert(0, str(REGISTRY_TOOLS))

from registry_core import (  # noqa: E402
    add_artifact,
    add_quality_flag,
    add_source_record,
    add_tag,
    connect,
    init_db,
    slug,
    upsert_dataset,
    upsert_eval_run,
    upsert_experiment,
    upsert_metric,
    upsert_model,
    upsert_project,
)


IMPORTER = "import_dual_rollout_4abc_v1"
BRANCH = "feature/on-policy-wdl-sft-dual-rollout"
PROJECT_NAME = f"verl:{BRANCH}"
REPO = Path("/root/buaa/local_data1/verl07/verl-dual-rollout")
DB_DEFAULT = "/data-1/experiment_registry/experiment_registry.sqlite"


RUNS: list[dict[str, Any]] = [
    {
        "label": "4A",
        "run_name": "WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS_1779761732",
        "experiment_key": "verl.branch.feature_on_policy_wdl_sft_dual_rollout.dual_model2_group_adv_is.4a",
        "display_name": "Dual rollout 4A: model2-only rollout + fused group-advantage IS",
        "method_variant": "model2_rollout_fused_group_adv_tis_binary_mask",
        "method_version": "4A",
        "variant": "dual_model2_group_adv_is_4a",
        "trust_level": "usable_with_caution",
        "trust_reason": "Completed 115-step production-context run, but online MATH-500 mean@3 regressed after step 25 and raw gradients stayed high.",
        "run_prefix": "WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS",
        "metrics": REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/metrics/OnPolicyWDLSFT/WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS_1779761732.jsonl",
        "log": REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS_1779761732.log",
        "validation_dir": REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/validation/WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS_1779761732",
        "launcher_scripts": [
            REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/run_4a_model2_group_adv_is.sh",
            REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/_common_dual_rollout.sh",
            REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/monitor_4abc_math_queue.sh",
        ],
        "checkpoint_dir": Path("/data-1/checkpoints/WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS_1779761732"),
        "input_model_path": "/data-1/.cache/huggingface/QwenJoint-4B-WDL-SFT-Qwen3-4B-Base-SFT-stage-1",
        "input_model_role": "joint_input",
        "output_model_role": "joint_checkpoint",
        "loss_mode": "dual_model2_group_adv_is",
        "method_family": "on_policy_wdl_sft",
        "model_topology": "joint_base_plus_sft",
        "joint_rollout_sources": "[sub_model_1]",
        "joint_rollout_select": "sub_model_1",
        "joint_rollout_train_on_selected_only": True,
        "rollout_is": None,
        "gamma_pos_sft": 1.0,
        "tis_threshold": 5.0,
        "max_actor_ckpts_to_keep": 1,
    },
    {
        "label": "4B",
        "run_name": "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1_1779795940",
        "experiment_key": "verl.branch.feature_on_policy_wdl_sft_dual_rollout.wdl_group_adv_is.4b_math_base",
        "display_name": "Single-model 4B: Qwen3-4B-Base + wdl_group_adv_is on MATH train",
        "method_variant": "single_model_group_adv_is_base",
        "method_version": "4B",
        "variant": "single_model_base_group_adv_is",
        "trust_level": "trusted",
        "trust_reason": "Completed 115-step MATH train run with source metrics, checkpoints, and validation records present.",
        "run_prefix": "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1",
        "metrics": REPO / "recipe/on_policy_wdl_sft/ablation_single_model/metrics/OnPolicyWDLSFT/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1_1779795940.jsonl",
        "log": REPO / "recipe/on_policy_wdl_sft/ablation_single_model/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1_1779795940.log",
        "validation_dir": REPO / "recipe/on_policy_wdl_sft/ablation_single_model/validation/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1_1779795940",
        "launcher_scripts": [
            REPO / "recipe/on_policy_wdl_sft/ablation_single_model/run_4b_math_base.sh",
            REPO / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh",
            REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/monitor_4abc_math_queue.sh",
        ],
        "checkpoint_dir": Path("/data-1/checkpoints/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1_1779795940"),
        "input_model_path": "/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "input_model_role": "base_initialization",
        "output_model_role": "single_model_checkpoint",
        "loss_mode": "wdl_group_adv_is",
        "method_family": "single_model_ablation",
        "model_topology": "single_model_base",
        "rollout_is": None,
        "gamma_pos_sft": 1.0,
        "tis_threshold": None,
        "max_actor_ckpts_to_keep": 13,
    },
    {
        "label": "4C",
        "run_name": "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1_1779807346",
        "experiment_key": "verl.branch.feature_on_policy_wdl_sft_dual_rollout.wdl_group_adv_is.4c_math_sft",
        "display_name": "Single-model 4C: Qwen3-4B-Base-SFT-stage-1 + wdl_group_adv_is on MATH train",
        "method_variant": "single_model_group_adv_is_sft",
        "method_version": "4C",
        "variant": "single_model_sft_group_adv_is",
        "trust_level": "trusted",
        "trust_reason": "Completed 115-step MATH train run with source metrics, checkpoints, and validation records present.",
        "run_prefix": "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1",
        "metrics": REPO / "recipe/on_policy_wdl_sft/ablation_single_model/metrics/OnPolicyWDLSFT/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1_1779807346.jsonl",
        "log": REPO / "recipe/on_policy_wdl_sft/ablation_single_model/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1_1779807346.log",
        "validation_dir": REPO / "recipe/on_policy_wdl_sft/ablation_single_model/validation/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1_1779807346",
        "launcher_scripts": [
            REPO / "recipe/on_policy_wdl_sft/ablation_single_model/run_4c_math_sft.sh",
            REPO / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh",
            REPO / "recipe/on_policy_wdl_sft/dual_submodel_rollout/monitor_4abc_math_queue.sh",
        ],
        "checkpoint_dir": Path("/data-1/checkpoints/WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1_1779807346"),
        "input_model_path": "/data-1/.cache/Qwen3-4B-Base-SFT-stage-1",
        "input_model_role": "sft_stage1_initialization",
        "output_model_role": "single_model_checkpoint",
        "loss_mode": "wdl_group_adv_is",
        "method_family": "single_model_ablation",
        "model_topology": "single_model_sft",
        "rollout_is": None,
        "gamma_pos_sft": 1.0,
        "tis_threshold": None,
        "max_actor_ckpts_to_keep": 13,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def metric_scope(name: str) -> str:
    if name.startswith("val-"):
        return "online_validation"
    if name.startswith("perf/"):
        return "performance"
    if name.startswith("response"):
        return "rollout"
    if name.startswith("actor/") or name.startswith("critic/"):
        return "actor_train"
    if name.startswith("rollout_corr/"):
        return "rollout_correction"
    if name.startswith("dual_") or name.startswith("wdl_group_adv_is/") or name.startswith("jointTraining/"):
        return "method_diagnostics"
    return "training"


def dataset_from_val_metric(name: str) -> tuple[str | None, str | None, str | None]:
    parts = name.split("/")
    if len(parts) < 4 or parts[0] not in {"val-core", "val-aux"}:
        return None, None, None
    if parts[1] == "HuggingFaceH4" and len(parts) >= 5:
        dataset = f"{parts[1]}/{parts[2]}"
        metric = "/".join(parts[3:])
    else:
        dataset = parts[1]
        metric = "/".join(parts[2:])
    return parts[0], dataset, metric


def dataset_key(name: str) -> str:
    return "math." + slug(name.replace("/", "_"))


def artifact_with_stat(
    conn: sqlite3.Connection,
    artifact_kind: str,
    path: str | Path,
    *,
    experiment_id: int | None = None,
    training_run_id: int | None = None,
    eval_run_id: int | None = None,
    model_id: int | None = None,
    description: str | None = None,
    notes: str | None = None,
) -> None:
    path_str = str(path)
    add_artifact(
        conn,
        artifact_kind,
        path_str,
        experiment_id=experiment_id,
        training_run_id=training_run_id,
        eval_run_id=eval_run_id,
        model_id=model_id,
        description=description,
        notes=notes,
    )
    p = Path(path_str)
    exists = p.exists()
    size = None
    mtime = None
    if exists:
        st = p.stat()
        size = st.st_size
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).replace(microsecond=0).isoformat()
    conn.execute(
        """
        update artifacts
        set exists_checked_at=?, artifact_exists=?, size_bytes=coalesce(?, size_bytes), mtime=coalesce(?, mtime),
            description=coalesce(?, description), notes=coalesce(?, notes)
        where artifact_kind=? and path=? and coalesce(experiment_id,-1)=coalesce(?,-1)
          and coalesce(training_run_id,-1)=coalesce(?,-1)
          and coalesce(eval_run_id,-1)=coalesce(?,-1)
          and coalesce(model_id,-1)=coalesce(?,-1)
        """,
        (
            utc_now(),
            1 if exists else 0,
            size,
            mtime,
            description,
            notes,
            artifact_kind,
            path_str,
            experiment_id,
            training_run_id,
            eval_run_id,
            model_id,
        ),
    )


def latest_step(ckpt_dir: Path) -> int | None:
    p = ckpt_dir / "latest_checkpointed_iteration.txt"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return int(text) if text else None


def best_checkpoint(ckpt_dir: Path) -> dict[str, Any]:
    p = ckpt_dir / "best_checkpoint.json"
    return read_json(p) if p.exists() else {}


def summarize(rows: list[dict[str, Any]], best: dict[str, Any]) -> dict[str, Any]:
    train_rows = [r for r in rows if "actor/pg_loss" in r.get("data", {})]
    val_rows = [r for r in rows if "val-core/HuggingFaceH4/MATH-500/acc/mean@3" in r.get("data", {})]
    final_train = train_rows[-1] if train_rows else {}
    first_train = train_rows[0] if train_rows else {}
    final_val = val_rows[-1] if val_rows else {}
    best_val = max(val_rows, key=lambda r: r["data"]["val-core/HuggingFaceH4/MATH-500/acc/mean@3"]) if val_rows else {}
    return {
        "records": len(rows),
        "train_rows": len(train_rows),
        "val_steps": [r["step"] for r in val_rows],
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "first_pg_loss": first_train.get("data", {}).get("actor/pg_loss"),
        "final_pg_loss": final_train.get("data", {}).get("actor/pg_loss"),
        "final_reward_mean": final_train.get("data", {}).get("critic/rewards/mean"),
        "final_grad_norm": final_train.get("data", {}).get("actor/grad_norm"),
        "max_grad_norm": max((r["data"].get("actor/grad_norm") for r in train_rows if is_number(r["data"].get("actor/grad_norm"))), default=None),
        "final_response_len_mean": final_train.get("data", {}).get("response_length/mean"),
        "max_response_len_mean": max((r["data"].get("response_length/mean") for r in train_rows if is_number(r["data"].get("response_length/mean"))), default=None),
        "max_aborted_ratio": max((r["data"].get("response/aborted_ratio") for r in train_rows if is_number(r["data"].get("response/aborted_ratio"))), default=None),
        "best_step": best.get("step") or best_val.get("step"),
        "best_metric_key": best.get("metric_key") or "val-core/HuggingFaceH4/MATH-500/acc/mean@3",
        "best_metric_value": best.get("metric_value") or best_val.get("data", {}).get("val-core/HuggingFaceH4/MATH-500/acc/mean@3"),
        "final_math500_mean3": final_val.get("data", {}).get("val-core/HuggingFaceH4/MATH-500/acc/mean@3"),
        "final_math500_best3": final_val.get("data", {}).get("val-core/HuggingFaceH4/MATH-500/acc/best@3/mean"),
        "final_math500_maj3": final_val.get("data", {}).get("val-core/HuggingFaceH4/MATH-500/acc/maj@3/mean"),
        "final_aime25_mean3": final_val.get("data", {}).get("val-core/aime25/acc/mean@3"),
        "nonfinite_count": sum(
            1
            for r in rows
            for v in r.get("data", {}).values()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(float(v))
        ),
    }


def upsert_training_run(
    conn: sqlite3.Connection,
    *,
    key: str,
    experiment_id: int,
    input_model_id: int,
    output_model_id: int,
    train_dataset_id: int,
    run: dict[str, Any],
    summary: dict[str, Any],
    hyperparams: dict[str, Any],
    git_commit: str,
) -> int:
    extra = {
        "summary": summary,
        "run_name": run["run_name"],
        "checkpoint_dir": str(run["checkpoint_dir"]),
        "best_checkpoint_step": summary["best_step"],
        "best_checkpoint_metric_key": summary["best_metric_key"],
        "best_checkpoint_metric_value": summary["best_metric_value"],
    }
    conn.execute(
        """
        insert into training_runs(
          training_run_key, experiment_id, input_model_id, output_model_id, train_dataset_id,
          method, framework, framework_version, beta, learning_rate, num_epochs,
          effective_batch_size, max_length, weight_decay, lr_scheduler, distributed_backend,
          distributed_config_json, hyperparams_json, num_gpus, total_steps, final_train_loss,
          final_step_loss, first_step_loss, raw_summary_path, wandb_run, git_branch, git_commit,
          extra_json, notes
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(training_run_key) do update set
          experiment_id=excluded.experiment_id,
          input_model_id=excluded.input_model_id,
          output_model_id=excluded.output_model_id,
          train_dataset_id=excluded.train_dataset_id,
          method=excluded.method,
          framework=excluded.framework,
          framework_version=excluded.framework_version,
          beta=excluded.beta,
          learning_rate=excluded.learning_rate,
          num_epochs=excluded.num_epochs,
          effective_batch_size=excluded.effective_batch_size,
          max_length=excluded.max_length,
          weight_decay=excluded.weight_decay,
          lr_scheduler=excluded.lr_scheduler,
          distributed_backend=excluded.distributed_backend,
          distributed_config_json=excluded.distributed_config_json,
          hyperparams_json=excluded.hyperparams_json,
          num_gpus=excluded.num_gpus,
          total_steps=excluded.total_steps,
          final_train_loss=excluded.final_train_loss,
          final_step_loss=excluded.final_step_loss,
          first_step_loss=excluded.first_step_loss,
          raw_summary_path=excluded.raw_summary_path,
          wandb_run=excluded.wandb_run,
          git_branch=excluded.git_branch,
          git_commit=excluded.git_commit,
          extra_json=excluded.extra_json,
          notes=excluded.notes
        """,
        (
            key,
            experiment_id,
            input_model_id,
            output_model_id,
            train_dataset_id,
            run["loss_mode"],
            "verl",
            "local docker/uv; vLLM 0.12.0; PyTorch 2.9.1+cu126",
            0.0,
            5e-7,
            1.0,
            64,
            4096,
            0.1,
            "constant_after_warmup",
            "ray+fsdp+vllm",
            json.dumps({"nnodes": 1, "n_gpus_per_node": 8, "fsdp_size": -1, "rollout_tp_size": 1}, ensure_ascii=False, sort_keys=True),
            json.dumps(hyperparams, ensure_ascii=False, sort_keys=True),
            8,
            summary["last_step"],
            summary["final_pg_loss"],
            summary["final_pg_loss"],
            summary["first_pg_loss"],
            str(run["metrics"]),
            run["run_name"],
            BRANCH,
            git_commit,
            json.dumps(extra, ensure_ascii=False, sort_keys=True),
            run["trust_reason"],
        ),
    )
    row = conn.execute("select id from training_runs where training_run_key=?", (key,)).fetchone()
    return int(row["id"])


def insert_training_metrics(conn: sqlite3.Connection, training_run_id: int, rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        step = int(row["step"])
        for name, value in row.get("data", {}).items():
            if not is_number(value):
                continue
            conn.execute(
                """
                insert into training_metrics(training_run_id, metric_name, metric_value, step, metric_scope)
                values (?, ?, ?, ?, ?)
                on conflict(training_run_id, metric_name, step, metric_scope) do update set
                  metric_value=excluded.metric_value
                """,
                (training_run_id, name, float(value), step, metric_scope(name)),
            )
            count += 1
    return count


def import_online_validation_evals(
    conn: sqlite3.Connection,
    *,
    run: dict[str, Any],
    rows: list[dict[str, Any]],
    experiment_id: int,
    project_id: int,
    git_commit: str,
) -> int:
    eval_count = 0
    val_rows = [r for r in rows if any(k.startswith("val-") for k in r.get("data", {}))]
    for val in val_rows:
        step = int(val["step"])
        model_path = str(run["checkpoint_dir"] / f"global_step_{step}") if (run["checkpoint_dir"] / f"global_step_{step}").exists() else f"verl_online_validation://{run['run_name']}/step_{step}"
        model_id = upsert_model(
            conn,
            f"{run['experiment_key']}.online_validation_step_{step}",
            model_path,
            display_name=f"{run['run_name']} online validation step {step}",
            base_model="Qwen3-4B",
            checkpoint_step=step,
            checkpoint_kind="online_validation_step",
            model_role=run["output_model_role"],
            project_id=project_id,
            git_branch=BRANCH,
            git_commit=git_commit,
            is_best=1 if step == run.get("best_step") else 0,
            is_latest=1 if step == run.get("latest_step") else 0,
            notes="Online validation policy state; retained checkpoint may not exist for intermediate steps.",
        )
        eval_id = upsert_eval_run(
            conn,
            eval_run_key=f"{run['experiment_key']}.online_validation.step_{step}.n3",
            experiment_id=experiment_id,
            model_id=model_id,
            eval_name=f"{run['label']} online validation step {step} n=3",
            domain="math",
            eval_harness="verl_online_validation",
            framework="verl",
            output_dir=str(run["validation_dir"] / str(step)),
            raw_metrics_path=str(run["metrics"]),
            raw_samples_path=str(run["validation_dir"] / f"{step}.jsonl"),
            n=3,
            num_samples=3,
            repeat_count=3,
            temperature=1.0,
            top_p=0.95,
            top_k=-1,
            do_sample=1,
            max_tokens=4096,
            max_new_tokens=4096,
            prompt_mode="system_prompt_parquet",
            command=f"bash {run['launcher_scripts'][0]}",
            cwd=str(REPO),
            git_branch=BRANCH,
            git_commit=git_commit,
            eval_datetime=utc_now(),
            trust_level=run["trust_level"],
            trust_reason=run["trust_reason"],
            extra_json=json.dumps({"training_run_name": run["run_name"], "global_step": step, "source": "training_metrics_jsonl"}, ensure_ascii=False, sort_keys=True),
            notes="Online validation metrics emitted during training; not a separate offline vLLM eval.",
        )
        add_source_record(conn, str(run["metrics"]), "training_metrics_jsonl", f"validation_step_{step}", IMPORTER, "eval_runs", eval_id)
        artifact_with_stat(conn, "validation_jsonl", run["validation_dir"] / f"{step}.jsonl", experiment_id=experiment_id, eval_run_id=eval_id, model_id=model_id, description=f"Online validation generations for step {step}")
        for full_name, value in val.get("data", {}).items():
            if not is_number(value):
                continue
            scope, dataset, metric = dataset_from_val_metric(full_name)
            if dataset is None or metric is None:
                continue
            ds_id = upsert_dataset(conn, dataset_key(dataset), dataset, domain="math")
            conn.execute(
                """
                insert or ignore into eval_run_datasets(eval_run_id, dataset_id, split, subset, num_examples, notes)
                values (?, ?, ?, ?, ?, ?)
                """,
                (eval_id, ds_id, "validation", None, None, "Online validation from trainer metrics JSONL"),
            )
            upsert_metric(conn, eval_id, ds_id, metric, float(value), metric_scope=scope)
        eval_count += 1
    return eval_count


def add_validation_check(conn: sqlite3.Connection, name: str, source_path: str, source_value: Any, database_value: Any, passed: bool, notes: str) -> None:
    conn.execute(
        "insert into validation_checks(check_name, source_path, source_value, database_value, passed, checked_at, notes) values (?, ?, ?, ?, ?, ?, ?)",
        (name, source_path, str(source_value), str(database_value), 1 if passed else 0, utc_now(), notes),
    )


def import_run(conn: sqlite3.Connection, project_id: int, run: dict[str, Any], git_commit: str) -> dict[str, Any]:
    rows = load_jsonl(run["metrics"])
    best = best_checkpoint(run["checkpoint_dir"])
    run["latest_step"] = latest_step(run["checkpoint_dir"])
    run["best_step"] = best.get("step")
    summary = summarize(rows, best)
    hyperparams = {
        "run_prefix": run["run_prefix"],
        "loss_mode": run["loss_mode"],
        "loss_agg_mode": "seq-mean-token-sum",
        "train_file": "/data-1/dataset/math/train_rl_format.parquet",
        "validation_files": [
            "/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet",
            "/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet",
        ],
        "total_training_steps": 115,
        "total_epochs": 1,
        "train_prompt_bsz": 64,
        "train_prompt_mini_bsz": 8,
        "n_resp_per_prompt": 8,
        "max_prompt_length": 500,
        "max_response_length": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "val_n": 3,
        "val_top_p": 0.95,
        "learning_rate": 5e-7,
        "lr_warmup_steps": 5,
        "grad_clip": 500.0,
        "weight_decay": 0.1,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.27,
        "norm_adv_by_std_in_grpo": False,
        "rollout_is": run["rollout_is"],
        "gamma_pos_sft": run["gamma_pos_sft"],
        "tis_threshold": run["tis_threshold"],
        "use_kl_loss": False,
        "kl_loss_coef": 0.0,
        "reward_manager": "dapo",
        "custom_reward_function": "recipe/on_policy_wdl_sft/custom_reward_function_latex_verify.py:compute_score_latex_verify",
        "vllm_attention_backend": "FLASHINFER",
        "fsdp_attn_implementation": "flash_attention_2",
        "rollout_engine": "vllm",
        "rollout_mode": "async",
        "rollout_tp_size": 1,
        "n_gpus_per_node": 8,
        "max_actor_ckpts_to_keep": run["max_actor_ckpts_to_keep"],
        "keep_best_ckpt": True,
        "best_ckpt_metric_key": "val-core/HuggingFaceH4/MATH-500/acc/mean@3",
        "best_ckpt_metric_mode": "max",
        "model_topology": run["model_topology"],
    }
    if run.get("joint_rollout_sources"):
        hyperparams.update(
            {
                "joint_rollout_sources": run["joint_rollout_sources"],
                "joint_rollout_select": run["joint_rollout_select"],
                "joint_rollout_train_on_selected_only": run["joint_rollout_train_on_selected_only"],
            }
        )

    exp_id = upsert_experiment(
        conn,
        project_id,
        run["experiment_key"],
        run["display_name"],
        method="rl",
        method_family=run["method_family"],
        method_variant=run["method_variant"],
        method_version=run["method_version"],
        domain="math",
        variant=run["variant"],
        status="completed",
        trust_level=run["trust_level"],
        trust_reason=run["trust_reason"],
        extra_json=json.dumps({"hyperparams": hyperparams, "summary": summary}, ensure_ascii=False, sort_keys=True),
        notes=run["trust_reason"],
    )
    conn.execute("update experiments set git_branch=?, git_commit=? where id=?", (BRANCH, git_commit, exp_id))
    for tag in [run["label"], run["loss_mode"], "4abc", "math_train", "online_validation"]:
        add_tag(conn, "experiment", exp_id, tag)

    input_model_id = upsert_model(
        conn,
        f"{run['experiment_key']}.input",
        run["input_model_path"],
        display_name=f"{run['label']} input model",
        base_model="Qwen3-4B",
        checkpoint_kind="input_initialization",
        model_role=run["input_model_role"],
        project_id=project_id,
        git_branch=BRANCH,
        git_commit=git_commit,
        is_best=0,
        is_latest=0,
        notes=f"Input model for {run['label']}",
    )
    latest = summary["last_step"]
    best_step = summary["best_step"]
    latest_model_id = upsert_model(
        conn,
        f"{run['experiment_key']}.latest_step_{latest}",
        str(run["checkpoint_dir"] / f"global_step_{latest}"),
        display_name=f"{run['run_name']} latest checkpoint step {latest}",
        base_model="Qwen3-4B",
        checkpoint_step=latest,
        checkpoint_kind="latest",
        model_role=run["output_model_role"],
        project_id=project_id,
        git_branch=BRANCH,
        git_commit=git_commit,
        is_best=1 if latest == best_step else 0,
        is_latest=1,
        extra_json=json.dumps({"selection_metric_name": summary["best_metric_key"], "selection_metric_value": summary["best_metric_value"]}, sort_keys=True),
    )
    if best_step is not None:
        best_model_id = upsert_model(
            conn,
            f"{run['experiment_key']}.best_step_{best_step}",
            str(run["checkpoint_dir"] / f"global_step_{best_step}"),
            display_name=f"{run['run_name']} best checkpoint step {best_step}",
            base_model="Qwen3-4B",
            checkpoint_step=best_step,
            checkpoint_kind="best",
            model_role=run["output_model_role"],
            project_id=project_id,
            git_branch=BRANCH,
            git_commit=git_commit,
            is_best=1,
            is_latest=1 if best_step == latest else 0,
            extra_json=json.dumps({"selection_metric_name": summary["best_metric_key"], "selection_metric_value": summary["best_metric_value"]}, sort_keys=True),
        )
    else:
        best_model_id = latest_model_id

    train_ds_id = upsert_dataset(
        conn,
        "math.train_rl_format",
        "MATH train RL format",
        domain="math",
        path="/data-1/dataset/math/train_rl_format.parquet",
        split="train",
        row_count=7405,
        notes="7500 raw rows; max_prompt_length=500 filtering keeps 7405 prompts; drop_last gives 115 steps at batch 64.",
    )
    training_key = f"{run['experiment_key']}.{run['run_name'].split('_')[-1]}.train"
    training_id = upsert_training_run(
        conn,
        key=training_key,
        experiment_id=exp_id,
        input_model_id=input_model_id,
        output_model_id=latest_model_id,
        train_dataset_id=train_ds_id,
        run=run,
        summary=summary,
        hyperparams=hyperparams,
        git_commit=git_commit,
    )
    conn.execute(
        "insert or ignore into training_run_datasets(training_run_id, dataset_id, role, row_count, notes) values (?, ?, ?, ?, ?)",
        (training_id, train_ds_id, "train", 7405, "Filtered prompt count used for one-epoch 115-step schedule."),
    )
    for ds_key, ds_name, ds_path in [
        ("math.math500_with_system_prompt", "HuggingFaceH4/MATH-500", "/data-1/dataset/MATH-500/math500-test_with_system_prompt.parquet"),
        ("math.aime2025_with_system_prompt", "aime25", "/data-1/dataset/AIME-2025/aime-2025_with_system_prompt.parquet"),
    ]:
        ds_id = upsert_dataset(conn, ds_key, ds_name, domain="math", path=ds_path, split="validation")
        conn.execute(
            "insert or ignore into training_run_datasets(training_run_id, dataset_id, role, notes) values (?, ?, ?, ?)",
            (training_id, ds_id, "validation", "Trainer online validation dataset."),
        )

    metric_count = insert_training_metrics(conn, training_id, rows)
    eval_count = import_online_validation_evals(conn, run=run, rows=rows, experiment_id=exp_id, project_id=project_id, git_commit=git_commit)

    for kind, path, desc in [
        ("training_metrics_jsonl", run["metrics"], "File logger metrics JSONL used for scalar training/validation import."),
        ("training_log", run["log"], "Training stdout/stderr log."),
        ("validation_generation_dir", run["validation_dir"], "Online validation generation JSONL directory."),
        ("checkpoint_dir", run["checkpoint_dir"], "Training checkpoint directory."),
        ("latest_checkpoint_txt", run["checkpoint_dir"] / "latest_checkpointed_iteration.txt", "Latest checkpoint step metadata."),
        ("best_checkpoint_json", run["checkpoint_dir"] / "best_checkpoint.json", "Best checkpoint selection metadata."),
    ]:
        artifact_with_stat(conn, kind, path, experiment_id=exp_id, training_run_id=training_id, description=desc)
        add_source_record(conn, str(path), kind, None, IMPORTER, "training_runs", training_id, notes=desc)
    for script in run["launcher_scripts"]:
        artifact_with_stat(conn, "launcher_script", script, experiment_id=exp_id, training_run_id=training_id, description=f"Launcher/config source for {run['label']}")
        add_source_record(conn, str(script), "launcher_script", None, IMPORTER, "experiments", exp_id)
    artifact_with_stat(conn, "checkpoint_dir", run["checkpoint_dir"] / f"global_step_{latest}", experiment_id=exp_id, training_run_id=training_id, model_id=latest_model_id, description="Latest retained checkpoint")
    if best_model_id != latest_model_id:
        artifact_with_stat(conn, "checkpoint_dir", run["checkpoint_dir"] / f"global_step_{best_step}", experiment_id=exp_id, training_run_id=training_id, model_id=best_model_id, description="Best retained checkpoint")

    if run["label"] == "4A":
        add_quality_flag(conn, "experiment", exp_id, "online_metric_regression", "warning", "MATH-500 mean@3 peaked at step 25 and regressed by final step.")
        add_quality_flag(conn, "training_run", training_id, "high_raw_grad_norm_long_response", "warning", "Raw pre-clip grad_norm reached 6265 and response lengths stayed near 3.3k tokens.")

    db_latest = conn.execute("select total_steps from training_runs where id=?", (training_id,)).fetchone()[0]
    add_validation_check(
        conn,
        f"4abc_{run['label'].lower()}_latest_step",
        str(run["checkpoint_dir"] / "latest_checkpointed_iteration.txt"),
        run["latest_step"],
        db_latest,
        run["latest_step"] == db_latest == 115,
        "Checkpoint latest step must match training_runs.total_steps.",
    )
    db_best = conn.execute(
        "select metric_value from training_metrics where training_run_id=? and metric_name=? and step=? and metric_scope='online_validation'",
        (training_id, summary["best_metric_key"], summary["best_step"]),
    ).fetchone()
    db_best_value = db_best[0] if db_best else None
    add_validation_check(
        conn,
        f"4abc_{run['label'].lower()}_best_checkpoint_metric",
        str(run["checkpoint_dir"] / "best_checkpoint.json"),
        summary["best_metric_value"],
        db_best_value,
        db_best_value is not None and abs(float(summary["best_metric_value"]) - float(db_best_value)) < 1e-12,
        "Best checkpoint metric must match imported training metric at the selected step.",
    )
    db_final = conn.execute(
        "select metric_value from training_metrics where training_run_id=? and metric_name='val-core/HuggingFaceH4/MATH-500/acc/mean@3' and step=115 and metric_scope='online_validation'",
        (training_id,),
    ).fetchone()
    db_final_value = db_final[0] if db_final else None
    add_validation_check(
        conn,
        f"4abc_{run['label'].lower()}_final_math500_mean3",
        str(run["metrics"]),
        summary["final_math500_mean3"],
        db_final_value,
        db_final_value is not None and abs(float(summary["final_math500_mean3"]) - float(db_final_value)) < 1e-12,
        "Final MATH-500 mean@3 must match metrics JSONL step 115.",
    )

    return {
        "label": run["label"],
        "experiment_id": exp_id,
        "training_run_id": training_id,
        "metrics_inserted": metric_count,
        "eval_runs_inserted": eval_count,
        "latest_model_id": latest_model_id,
        "best_model_id": best_model_id,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import dual-rollout 4ABC training results into experiment registry")
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--repo", default=str(REPO))
    args = parser.parse_args()

    init_db(args.db)
    git_commit = os.popen(f"git -C {args.repo} rev-parse HEAD").read().strip() or None
    with connect(args.db) as conn:
        project_id = upsert_project(
            conn,
            PROJECT_NAME,
            str(REPO),
            BRANCH,
            "Branch-scoped form for dual-rollout WDL-SFT and group-advantage IS experiments.",
        )
        conn.execute(
            "update projects set remote_url=? where id=?",
            ("https://github.com/AlexJJ009/verl-v0.7.git", project_id),
        )
        conn.execute("delete from validation_checks where check_name like '4abc_%'")
        imported = [import_run(conn, project_id, run, git_commit) for run in RUNS]
        conn.commit()
    print(json.dumps({"project": PROJECT_NAME, "imported": imported}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
