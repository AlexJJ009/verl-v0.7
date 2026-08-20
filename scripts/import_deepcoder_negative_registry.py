#!/usr/bin/env python3
"""Import DeepCoder 4K negative-result code-task runs into SQLite registry.

This is a supplemental importer for source-backed metrics that the generic
verl importer records as training rows but not as queryable eval rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

REGISTRY_TOOLS = Path("/data-1/agent-tools/experiment_registry")
if str(REGISTRY_TOOLS) not in sys.path:
    sys.path.insert(0, str(REGISTRY_TOOLS))

from registry_core import (  # noqa: E402
    add_artifact,
    add_quality_flag,
    add_source_record,
    add_tag,
    connect,
    init_db,
    upsert_dataset,
    upsert_eval_run,
    upsert_experiment,
    upsert_metric,
    upsert_model,
)

DB = "/data-1/experiment_registry/experiment_registry.sqlite"
REPO = Path("/data-1/verl07/verl")
PROJECT_NAME = "verl:feature/on-policy-wdl-sft"
PROJECT_KEY = "verl_feature_on_policy_wdl_sft"
BRANCH = "feature/on-policy-wdl-sft"
IMPORTER = "import_deepcoder_negative_registry_v1"
RELEASE_GATE_SCRIPT = REPO / "scripts/training_result_release_gate.py"

METRICS_ROOT = REPO / "recipe/on_policy_wdl_sft/code_task/metrics/OnPolicyWDLSFT-CodeTask"
VALIDATION_ROOT = REPO / "recipe/on_policy_wdl_sft/code_task/validation"
REPORT_PATH = REPO / "docs/joint_training/reports/deepcoder_kodcode_failure_analysis.md"
TRAIN_DATASET = Path("/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_rl_format.parquet")
TRAIN_MANIFEST = Path("/data-1/dataset/code/verl_rl/deepcoder_preview_train_prompt1024_manifest.json")
HUMANEVAL_PLUS = Path("/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet")
MBPP_PLUS = Path("/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet")
BASE_MODEL = "/data-1/.cache/huggingface/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539"


RUNS = [
    {
        "run_name": "ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780976139",
        "experiment_key": "verl.on_policy_wdl_sft.code.deepcoder_s1_beta0_v1_retention.step134",
        "display_name": "On-Policy SFT Code DeepCoder S1 beta=0.0 V1 retention step134",
        "beta": 0.0,
        "beta_label": "beta0",
        "status": "partial",
        "trust_level": "needs_review",
        "trust_reason": "DeepCoder beta=0.0 stopped at train step 134 with final validation at step 130; negative data-switch result, not reliable as a completed run.",
        "total_steps": 134,
        "planned_total_steps": 150,
        "final_eval_step": 130,
        "checkpoint_root": "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780976139",
        "log_rel": "recipe/on_policy_wdl_sft/code_task/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA0-V1-RETENTION_1780976139_resumed_1780977456.log",
        "reliability": "eval_insufficient",
        "notes": "Incomplete DeepCoder data-switch negative result; final training correct_ratio is near zero and the run lacks step-150 evaluation.",
    },
    {
        "run_name": "ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781282660",
        "experiment_key": "verl.on_policy_wdl_sft.code.deepcoder_s1_beta01_v1_retention.step150",
        "display_name": "On-Policy SFT Code DeepCoder S1 beta=0.1 V1 retention step150",
        "beta": 0.1,
        "beta_label": "beta01",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "trust_reason": "Completed DeepCoder data-switch negative result with low correct-rollout density and high response clipping; not a drop-in KodCode replacement.",
        "total_steps": 150,
        "planned_total_steps": 150,
        "final_eval_step": 150,
        "checkpoint_root": "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781282660",
        "log_rel": "recipe/on_policy_wdl_sft/code_task/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781282660.log",
        "reliability": "usable_with_caution",
        "notes": "Completed DeepCoder data-switch negative result; use for failure analysis and future controlled ablations.",
    },
    {
        "run_name": "ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA05-V1-FULL_1782059353",
        "experiment_key": "verl.on_policy_wdl_sft.code.deepcoder_s1_beta05_v1_full.step150",
        "display_name": "On-Policy SFT Code DeepCoder S1 beta=0.5 V1 full step150",
        "beta": 0.5,
        "beta_label": "beta05",
        "status": "completed",
        "trust_level": "usable_with_caution",
        "trust_reason": "Completed DeepCoder data-switch negative result; beta=0.5 did not improve HumanEval+/MBPP+ and ended with high response clipping.",
        "total_steps": 150,
        "planned_total_steps": 150,
        "final_eval_step": 150,
        "checkpoint_root": "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA05-V1-FULL_1782059353",
        "log_rel": "recipe/on_policy_wdl_sft/code_task/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA05-V1-FULL_1782059353.log",
        "reliability": "usable_with_caution",
        "notes": "Completed beta=0.5 DeepCoder ablation; negative result because higher beta changed loss scale but did not repair sparse positives.",
    },
    {
        "run_name": "ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781184148",
        "experiment_key": "verl.on_policy_wdl_sft.code.deepcoder_s1_beta01_v1_retention.collapsed_1781184148",
        "display_name": "On-Policy SFT Code DeepCoder S1 beta=0.1 collapsed run 1781184148",
        "beta": 0.1,
        "beta_label": "beta01_collapsed",
        "status": "completed",
        "trust_level": "buggy",
        "trust_reason": "Collapsed DeepCoder failure-mode run: final extraction failure is about 98% and response clipping about 97%; useful only as a negative failure example.",
        "total_steps": 150,
        "planned_total_steps": 150,
        "final_eval_step": 150,
        "checkpoint_root": "/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781184148",
        "log_rel": "recipe/on_policy_wdl_sft/code_task/ONPOLICY-SFT-Qwen3-4B-CODE-DEEPCODER-S1-BETA01-V1-RETENTION_1781184148_resumed_1781281912.log",
        "reliability": "buggy",
        "notes": "Near-total format-collapse run retained as a queryable failure mode; not representative of later stable DeepCoder beta=0.1.",
    },
]


def check_release_gate(run_names: list[str]) -> None:
    for run_name in run_names:
        subprocess.check_call([sys.executable, str(RELEASE_GATE_SCRIPT), "check", "--run-name", run_name])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_at(rows: list[dict[str, Any]], step: int, key: str) -> float | None:
    for row in rows:
        if int(row.get("step", -1)) == int(step):
            value = row.get("data", {}).get(key)
            if isinstance(value, int | float):
                return float(value)
    return None


def best_metric(rows: list[dict[str, Any]], key: str) -> tuple[int | None, float | None]:
    best_step = None
    best_value = None
    for row in rows:
        value = row.get("data", {}).get(key)
        if isinstance(value, int | float) and (best_value is None or value > best_value):
            best_step = int(row["step"])
            best_value = float(value)
    return best_step, best_value


def count_validation(path: Path) -> tuple[int, dict[str, int], dict[str, dict[str, int]]]:
    total = 0
    by_status: dict[str, int] = {}
    by_dataset_status: dict[str, dict[str, int]] = {}
    if not path.exists():
        return 0, by_status, by_dataset_status
    for row in load_jsonl(path):
        total += 1
        dataset = str(row.get("data_source") or row.get("dataset") or row.get("source") or "unknown")
        acc = row.get("acc")
        status = "pass" if acc == 1 or acc == 1.0 else "non_pass"
        by_status[status] = by_status.get(status, 0) + 1
        by_dataset_status.setdefault(dataset, {})
        by_dataset_status[dataset][status] = by_dataset_status[dataset].get(status, 0) + 1
    return total, by_status, by_dataset_status


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
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
          notes=coalesce(projects.notes, excluded.notes)
        """,
        (
            PROJECT_KEY,
            PROJECT_NAME,
            str(REPO),
            BRANCH,
            "Branch-scoped registry for feature/on-policy-wdl-sft results.",
        ),
    )
    return int(conn.execute("select id from projects where name=?", (PROJECT_NAME,)).fetchone()["id"])


def checkpoint_steps(root: Path) -> list[int]:
    if not root.exists():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("global_step_"):
            try:
                out.append(int(child.name.removeprefix("global_step_")))
            except ValueError:
                pass
    return sorted(out)


def add_validation_check(
    conn: sqlite3.Connection,
    name: str,
    source_path: Path,
    source_value: Any,
    db_value: Any,
    passed: bool,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        insert into validation_checks(check_name, source_path, source_value, database_value, passed, checked_at, notes)
        values (?, ?, ?, ?, ?, datetime('now'), ?)
        """,
        (name, str(source_path), str(source_value), str(db_value), 1 if passed else 0, notes),
    )


def import_run(conn: sqlite3.Connection, project_id: int, run: dict[str, Any], commit: str | None) -> int:
    run_name = run["run_name"]
    metrics_path = METRICS_ROOT / f"{run_name}.jsonl"
    validation_path = VALIDATION_ROOT / run_name / f"{run['final_eval_step']}.jsonl"
    log_path = REPO / run["log_rel"]
    checkpoint_root = Path(run["checkpoint_root"])
    rows = load_jsonl(metrics_path)
    final_step = int(run["final_eval_step"])
    actual_last_step = int(rows[-1]["step"])

    best_he_step, best_he = best_metric(rows, "val-core/HumanEval+/acc/pass@1")
    best_mbpp_step, best_mbpp = best_metric(rows, "val-core/MBPP+/acc/pass@1")
    final_he = metric_at(rows, final_step, "val-core/HumanEval+/acc/pass@1")
    final_mbpp = metric_at(rows, final_step, "val-core/MBPP+/acc/pass@1")
    final_correct = metric_at(rows, actual_last_step, "wdl_sft/correct_ratio")
    best_correct_step, best_correct = best_metric(rows, "wdl_sft/correct_ratio")
    final_clip = metric_at(rows, actual_last_step, "response_length/clip_ratio")
    final_response_mean = metric_at(rows, actual_last_step, "response_length/mean")
    final_he_extract = metric_at(rows, final_step, "val-aux/HumanEval+/code_reward_extraction_fail/mean@1")
    final_mbpp_extract = metric_at(rows, final_step, "val-aux/MBPP+/code_reward_extraction_fail/mean@1")
    best_ckpt_path = checkpoint_root / "best_checkpoint.json"
    best_ckpt = json.loads(best_ckpt_path.read_text(encoding="utf-8")) if best_ckpt_path.exists() else {}
    steps = checkpoint_steps(checkpoint_root)
    latest_step = max(steps) if steps else None

    exp_id = upsert_experiment(
        conn,
        project_id,
        run["experiment_key"],
        run["display_name"],
        method="on_policy_sft",
        method_family="on_policy_wdl_sft",
        method_variant="stage1_deepcoder_4k_negative_result",
        method_version="code_task_deepcoder_v1",
        domain="code",
        variant=run["beta_label"],
        status=run["status"],
        trust_level=run["trust_level"],
        trust_reason=run["trust_reason"],
        extra_json=json.dumps(
            {
                "run_name": run_name,
                "deepcoder_data_switch_negative_result": True,
                "checkpoint_prefix": str(checkpoint_root),
                "actual_last_step": actual_last_step,
                "planned_total_steps": run["planned_total_steps"],
                "final_eval_step": final_step,
                "best_checkpoint": best_ckpt,
                "reliability": run["reliability"],
                "report": str(REPORT_PATH),
            },
            ensure_ascii=False,
        ),
        notes=run["notes"],
    )

    for tag in ("code_task", "deepcoder", "negative_result", "deepcoder_data_switch"):
        add_tag(conn, "experiment", exp_id, tag)
    add_quality_flag(
        conn,
        "experiment",
        exp_id,
        "negative_transfer",
        "warning",
        "DeepCoder data switch was a negative result under the current 4K Stage1 setup.",
    )
    add_quality_flag(
        conn,
        "experiment",
        exp_id,
        "not_drop_in_replacement",
        "warning",
        "Do not treat DeepCoder-Preview as a drop-in replacement for KodCode in this setup.",
    )
    if run["trust_level"] == "buggy":
        add_quality_flag(conn, "experiment", exp_id, "format_collapse", "high", run["trust_reason"])
    if run["status"] != "completed":
        add_quality_flag(conn, "experiment", exp_id, "eval_insufficient", "warning", run["trust_reason"])

    dataset_id = upsert_dataset(
        conn,
        "code.deepcoder_preview_prompt1024.train",
        TRAIN_DATASET.name,
        domain="code",
        path=str(TRAIN_DATASET),
        split="train",
        row_count=19241,
        notes=f"DeepCoder-Preview prompt1024 train split; manifest {TRAIN_MANIFEST}",
    )
    he_ds = upsert_dataset(
        conn,
        "code.official_humaneval_plus.full",
        "HumanEval+",
        domain="code",
        path=str(HUMANEVAL_PLUS),
        split="validation",
        row_count=164,
    )
    mbpp_ds = upsert_dataset(
        conn,
        "code.official_mbpp_plus.full",
        "MBPP+",
        domain="code",
        path=str(MBPP_PLUS),
        split="validation",
        row_count=378,
    )

    model_id = None
    if latest_step is not None:
        model_row = conn.execute(
            "select id from models where model_key=?",
            (f"{run['experiment_key']}.global_step_{latest_step}",),
        ).fetchone()
        if model_row:
            model_id = int(model_row["id"])
        else:
            model_id = upsert_model(
                conn,
                f"{run['experiment_key']}.global_step_{latest_step}",
                str(checkpoint_root / f"global_step_{latest_step}" / "actor"),
                display_name=f"{run['display_name']} global_step_{latest_step}",
                base_model=BASE_MODEL,
                checkpoint_step=latest_step,
                checkpoint_kind="stage1_actor",
                model_role="actor",
                project_id=project_id,
                git_branch=BRANCH,
                git_commit=commit,
                is_best=1 if latest_step == best_ckpt.get("step") else 0,
                is_latest=1,
            )
    else:
        model_id = upsert_model(
            conn,
            f"{run['experiment_key']}.trainer_policy_step_{actual_last_step}",
            f"verl_training_run://{run_name}/trainer_policy_step_{actual_last_step}",
            display_name=f"{run['display_name']} trainer policy step {actual_last_step}",
            base_model=BASE_MODEL,
            checkpoint_step=actual_last_step,
            checkpoint_kind="trainer_policy_no_retained_checkpoint",
            model_role="trainer_policy",
            project_id=project_id,
            git_branch=BRANCH,
            git_commit=commit,
            extra_json=json.dumps(
                {"checkpoint_root_checked": str(checkpoint_root), "checkpoint_steps": steps}, ensure_ascii=False
            ),
            notes="No retained checkpoint directory found; model row anchors metrics to the trainer policy state reported by JSONL.",
        )

    train_run_key = f"verl.code_task.stage1.{run['beta_label']}.{run_name}.train"
    conn.execute(
        """
        insert into training_runs(
          training_run_key, experiment_id, output_model_id, train_dataset_id, method, framework, framework_version,
          beta, learning_rate, per_device_batch_size, gradient_accumulation_steps, effective_batch_size,
          max_length, weight_decay, lr_scheduler, distributed_backend, distributed_config_json, hyperparams_json,
          num_gpus, total_steps, raw_summary_path, wandb_run, git_branch, git_commit, extra_json, notes
        )
        values (?, ?, ?, ?, 'on_policy_sft', 'verl', '0.7-local', ?, 5e-7, 64, 1, 512, 4096, 0.1,
                'constant_with_warmup', 'fsdp+ray+vllm', ?, ?, 8, ?, ?, ?, ?, ?, ?, ?)
        on conflict(training_run_key) do update set
          experiment_id=excluded.experiment_id,
          output_model_id=excluded.output_model_id,
          train_dataset_id=excluded.train_dataset_id,
          beta=excluded.beta,
          total_steps=excluded.total_steps,
          raw_summary_path=excluded.raw_summary_path,
          wandb_run=excluded.wandb_run,
          git_branch=excluded.git_branch,
          git_commit=excluded.git_commit,
          extra_json=excluded.extra_json,
          notes=excluded.notes
        """,
        (
            train_run_key,
            exp_id,
            model_id,
            dataset_id,
            run["beta"],
            json.dumps({"backend": "fsdp+ray+vllm", "num_gpus": 8}, ensure_ascii=False),
            json.dumps(
                {
                    "loss_mode": "wdl_sft",
                    "beta": run["beta"],
                    "learning_rate": 5e-7,
                    "rollout_n": 8,
                    "max_response_length": 4096,
                    "max_prompt_length": 1024,
                    "val_n": 1,
                    "val_temperature": 0.2,
                    "val_top_p": 0.95,
                    "train_file": str(TRAIN_DATASET),
                    "reward_fn": "compute_score_code_official_aligned",
                },
                ensure_ascii=False,
            ),
            actual_last_step,
            str(metrics_path),
            run_name,
            BRANCH,
            commit,
            json.dumps(
                {
                    "checkpoint_root": str(checkpoint_root),
                    "log_path": str(log_path),
                    "final_correct_ratio": final_correct,
                    "final_response_clip_ratio": final_clip,
                    "final_response_length_mean": final_response_mean,
                    "best_correct_ratio": best_correct,
                    "best_correct_ratio_step": best_correct_step,
                },
                ensure_ascii=False,
            ),
            run["notes"],
        ),
    )
    tr_id = int(
        conn.execute("select id from training_runs where training_run_key=?", (train_run_key,)).fetchone()["id"]
    )

    eval_key = f"verl.code_task.deepcoder_4k.{run_name}.online_final_step{final_step}"
    eval_id = upsert_eval_run(
        conn,
        eval_run_key=eval_key,
        experiment_id=exp_id,
        model_id=model_id,
        eval_name=f"{run_name}_online_final_step{final_step}_n1",
        domain="code",
        script_path="recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py",
        parser_version="code-think-answer-python-v1",
        eval_harness="verl_online_code_validation",
        framework="verl",
        output_dir=str(validation_path.parent),
        raw_metrics_path=str(metrics_path),
        raw_samples_path=str(validation_path),
        n=1,
        num_samples=542,
        repeat_count=1,
        temperature=0.2,
        top_p=0.95,
        top_k=-1,
        do_sample=1,
        max_tokens=4096,
        max_prompt_tokens=1024,
        max_new_tokens=4096,
        seed=20260604,
        prompt_mode="code-think-answer-python-v1",
        command="trainer online validation",
        cwd=str(REPO),
        git_branch=BRANCH,
        git_commit=commit,
        trust_level=run["trust_level"],
        trust_reason=run["trust_reason"],
        extra_json=json.dumps(
            {
                "run_name": run_name,
                "checkpoint_prefix": str(checkpoint_root),
                "final_eval_step": final_step,
                "actual_last_step": actual_last_step,
                "best_humaneval_plus_step": best_he_step,
                "best_mbpp_plus_step": best_mbpp_step,
                "reliability": run["reliability"],
                "metric_source": "training JSONL online validation metrics plus final validation JSONL",
            },
            ensure_ascii=False,
        ),
        notes="DeepCoder data switch negative result; HumanEval+/MBPP+ are online EvalPlus validation pass@1 at n=1.",
    )

    metrics = [
        (he_ds, "pass@1", final_he, "final", "HumanEval+ final pass@1"),
        (he_ds, "best_pass@1", best_he, "best", f"HumanEval+ best pass@1 at step {best_he_step}"),
        (mbpp_ds, "pass@1", final_mbpp, "final", "MBPP+ final pass@1"),
        (mbpp_ds, "best_pass@1", best_mbpp, "best", f"MBPP+ best pass@1 at step {best_mbpp_step}"),
        (he_ds, "extraction_fail", final_he_extract, "final", "HumanEval+ final extraction failure rate"),
        (mbpp_ds, "extraction_fail", final_mbpp_extract, "final", "MBPP+ final extraction failure rate"),
        (
            dataset_id,
            "train_correct_ratio",
            final_correct,
            "final_train",
            f"Training correct_ratio at step {actual_last_step}",
        ),
        (
            dataset_id,
            "best_train_correct_ratio",
            best_correct,
            "best_train",
            f"Best training correct_ratio at step {best_correct_step}",
        ),
        (
            dataset_id,
            "response_clip_ratio",
            final_clip,
            "final_train",
            f"Training response clip_ratio at step {actual_last_step}",
        ),
        (
            dataset_id,
            "response_length_mean",
            final_response_mean,
            "final_train",
            f"Training response length mean at step {actual_last_step}",
        ),
    ]
    for ds_id, name, value, scope, notes in metrics:
        if value is not None:
            upsert_metric(conn, eval_id, ds_id, name, value, metric_scope=scope, notes=notes)

    total, status_counts, dataset_status = count_validation(validation_path)
    upsert_metric(
        conn, eval_id, he_ds, "num_examples", 164, metric_scope="final", notes="HumanEval+ online validation examples"
    )
    upsert_metric(
        conn, eval_id, mbpp_ds, "num_examples", 378, metric_scope="final", notes="MBPP+ online validation examples"
    )
    upsert_metric(
        conn,
        eval_id,
        None,
        "validation_dump_rows",
        total,
        metric_scope="final",
        notes="Final validation JSONL line count",
    )
    if total:
        upsert_metric(conn, eval_id, None, "validation_pass_count", status_counts.get("pass", 0), metric_scope="final")
        upsert_metric(
            conn, eval_id, None, "validation_non_pass_count", status_counts.get("non_pass", 0), metric_scope="final"
        )
    for dataset, counts in dataset_status.items():
        ds_id = he_ds if dataset == "HumanEval+" else mbpp_ds if dataset == "MBPP+" else None
        if ds_id is not None:
            upsert_metric(conn, eval_id, ds_id, "pass_count", counts.get("pass", 0), metric_scope="final_jsonl")
            upsert_metric(conn, eval_id, ds_id, "non_pass_count", counts.get("non_pass", 0), metric_scope="final_jsonl")

    add_artifact(
        conn,
        "training_metrics_jsonl",
        str(metrics_path),
        experiment_id=exp_id,
        training_run_id=tr_id,
        eval_run_id=eval_id,
        model_id=model_id,
    )
    add_artifact(
        conn,
        "validation_dump_jsonl",
        str(validation_path),
        experiment_id=exp_id,
        training_run_id=tr_id,
        eval_run_id=eval_id,
        model_id=model_id,
    )
    add_artifact(conn, "training_log", str(log_path), experiment_id=exp_id, training_run_id=tr_id)
    add_artifact(
        conn, "checkpoint_root", str(checkpoint_root), experiment_id=exp_id, training_run_id=tr_id, model_id=model_id
    )
    add_artifact(
        conn,
        "failure_analysis_report",
        str(REPORT_PATH),
        experiment_id=exp_id,
        training_run_id=tr_id,
        eval_run_id=eval_id,
    )
    if best_ckpt_path.exists():
        add_artifact(
            conn,
            "best_checkpoint_json",
            str(best_ckpt_path),
            experiment_id=exp_id,
            training_run_id=tr_id,
            model_id=model_id,
        )

    add_source_record(
        conn,
        str(metrics_path),
        "jsonl",
        "training_metrics_and_online_eval",
        IMPORTER,
        "eval_runs",
        eval_id,
        "Source for best/final HumanEval+/MBPP+, correct_ratio, and response clip ratio.",
    )
    add_source_record(
        conn,
        str(validation_path),
        "jsonl",
        "final_validation_dump",
        IMPORTER,
        "eval_runs",
        eval_id,
        "Final validation samples for HumanEval+/MBPP+ online validation.",
    )
    add_source_record(
        conn,
        str(log_path),
        "log",
        "training_log",
        IMPORTER,
        "training_runs",
        tr_id,
        "Runtime log for DeepCoder 4K negative-result run.",
    )
    add_source_record(
        conn,
        str(REPORT_PATH),
        "markdown",
        "negative_result_report",
        IMPORTER,
        "experiments",
        exp_id,
        "DeepCoder vs KodCode negative-result report.",
    )
    if best_ckpt_path.exists():
        add_source_record(
            conn,
            str(best_ckpt_path),
            "json",
            "best_checkpoint",
            IMPORTER,
            "models",
            model_id,
            "Best checkpoint marker.",
        )

    add_quality_flag(
        conn,
        "eval_run",
        eval_id,
        run["trust_level"],
        "warning" if run["trust_level"] != "buggy" else "high",
        run["trust_reason"],
    )
    add_quality_flag(
        conn,
        "eval_run",
        eval_id,
        "negative_result",
        "warning",
        "DeepCoder data switch negative result; compare with KodCode before interpreting.",
    )
    if final_clip is not None and final_clip > 0.25:
        add_quality_flag(
            conn,
            "eval_run",
            eval_id,
            "high_response_clip_ratio",
            "warning",
            f"Final response clip ratio is {final_clip:.4f}.",
        )
    if final_correct is not None and final_correct < 0.02:
        add_quality_flag(
            conn,
            "eval_run",
            eval_id,
            "low_correct_ratio",
            "warning",
            f"Final training correct_ratio is {final_correct:.4f}.",
        )
    if run["trust_level"] == "buggy":
        add_quality_flag(
            conn, "eval_run", eval_id, "format_collapse", "high", "Near-total extraction failure and response clipping."
        )

    add_validation_check(
        conn, f"{run_name}.final_he_pass1_metric", metrics_path, final_he, final_he, final_he is not None
    )
    add_validation_check(
        conn, f"{run_name}.final_mbpp_pass1_metric", metrics_path, final_mbpp, final_mbpp, final_mbpp is not None
    )
    add_validation_check(
        conn,
        f"{run_name}.final_correct_ratio_metric",
        metrics_path,
        final_correct,
        final_correct,
        final_correct is not None,
    )
    add_validation_check(
        conn,
        f"{run_name}.final_response_clip_ratio_metric",
        metrics_path,
        final_clip,
        final_clip,
        final_clip is not None,
    )
    add_validation_check(
        conn,
        f"{run_name}.final_validation_line_count",
        validation_path,
        total,
        total,
        total == 542,
        "Expected 164 HumanEval+ + 378 MBPP+ rows.",
    )

    return exp_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    if args.repo != REPO:
        raise SystemExit(f"This importer is scoped to {REPO}; got {args.repo}")
    check_release_gate([run["run_name"] for run in RUNS])

    init_db(args.db)
    commit = git_commit()
    with connect(args.db) as conn:
        project_id = upsert_project(conn)
        imported = [import_run(conn, project_id, run, commit) for run in RUNS]
        conn.commit()
    print(json.dumps({"db": args.db, "importer": IMPORTER, "experiment_ids": imported}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
