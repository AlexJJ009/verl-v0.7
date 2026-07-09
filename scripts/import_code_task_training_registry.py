#!/usr/bin/env python3
"""Import a completed code-task training run into the local experiment registry."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    slug,
    upsert_dataset,
    upsert_experiment,
    upsert_model,
)


REPO = Path("/data-1/verl07/verl")
DB = "/data-1/experiment_registry/experiment_registry.sqlite"
PROJECT_NAME = "verl:feature/on-policy-wdl-sft"
BRANCH = "feature/on-policy-wdl-sft"
IMPORTER = "import_code_task_training_registry_v1"
RELEASE_GATE_SCRIPT = REPO / "scripts/training_result_release_gate.py"
BASE_MODEL_INSTRUCT2507 = "/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_at(rows: list[dict[str, Any]], step: int, key: str) -> float | None:
    for row in rows:
        if int(row.get("step", -1)) == step and key in row.get("data", {}):
            return row["data"][key]
    return None


def best_metric(rows: list[dict[str, Any]], key: str) -> tuple[int | None, float | None]:
    best_step = None
    best_val = None
    for row in rows:
        data = row.get("data", {})
        if key not in data:
            continue
        val = data[key]
        if best_val is None or val > best_val:
            best_step = int(row["step"])
            best_val = val
    return best_step, best_val


def final_training_step(rows: list[dict[str, Any]]) -> int:
    train_steps = [int(r["step"]) for r in rows if "wdl_sft/correct_ratio" in r.get("data", {})]
    return max(train_steps) if train_steps else int(rows[-1]["step"])


def validation_steps(rows: list[dict[str, Any]]) -> list[int]:
    return sorted(
        int(r["step"])
        for r in rows
        if any(k.startswith("val-core/") for k in r.get("data", {}))
    )


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def infer_run(run_name: str) -> dict[str, Any]:
    beta = None
    beta_label = "unknown"
    if "BETA05" in run_name:
        beta = 0.5
        beta_label = "beta05"
    elif "BETA01" in run_name:
        beta = 0.1
        beta_label = "beta01"
    elif "BETA0" in run_name:
        beta = 0.0
        beta_label = "beta0"

    stage = "stage2" if "CODE-S2-" in run_name else "stage1"
    handoff = None
    m = re.search(r"-P(\d+)-", run_name)
    if m:
        handoff = int(m.group(1))
    dataset = "kodcode" if "KODCODE" in run_name else "deepcoder" if "DEEPCODER" in run_name else "code"
    init = "qwen3-4b-instruct-2507" if "INSTRUCT2507" in run_name else "qwen3-4b-base"
    ctx = "ctx8k" if "CTX8K" in run_name or "R8K" in run_name else "ctx4k"
    suffix = slug(run_name)
    return {
        "beta": beta,
        "beta_label": beta_label,
        "stage": stage,
        "handoff": handoff,
        "dataset": dataset,
        "init": init,
        "ctx": ctx,
        "experiment_key": f"verl.on_policy_wdl_sft.code.{dataset}.{init}.{ctx}.{stage}.{beta_label}.{suffix}",
        "display_name": f"{stage.upper()} {dataset} {init} {ctx} {beta_label} {run_name}",
    }


def upsert_project_local(conn) -> int:
    conn.execute(
        """
        insert into projects(project_key, name, repo_path, default_branch, notes)
        values (?, ?, ?, ?, ?)
        on conflict(name) do update set
          repo_path=excluded.repo_path,
          default_branch=excluded.default_branch,
          notes=coalesce(excluded.notes, projects.notes)
        """,
        ("verl_feature_on_policy_wdl_sft", PROJECT_NAME, str(REPO), BRANCH, "Branch-scoped registry for On-Policy WDL-SFT runs."),
    )
    return int(conn.execute("select id from projects where name=?", (PROJECT_NAME,)).fetchone()["id"])


def find_validation_path(run_name: str, step: int) -> Path | None:
    roots = [
        REPO / "recipe/on_policy_wdl_sft/code_task/validation",
        REPO / "recipe/on_policy_wdl_sft/staged_v1/validation",
    ]
    for root in roots:
        p = root / run_name / f"{step}.jsonl"
        if p.exists():
            return p
    return None


def find_wandb_run(run_name: str, run_prefix: str | None) -> str | None:
    roots = []
    if run_prefix:
        roots.append(Path("/data-1/wandb_runs") / run_prefix / "wandb")
    roots.append(Path("/data-1/wandb_runs"))
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/offline-run-*"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def import_run(args: argparse.Namespace) -> int:
    run_name = args.run_name
    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        raise SystemExit(f"metrics path not found: {metrics_path}")
    if args.release_gate:
        subprocess.check_call([sys.executable, str(RELEASE_GATE_SCRIPT), "check", "--run-name", run_name], cwd=REPO)

    rows = load_jsonl(metrics_path)
    info = infer_run(run_name)
    final_step = args.final_step or final_training_step(rows)
    val_steps = validation_steps(rows)
    final_eval_step = max([s for s in val_steps if s <= final_step], default=final_step)
    commit = git_commit()
    run_prefix = args.run_prefix or run_name.rsplit("_", 1)[0]

    best_he_step, best_he = best_metric(rows, "val-core/HumanEval+/acc/pass@1")
    best_mbpp_step, best_mbpp = best_metric(rows, "val-core/MBPP+/acc/pass@1")
    best_lcb_step, best_lcb = best_metric(rows, "val-core/LiveCodeBench/acc/pass@1")
    final_metrics = {
        "HumanEval+": metric_at(rows, final_eval_step, "val-core/HumanEval+/acc/pass@1"),
        "MBPP+": metric_at(rows, final_eval_step, "val-core/MBPP+/acc/pass@1"),
        "LiveCodeBench": metric_at(rows, final_eval_step, "val-core/LiveCodeBench/acc/pass@1"),
    }
    final_train = next((r.get("data", {}) for r in reversed(rows) if "wdl_sft/correct_ratio" in r.get("data", {})), {})
    best_ckpt_path = checkpoint_dir / "best_checkpoint.json"
    best_ckpt = json.loads(best_ckpt_path.read_text(encoding="utf-8")) if best_ckpt_path.exists() else {}
    latest_step_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    latest_step = int(latest_step_path.read_text().strip()) if latest_step_path.exists() else final_step
    validation_path = find_validation_path(run_name, final_eval_step)
    wandb_run = args.wandb_run or find_wandb_run(run_name, run_prefix)

    init_db(args.db)
    with connect(args.db) as conn:
        project_id = upsert_project_local(conn)
        exp_id = upsert_experiment(
            conn,
            project_id,
            info["experiment_key"],
            info["display_name"],
            method="on_policy_wdl_sft",
            method_family="on_policy_wdl_sft",
            method_variant=f"{info['stage']}_{info['dataset']}_{info['ctx']}",
            method_version="code_task_v1",
            domain="code",
            variant=info["beta_label"],
            status="completed",
            trust_level="usable_with_caution",
            trust_reason="Online validation and training metrics imported automatically after release gate; offline official eval remains the stronger decision evidence.",
            extra_json=json.dumps(
                {
                    "run_name": run_name,
                    "run_prefix": run_prefix,
                    "stage": info["stage"],
                    "handoff_step": info["handoff"],
                    "final_step": final_step,
                    "final_eval_step": final_eval_step,
                    "best_checkpoint": best_ckpt,
                    "git_branch": BRANCH,
                    "git_commit": commit,
                },
                ensure_ascii=False,
            ),
            notes="Auto-imported by code-task training release hook.",
        )
        for tag in ["code_task", info["dataset"], info["stage"], info["beta_label"], "auto_release_hook"]:
            add_tag(conn, "experiment", exp_id, tag)

        dataset_id = upsert_dataset(
            conn,
            f"code.{info['dataset']}.train",
            f"{info['dataset']} train",
            domain="code",
            path=str(args.train_file) if args.train_file else None,
            split="train",
            notes="Training dataset inferred/imported by code-task training release hook.",
        )
        model_id = upsert_model(
            conn,
            f"{info['experiment_key']}.global_step_{latest_step}",
            str(checkpoint_dir / f"global_step_{latest_step}" / "actor"),
            display_name=f"{info['display_name']} global_step_{latest_step}",
            base_model=BASE_MODEL_INSTRUCT2507 if info["init"] == "qwen3-4b-instruct-2507" else None,
            checkpoint_step=latest_step,
            checkpoint_kind=f"{info['stage']}_actor",
            model_role="actor",
            project_id=project_id,
            git_branch=BRANCH,
            git_commit=commit,
            is_best=1 if best_ckpt.get("step") == latest_step else 0,
            is_latest=1,
            extra_json=json.dumps({"checkpoint_dir": str(checkpoint_dir)}, ensure_ascii=False),
        )

        train_run_key = f"verl.code_task.training.{slug(run_name)}"
        hyperparams = {
            "loss_mode": "wdl_sft",
            "beta": info["beta"],
            "learning_rate": 5e-7,
            "rollout_n": 8,
            "max_response_length": 4096,
            "max_prompt_length": 1024,
            "val_n": 1,
            "val_temperature": 0.2,
            "val_top_p": 0.95,
            "online_validation": ["HumanEval+", "MBPP+", "LiveCodeBench"],
        }
        conn.execute(
            """
            insert into training_runs(
              training_run_key, experiment_id, output_model_id, train_dataset_id, method, framework, framework_version,
              beta, learning_rate, per_device_batch_size, gradient_accumulation_steps, effective_batch_size,
              max_length, weight_decay, lr_scheduler, distributed_backend, distributed_config_json, hyperparams_json,
              num_gpus, total_steps, raw_summary_path, wandb_run, git_branch, git_commit, extra_json, notes
            )
            values (?, ?, ?, ?, 'on_policy_wdl_sft', 'verl', '0.7-local', ?, 5e-7, 64, 1, 512, 4096, 0.1,
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
                info["beta"],
                json.dumps({"backend": "fsdp+ray+vllm", "num_gpus": 8}, ensure_ascii=False),
                json.dumps(hyperparams, ensure_ascii=False),
                final_step,
                str(metrics_path),
                wandb_run,
                BRANCH,
                commit,
                json.dumps(
                    {
                        "checkpoint_dir": str(checkpoint_dir),
                        "validation_path": str(validation_path) if validation_path else None,
                        "best_humaneval_step": best_he_step,
                        "best_humaneval": best_he,
                        "best_mbpp_step": best_mbpp_step,
                        "best_mbpp": best_mbpp,
                        "best_lcb_step": best_lcb_step,
                        "best_lcb": best_lcb,
                        "final_validation": final_metrics,
                        "final_correct_ratio": final_train.get("wdl_sft/correct_ratio"),
                        "final_response_clip_ratio": final_train.get("response_length/clip_ratio"),
                        "final_response_length_mean": final_train.get("response_length/mean"),
                    },
                    ensure_ascii=False,
                ),
                "Auto-imported after training release gate success. Treat as online-training evidence, not offline official eval.",
            ),
        )
        tr_id = int(conn.execute("select id from training_runs where training_run_key=?", (train_run_key,)).fetchone()["id"])
        conn.execute(
            """
            insert into training_run_datasets(training_run_id, dataset_id, role, notes)
            values (?, ?, 'train', 'Auto-linked by code-task training release hook.')
            on conflict(training_run_id, dataset_id, role) do update set notes=excluded.notes
            """,
            (tr_id, dataset_id),
        )
        metric_rows = []
        for row in rows:
            step = int(row["step"])
            for key, value in row.get("data", {}).items():
                if isinstance(value, (int, float)):
                    metric_rows.append((tr_id, key, float(value), step, "training_metrics_jsonl", "auto-imported from metrics jsonl"))
        conn.executemany(
            """
            insert into training_metrics(training_run_id, metric_name, metric_value, step, metric_scope, notes)
            values (?, ?, ?, ?, ?, ?)
            on conflict(training_run_id, metric_name, step, metric_scope) do update set
              metric_value=excluded.metric_value,
              notes=excluded.notes
            """,
            metric_rows,
        )
        add_source_record(conn, str(metrics_path), "jsonl", "full", IMPORTER, "training_runs", tr_id, "metrics JSONL")
        if validation_path:
            add_artifact(conn, "online_validation_samples", str(validation_path), experiment_id=exp_id, training_run_id=tr_id, model_id=model_id)
            add_source_record(conn, str(validation_path), "jsonl", f"step_{final_eval_step}", IMPORTER, "training_runs", tr_id, "online validation samples")
        add_artifact(conn, "checkpoint_dir", str(checkpoint_dir), experiment_id=exp_id, training_run_id=tr_id, model_id=model_id)
        if info["stage"] == "stage2":
            add_quality_flag(conn, "experiment", exp_id, "stage2_online_only", "info", "Stage2 result is online validation evidence until offline official eval is run.")
        conn.commit()

    print(json.dumps({"training_run_id": tr_id, "training_run_key": train_run_key, "run_name": run_name}, ensure_ascii=False))
    return tr_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"), required=not os.environ.get("RUN_NAME"))
    parser.add_argument("--run-prefix", default=os.environ.get("RUN_PREFIX"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR"), required=not os.environ.get("CHECKPOINT_DIR"))
    parser.add_argument("--metrics-path", default=os.environ.get("METRICS_PATH"), required=not os.environ.get("METRICS_PATH"))
    parser.add_argument("--final-step", type=int, default=int(os.environ["FINAL_STEP"]) if os.environ.get("FINAL_STEP", "").isdigit() else None)
    parser.add_argument("--train-file", default=os.environ.get("TRAIN_FILE"))
    parser.add_argument("--wandb-run", default=os.environ.get("WANDB_RUN_DIR"))
    parser.add_argument("--db", default=DB)
    parser.add_argument("--no-release-gate", action="store_false", dest="release_gate")
    parser.set_defaults(release_gate=True)
    args = parser.parse_args()
    import_run(args)


if __name__ == "__main__":
    main()
