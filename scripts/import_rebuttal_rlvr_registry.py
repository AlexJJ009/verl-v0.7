#!/usr/bin/env python3
"""Import one release-gated rebuttal RLVR run into the local registry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REGISTRY_TOOLS = Path(os.environ.get("EXPERIMENT_REGISTRY_TOOLS", "/data-1/agent-tools/experiment_registry"))
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


REPO = Path(os.environ.get("VERL_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
DB = os.environ.get("EXPERIMENT_REGISTRY_DB", "/data-1/experiment_registry/experiment_registry.sqlite")
RELEASE_GATE_SCRIPT = REPO / "scripts/training_result_release_gate.py"
RELEASE_GATE_STATE = os.environ.get("TRAINING_RELEASE_GATE_STATE")
PROJECT_NAME = "verl:rebuttal-rlvr"
IMPORTER = "import_rebuttal_rlvr_registry_v1"


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"metrics file is empty: {path}")
    return rows


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def upsert_project_local(conn: Any) -> int:
    branch = git_value("branch", "--show-current")
    conn.execute(
        """
        insert into projects(project_key, name, repo_path, default_branch, notes)
        values (?, ?, ?, ?, ?)
        on conflict(name) do update set repo_path=excluded.repo_path,
          default_branch=excluded.default_branch, notes=excluded.notes
        """,
        ("verl_rebuttal_rlvr", PROJECT_NAME, str(REPO), branch, "Formal ordinary-SFT versus WDL-SFT RLVR comparison."),
    )
    return int(conn.execute("select id from projects where name=?", (PROJECT_NAME,)).fetchone()["id"])


def import_run(args: argparse.Namespace) -> int:
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    metrics_path = Path(args.metrics_path).resolve()
    init_model = Path(args.init_model_path).resolve()
    train_file = Path(args.train_file).resolve()
    for path, label in ((checkpoint_dir, "checkpoint"), (init_model, "init model")):
        if not path.is_dir():
            raise SystemExit(f"{label} directory missing: {path}")
    for path, label in ((metrics_path, "metrics"), (train_file, "train file")):
        if not path.is_file():
            raise SystemExit(f"{label} file missing: {path}")
    gate_command = [sys.executable, str(RELEASE_GATE_SCRIPT)]
    if RELEASE_GATE_STATE:
        gate_command.extend(["--state", RELEASE_GATE_STATE])
    gate_command.extend(["check", "--run-name", args.run_name])
    subprocess.check_call(gate_command, cwd=REPO)

    rows = load_metrics(metrics_path)
    commit = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    classifier = "ordinary_sft" if args.arm == "sft" else "offline_wdl_sft"
    exp_key = f"verl.rebuttal_rlvr.{args.arm}.{slug(args.init_pair)}.r{args.rl_seed}"
    training_key = f"verl.rebuttal_rlvr.training.{slug(args.run_name)}"
    final_values = next((row.get("data", {}) for row in reversed(rows) if int(row.get("step", -1)) <= args.final_step), {})

    init_db(args.db)
    with connect(args.db) as conn:
        project_id = upsert_project_local(conn)
        experiment_id = upsert_experiment(
            conn,
            project_id,
            exp_key,
            args.run_name,
            method="grpo",
            method_family="rebuttal_rlvr",
            method_variant=classifier,
            method_version="rebuttal-standard-grpo-v2",
            domain="math",
            variant=f"{args.init_pair}-r{args.rl_seed}",
            status="completed",
            trust_level="usable_with_caution",
            trust_reason="Release-gated online training evidence; confirmatory offline Math-7 analysis is deferred.",
            extra_json=json.dumps({"arm": args.arm, "init_pair": args.init_pair, "rl_seed": args.rl_seed, "final_step": args.final_step}),
            notes="Automatically imported after terminal success_complete.",
        )
        for tag in ("rebuttal_rlvr", args.arm, classifier, "auto_release_hook", "math"):
            add_tag(conn, "experiment", experiment_id, tag)

        dataset_id = upsert_dataset(
            conn,
            "math.hendrycks_math_7500_rlvr",
            "Hendrycks MATH 7,500-row RLVR train",
            domain="math",
            path=str(train_file),
            split="train",
            notes="Shared downstream GRPO file for both rebuttal arms; distinct from AM-1.4M initialization SFT data.",
        )
        input_model_id = upsert_model(
            conn,
            f"rebuttal_rlvr.init.{args.arm}.{args.init_pair}",
            str(init_model),
            display_name=f"{args.init_pair} {classifier} initialization",
            checkpoint_kind="initialization",
            model_role="actor_init",
            project_id=project_id,
            git_branch=branch,
            git_commit=commit,
            extra_json=json.dumps({"classifier": classifier, "initialization_dataset": "AM-1.4M"}),
        )
        output_model_id = upsert_model(
            conn,
            f"{exp_key}.global_step_{args.final_step}",
            str(checkpoint_dir / f"global_step_{args.final_step}" / "actor"),
            display_name=f"{args.run_name} final step {args.final_step}",
            base_model=str(init_model),
            checkpoint_step=args.final_step,
            checkpoint_kind="grpo_actor",
            model_role="actor",
            project_id=project_id,
            git_branch=branch,
            git_commit=commit,
            is_best=0,
            is_latest=1,
            extra_json=json.dumps({"checkpoint_dir": str(checkpoint_dir)}),
        )
        conn.execute(
            """
            insert into training_runs(
              training_run_key, experiment_id, input_model_id, output_model_id,
              train_dataset_id, method, framework, framework_version,
              learning_rate, effective_batch_size, max_length, weight_decay,
              lr_scheduler, distributed_backend, distributed_config_json,
              hyperparams_json, num_gpus, total_steps, raw_summary_path,
              wandb_run, git_branch, git_commit, extra_json, notes
            ) values (?, ?, ?, ?, ?, 'grpo', 'verl', '0.7-local', 5e-7, 512,
              4096, 0.1, 'constant', 'fsdp+ray+vllm', ?, ?, 8, ?, ?, ?, ?, ?, ?, ?)
            on conflict(training_run_key) do update set
              experiment_id=excluded.experiment_id,
              input_model_id=excluded.input_model_id,
              output_model_id=excluded.output_model_id,
              train_dataset_id=excluded.train_dataset_id,
              total_steps=excluded.total_steps,
              raw_summary_path=excluded.raw_summary_path,
              wandb_run=excluded.wandb_run,
              extra_json=excluded.extra_json,
              notes=excluded.notes
            """,
            (
                training_key, experiment_id, input_model_id, output_model_id, dataset_id,
                json.dumps({"backend": "fsdp+ray+vllm", "num_gpus": 8}),
                json.dumps({"config_version": "rebuttal-standard-grpo-v2", "rollout_n": 8, "val_n": 3, "data_shuffle": False}),
                args.final_step, str(metrics_path), args.wandb_run, branch, commit,
                json.dumps({"arm": args.arm, "init_pair": args.init_pair, "rl_seed": args.rl_seed, "final_metrics": final_values}),
                "Auto-imported after release gate success; offline Math-7 result analysis remains deferred.",
            ),
        )
        training_run_id = int(conn.execute("select id from training_runs where training_run_key=?", (training_key,)).fetchone()["id"])
        metric_rows = []
        for row in rows:
            step = int(row.get("step", -1))
            for name, value in row.get("data", {}).items():
                if isinstance(value, (int, float)):
                    metric_rows.append((training_run_id, name, float(value), step, "training_metrics_jsonl", "auto-imported"))
        conn.executemany(
            """
            insert into training_metrics(training_run_id, metric_name, metric_value, step, metric_scope, notes)
            values (?, ?, ?, ?, ?, ?)
            on conflict(training_run_id, metric_name, step, metric_scope) do update set metric_value=excluded.metric_value, notes=excluded.notes
            """,
            metric_rows,
        )
        add_source_record(conn, str(metrics_path), "jsonl", "full", IMPORTER, "training_runs", training_run_id, "Online training and validation metrics.")
        add_artifact(conn, "checkpoint_dir", str(checkpoint_dir), experiment_id=experiment_id, training_run_id=training_run_id, model_id=output_model_id)
        add_quality_flag(conn, "experiment", experiment_id, "offline_math7_deferred", "info", "Confirmatory offline Math-7 analysis will be added after returned checkpoints/logs are analyzed.")
        conn.commit()

    print(json.dumps({"training_run_id": training_run_id, "training_run_key": training_key, "run_name": args.run_name}, sort_keys=True))
    return training_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arm", choices=("sft", "wdl"), required=True)
    parser.add_argument("--init-pair", required=True)
    parser.add_argument("--rl-seed", type=int, required=True)
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--wandb-run")
    parser.add_argument("--final-step", type=int, default=115)
    parser.add_argument("--db", default=DB)
    import_run(parser.parse_args())


if __name__ == "__main__":
    main()
