#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Wait for and validate step 1 of the strict-scorer single-model Math A run."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_METRICS = {
    "training/global_step",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/optimizer_step_applied",
    "wdl_sft/n_correct",
    "wdl_sft/n_incorrect",
    "wdl_sft/positive_supervised_response_count",
    "wdl_sft/positive_supervised_token_count",
    "response_length/clip_ratio",
}


def _slurm_job_alive(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() in {"RUNNING", "COMPLETING"}


def _find_step_one(metrics_root: Path, project: str, run_prefix: str) -> tuple[Path, dict] | None:
    directory = metrics_root / project
    paths = sorted(directory.glob(f"{run_prefix}_*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths:
        for raw_line in path.read_text().splitlines():
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            data = payload.get("data", {})
            if int(payload.get("step", -1)) == 1 and int(data.get("training/global_step", -1)) == 1:
                return path, data
    return None


def validate_step_one(data: dict, expected_responses: int = 512) -> dict[str, object]:
    missing = sorted(REQUIRED_METRICS - data.keys())
    grad = float(data.get("actor/grad_norm", float("nan")))
    loss = float(data.get("actor/pg_loss", float("nan")))
    n_correct = int(data.get("wdl_sft/n_correct", -1))
    n_incorrect = int(data.get("wdl_sft/n_incorrect", -1))
    checks = {
        "all_required_metrics_present": not missing,
        "optimizer_step_applied": float(data.get("actor/optimizer_step_applied", 0.0)) == 1.0,
        "finite_nonzero_actor_grad": math.isfinite(grad) and grad > 0.0,
        "finite_actor_loss": math.isfinite(loss),
        "full_rollout_batch_scored": n_correct + n_incorrect == expected_responses,
        "positive_supervised_signal": int(data.get("wdl_sft/positive_supervised_response_count", 0)) > 0
        and int(data.get("wdl_sft/positive_supervised_token_count", 0)) > 0,
    }
    return {**checks, "missing_metrics": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_seconds
    result = None
    failure_reason = None
    while time.monotonic() < deadline:
        result = _find_step_one(args.metrics_root, args.project, args.run_prefix)
        if result is not None:
            break
        if not _slurm_job_alive(args.slurm_job_id):
            failure_reason = "Slurm job exited before step 1 metrics appeared"
            break
        time.sleep(args.poll_seconds)

    receipt: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_prefix": args.run_prefix,
    }
    if result is None:
        receipt.update(status="fail", failure_reason=failure_reason or "timed out waiting for step 1")
    else:
        metrics_path, data = result
        checks = validate_step_one(data)
        passed = all(value for key, value in checks.items() if key != "missing_metrics")
        receipt.update(
            status="pass" if passed else "fail",
            metrics_path=str(metrics_path),
            checks=checks,
            observed={key: data.get(key) for key in sorted(REQUIRED_METRICS)},
        )

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
