#!/usr/bin/env python3
"""Wait for and validate the first optimizer step of a Math WDL causal arm."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_METRICS = {
    "training/global_step",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/grad_clip_event",
    "actor/optimizer_step_applied",
    "wdl_sft/n_correct",
    "wdl_sft/n_incorrect",
    "wdl_sft/all_incorrect_group_ratio",
    "wdl_sft/mixed_group_ratio",
    "wdl_sft/all_correct_group_ratio",
    "wdl_sft/positive_supervised_response_count",
    "wdl_sft/positive_supervised_token_count",
    "jointTraining/model1_grad_norm",
    "jointTraining/model2_grad_norm",
    "jointTraining/fused_vs_model2_chosen_token_logprob_delta_mean",
    "jointTraining/fused_vs_model2_chosen_token_logprob_abs_mean",
    "response_length/p50",
    "response_length/p95",
    "response_length/clip_ratio",
}


def _tmux_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _find_step_one(metrics_root: Path, project: str, run_prefix: str) -> tuple[Path, dict] | None:
    directory = metrics_root / project
    for path in sorted(directory.glob(f"{run_prefix}_*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
        for raw_line in path.read_text().splitlines():
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            data = payload.get("data", {})
            if int(payload.get("step", -1)) == 1 and int(data.get("training/global_step", -1)) == 1:
                return path, data
    return None


def validate_step_one(data: dict, expected_model1_gradient: str) -> dict[str, bool]:
    missing = sorted(REQUIRED_METRICS - data.keys())
    model1_grad = float(data.get("jointTraining/model1_grad_norm", float("nan")))
    model2_grad = float(data.get("jointTraining/model2_grad_norm", float("nan")))
    checks = {
        "all_required_metrics_present": not missing,
        "optimizer_step_applied": float(data.get("actor/optimizer_step_applied", 0.0)) == 1.0,
        "finite_nonzero_actor_grad": 0.0 < float(data.get("actor/grad_norm", 0.0)) < float("inf"),
        "nonzero_model2_gradient": 0.0 < model2_grad < float("inf"),
        "positive_supervised_signal": int(data.get("wdl_sft/positive_supervised_response_count", 0)) > 0
        and int(data.get("wdl_sft/positive_supervised_token_count", 0)) > 0,
    }
    if expected_model1_gradient == "zero":
        checks["model1_gradient_matches_arm"] = abs(model1_grad) <= 1e-12
    else:
        checks["model1_gradient_matches_arm"] = 0.0 < model1_grad < float("inf")
    checks["no_missing_metrics"] = not missing
    return {**checks, "missing_metrics": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--expected-model1-gradient", choices=("zero", "nonzero"), required=True)
    parser.add_argument("--queue-tmux", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
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
        if not _tmux_alive(args.queue_tmux):
            failure_reason = "queue tmux exited before step 1 metrics appeared"
            break
        time.sleep(args.poll_seconds)

    receipt: dict = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_prefix": args.run_prefix,
        "expected_model1_gradient": args.expected_model1_gradient,
    }
    if result is None:
        receipt.update(status="fail", failure_reason=failure_reason or "timed out waiting for step 1")
    else:
        metrics_path, data = result
        checks = validate_step_one(data, args.expected_model1_gradient)
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
