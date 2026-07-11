#!/usr/bin/env python3
"""Build the controlled adoption record for pre-existing Stage123 recipe files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED = (
    "on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh",
    "on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh",
    "on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh",
    "on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
    "on_policy_wdl_sft/code_task/stage123_gpu_idle_watchdog.py",
    "on_policy_wdl_sft/code_task/stage123_preflight.py",
)


def build(baseline: dict, recipe: Path) -> dict:
    source = {item["path"]: item for item in baseline["entries"]}
    if set(source) != set(ALLOWED):
        raise ValueError("recipe baseline dirty paths do not equal the controlled adoption set")
    entries = []
    for rel in ALLOWED:
        original = source[rel]; path = recipe / rel
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({
            "path": rel,
            "baseline": {key: original.get(key) for key in ("status", "type", "mode", "size", "sha256", "head_blob", "index_blob")},
            "result": {"type": "file", "mode": path.stat().st_mode & 0o777, "size": path.stat().st_size, "sha256": current},
            "changed": current != original["sha256"],
            "ownership": "experiment-execution-reliability-goal",
            "allowed_milestone": 2,
        })
    return {"schema_version": 1, "baseline_commit": baseline["head"], "baseline_aggregate_sha256": baseline["aggregate_sha256"], "allowed_paths": list(ALLOWED), "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True); parser.add_argument("--recipe", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = build(json.loads(args.baseline.read_text()), args.recipe)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "paths": len(result["entries"]), "changed": sum(item["changed"] for item in result["entries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
