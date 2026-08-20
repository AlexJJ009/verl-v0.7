#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibration_probe_zero_step import run_repetition, sha256, split_workload, write_json  # noqa: E402

DEFAULT_RUN_IDS = ("frac25-stage2-nokl", "frac25-stage2-m2kl")
EXPECTED_STAGE2_GENERATIONS = 64 * 3 * 2


def render_matrix(path: Path) -> dict:
    output = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/stage123_matrix_manifest.py"), "render", str(path)], text=True
    )
    return json.loads(output)


def render_base(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    base_path = Path(raw["base_manifest"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    output = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", str(base_path), "--format", "json"],
        text=True,
    )
    return json.loads(output)


def run_environment(run: dict, profile: dict | None = None) -> dict[str, str]:
    source = run["source"]
    kl = run["submodel_kl"]
    profile = profile or {
        "sha256": "",
        "rollout_gpu_memory_utilization": 0.4,
        "rollout_max_num_batched_tokens": 32768,
        "rollout_free_cache_engine": False,
        "rollout_enable_sleep_mode": False,
        "ref_fsdp_offload": True,
        "actor_param_offload": True,
        "actor_optimizer_offload": True,
        "ref_log_prob_micro_batch_size": 1,
        "ref_log_prob_max_token_len_per_gpu": 9216,
    }
    return {
        "BASE_MODEL_PATH": source["model1_path"],
        "EXPECTED_MODEL1_PATH": source["model1_path"],
        "EXPECTED_MODEL1_CONFIG_SHA256": source["model1_config_sha256"],
        "EXPECTED_MODEL1_TOKENIZER_CONFIG_SHA256": source["model1_tokenizer_config_sha256"],
        "EXPECTED_MODEL1_CHAT_TEMPLATE_SHA256": source["model1_chat_template_sha256"],
        "EXPECTED_MODEL1_PROVENANCE_PATH": source["model1_provenance_path"],
        "EXPECTED_MODEL1_PROVENANCE_SHA256": source["model1_provenance_sha256"],
        "CALIBRATION_STAGE1_CKPT_DIR": source["checkpoint_root"],
        "CALIBRATION_STAGE1_MODEL2": source["model2_path"],
        "CALIBRATION_STAGE1_RUN_PREFIX": source["run_prefix"],
        "CALIBRATION_STAGE1_HANDOFF_STEP": str(source["handoff_step"]),
        "CALIBRATION_TRAIN_FILE": run["train_file"],
        "SUBMODEL_KL_ENABLED": str(bool(kl["enabled"])).lower(),
        "SUBMODEL_KL_MODEL1_ENABLED": str(bool(kl["model1_enabled"])).lower(),
        "SUBMODEL_KL_MODEL1_COEF": str(kl["model1_coef"]),
        "SUBMODEL_KL_MODEL2_ENABLED": str(bool(kl["model2_enabled"])).lower(),
        "SUBMODEL_KL_MODEL2_COEF": str(kl["model2_coef"]),
        "SUBMODEL_KL_MODEL2_REF_PATH": kl.get("model2_ref_path", source["model2_path"]),
        "JOINT_VALIDATION_VIEWS": "[model1,model2]",
        "STAGE123_EXPECTED_VAL_N": "3",
        "VAL_N": "3",
        "VAL_TEMPERATURE": "0.2",
        "VAL_TOP_P": "0.95",
        "VAL_DO_SAMPLE": "True",
        "STAGE123_EXPECTED_PROFILE_HASH": str(profile["sha256"]),
        "ROLLOUT_GPU_MEMORY_UTILIZATION": f"{float(profile['rollout_gpu_memory_utilization']):.2f}",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(profile["rollout_max_num_batched_tokens"]),
        "ROLLOUT_FREE_CACHE_ENGINE": str(profile["rollout_free_cache_engine"]),
        "ROLLOUT_ENABLE_SLEEP_MODE": str(profile["rollout_enable_sleep_mode"]),
        "REF_FSDP_OFFLOAD": str(profile["ref_fsdp_offload"]),
        "FSDP_OFFLOAD": str(profile["actor_param_offload"]),
        "FSDP_OPTIMIZER_OFFLOAD": str(profile["actor_optimizer_offload"]),
        "REF_LOG_PROB_MICRO_BATCH_SIZE": str(profile["ref_log_prob_micro_batch_size"]),
        "REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": str(profile["ref_log_prob_max_token_len_per_gpu"]),
    }


def summarize(run: dict, repetitions: list[dict], minimum_headroom_mib: int) -> dict:
    resources = [item["resources"] for item in repetitions if item.get("resources")]
    peak = max((item.get("peak_gpu_memory_used_mib") or 0 for item in resources), default=0)
    totals = [
        gpu["total_memory_mib"]
        for item in resources
        for gpu in item.get("per_gpu_memory", [])
        if gpu.get("total_memory_mib")
    ]
    total = min(totals) if totals else 0
    headroom = total - peak if total else 0
    passed = bool(repetitions) and all(item["status"] == "passed" for item in repetitions)
    passed = passed and total > 0 and headroom >= minimum_headroom_mib
    return {
        "run_id": run["id"],
        "submodel_kl": run["submodel_kl"],
        "status": "passed" if passed else "failed",
        "peak_gpu_memory_used_mib": peak,
        "gpu_total_memory_mib": total,
        "minimum_gpu_headroom_mib": headroom,
        "required_minimum_headroom_mib": minimum_headroom_mib,
        "repetitions": repetitions,
    }


def qualify_matrix_repetition(result: dict) -> dict:
    views = {
        view
        for path in result.get("validation_generation_files", [])
        for view in ("model1", "model2")
        if f"/{view}/" in path
    }
    passed = (
        result.get("returncode") == 0
        and result.get("timed_out") is False
        and result.get("generation_count", 0) >= EXPECTED_STAGE2_GENERATIONS
        and views == {"model1", "model2"}
        and not result.get("formal_checkpoint_files")
        and result.get("cleanup", {}).get("resources_released") is True
        and (result.get("resources") or {}).get("peak_gpu_memory_used_mib") is not None
    )
    result = dict(result)
    result["matrix_score_complete"] = passed
    result["status"] = "passed" if passed else "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-ids", default=",".join(DEFAULT_RUN_IDS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--minimum-headroom-mib", type=int)
    parser.add_argument("--reuse-repetition", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 3:
        raise SystemExit("repetitions must be 1..3")
    if not str(args.scratch_root).startswith("/data-1/tmp/verl_agent_scratch/"):
        raise SystemExit("scratch root must be under /data-1/tmp/verl_agent_scratch")

    run_ids = args.run_ids.split(",")
    matrix = render_matrix(args.manifest)
    minimum_headroom_mib = args.minimum_headroom_mib or int(matrix["resource_profile"]["minimum_gpu_headroom_mib"])
    base = render_base(args.manifest)
    by_id = {run["id"]: run for run in matrix["runs"]}
    if any(run_id not in by_id for run_id in run_ids):
        raise SystemExit("unknown matrix run id")
    runs = [by_id[run_id] for run_id in run_ids]
    if any(run["phase"] != "stage2" for run in runs):
        raise SystemExit("matrix memory probe supports Stage2 runs only")

    run_root = args.scratch_root / ("probe-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_root.mkdir(parents=True)
    splits = split_workload(run_root)
    deadline = int(base["calibration_policy"]["validation_deadline_seconds"])
    reports = []
    reused = {}
    for value in args.reuse_repetition:
        run_id, separator, path = value.partition("=")
        if not separator:
            raise SystemExit("reuse repetition must use RUN_ID=PATH")
        reused[run_id] = Path(path)
    for run in runs:
        repetitions = []
        if run["id"] in reused:
            result = json.loads(reused[run["id"]].read_text())
            repetitions.append(qualify_matrix_repetition(result))
        for repetition in range(1, args.repetitions + 1 if not repetitions else 1):
            result = qualify_matrix_repetition(
                run_repetition(
                    base,
                    "stage2",
                    repetition,
                    run_root,
                    splits,
                    deadline,
                    environment_overrides=run_environment(run, matrix["resource_profile"]),
                    repetition_label=run["id"],
                )
            )
            repetitions.append(result)
            if result["status"] != "passed":
                break
        reports.append(summarize(run, repetitions, minimum_headroom_mib))
        if reports[-1]["status"] != "passed":
            break

    report = {
        "schema_version": 1,
        "result_type": "stage123_matrix_gpu_memory_probe",
        "manifest": str(args.manifest),
        "manifest_sha256": matrix["manifest_sha256"],
        "training_steps": 0,
        "optimizer_enabled": False,
        "rollout_gpu_memory_utilization": matrix["resource_profile"]["rollout_gpu_memory_utilization"],
        "rollout_max_num_batched_tokens": matrix["resource_profile"]["rollout_max_num_batched_tokens"],
        "rollout_free_cache_engine": matrix["resource_profile"]["rollout_free_cache_engine"],
        "rollout_enable_sleep_mode": matrix["resource_profile"]["rollout_enable_sleep_mode"],
        "ref_fsdp_offload": matrix["resource_profile"]["ref_fsdp_offload"],
        "actor_optimizer_offload": matrix["resource_profile"]["actor_optimizer_offload"],
        "actor_param_offload": matrix["resource_profile"]["actor_param_offload"],
        "minimum_gpu_headroom_mib": matrix["resource_profile"]["minimum_gpu_headroom_mib"],
        "ref_log_prob_micro_batch_size": matrix["resource_profile"]["ref_log_prob_micro_batch_size"],
        "ref_log_prob_max_token_len_per_gpu": matrix["resource_profile"]["ref_log_prob_max_token_len_per_gpu"],
        "model1_identity": {
            key: runs[0]["source"][key]
            for key in (
                "model1_path",
                "model1_config_sha256",
                "model1_tokenizer_config_sha256",
                "model1_chat_template_sha256",
                "model1_provenance_path",
                "model1_provenance_sha256",
            )
        },
        "run_root": str(run_root),
        "runs": reports,
        "status": "passed"
        if len(reports) == len(runs) and all(item["status"] == "passed" for item in reports)
        else "failed",
    }
    report_path = run_root / "matrix-memory-probe-report.json"
    write_json(report_path, report)
    write_json(
        args.scratch_root / "latest-matrix-memory-probe.json",
        {
            "schema_version": 1,
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "status": report["status"],
            "run_root": str(run_root),
        },
    )
    print(json.dumps({"ok": report["status"] == "passed", "report": str(report_path)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
